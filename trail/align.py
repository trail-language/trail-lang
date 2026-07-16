"""Cross-source, cross-frequency alignment.

Each source produces a panel at its own native frequency. To compute across them, every
source panel is brought onto a common **target-frequency grid** and merged on `(entity, time)`:

- a source **finer than or equal to** the target is **downsampled** - each target bucket is
  reduced by the field's `kind` (flow -> sum, stock/level/price/index/meta -> last,
  rate/ratio -> mean, return -> compound);
- a source **coarser than** the target is **upsampled by a backward as-of join** - the last
  value known as of each target instant is carried forward (lookahead-safe).

`time` is the period-end instant of each bucket, so all sources land on the same canonical grid.
"""
from __future__ import annotations

import warnings

import polars as pl

from trail.ast import FREQUENCIES
from trail.config import ConfigError
from trail.ops import _AGG, AGG_FOR_KIND, FREQ_DUR
from trail.schema import kind_of
from trail.source import BROADCAST_ENTITY, ENTITY_COL, TIME_COL

# coarse -> fine; shared with the parser/loader via trail.ast
FREQ_ORDER = list(FREQUENCIES)


def _canonical(col: str) -> str:
    """Strip a frequency prefix from a physical column so kind lookup uses the canonical field
    (annual.balance.total_assets -> balance.total_assets), else the column unchanged."""
    head, _, rest = col.partition(".")
    return rest if head in FREQ_ORDER and rest.count(".") >= 1 else col

# kinds whose per-period value must not be repeated onto a finer grid (a total/return mis-scales).
_UPSAMPLE_UNSAFE = {"flow", "return", "per_share"}

# a coarse entity dimension -> the grid meta field that maps a canonical entity to that dimension's
# key. A source keyed by "country" is remapped onto stocks through each stock's meta.country.
_DIM_MAP_COL = {"country": "meta.country"}


class AlignmentWarning(UserWarning):
    """A cross-source alignment produced a result that may be semantically surprising."""


def finest(freqs: list[str]) -> str:
    return max(freqs, key=FREQ_ORDER.index)


def _finer_or_equal(a: str, b: str) -> bool:
    return FREQ_ORDER.index(a) >= FREQ_ORDER.index(b)


def _period_end(freq: str) -> pl.Expr:
    """The canonical period-end instant of the target bucket containing each time."""
    dur = FREQ_DUR[freq]
    return pl.col(TIME_COL).dt.truncate(dur).dt.offset_by(dur).dt.offset_by("-1d")


def _kind_agg(field: str) -> pl.Expr:
    """Reduce a field's values within a target bucket by its kind (§4.4 automatic rule)."""
    agg = AGG_FOR_KIND.get(kind_of(_canonical(field)) or "flow", "last")
    c = pl.col(field)
    if agg == "last":  # order-dependent: take the latest by time, not group arrival order
        return c.sort_by(pl.col(TIME_COL)).last().alias(field)
    return _AGG[agg](c).alias(field)


def _fields(panel: pl.DataFrame) -> list[str]:
    return [c for c in panel.columns if c not in (ENTITY_COL, TIME_COL)]


def _downsample(panel: pl.DataFrame, target_freq: str) -> pl.DataFrame:
    fields = _fields(panel)
    return (
        panel.group_by([pl.col(ENTITY_COL), _period_end(target_freq).alias(TIME_COL)])
        .agg([_kind_agg(f) for f in fields])
        .sort([ENTITY_COL, TIME_COL])
    )


def _upsample_asof(panel: pl.DataFrame, grid: pl.DataFrame) -> pl.DataFrame:
    """Carry each coarse value forward onto the finer target grid (backward as-of, per entity)."""
    right = panel.sort([ENTITY_COL, TIME_COL])
    # both sides sorted on (entity, time) just above; skip polars' unverifiable re-check under `by`
    return grid.join_asof(right, on=TIME_COL, by=ENTITY_COL, strategy="backward", check_sortedness=False)


def _warn_upsample_flow(panel: pl.DataFrame, stacklevel: int = 3) -> None:
    for f in _fields(panel):
        k = kind_of(_canonical(f))
        if k in _UPSAMPLE_UNSAFE:
            warnings.warn(
                f"W-UPSAMPLE-FLOW field '{f}' (kind {k}) is carried forward by as-of onto a finer "
                f"grid; repeating a per-period {k} value mis-scales it - use an explicit resample.",
                AlignmentWarning, stacklevel=stacklevel,
            )


def is_broadcast(panel: pl.DataFrame) -> bool:
    """A panel whose entity axis is entirely the sentinel is a global broadcast series.

    A panel mixing the sentinel with real entities violates the source contract; reject it
    loudly rather than silently treating `*` as one more entity.
    """
    if panel.height == 0:
        return False
    ents = set(panel.get_column(ENTITY_COL).unique().to_list())
    if BROADCAST_ENTITY in ents and ents != {BROADCAST_ENTITY}:
        raise ConfigError(
            f"E-BROADCAST-MIXED a source panel mixes the broadcast sentinel '{BROADCAST_ENTITY}' "
            "with real entities; a broadcast series must have no other entity"
        )
    return ents == {BROADCAST_ENTITY}


def _align_broadcast(panel: pl.DataFrame, native: str, target_freq: str, grid: pl.DataFrame) -> pl.DataFrame:
    """Replicate a single-entity (sentinel) series across every grid entity by time alignment.

    The value in effect at each grid instant is joined onto every entity at that instant - a
    global macro rate meeting a per-entity grid. No `by`: the series has no entity axis of its own.
    """
    body = panel.drop(ENTITY_COL)  # time + fields only
    fields = [c for c in body.columns if c != TIME_COL]
    if _finer_or_equal(native, target_freq):
        agg = body.group_by(_period_end(target_freq).alias(TIME_COL)).agg([_kind_agg(f) for f in fields])
        return grid.join(agg, on=TIME_COL, how="left")
    _warn_upsample_flow(panel, stacklevel=4)  # +1 frame: this runs inside _align_broadcast
    right = body.sort(TIME_COL)
    return grid.sort(TIME_COL).join_asof(right, on=TIME_COL, strategy="backward", check_sortedness=False)


def _align_by_dimension(panel: pl.DataFrame, native: str, target_freq: str,
                        out: pl.DataFrame, dim: str) -> pl.DataFrame:
    """Remap a foreign-dimension source (e.g. country) onto canonical entities.

    The source's `entity` column holds the dimension key; `out` carries a bridge meta column
    (meta.country) giving each entity's key. Each entity gets the foreign row for its key,
    time-aligned (as-of for a coarser source). The map lives in the data plane (§5.4).
    """
    map_col = _DIM_MAP_COL.get(dim)
    if map_col is None:
        raise ConfigError(f"E-DIM-UNKNOWN source declares unsupported entity dimension '{dim}'")
    if map_col not in out.columns:
        raise ConfigError(
            f"E-DIM-UNMAPPED a source is keyed by dimension '{dim}' but no entity-bearing source "
            f"provides the bridge field '{map_col}'; add it to a source that supplies your entities"
        )
    body = panel.rename({ENTITY_COL: map_col})  # the foreign entity is the map key
    fields = [c for c in body.columns if c not in (map_col, TIME_COL)]
    if _finer_or_equal(native, target_freq):
        red = (body.group_by([pl.col(map_col), _period_end(target_freq).alias(TIME_COL)])
                   .agg([_kind_agg(f) for f in fields]))
        return out.join(red, on=[map_col, TIME_COL], how="left").select([ENTITY_COL, TIME_COL, *fields])
    _warn_upsample_flow(panel, stacklevel=4)  # +1 frame: runs inside _align_by_dimension
    left = out.select([ENTITY_COL, TIME_COL, map_col]).sort([map_col, TIME_COL])
    right = body.sort([map_col, TIME_COL])
    return (left.join_asof(right, on=TIME_COL, by=map_col, strategy="backward", check_sortedness=False)
                .select([ENTITY_COL, TIME_COL, *fields]))


def _left_join_fields(out: pl.DataFrame, aligned: pl.DataFrame) -> pl.DataFrame:
    cols = [c for c in aligned.columns if c not in (ENTITY_COL, TIME_COL)]
    return out.join(aligned.select([ENTITY_COL, TIME_COL, *cols]), on=[ENTITY_COL, TIME_COL], how="left")


def _unpack(item: tuple) -> tuple[pl.DataFrame, str, str]:
    """Accept (panel, freq) or (panel, freq, entity_dim); default dim to 'entity'."""
    panel, freq, *rest = item
    return panel, freq, (rest[0] if rest else "entity")


def align_and_merge(loaded: list[tuple], target_freq: str) -> pl.DataFrame:
    """Align each source panel to `target_freq` and merge on `(entity, time)`.

    Items are `(panel, native_freq[, entity_dim])`. Field sets are disjoint (assigned by
    precedence before loading), so each source is a clean left-join. Three source classes,
    merged in order so each pass sees what it needs:

    - **canonical** (entity-keyed) define the grid and are downsampled / as-of upsampled;
    - **broadcast** (sentinel `*`) are replicated across every entity by time alignment;
    - **foreign-dimension** (e.g. country) are remapped onto entities via a bridge meta column,
      which the canonical pass must populate first.
    """
    items = [_unpack(x) for x in loaded]
    canonical = [(p, f) for p, f, d in items if d == "entity" and not is_broadcast(p)]
    broadcast = [(p, f) for p, f, d in items if d == "entity" and is_broadcast(p)]
    foreign = [(p, f, d) for p, f, d in items if d != "entity"]
    # with no entity-keyed source, the foreign dimension IS the entity axis (e.g. a lone
    # country model): treat those panels as canonical rather than remapping onto nothing.
    # A broadcast source may still ride along (it replicates onto whatever axis results).
    # Coherent only for a single dimension; two distinct foreign axes have no common entity.
    if not canonical and foreign:
        dims = {d for _, _, d in foreign}
        if len(dims) > 1:
            raise ConfigError(
                f"E-DIM-AMBIGUOUS no entity-bearing source, and foreign sources span multiple "
                f"dimensions {sorted(dims)}; there is no single entity axis to compute on"
            )
        canonical = [(p, f) for p, f, _ in foreign]
        foreign = []

    pe = _period_end(target_freq)
    native = [p for p, f in canonical if _finer_or_equal(f, target_freq)]
    grid_src = native or [p for p, _ in canonical]  # coarser real-entity sources still define a grid
    if not grid_src:  # every source is a global broadcast: nothing to broadcast onto
        raise ConfigError(
            "E-NO-ENTITY every source providing these fields is a global broadcast series; "
            "a model needs at least one entity-bearing source to compute on"
        )
    grid = (
        pl.concat([p.select([pl.col(ENTITY_COL), pe.alias(TIME_COL)]) for p in grid_src])
        .unique()
        .sort([ENTITY_COL, TIME_COL])
    )
    out = grid
    for panel, nfreq in canonical:  # pass 1: entity-keyed sources (also populate any bridge column)
        if _finer_or_equal(nfreq, target_freq):
            aligned = _downsample(panel, target_freq)
        else:
            _warn_upsample_flow(panel)
            aligned = _upsample_asof(panel, grid)
        out = _left_join_fields(out, aligned)
    for panel, nfreq in broadcast:  # pass 2: global series replicated across entities
        out = _left_join_fields(out, _align_broadcast(panel, nfreq, target_freq, grid))
    for panel, nfreq, dim in foreign:  # pass 3: foreign dimensions need the bridge column present
        out = _left_join_fields(out, _align_by_dimension(panel, nfreq, target_freq, out, dim))
    return out.sort([ENTITY_COL, TIME_COL])
