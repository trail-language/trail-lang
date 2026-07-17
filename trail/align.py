"""Cross-source, cross-frequency alignment (point-in-time aware).

Each source produces a panel at its own native frequency. To compute across them, every
source panel is brought onto a common **target-frequency grid** and merged on `(entity, time)`:

- a source **finer than or equal to** the target is **downsampled** - each target bucket is
  reduced by the field's `kind` (flow -> sum, stock/level/price/index/meta -> last,
  rate/ratio -> mean, return -> compound);
- a source **coarser than** the target is **upsampled by a backward as-of join** - the last
  value known as of each target instant is carried forward (lookahead-safe).

`time` is the period-end instant of each bucket (the decision calendar). A field may carry a
separate **alignment coordinate** - a reserved `__date:*` column giving when the value became
knowable (filing date, trade date, ...). When present, the value is PLACED by its coordinate
(row-shift), not its period-end: FY2023 filed 2024-02 lands on the annual-2024 row. Coordinates
are consumed here and never reach the compiler. A field with no coordinate is naive
(coordinate = `time`), which reduces every op to its pre-PIT behavior byte-for-byte.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import polars as pl

from trail.ast import FREQUENCIES
from trail import fieldname
from trail.config import ConfigError
from trail.ops import _AGG, AGG_FOR_KIND, FREQ_DUR
from trail.schema import kind_of
from trail.source import BROADCAST_ENTITY, ENTITY_COL, TIME_COL, is_date_col

# coarse -> fine; shared with the parser/loader via trail.ast
FREQ_ORDER = list(FREQUENCIES)


_canonical = fieldname.canonical  # strip qualifiers for kind lookup (see trail.fieldname)

# kinds whose per-period value must not be repeated onto a finer grid (a total/return mis-scales).
_UPSAMPLE_UNSAFE = {"flow", "return", "per_share"}

# a coarse entity dimension -> the grid meta field that maps a canonical entity to that dimension's
# key. A source keyed by "country" is remapped onto stocks through each stock's meta.country.
_DIM_MAP_COL = {"country": "meta.country"}

_DECISION = "__decision"  # transient row-shift label column (never leaves align)


class AlignmentWarning(UserWarning):
    """A cross-source alignment produced a result that may be semantically surprising."""


@dataclass(frozen=True)
class LoadedPanel:
    """A source panel ready for alignment: its native frequency, entity dimension, and the
    per-field alignment coordinate map (``final_col -> __date:col`` or ``None`` = naive)."""

    panel: pl.DataFrame
    freq: str
    dim: str = "entity"
    coord_map: dict = field(default_factory=dict)

    def coord_of(self, fld: str) -> str:
        """The physical coordinate column for a field, or ``time`` when the field is naive."""
        return self.coord_map.get(fld) or TIME_COL


def _as_loaded(item) -> LoadedPanel:
    """Accept a LoadedPanel or a legacy ``(panel, freq[, dim])`` tuple (all-naive coord_map)."""
    if isinstance(item, LoadedPanel):
        return item
    panel, freq, *rest = item
    return LoadedPanel(panel, freq, rest[0] if rest else "entity", {})


def finest(freqs: list[str]) -> str:
    return max(freqs, key=FREQ_ORDER.index)


def _finer_or_equal(a: str, b: str) -> bool:
    return FREQ_ORDER.index(a) >= FREQ_ORDER.index(b)


def _period_end(freq: str, col: str = TIME_COL) -> pl.Expr:
    """The canonical period-end instant of the target bucket containing each value of `col`."""
    dur = FREQ_DUR[freq]
    return pl.col(col).dt.truncate(dur).dt.offset_by(dur).dt.offset_by("-1d")


def _kind_agg(field: str) -> pl.Expr:
    """Reduce a field's values within a bucket by its kind (§4.4). `last` takes the value of the
    latest economic period `time`. Same-period restatements are collapsed by the coordinate
    upstream (`_reduce_shift` pre-dedups keep-latest), so `time` alone orders correctly here."""
    agg = AGG_FOR_KIND.get(kind_of(_canonical(field)) or "flow", "last")
    c = pl.col(field)
    if agg == "last":  # order-dependent: latest economic period, not group arrival order
        return c.sort_by(pl.col(TIME_COL)).last().alias(field)
    return _AGG[agg](c).alias(field)


def _fields(panel: pl.DataFrame) -> list[str]:
    """Value fields: everything but the index and the reserved `__date:*` coordinates."""
    return [c for c in panel.columns if c not in (ENTITY_COL, TIME_COL) and not is_date_col(c)]


def _coord_groups(lp: LoadedPanel):
    """Split a panel into (coordinate, sub-panel) groups - fields sharing a coordinate align
    together. Deterministic order (by coordinate name). A naive panel yields one `time` group
    equal to the whole panel, so the caller's per-group path reduces to the pre-PIT single path."""
    groups: dict[str, list[str]] = {}
    for f in _fields(lp.panel):
        groups.setdefault(lp.coord_of(f), []).append(f)
    for coord in sorted(groups):
        gfields = groups[coord]  # source column order (a naive single group == the panel's order)
        cols = [ENTITY_COL, TIME_COL, *gfields]
        if coord != TIME_COL:
            cols.append(coord)
        yield coord, lp.panel.select(cols)


def _fill_null_coords(lp: LoadedPanel) -> LoadedPanel:
    """A value with no known-date falls back to its period-end (naive for that row). Done on the
    whole panel up front so grid-seeding and group-splitting both see non-null coordinates."""
    fills = []
    for c in (c for c in lp.panel.columns if is_date_col(c)):
        n = lp.panel.get_column(c).null_count()
        if n:
            warnings.warn(
                f"W-PIT-PARTIAL {n} row(s) carry no '{c}' coordinate; placing those at period-end "
                "(naive)", AlignmentWarning, stacklevel=3)
            fills.append(pl.coalesce([pl.col(c), pl.col(TIME_COL)]).alias(c))
    return LoadedPanel(lp.panel.with_columns(fills), lp.freq, lp.dim, lp.coord_map) if fills else lp


def _reduce_shift(body: pl.DataFrame, target_freq: str, coord: str, keys: list[str]) -> pl.DataFrame:
    """Fiscal-bucket + kind-reduce, shifting the row to the decision bucket under PIT.

    Group by the economic period-end (`keys` + `period_end(time)`) so a fiscal period's rows
    aggregate together; reduce each field by kind. Naive (coord = time): that IS the result.
    PIT (coord != time): also carry `max(coordinate)` per bucket, RELABEL the row to
    `period_end(max coordinate)` (the instant the bucket became fully known), and re-reduce any
    buckets that a late filing bunches into the same decision bucket. A same-period restatement
    is pre-deduped (keep latest coordinate) so it replaces rather than double-sums.
    """
    fields = [c for c in body.columns if c not in ({TIME_COL, coord} | set(keys)) and not is_date_col(c)]
    kcols = [pl.col(k) for k in keys]
    pe_time = _period_end(target_freq).alias(TIME_COL)
    if coord == TIME_COL:
        return body.group_by([*kcols, pe_time]).agg([_kind_agg(f) for f in fields])
    dedup = body.sort([*keys, TIME_COL, coord]).unique(subset=[*keys, TIME_COL], keep="last")
    fiscal = dedup.group_by([*kcols, pe_time]).agg(
        [_kind_agg(f) for f in fields] + [pl.col(coord).max().alias(coord)])
    shifted = fiscal.with_columns(_period_end(target_freq, coord).alias(_DECISION))
    return (shifted.group_by([*kcols, pl.col(_DECISION)])
                   .agg([_kind_agg(f) for f in fields])
                   .rename({_DECISION: TIME_COL}))


def _asof_coord(left: pl.DataFrame, right_body: pl.DataFrame, coord: str, by: str | None) -> pl.DataFrame:
    """Backward as-of: carry each right value onto `left` at instants >= its coordinate.

    `left` is keyed by `time` (the grid); the right value is placed by `coord` (its known-date),
    so it is invisible until knowable. `by` is the join partition (entity / bridge key / None).
    The right panel's own period-end `time` is dropped (irrelevant post-placement) and the
    coordinate column is dropped from the result.
    """
    on_kwargs = {"by": by} if by else {}
    if coord == TIME_COL:
        r = right_body.sort([*([by] if by else []), TIME_COL])
        left = left if by else left.sort(TIME_COL)
        return left.join_asof(r, on=TIME_COL, strategy="backward", check_sortedness=False, **on_kwargs)
    r = right_body.drop(TIME_COL).sort([*([by] if by else []), coord])
    left = left if by else left.sort(TIME_COL)
    j = left.join_asof(r, left_on=TIME_COL, right_on=coord, strategy="backward",
                       check_sortedness=False, **on_kwargs)
    return j.drop(coord) if coord in j.columns else j


def _warn_upsample_flow(fields: list[str], stacklevel: int = 3) -> None:
    for f in fields:
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


def _downsample_group(sub: pl.DataFrame, target_freq: str, coord: str) -> pl.DataFrame:
    return _reduce_shift(sub, target_freq, coord, [ENTITY_COL]).sort([ENTITY_COL, TIME_COL])


def _align_broadcast(sub: pl.DataFrame, native: str, target_freq: str, grid: pl.DataFrame,
                     coord: str) -> pl.DataFrame:
    """Replicate a single-entity (sentinel) series across every grid entity by time alignment.

    The value in effect at each grid instant is joined onto every entity at that instant - a
    global macro rate meeting a per-entity grid. No `by`: the series has no entity axis of its own.
    """
    body = sub.drop(ENTITY_COL)  # time (+coord) + fields
    fields = [c for c in body.columns if c not in (TIME_COL, coord) and not is_date_col(c)]
    if _finer_or_equal(native, target_freq):
        red = _reduce_shift(body, target_freq, coord, [])
        return grid.join(red, on=TIME_COL, how="left")
    _warn_upsample_flow(fields, stacklevel=5)  # +2 frames: runs inside _align_broadcast/group loop
    return _asof_coord(grid, body, coord, by=None)


def _align_by_dimension(sub: pl.DataFrame, native: str, target_freq: str,
                        out: pl.DataFrame, dim: str, coord: str) -> pl.DataFrame:
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
    body = sub.rename({ENTITY_COL: map_col})  # the foreign entity is the map key
    fields = [c for c in body.columns if c not in (map_col, TIME_COL, coord) and not is_date_col(c)]
    if _finer_or_equal(native, target_freq):
        red = _reduce_shift(body, target_freq, coord, [map_col])
        return out.join(red, on=[map_col, TIME_COL], how="left").select([ENTITY_COL, TIME_COL, *fields])
    _warn_upsample_flow(fields, stacklevel=5)
    left = out.select([ENTITY_COL, TIME_COL, map_col]).sort([map_col, TIME_COL])
    joined = _asof_coord(left, body, coord, by=map_col)
    return joined.select([ENTITY_COL, TIME_COL, *fields])


def _left_join_fields(out: pl.DataFrame, aligned: pl.DataFrame) -> pl.DataFrame:
    cols = [c for c in aligned.columns if c not in (ENTITY_COL, TIME_COL)]
    return out.join(aligned.select([ENTITY_COL, TIME_COL, *cols]), on=[ENTITY_COL, TIME_COL], how="left")


def align_and_merge(loaded: list, target_freq: str) -> pl.DataFrame:
    """Align each source panel to `target_freq` and merge on `(entity, time)`.

    Items are :class:`LoadedPanel` (or legacy ``(panel, native_freq[, entity_dim])`` tuples,
    treated as all-naive). Field sets are disjoint (assigned by precedence before loading), so
    each source is a clean left-join. Three source classes, merged in order so each pass sees
    what it needs:

    - **canonical** (entity-keyed) define the grid and are downsampled / as-of upsampled;
    - **broadcast** (sentinel `*`) are replicated across every entity by time alignment;
    - **foreign-dimension** (e.g. country) are remapped onto entities via a bridge meta column,
      which the canonical pass must populate first.

    Each panel is split into per-coordinate groups; a group with a `__date:*` coordinate places
    its values by knowability (row-shift), a naive group behaves exactly as before PIT.
    """
    items = [_fill_null_coords(_as_loaded(x)) for x in loaded]
    canonical = [lp for lp in items if lp.dim == "entity" and not is_broadcast(lp.panel)]
    broadcast = [lp for lp in items if lp.dim == "entity" and is_broadcast(lp.panel)]
    foreign = [lp for lp in items if lp.dim != "entity"]
    # with no entity-keyed source, the foreign dimension IS the entity axis (e.g. a lone
    # country model): treat those panels as canonical rather than remapping onto nothing.
    # A broadcast source may still ride along (it replicates onto whatever axis results).
    # Coherent only for a single dimension; two distinct foreign axes have no common entity.
    if not canonical and foreign:
        dims = {lp.dim for lp in foreign}
        if len(dims) > 1:
            raise ConfigError(
                f"E-DIM-AMBIGUOUS no entity-bearing source, and foreign sources span multiple "
                f"dimensions {sorted(dims)}; there is no single entity axis to compute on"
            )
        canonical = [LoadedPanel(lp.panel, lp.freq, "entity", lp.coord_map) for lp in foreign]
        foreign = []

    native = [lp for lp in canonical if _finer_or_equal(lp.freq, target_freq)]
    if not native and canonical:
        finest_avail = finest([lp.freq for lp in canonical])
        warnings.warn(
            f"W-GRID-COARSER no source natively populates target frequency '{target_freq}'; "
            f"the grid is {finest_avail} rows wearing {target_freq} bucket labels - one step of "
            f"lag()/roll_*() is one {finest_avail} period, not one {target_freq} period",
            AlignmentWarning, stacklevel=2,
        )
    grid_src = native or canonical  # coarser real-entity sources still define a grid
    if not grid_src:  # every source is a global broadcast: nothing to broadcast onto
        raise ConfigError(
            "E-NO-ENTITY every source providing these fields is a global broadcast series; "
            "a model needs at least one entity-bearing source to compute on"
        )
    # seed the grid from each field's OWN coordinate period-end (a shifted row must have somewhere
    # to land); naive fields seed period_end(time), so the grid reduces to the pre-PIT grid.
    grid_frames = []
    for lp in grid_src:
        # a panel with no value fields still seeds its period-end (`or {TIME_COL}`)
        for coord in ({lp.coord_of(f) for f in _fields(lp.panel)} or {TIME_COL}):
            grid_frames.append(lp.panel.select(
                [pl.col(ENTITY_COL), _period_end(target_freq, coord).alias(TIME_COL)]))
    grid = pl.concat(grid_frames).unique().sort([ENTITY_COL, TIME_COL])

    out = grid
    for lp in canonical:  # pass 1: entity-keyed sources (also populate any bridge column)
        for coord, sub in _coord_groups(lp):
            if _finer_or_equal(lp.freq, target_freq):
                aligned = _downsample_group(sub, target_freq, coord)
            else:
                _warn_upsample_flow(_fields(sub), stacklevel=3)
                aligned = _asof_coord(grid, sub, coord, by=ENTITY_COL)
            out = _left_join_fields(out, aligned)
    for lp in broadcast:  # pass 2: global series replicated across entities
        for coord, sub in _coord_groups(lp):
            out = _left_join_fields(out, _align_broadcast(sub, lp.freq, target_freq, grid, coord))
    for lp in foreign:  # pass 3: foreign dimensions need the bridge column present
        for coord, sub in _coord_groups(lp):
            out = _left_join_fields(out, _align_by_dimension(sub, lp.freq, target_freq, out, lp.dim, coord))
    return out.sort([ENTITY_COL, TIME_COL])
