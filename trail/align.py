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

from trail.config import ConfigError
from trail.ops import _AGG, AGG_FOR_KIND, FREQ_DUR
from trail.schema import kind_of
from trail.source import BROADCAST_ENTITY, ENTITY_COL, TIME_COL

# coarse -> fine
FREQ_ORDER = ["annual", "quarterly", "monthly", "weekly", "daily", "hourly", "minute"]

# kinds whose per-period value must not be repeated onto a finer grid (a total/return mis-scales).
_UPSAMPLE_UNSAFE = {"flow", "return", "per_share"}


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
    agg = AGG_FOR_KIND.get(kind_of(field) or "flow", "last")
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
        k = kind_of(f)
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


def align_and_merge(loaded: list[tuple[pl.DataFrame, str]], target_freq: str) -> pl.DataFrame:
    """Align each `(panel, native_freq)` to `target_freq` and merge on `(entity, time)`.

    Field sets across sources are assumed disjoint (assigned by precedence before loading),
    so the merge is a clean left-join per source with no per-cell coalescing.
    """
    pe = _period_end(target_freq)
    # the grid is defined by real-entity sources at or finer than the target - the ones that
    # natively populate this frequency. Coarser sources are as-of enrichment and add no rows;
    # broadcast (sentinel-entity) sources have no entity axis, so they never define the grid
    # (a lone macro value must not stamp a phantom row onto a daily price grid, and the sentinel
    # '*' must never surface as a real output entity).
    non_bcast = [(p, f) for p, f in loaded if not is_broadcast(p)]
    native = [p for p, f in non_bcast if _finer_or_equal(f, target_freq)]
    grid_src = native or [p for p, _ in non_bcast]  # coarser real-entity sources still define a grid
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
    for panel, nfreq in loaded:
        if is_broadcast(panel):
            aligned = _align_broadcast(panel, nfreq, target_freq, grid)
        elif _finer_or_equal(nfreq, target_freq):
            aligned = _downsample(panel, target_freq)
        else:
            _warn_upsample_flow(panel)
            aligned = _upsample_asof(panel, grid)
        cols = [c for c in aligned.columns if c not in (ENTITY_COL, TIME_COL)]
        out = out.join(aligned.select([ENTITY_COL, TIME_COL, *cols]), on=[ENTITY_COL, TIME_COL], how="left")
    return out.sort([ENTITY_COL, TIME_COL])
