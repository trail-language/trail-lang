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

import polars as pl

from trail.ops import FREQ_DUR
from trail.schema import kind_of
from trail.source import ENTITY_COL, TIME_COL

# coarse -> fine
FREQ_ORDER = ["annual", "quarterly", "monthly", "weekly", "daily", "hourly", "minute"]


def finest(freqs: list[str]) -> str:
    return max(freqs, key=FREQ_ORDER.index)


def _finer_or_equal(a: str, b: str) -> bool:
    return FREQ_ORDER.index(a) >= FREQ_ORDER.index(b)


def _period_end(freq: str) -> pl.Expr:
    """The canonical period-end instant of the target bucket containing each time."""
    dur = FREQ_DUR[freq]
    return pl.col(TIME_COL).dt.truncate(dur).dt.offset_by(dur).dt.offset_by("-1d")


def _kind_agg(field: str) -> pl.Expr:
    """Reduce a field's values within a target bucket according to its kind."""
    k = kind_of(field) or "flow"
    c = pl.col(field)
    if k in ("flow", "per_share"):
        return c.sum().alias(field)
    if k in ("rate", "ratio"):
        return c.mean().alias(field)
    if k == "return":
        return ((c + 1).product() - 1).alias(field)
    return c.sort_by(pl.col(TIME_COL)).last().alias(field)  # stock/level/price/index/meta


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


def align_and_merge(loaded: list[tuple[pl.DataFrame, str]], target_freq: str) -> pl.DataFrame:
    """Align each `(panel, native_freq)` to `target_freq` and merge on `(entity, time)`.

    Field sets across sources are assumed disjoint (assigned by precedence before loading),
    so the merge is a clean left-join per source with no per-cell coalescing.
    """
    pe = _period_end(target_freq)
    # the grid is defined by sources at or finer than the target - the ones that natively
    # populate this frequency. Coarser sources are as-of enrichment and add no rows of their
    # own (a lone annual macro value must not stamp a phantom row onto a daily price grid).
    # Degenerate guard: if every source is coarser than the target, fall back to their union.
    native = [p for p, f in loaded if _finer_or_equal(f, target_freq)]
    grid_src = native or [p for p, _ in loaded]
    grid = (
        pl.concat([p.select([pl.col(ENTITY_COL), pe.alias(TIME_COL)]) for p in grid_src])
        .unique()
        .sort([ENTITY_COL, TIME_COL])
    )
    out = grid
    for panel, native in loaded:
        if _finer_or_equal(native, target_freq):
            aligned = _downsample(panel, target_freq)
        else:
            aligned = _upsample_asof(panel, grid)
        cols = [c for c in aligned.columns if c not in (ENTITY_COL, TIME_COL)]
        out = out.join(aligned.select([ENTITY_COL, TIME_COL, *cols]), on=[ENTITY_COL, TIME_COL], how="left")
    return out.sort([ENTITY_COL, TIME_COL])
