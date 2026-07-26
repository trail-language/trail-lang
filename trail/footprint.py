"""Static footprint engine: given the dirty INPUT cells of an expression, compute the exact set of
OUTPUT (entity, period) cells that must be recomputed, by walking the post-expand_program op-graph.

Per-op rule (keyed on OPS[name].axis): elementwise/model -> identity; time-series -> entity-local
forward window; cross-sectional -> whole `by`-group at the period. See
docs 2026-07-27-tracked-views-p2-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from trail import ast
from trail.ops import OPS
from trail.source import ENTITY_COL, TIME_COL

INF = float("inf")

# time-series ops whose lookback is unbounded / whole-entity-tail (no fixed int window arg)
_UNBOUNDED_TS = frozenset({
    "cummax", "cumsum", "cumprod", "cummin", "ts_mean", "ts_std", "ts_min", "asof",
    "ttm", "trailing", "resample", "to_annual", "to_quarterly", "to_monthly", "to_daily",
})


def window_of(name: str, args: tuple) -> int | float:
    """Static forward window for a time-series op: an int-literal count, or INF for unbounded /
    duration-string / non-literal windows (conservative). 0 for non-time-series ops."""
    spec = OPS.get(name)
    if spec is None or spec.axis != "time-series":
        return 0
    if name in _UNBOUNDED_TS:
        return INF
    if (len(args) >= 2 and isinstance(args[1], ast.Literal)
            and isinstance(args[1].value, int) and not isinstance(args[1].value, bool)):
        return int(args[1].value)
    return INF  # duration string / computed arg -> conservative whole-tail


@dataclass
class PanelIndex:
    periods_by_entity: dict   # entity -> [time, ...] sorted ascending
    pos: dict                 # (entity, time) -> index into that list
    all_at: dict              # time -> [entity, ...]  (used when by is None)
    cell_group: dict          # by_col -> {(entity, time): groupval}
    group_members: dict       # by_col -> {(time, groupval): [entity, ...]}


def build_index(panel: pl.DataFrame, by_cols: set[str]) -> PanelIndex:
    """Precompute the per-entity period lists and per-period group membership the propagation needs."""
    ents = panel.get_column(ENTITY_COL).to_list()
    times = panel.get_column(TIME_COL).to_list()
    periods_by_entity: dict = {}
    all_at: dict = {}
    for e, t in zip(ents, times):
        periods_by_entity.setdefault(e, []).append(t)
        all_at.setdefault(t, []).append(e)
    for e in periods_by_entity:
        periods_by_entity[e].sort()
    pos = {(e, t): i for e, ts in periods_by_entity.items() for i, t in enumerate(ts)}
    cell_group: dict = {}
    group_members: dict = {}
    for col in by_cols:
        if col not in panel.columns:
            continue
        vals = panel.get_column(col).to_list()
        cg: dict = {}
        gm: dict = {}
        for e, t, g in zip(ents, times, vals):
            cg[(e, t)] = g
            gm.setdefault((t, g), []).append(e)
        cell_group[col] = cg
        group_members[col] = gm
    return PanelIndex(periods_by_entity, pos, all_at, cell_group, group_members)
