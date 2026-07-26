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

# time-series ops whose lookback is unbounded (any prior input affects every later output)
_TS_UNBOUNDED = frozenset({
    "cummax", "cumsum", "cumprod", "cummin", "ts_mean", "ts_std", "ts_min", "asof",
    "ewm_mean", "ewm_std", "ttm", "trailing", "resample",
    "to_annual", "to_quarterly", "to_monthly", "to_daily",
})
_TS_LAG = frozenset({"lag"})  # output t reads input t-k -> a dirty input t reaches output t+k
# windowed ops: output t reads inputs [t-w+1 .. t] -> a dirty input t reaches output t+(w-1)
_TS_WINDOWED = frozenset({
    "roll_mean", "roll_sum", "roll_std", "roll_var", "roll_max", "roll_min",
    "roll_quantile", "roll_median", "roll_skew", "decay_linear",
})


def forward_reach(name: str, args: tuple) -> int | float:
    """The maximum forward offset (in periods) from a dirty INPUT cell to a tainted OUTPUT cell:
    `lag(k)` -> k; a `roll_*`/`decay_linear` window of w -> w-1; unbounded ops -> INF; a
    duration/non-literal window -> INF (conservative). 0 for non-time-series ops."""
    spec = OPS.get(name)
    if spec is None or spec.axis != "time-series":
        return 0
    if name in _TS_UNBOUNDED:
        return INF
    k = None
    if (len(args) >= 2 and isinstance(args[1], ast.Literal)
            and isinstance(args[1].value, int) and not isinstance(args[1].value, bool)):
        k = int(args[1].value)
    if k is None:
        return INF  # duration string / computed window -> conservative whole-tail
    if name in _TS_LAG:
        return k
    return max(k - 1, 0)  # windowed op reaches t + (w-1)


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


def expand_timeseries(cells: set, reach, index: PanelIndex) -> set:
    """Entity-local forward expansion: a dirty input at (e, t) taints (e, t .. t+reach); reach == INF
    taints the whole tail of entity e. `reach` is the op's forward reach (see forward_reach)."""
    if reach == 0:
        return set(cells)
    out: set = set()
    for e, t in cells:
        ts = index.periods_by_entity.get(e)
        i = index.pos.get((e, t)) if ts is not None else None
        if i is None:
            out.add((e, t))
            continue
        end = len(ts) if reach == INF else min(len(ts), i + int(reach) + 1)
        for j in range(i, end):
            out.add((e, ts[j]))
    return out


def expand_group(cells: set, by_col, index: PanelIndex) -> set:
    """Cross-sectional expansion: a dirty (e, t) taints every entity sharing e's `by`-group at t;
    by_col is None -> every entity at period t (whole-period)."""
    out: set = set()
    for e, t in cells:
        if by_col is None:
            out.update((e2, t) for e2 in index.all_at.get(t, ()))
            continue
        g = index.cell_group.get(by_col, {}).get((e, t))
        members = index.group_members.get(by_col, {}).get((t, g))
        if members is None:
            out.add((e, t))
        else:
            out.update((e2, t) for e2 in members)
    return out


def cell_footprint(expr, dirty: dict, index: PanelIndex, locals: dict, memo: dict) -> set:
    """Output cells of `expr` tainted by the dirty INPUT cells in `dirty` (field-column -> cell set).
    Post-order walk; each op maps its children's cells per its axis rule."""
    if expr is None:
        return set()
    key = id(expr)
    if key in memo:
        return memo[key]
    match expr:
        case ast.Literal():
            result: set = set()
        case ast.FieldRef():
            result = set(dirty.get(expr.column, ()))
        case ast.NameRef():
            loc = locals.get(expr.name)
            result = cell_footprint(loc, dirty, index, locals, memo) if loc is not None else set()
        case ast.BinOp() | ast.Compare() | ast.BoolOp() | ast.Coalesce():
            result = (cell_footprint(expr.left, dirty, index, locals, memo)
                      | cell_footprint(expr.right, dirty, index, locals, memo))
        case ast.In():
            result = cell_footprint(expr.item, dirty, index, locals, memo)
            for o in expr.options:
                result |= cell_footprint(o, dirty, index, locals, memo)
        case ast.Not() | ast.Neg():
            result = cell_footprint(expr.operand, dirty, index, locals, memo)
        case ast.Ternary():
            result = (cell_footprint(expr.value, dirty, index, locals, memo)
                      | cell_footprint(expr.cond, dirty, index, locals, memo)
                      | cell_footprint(expr.orelse, dirty, index, locals, memo))
        case ast.Call():
            child: set = set()
            for a in expr.args:
                child |= cell_footprint(a, dirty, index, locals, memo)
            for _, v in expr.kwargs:
                child |= cell_footprint(v, dirty, index, locals, memo)
            axis = OPS[expr.name].axis if expr.name in OPS else "elementwise"
            if axis == "time-series":
                result = expand_timeseries(child, forward_reach(expr.name, expr.args), index)
            elif axis == "cross-sectional":
                result = expand_group(child, ".".join(expr.by) if expr.by else None, index)
            else:  # elementwise / model
                result = child
        case _:
            result = set()
    memo[key] = result
    return result


def model_footprint(decl, dirty: dict, index: PanelIndex) -> set:
    """Union of the footprints of a model's exports (locals resolved), or a signal's expression."""
    memo: dict = {}
    if isinstance(decl, ast.SignalDecl):
        return cell_footprint(decl.expr, dirty, index, {}, memo)
    locals = {s.name: s.expr for s in decl.statements
              if isinstance(s, ast.Assignment) and not s.export}
    out: set = set()
    for s in decl.statements:
        if isinstance(s, ast.Assignment) and s.export:
            expr = s.expr if s.expr is not None else locals.get(s.name)
            out |= cell_footprint(expr, dirty, index, locals, memo)
        elif isinstance(s, ast.ScoreDecl):  # a scored output column; cases + default are elementwise
            for c in s.cases:
                out |= cell_footprint(c.value, dirty, index, locals, memo)
                out |= cell_footprint(c.cond, dirty, index, locals, memo)
            out |= cell_footprint(s.default, dirty, index, locals, memo)
    return out


def replace_rows(stored: pl.DataFrame, fresh: pl.DataFrame, keys: set) -> pl.DataFrame:
    """Overwrite exactly the (entity, time) rows in `keys` with `fresh`'s values; keep every other
    `stored` row. Only `keys` rows of `fresh` are used (non-footprint fresh rows are discarded); a
    `keys` row absent from `fresh` (e.g. on_missing skip) is simply dropped, as a full recompute would."""
    if not keys:
        return stored
    kf = pl.DataFrame(
        {ENTITY_COL: [e for e, _ in keys], TIME_COL: [t for _, t in keys]}
    ).with_columns(pl.col(TIME_COL).cast(stored.schema[TIME_COL]))
    kept = stored.join(kf, on=[ENTITY_COL, TIME_COL], how="anti")
    take = fresh.join(kf, on=[ENTITY_COL, TIME_COL], how="semi")
    return pl.concat([kept, take.select(stored.columns)], how="vertical").sort([ENTITY_COL, TIME_COL])
