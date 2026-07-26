import datetime as dt

import polars as pl

from trail import ast
from trail.footprint import (
    INF, build_index, cell_footprint, expand_group, expand_timeseries, expand_whole_entity,
    forward_reach, model_footprint,
)
from trail.pipeline import prepare


def test_forward_reach_literal_and_unbounded():
    fx = ast.FieldRef(("x",))
    assert forward_reach("lag", (fx, ast.Literal(2))) == 2          # output t reads t-2 -> reach 2
    assert forward_reach("roll_mean", (fx, ast.Literal(4))) == 3    # window 4 -> reaches t+(w-1)
    assert forward_reach("roll_mean", (fx, ast.Literal(1))) == 0    # window 1 = self
    assert forward_reach("ts_mean", (fx,)) == INF
    assert forward_reach("ewm_mean", (fx, ast.Literal(3))) == INF   # exponential -> unbounded
    assert forward_reach("roll_mean", (fx, ast.Literal("1y"))) == INF   # duration -> conservative
    assert forward_reach("zscore", (fx,)) == 0                      # not time-series


def _panel3():
    t = [dt.datetime(2019, 12, 31), dt.datetime(2020, 12, 31), dt.datetime(2021, 12, 31)]
    panel = pl.DataFrame({
        "entity": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "time": t * 3,
        "meta.sector": ["Tech"] * 6 + ["Energy"] * 3,
    })
    return panel, t


def test_build_index_groups_and_periods():
    panel, t = _panel3()
    idx = build_index(panel, {"meta.sector"})
    assert idx.periods_by_entity["A"] == t
    assert idx.pos[("B", t[2])] == 2
    assert set(idx.all_at[t[2]]) == {"A", "B", "C"}
    assert idx.cell_group["meta.sector"][("B", t[2])] == "Tech"
    assert set(idx.group_members["meta.sector"][(t[2], "Tech")]) == {"A", "B"}
    assert set(idx.group_members["meta.sector"][(t[2], "Energy")]) == {"C"}


def test_expand_timeseries_forward_reach():
    panel, t = _panel3()
    idx = build_index(panel, {"meta.sector"})
    assert expand_timeseries({("A", t[0])}, 1, idx) == {("A", t[0]), ("A", t[1])}   # reach 1 -> t..t+1
    assert expand_timeseries({("A", t[0])}, 2, idx) == {("A", t[0]), ("A", t[1]), ("A", t[2])}
    assert expand_timeseries({("A", t[1])}, INF, idx) == {("A", t[1]), ("A", t[2])}
    assert expand_timeseries({("A", t[2])}, 5, idx) == {("A", t[2])}   # clamped at series end
    assert expand_timeseries({("A", t[0])}, 0, idx) == {("A", t[0])}   # reach 0 = self only


def test_expand_group_by_sector_and_whole_period():
    panel, t = _panel3()
    idx = build_index(panel, {"meta.sector"})
    assert expand_group({("B", t[2])}, "meta.sector", idx) == {("A", t[2]), ("B", t[2])}
    assert expand_group({("B", t[2])}, None, idx) == {("A", t[2]), ("B", t[2]), ("C", t[2])}
    assert expand_group({("C", t[2])}, "meta.sector", idx) == {("C", t[2])}   # alone in Energy


def test_expand_whole_entity_is_bidirectional():
    panel, t = _panel3()
    idx = build_index(panel, {"meta.sector"})
    assert expand_whole_entity({("A", t[1])}, idx) == {("A", t[0]), ("A", t[1]), ("A", t[2])}


def test_expand_group_missing_group_widens_to_period():
    panel, t = _panel3()
    idx = build_index(panel, {"meta.sector"})
    # an entity/period not in the index resolves to no group -> widen to the whole period, not one cell
    assert expand_group({("Z", t[2])}, "meta.sector", idx) == {("A", t[2]), ("B", t[2]), ("C", t[2])}


def test_time_series_ops_never_misclassified_as_bounded():
    from trail.footprint import _TS_LAG, _TS_WHOLE_ENTITY, _TS_WINDOWED
    from trail.ops import OPS
    ts = {n for n, s in OPS.items() if s.axis == "time-series"}
    assert _TS_WINDOWED <= ts and _TS_LAG <= ts and _TS_WHOLE_ENTITY <= ts
    assert not (_TS_WINDOWED & _TS_WHOLE_ENTITY) and not (_TS_LAG & _TS_WHOLE_ENTITY)
    # bidirectional ops must be whole-entity, never treated as a forward window
    assert {"ts_mean", "resample", "to_annual", "to_quarterly"} <= _TS_WHOLE_ENTITY
    # any op not explicitly windowed/lag falls through forward_reach to INF (safe widen), never bounded
    for n in ts - _TS_WINDOWED - _TS_LAG:
        assert forward_reach(n, ()) == INF


def test_cell_footprint_composition():
    panel, t = _panel3()
    idx = build_index(panel, {"meta.sector"})
    dirty = {"income.revenue": {("B", t[2])}}
    prog = prepare("signal s at annual = zscore(lag(income.revenue, 1)) by meta.sector", stdlib=False)
    sig = next(d for d in prog.decls if isinstance(d, ast.SignalDecl))
    # lag keeps B/2021 entity-local (t is the last period, so no forward spill), then sector -> A,B
    assert cell_footprint(sig.expr, dirty, idx, {}, {}) == {("A", t[2]), ("B", t[2])}


def test_model_footprint_unions_exports():
    panel, t = _panel3()
    idx = build_index(panel, {"meta.sector"})
    dirty = {"income.revenue": {("C", t[2])}}
    prog = prepare("model m at annual { export v = zscore(income.revenue) by meta.sector }",
                   stdlib=False)
    m = next(d for d in prog.decls if isinstance(d, ast.ModelDecl))
    assert model_footprint(m, dirty, idx) == {("C", t[2])}   # C alone in Energy
