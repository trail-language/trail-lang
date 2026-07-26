import datetime as dt

import polars as pl

from trail import ast
from trail.footprint import INF, build_index, window_of


def test_window_of_literal_and_unbounded():
    fx = ast.FieldRef(("x",))
    assert window_of("lag", (fx, ast.Literal(2))) == 2
    assert window_of("roll_mean", (fx, ast.Literal(4))) == 4
    assert window_of("ts_mean", (fx,)) == INF
    assert window_of("roll_mean", (fx, ast.Literal("1y"))) == INF   # duration -> conservative
    assert window_of("zscore", (fx,)) == 0                          # not time-series


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
