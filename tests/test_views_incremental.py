"""Incremental recompute invariant: an incremental serve must equal a full recompute against the new
data, cell for cell, across op axes + restatement + no-changefeed fallback.

A module-level test source whose DATA is controlled by `_STATE` (NOT by its config options, so
`panel_key` stays constant and the changefeed — not a recipe change — drives recompute). Resolved as a
dotted driver: `driver: tests.test_views_incremental.versioned`.
"""
import datetime as dt

import polars as pl

from trail.mcp.tools import drop_tool, run_tool
from trail.source import Capabilities, DataSource, ENTITY_COL, TIME_COL

_T = [dt.datetime(2020, 12, 31), dt.datetime(2021, 12, 31)]
_SEC = {"A": "Tech", "B": "Tech", "C": "Energy"}
_EXCH = {"A": "NYSE", "B": "NASDAQ", "C": "NYSE"}
_BASE = {"A": 10.0, "B": 20.0, "C": 5.0}
_STATE = {"token": "w0", "overrides": {}, "dirty": set(), "changefeed": True, "sector_overrides": {}}


def _reset():
    _STATE.update(token="w0", overrides={}, dirty=set(), changefeed=True, sector_overrides={})


def _rev(e, t):
    return _STATE["overrides"].get((e, t), _BASE[e])


def _sec(e):
    return _STATE["sector_overrides"].get(e, _SEC[e])


class _Versioned(DataSource):
    name = "vsrc"

    def available_fields(self, frequency=None):
        return {"income.revenue", "meta.sector", "meta.exchange"}

    def capabilities(self):
        return Capabilities(frequency="annual", provides_meta=True)

    def freshness_token(self):
        return _STATE["token"]

    def changed_since(self, cursor):
        if not _STATE["changefeed"]:
            return None
        return set(_STATE["dirty"]) if cursor != _STATE["token"] else set()

    def load(self, request):
        rows = [{ENTITY_COL: e, TIME_COL: t, "income.revenue": _rev(e, t),
                 "meta.sector": _sec(e), "meta.exchange": _EXCH[e]}
                for e in _SEC for t in _T]
        df = pl.DataFrame(rows)
        if request.entities:
            df = df.filter(pl.col(ENTITY_COL).is_in(list(request.entities)))
        return df


def versioned(options):
    return _Versioned(options)


def _cfg(tmp_path):
    p = tmp_path / "trail.yaml"
    p.write_text("sources:\n  vsrc:\n    driver: tests.test_views_incremental.versioned\n"
                 "precedence:\n  default: [vsrc]\n"
                 "panel:\n  periods: [2020, 2021]\n")
    return str(p)


def _rows(res, col):
    return {(r["entity"], r["time"]): (round(v, 6) if (v := r[col]) is not None else None)
            for r in res["records"]}


def _inc_vs_full(tmp_path, model, col, token, overrides, dirty, changefeed=True, sector_overrides=None):
    _reset()
    cfg = _cfg(tmp_path)
    run_tool("m", {"config": cfg}, program=model, format="records")            # build at w0
    _STATE.update(token=token, overrides=overrides, dirty=set(dirty), changefeed=changefeed,
                  sector_overrides=sector_overrides or {})
    inc = _rows(run_tool("m", {"config": cfg}, program=model, format="records"), col)  # incremental
    drop_tool("m", {"config": cfg})                                            # oracle: full rebuild
    full = _rows(run_tool("m", {"config": cfg}, program=model, format="records"), col)
    _reset()
    return inc, full


ZS = "track model m at annual { export z = zscore(income.revenue) by meta.sector }"
LAG = "track model m at annual { export l = lag(income.revenue, 1) }"
COMP = "track model m at annual { export c = zscore(lag(income.revenue, 1)) by meta.sector }"


def test_zscore_by_sector(tmp_path):
    inc, full = _inc_vs_full(tmp_path, ZS, "views.m.z", "w1", {("B", _T[1]): 99.0}, {("B", _T[1])})
    assert inc == full and inc


def test_zscore_reporter_alone_in_sector(tmp_path):
    inc, full = _inc_vs_full(tmp_path, ZS, "views.m.z", "w1", {("C", _T[1]): 42.0}, {("C", _T[1])})
    assert inc == full and inc


def test_lag_change_at_prior_period(tmp_path):
    # a change to B's 2020 revenue must update the lag output at 2021 (which reads 2020)
    inc, full = _inc_vs_full(tmp_path, LAG, "views.m.l", "w2", {("B", _T[0]): 88.0}, {("B", _T[0])})
    assert inc == full and inc


def test_composition_zscore_of_lag(tmp_path):
    inc, full = _inc_vs_full(tmp_path, COMP, "views.m.c", "w2", {("B", _T[0]): 88.0}, {("B", _T[0])})
    assert inc == full and inc


def test_restatement_old_period(tmp_path):
    # a restatement re-files an OLD period; changed_since reports that cell -> its sector recomputes
    inc, full = _inc_vs_full(tmp_path, ZS, "views.m.z", "w3", {("A", _T[0]): 77.0}, {("A", _T[0])})
    assert inc == full and inc


def test_no_changefeed_falls_back_to_full(tmp_path):
    # source with no changefeed but a moved token -> COARSE -> whole-view rebuild, still == full
    inc, full = _inc_vs_full(tmp_path, ZS, "views.m.z", "w1", {("B", _T[1]): 99.0}, set(),
                             changefeed=False)
    assert inc == full and inc


def test_whole_entity_op_bidirectional(tmp_path):
    # ts_mean is a whole-series stat: a change at 2021 also changes the 2020 output (both directions)
    ts = "track model m at annual { export a = ts_mean(income.revenue) }"
    inc, full = _inc_vs_full(tmp_path, ts, "views.m.a", "w1", {("B", _T[1]): 99.0}, {("B", _T[1])})
    assert inc == full and inc


def test_sector_reassignment_stays_correct(tmp_path):
    # B files AND moves Tech->Energy: the old Tech peer A must not keep a stale z-score (group_hash
    # change forces a full rebuild)
    inc, full = _inc_vs_full(tmp_path, ZS, "views.m.z", "w1", {("B", _T[1]): 99.0}, {("B", _T[1])},
                             sector_overrides={"B": "Energy"})
    assert inc == full and inc


def test_nested_distinct_by_stays_correct(tmp_path):
    # zscore(... by exchange) nested inside (... by sector): two distinct groupings -> full rebuild
    nest = ("track model m at annual { "
            "export n = zscore(xs_mean(income.revenue) by meta.exchange) by meta.sector }")
    inc, full = _inc_vs_full(tmp_path, nest, "views.m.n", "w1", {("B", _T[1]): 99.0}, {("B", _T[1])})
    assert inc == full and inc


def test_entity_scoped_run_does_not_clobber_stored_view(tmp_path):
    # a scoped run must not overwrite the persisted full-universe frame
    _reset()
    cfg = _cfg(tmp_path)
    run_tool("m", {"config": cfg}, program=ZS, format="records")
    run_tool("m", {"config": cfg}, program=ZS, format="records", entities=["A"])  # scoped rescore
    after = run_tool("m", {"config": cfg}, program=ZS, format="records")          # unfiltered
    assert {r["entity"] for r in after["records"]} == {"A", "B", "C"}             # nothing vanished
    _reset()
