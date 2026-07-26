"""Incremental recompute: assert an incremental serve equals a full recompute against the new data.

A module-level test source whose DATA version is controlled by `_STATE` (NOT by its config options, so
`panel_key` stays constant and the changefeed — not a recipe change — drives recompute). It is resolved
as a dotted driver: `driver: tests.test_views_incremental.versioned`.
"""
import datetime as dt

import polars as pl

from trail.mcp.tools import drop_tool, run_tool
from trail.source import Capabilities, DataSource, ENTITY_COL, TIME_COL

_T = [dt.datetime(2020, 12, 31), dt.datetime(2021, 12, 31)]
_SEC = {"A": "Tech", "B": "Tech", "C": "Energy"}
_STATE = {"token": "w0"}  # tests flip to "w1" to simulate B filing a new 2021 value


def _rev(e, t):
    if _STATE["token"] == "w1" and e == "B" and t == _T[1]:
        return 99.0
    return {"A": 10.0, "B": 20.0, "C": 5.0}[e]


class _Versioned(DataSource):
    name = "vsrc"

    def available_fields(self, frequency=None):
        return {"income.revenue", "meta.sector"}

    def capabilities(self):
        return Capabilities(frequency="annual", provides_meta=True)

    def freshness_token(self):
        return _STATE["token"]

    def changed_since(self, cursor):
        if _STATE["token"] == "w1" and cursor == "w0":
            return {("B", _T[1])}          # exactly the cell that changed
        return set()

    def load(self, request):
        rows = [{ENTITY_COL: e, TIME_COL: t, "income.revenue": _rev(e, t), "meta.sector": _SEC[e]}
                for e in _SEC for t in _T]
        df = pl.DataFrame(rows)
        if request.entities:
            df = df.filter(pl.col(ENTITY_COL).is_in(list(request.entities)))
        return df


def versioned(options):
    return _Versioned(options)


MODEL = "track model m at annual { export z = zscore(income.revenue) by meta.sector }"


def _cfg(tmp_path):
    p = tmp_path / "trail.yaml"
    p.write_text("sources:\n  vsrc:\n    driver: tests.test_views_incremental.versioned\n"
                 "precedence:\n  default: [vsrc]\n"
                 "panel:\n  periods: [2020, 2021]\n")
    return str(p)


def _rows(res):
    return {(r["entity"], r["time"]): (round(v, 6) if (v := r["views.m.z"]) is not None else None)
            for r in res["records"]}


def test_incremental_equals_full(tmp_path):
    _STATE["token"] = "w0"
    cfg = _cfg(tmp_path)
    run_tool("m", {"config": cfg}, program=MODEL, format="records")        # build at w0
    _STATE["token"] = "w1"                                                 # B's 2021 value changes
    inc = _rows(run_tool("m", {"config": cfg}, program=MODEL, format="records"))  # incremental serve
    # oracle: drop + rebuild fully at w1
    drop_tool("m", {"config": cfg})
    full = _rows(run_tool("m", {"config": cfg}, program=MODEL, format="records"))
    assert inc == full and inc                                             # bit-for-bit, non-empty
    _STATE["token"] = "w0"                                                 # reset for other tests
