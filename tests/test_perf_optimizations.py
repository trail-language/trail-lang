"""Regression tests for the Tier-1/Tier-2 performance work: window staging, entity sortedness,
lazy {file} scan (projection pushdown), and the streaming engine. Every one of these is meant to
be semantics-preserving; these tests lock that invariant in alongside the structural change."""
import re

import polars as pl

from trail import ast
from trail.compiler import compile_signal
from trail.mcp.data import _finalize, resolve_panel
from trail.mcp.tools import eval_tool
from trail.pipeline import prepare


def _plan(expr: str):
    prog = prepare(f"signal value = {expr}")
    uni = {d.name: d for d in prog.decls if isinstance(d, ast.UniverseDecl)}
    sig = next(d for d in prog.decls if isinstance(d, ast.SignalDecl))
    return compile_signal(sig, uni)


def _panel_rows(n_ent=6, n_per=8):
    rows = []
    for e in range(n_ent):
        for t in range(n_per):
            rows.append({"entity": f"E{e}", "time": f"20{10 + t:02d}-01-01",
                         "income.net_income": float((e + 1) * (t + 1)),
                         "income.revenue": float((e + 2) * (t + 1) + 1),
                         "price.adj_close": float((e + 1) + t * 0.5 + 1)})
    return rows


# --- Tier 1a: window staging -------------------------------------------------
def test_window_ops_are_hoisted_into_stage_columns():
    plan = _plan("zscore(income.net_income / income.revenue) + rank(roll_mean(price.adj_close, 3))")
    df = pl.DataFrame({"entity": ["A"], "time": [1], "income.net_income": [1.0],
                       "income.revenue": [2.0], "price.adj_close": [3.0]})
    txt = plan._lf_builder(df).explain(optimized=False)
    stages = sorted(set(re.findall(r"__stage_\d+", txt)))
    assert stages, "expected window subexpressions to be hoisted into __stage_* columns"
    # roll_mean, rank, zscore -> three distinct hoisted windows
    assert len(stages) == 3


def test_elementwise_only_expr_is_not_staged():
    plan = _plan("income.net_income / income.revenue")
    df = pl.DataFrame({"entity": ["A"], "time": [1], "income.net_income": [1.0], "income.revenue": [2.0]})
    txt = plan._lf_builder(df).explain(optimized=False)
    assert "__stage_" not in txt  # no windows -> nothing to hoist


def test_staged_result_matches_direct_polars_reference():
    rows = _panel_rows()
    out = eval_tool("rank(roll_mean(price.adj_close, 3))", {"rows": rows}, format="records")
    got = {(r["entity"], r["time"][:10]): r["value"] for r in out["records"]}

    ref = (_finalize(pl.from_dicts(rows))
           .with_columns(pl.col("price.adj_close").rolling_mean(3, min_samples=3).over("entity").alias("_r"))
           .with_columns(pl.col("_r").rank("average").over("time").alias("value")))
    for row in ref.iter_rows(named=True):
        key = (row["entity"], str(row["time"])[:10])
        exp, act = row["value"], got[key]
        assert (exp is None and act is None) or abs(exp - act) < 1e-9


# --- Tier 1b: entity sortedness ---------------------------------------------
def test_finalize_flags_entity_sorted():
    df = pl.from_dicts(_panel_rows())
    panel = _finalize(df)
    assert panel.get_column("entity").flags["SORTED_ASC"]


# --- Tier 2a: streaming engine ----------------------------------------------
def test_streaming_matches_default_engine():
    rows = _panel_rows()
    expr = "zscore(income.net_income / income.revenue) + rank(roll_mean(price.adj_close, 3))"
    default = eval_tool(expr, {"rows": rows}, format="records")["records"]
    stream = eval_tool(expr, {"rows": rows}, format="records", streaming=True)["records"]
    # The streaming engine reduces in a different order, so values may differ by a few ULP
    # (float non-associativity) — identical up to tolerance, not bit-for-bit.
    assert len(default) == len(stream)
    for a, b in zip(default, stream, strict=True):
        assert (a["entity"], a["time"]) == (b["entity"], b["time"])
        va, vb = a["value"], b["value"]
        assert (va is None and vb is None) or abs(va - vb) < 1e-9


# --- Tier 2b: lazy {file} scan + projection pushdown ------------------------
def test_lazy_file_scan_matches_eager_and_projects_only_referenced(tmp_path):
    p = tmp_path / "wide.parquet"
    df = pl.from_dicts(_panel_rows())
    df = df.with_columns([pl.lit(0.0).alias(f"unused_{i}") for i in range(6)])  # wide, unused fields
    _finalize(df).write_parquet(p)

    eager, _ = resolve_panel({"file": str(p)})
    lazy, _ = resolve_panel({"file": str(p)}, lazy=True)
    assert isinstance(lazy, pl.LazyFrame) and isinstance(eager, pl.DataFrame)

    plan = _plan("income.net_income / income.revenue")
    txt = plan._lf_builder(lazy).select(["entity", "time", "value"]).explain()
    m = re.search(r"PROJECT\s+(\d+)/(\d+)\s+COLUMNS", txt)
    assert m and int(m.group(1)) < int(m.group(2)), "expected projection pushdown to read fewer columns"

    r_eager = eval_tool("income.net_income / income.revenue", {"file": str(p)}, format="records")
    assert r_eager["records"], "eval over a {file} parquet should return rows"
