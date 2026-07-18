"""Regression tests for `by`-on-def propagation (fix/by-on-def).

A `by` grouping clause attached to a stdlib/user `def` call used to be silently dropped:
the def body's cross-sectional ops fell back to the whole-period group, so
`pctile(x) by g` was bit-identical to `pctile(x)`. These tests pin the fix - the call-site
`by` fills every body cross-sectional op that carries no group of its own, while ops that
pin their own group, and the caller's own argument expressions, are left untouched.
"""
import datetime as dt

import polars as pl
import pytest

from trail import ast
from trail.compiler import compile_model
from trail.library import stdlib_source
from trail.macro import collect_functions, expand
from trail.parser import parse_program
from trail.pipeline import prepare

FUNCS = collect_functions(parse_program(stdlib_source()))
_T = dt.datetime(2024, 12, 31)


def _calls(e, acc):
    """Collect (name, by) for every Call node reachable from `e`."""
    match e:
        case ast.Call():
            acc.append((e.name, e.by))
            for a in e.args:
                _calls(a, acc)
            for _, v in e.kwargs:
                _calls(v, acc)
        case ast.BinOp() | ast.Compare() | ast.BoolOp() | ast.Coalesce():
            _calls(e.left, acc)
            _calls(e.right, acc)
        case ast.Not() | ast.Neg():
            _calls(e.operand, acc)
        case ast.Ternary():
            _calls(e.value, acc)
            _calls(e.cond, acc)
            _calls(e.orelse, acc)
        case ast.In():
            _calls(e.item, acc)


def _run(src: str, panel: pl.DataFrame) -> pl.DataFrame:
    prog = prepare(src)
    model = next(d for d in prog.decls if isinstance(d, ast.ModelDecl))
    return compile_model(model, {}).run(panel)


# --- AST-level: the propagation mechanism --------------------------------------------

def test_call_site_by_fills_all_body_cross_ops():
    g = ("meta", "sector")
    call = ast.Call("pctile", (ast.FieldRef(("income", "revenue")),), (), g)
    ops = []
    _calls(expand(call, FUNCS), ops)
    xs = {name: by for name, by in ops if name in ("rank", "xs_count")}
    assert xs == {"rank": g, "xs_count": g}


def test_ungrouped_def_call_leaves_body_cross_ops_ungrouped():
    call = ast.Call("pctile", (ast.FieldRef(("income", "revenue")),))
    ops = []
    _calls(expand(call, FUNCS), ops)
    xs = {name: by for name, by in ops if name in ("rank", "xs_count")}
    assert xs == {"rank": None, "xs_count": None}


def test_inner_op_own_by_is_not_overridden():
    # a def body pins its OWN group; the call-site `by` must not overwrite it.
    funcs = collect_functions(parse_program("def foo(x) = rank(x) by meta.exchange"))
    call = ast.Call("foo", (ast.FieldRef(("income", "revenue")),), (), ("meta", "sector"))
    ops = []
    _calls(expand(call, funcs), ops)
    assert ("rank", ("meta", "exchange")) in ops
    assert ("rank", ("meta", "sector")) not in ops


def test_call_site_by_does_not_leak_into_argument():
    # zscore(...) is the CALLER's own argument expression: grouping the def must not regroup it,
    # mirroring how a builtin `by` binds to one op and never to its operands.
    g = ("meta", "sector")
    arg = ast.Call("zscore", (ast.FieldRef(("income", "revenue")),))  # by=None
    call = ast.Call("pctile", (arg,), (), g)
    ops = []
    _calls(expand(call, FUNCS), ops)
    by_of = {name: by for name, by in ops}
    assert by_of["zscore"] is None
    assert by_of["rank"] == g and by_of["xs_count"] == g


def test_two_level_dotted_by_propagates_whole_path():
    # the grammar's `by` is one dotted field; propagation copies the full multi-part tuple.
    g = ("a", "b")
    call = ast.Call("pctile", (ast.FieldRef(("income", "revenue")),), (), g)
    ops = []
    _calls(expand(call, FUNCS), ops)
    assert all(by == g for name, by in ops if name in ("rank", "xs_count"))


# --- numeric end-to-end --------------------------------------------------------------

def test_pctile_by_group_differs_from_ungrouped():
    panel = pl.DataFrame({
        "entity": ["A", "B", "C", "D"],
        "time": [_T] * 4,
        "income.revenue": [1.0, 2.0, 3.0, 4.0],
        "meta.sector": ["S1", "S1", "S2", "S2"],
    }).sort(["entity", "time"])
    res = _run(
        "model m {\n"
        "  export a = pctile(income.revenue)\n"
        "  export b = pctile(income.revenue) by meta.sector\n"
        "}\n", panel).sort("entity")
    a, b = res["a"].to_list(), res["b"].to_list()
    assert a != b
    assert a == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])
    assert b == pytest.approx([0.0, 1.0, 0.0, 1.0])  # each 2-entity sector -> {0, 1}


def test_neutralize_by_group_residual_zero_within_group():
    import random
    random.seed(0)
    n = 40
    ents = [f"E{i:03d}" for i in range(n)]
    sect = ["S1" if i < n // 2 else "S2" for i in range(n)]
    panel = pl.DataFrame({
        "entity": ents, "time": [_T] * n,
        "income.revenue": [random.gauss(0, 1) for _ in range(n)],
        "income.net_income": [random.gauss(0, 1) for _ in range(n)],
        "meta.sector": sect,
    }).sort(["entity", "time"])
    res = _run(
        "model m {\n"
        "  export r = neutralize(income.revenue, income.net_income) by meta.sector\n"
        "}\n", panel)
    res = res.join(panel.select(["entity", "meta.sector"]), on="entity")
    per_group = res.group_by("meta.sector").agg(pl.col("r").mean())
    for v in per_group["r"].to_list():
        assert v == pytest.approx(0.0, abs=1e-9)
    # a GLOBAL (ungrouped) neutralize does NOT zero the per-group means - proves grouping bit.
    res_g = _run(
        "model m { export r = neutralize(income.revenue, income.net_income) }", panel)
    res_g = res_g.join(panel.select(["entity", "meta.sector"]), on="entity")
    global_means = res_g.group_by("meta.sector").agg(pl.col("r").mean())["r"].to_list()
    assert any(abs(v) > 1e-6 for v in global_means)


def test_two_level_dotted_by_groups_end_to_end():
    # a 2-part dotted grouping column drives per-group pctile end-to-end.
    panel = pl.DataFrame({
        "entity": ["A", "B", "C", "D"],
        "time": [_T] * 4,
        "income.revenue": [1.0, 2.0, 3.0, 4.0],
        "meta.group_key": ["g1", "g1", "g2", "g2"],
    }).sort(["entity", "time"])
    res = _run(
        "model m { export b = pctile(income.revenue) by meta.group_key }", panel).sort("entity")
    assert res["b"].to_list() == pytest.approx([0.0, 1.0, 0.0, 1.0])
