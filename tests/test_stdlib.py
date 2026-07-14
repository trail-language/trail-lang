import math

import polars as pl
import pytest

from trail import ast
from trail.compiler import compile_model
from trail.fixtures import load_panel
from trail.library import stdlib_source
from trail.macro import collect_functions, expand
from trail.parser import parse_program
from trail.validate import KNOWN_FUNCTIONS

FUNCS = collect_functions(parse_program(stdlib_source()))


def _call_names(e, acc):
    match e:
        case ast.Call():
            acc.add(e.name)
            for a in e.args:
                _call_names(a, acc)
        case ast.BinOp() | ast.Compare() | ast.BoolOp() | ast.Coalesce():
            _call_names(e.left, acc)
            _call_names(e.right, acc)
        case ast.Not() | ast.Neg():
            _call_names(e.operand, acc)
        case ast.In():
            _call_names(e.item, acc)
        case ast.Ternary():
            _call_names(e.value, acc)
            _call_names(e.cond, acc)
            _call_names(e.orelse, acc)


def test_stdlib_parses_and_has_no_duplicate_names():
    assert len(FUNCS) >= 60  # sanity: the whole library collected


def test_stdlib_does_not_shadow_primitives():
    assert set(FUNCS) & set(KNOWN_FUNCTIONS) == set()  # no accidental shadowing of a builtin


def test_every_stdlib_function_expands_to_primitives_only():
    # Passing a literal for every parameter is enough to fully expand each function;
    # the result must contain ONLY primitive calls (no leftover user-function names).
    for name, fd in FUNCS.items():
        call = ast.Call(name, tuple(ast.Literal(2.0) for _ in fd.params))
        expanded = expand(call, FUNCS)  # raises on recursion / arity self-inconsistency
        names: set[str] = set()
        _call_names(expanded, names)
        leftover = names - set(KNOWN_FUNCTIONS)
        assert not leftover, f"{name} expands to non-primitives: {leftover}"


# --- representative numeric correctness (stdlib + fixture) ---

def _run(exports: str):
    src = stdlib_source() + "\nmodel m {\n" + exports + "\n}\n"
    from trail.macro import expand_program
    prog = expand_program(parse_program(src))
    model = next(d for d in prog.decls if isinstance(d, ast.ModelDecl))
    return compile_model(model, {}).run(load_panel())


def _cell(df, col, sec, period):
    return df.filter((pl.col("entity") == sec) & (pl.col("time").dt.year() == period))[col][0]


def test_math_and_transform_values():
    df = _run(
        "export p = pi()\n"
        "export l = log10(income.revenue)\n"          # rev 100 -> 2.0
        "export sg = sigmoid(income.revenue - income.revenue)\n"   # sigmoid(0)=0.5
        "export th = tanh(income.revenue - income.revenue)\n"      # tanh(0)=0
        "export h = hypot(3, 4)\n"                     # 5
        "export sq = square(income.eps_diluted)\n"
        "export sn = sign(income.revenue - 200)\n"     # AAA rev 100 -> -1 ; BBB rev 200 -> 0
    )
    assert _cell(df, "p", "AAA", 2017) == pytest.approx(math.pi)
    assert _cell(df, "l", "AAA", 2017) == pytest.approx(2.0)
    assert _cell(df, "sg", "AAA", 2017) == pytest.approx(0.5)
    assert _cell(df, "th", "AAA", 2017) == pytest.approx(0.0)
    assert _cell(df, "h", "AAA", 2017) == pytest.approx(5.0)
    eps = _cell(load_panel(), "income.eps_diluted", "AAA", 2017)
    assert _cell(df, "sq", "AAA", 2017) == pytest.approx(eps * eps)
    assert _cell(df, "sn", "AAA", 2017) == pytest.approx(-1.0)
    assert _cell(df, "sn", "BBB", 2017) == pytest.approx(0.0)


def test_calculus_values():
    df = _run(
        "export d = diff(income.revenue)\n"            # AAA 2018: 110-100 = 10
        "export ig = integral(income.revenue)\n"       # AAA 2018: 100+110 = 210
    )
    assert _cell(df, "d", "AAA", 2018) == pytest.approx(10.0)
    assert _cell(df, "ig", "AAA", 2018) == pytest.approx(100.0 + 110.0)


def test_cross_sectional_transforms():
    df = _run(
        "export dm = demean(income.revenue)\n"
        "export rz = robust_zscore(income.revenue)\n"
    )
    # demean: each period's cross-sectional mean of the residual is ~0
    per_period = df.group_by("time").agg(pl.col("dm").mean())
    for v in per_period["dm"].to_list():
        assert v == pytest.approx(0.0, abs=1e-9)
    # robust_zscore of the cross-sectional median entity is ~0
    med = df.filter(pl.col("time").dt.year() == 2024)["rz"]
    assert med.median() == pytest.approx(0.0, abs=1e-9)
