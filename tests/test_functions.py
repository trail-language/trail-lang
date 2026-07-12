import pytest

from trail import ast
from trail.compiler import compile_model
from trail.fixtures import load_panel
from trail.macro import TrailFunctionError, expand, expand_program
from trail.parser import parse_program
from trail.pipeline import prepare
from trail.validate import validate


def _compile(src: str, model: str = "m"):
    prog = prepare(src)  # stdlib implicit, so builtin-like functions (yoy, ...) resolve
    universes = {d.name: d for d in prog.decls if isinstance(d, ast.UniverseDecl)}
    m = next(d for d in prog.decls if isinstance(d, ast.ModelDecl) and d.name == model)
    return compile_model(m, universes).run(load_panel())


def _first(df, col):
    return df.sort(["security", "period"])[col][0]  # AAA, 2017


def test_parse_func_def():
    fd = parse_program("def sq(x) = x * x").decls[0]
    assert isinstance(fd, ast.FuncDef) and fd.name == "sq" and fd.params == ("x",)


def test_expand_strips_defs_and_inlines():
    prog = expand_program(parse_program("def sq(x) = x * x\nmodel m { export y = sq(income.revenue) }"))
    assert all(not isinstance(d, ast.FuncDef) for d in prog.decls)  # defs stripped
    res = _compile("def sq(x) = x * x\nmodel m { export y = sq(income.revenue) }")
    assert _first(res, "y") == pytest.approx(100.0**2)  # AAA 2017 revenue = 100


def test_derived_matches_builtin_yoy():
    res = _compile(
        "def yoy2(x) = x / lag(x, 1) - 1\n"
        "model m { export a = yoy2(income.revenue)\n export b = yoy(income.revenue) }"
    )
    for a, b in zip(res["a"].to_list(), res["b"].to_list(), strict=True):
        assert (a is None and b is None) or a == pytest.approx(b)


def test_nested_composition():
    res = _compile("def add1(x) = x + 1\ndef dbl(x) = add1(x) * 2\nmodel m { export y = dbl(income.revenue) }")
    assert _first(res, "y") == pytest.approx((100.0 + 1) * 2)


def test_recursion_rejected():
    with pytest.raises(TrailFunctionError, match="recursive"):
        expand_program(parse_program("def f(x) = f(x)\nmodel m { export y = f(income.revenue) }"))


def test_indirect_recursion_rejected():
    with pytest.raises(TrailFunctionError, match="recursive"):
        expand_program(parse_program(
            "def a(x) = b(x)\ndef b(x) = a(x)\nmodel m { export y = a(income.revenue) }"
        ))


def test_arity_mismatch():
    with pytest.raises(TrailFunctionError, match="argument"):
        expand_program(parse_program("def f(x, y) = x + y\nmodel m { export z = f(income.revenue) }"))


def test_unknown_function_survives_to_validation():
    prog = expand_program(parse_program("model m { a = nope(income.revenue) }"))
    assert "E-FUNC-UNKNOWN" in [i.code for i in validate(prog)]


def test_substitution_is_hygienic_for_fields():
    # a bare expression: expanding sq(income.revenue) yields income.revenue * income.revenue
    prog = parse_program("def sq(x) = x * x")
    funcs = {d.name: d for d in prog.decls}
    call = ast.Call("sq", (ast.FieldRef(("income", "revenue")),))
    out = expand(call, funcs)
    assert out == ast.BinOp("mul", ast.FieldRef(("income", "revenue")), ast.FieldRef(("income", "revenue")))


def test_stdlib_core_functions():
    # stdlib is implicit via _compile/prepare - call core functions directly.
    res = _compile(
        "model m {\n"
        "  export gm = gross_margin(income.gross_profit, income.revenue)\n"
        "  export cr = current_ratio(balance.current_assets, balance.current_liabilities)\n"
        "}\n"
    )
    assert res["gm"].to_list() == pytest.approx([0.45] * 48)          # fixture: gross_profit = 0.45*rev
    assert res["cr"].to_list() == pytest.approx([0.8 / 0.5] * 48)     # current_assets 0.8*rev / cl 0.5*rev
