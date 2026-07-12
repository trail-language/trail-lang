from trail import ast
from trail.parser import parse_expr


def test_arithmetic_precedence():
    e = parse_expr("income.net_income / income.revenue + 1")
    assert e == ast.BinOp(
        "add",
        ast.BinOp("div", ast.FieldRef(("income", "net_income")), ast.FieldRef(("income", "revenue"))),
        ast.Literal(1),
    )


def test_call_with_kwarg_and_by():
    e = parse_expr("zscore(cagr(income.revenue, 5, clamp=true)) by meta.sector")
    assert isinstance(e, ast.Call) and e.name == "zscore" and e.by == ("meta", "sector")
    inner = e.args[0]
    assert inner == ast.Call(
        "cagr",
        (ast.FieldRef(("income", "revenue")), ast.Literal(5)),
        kwargs=(("clamp", ast.Literal(True)),),
    )


def test_ternary_chain_first_match():
    e = parse_expr("2 if x > 0.12 else 1 if x > 0.05 else 0")
    assert e == ast.Ternary(
        ast.Literal(2),
        ast.Compare("gt", ast.NameRef("x"), ast.Literal(0.12)),
        ast.Ternary(ast.Literal(1), ast.Compare("gt", ast.NameRef("x"), ast.Literal(0.05)), ast.Literal(0)),
    )


def test_coalesce_pin_and_in():
    e = parse_expr("cash.stock_issued @ fmp ?? 0")
    assert e == ast.Coalesce(ast.FieldRef(("cash", "stock_issued"), source="fmp"), ast.Literal(0))
    e2 = parse_expr('meta.exchange in ("NYSE", "NASDAQ")')
    assert e2 == ast.In(ast.FieldRef(("meta", "exchange")), (ast.Literal("NYSE"), ast.Literal("NASDAQ")))


def test_boolean_ops_and_not():
    e = parse_expr("not (a > 1 and b < 2)")
    assert isinstance(e, ast.Not) and isinstance(e.operand, ast.BoolOp)


def test_scientific_number():
    assert parse_expr("200e6") == ast.Literal(200e6)
