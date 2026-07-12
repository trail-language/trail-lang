from trail.deps import extract
from trail.parser import parse_expr, parse_program


def test_expr_deps():
    r = extract(parse_expr("zscore(income.net_income / avg2(balance.total_assets)) by meta.sector"))
    assert r.fields == frozenset({"income.net_income", "balance.total_assets", "meta.sector"})
    assert r.functions == frozenset({"zscore", "avg2"})


def test_model_locals_are_not_fields():
    prog = parse_program(
        "model m { roa = income.net_income / balance.total_assets\n export good = roa > 0 }"
    )
    r = extract(prog)
    assert "roa" in r.locals_used and "roa" not in r.fields


def test_pins_reported():
    r = extract(parse_expr("income.revenue @ fmp - income.revenue @ edgar"))
    assert r.pins == frozenset({("income.revenue", "fmp"), ("income.revenue", "edgar")})
