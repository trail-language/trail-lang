"""Frequency-qualified field references: annual.income.revenue / quarterly.income.revenue."""
from trail import ast
from trail.compiler import compile_expr
from trail.deps import extract
from trail.parser import parse_expr, parse_program
from trail.validate import validate


def test_frequency_prefix_parses_to_qualifier():
    e = parse_expr("quarterly.income.revenue")
    assert isinstance(e, ast.FieldRef)
    assert e.path == ("income", "revenue") and e.frequency == "quarterly"
    assert e.column == "income.revenue"  # canonical
    assert e.qualified_column == "quarterly.income.revenue"  # physical column


def test_daily_price_qualifier():
    e = parse_expr("daily.price.adj_close")
    assert e.path == ("price", "adj_close") and e.frequency == "daily"


def test_bare_field_unchanged():
    e = parse_expr("income.revenue")
    assert e.path == ("income", "revenue") and e.frequency is None
    assert e.qualified_column == "income.revenue"


def test_frequency_and_source_pin_compose():
    e = parse_expr("annual.income.revenue @ fmp")
    assert e.path == ("income", "revenue")
    assert e.frequency == "annual" and e.source == "fmp"


def _codes(src):
    return [i.code for i in validate(parse_program(src))]


def test_qualified_field_is_valid():
    assert "E-FIELD-UNKNOWN" not in _codes("model m { export a = quarterly.income.revenue }")


def test_qualified_unknown_canonical_still_errors():
    assert "E-FIELD-UNKNOWN" in _codes("model m { export a = quarterly.income.bogus }")


def test_deps_surfaces_qualified_and_bare():
    fields = extract(parse_program(
        "model m { export a = quarterly.income.revenue + income.revenue }"
    )).fields
    assert "quarterly.income.revenue" in fields and "income.revenue" in fields


def test_compiler_emits_qualified_column():
    expr = compile_expr(parse_expr("quarterly.income.revenue"), set())
    assert expr.meta.output_name() == "quarterly.income.revenue"
    assert compile_expr(parse_expr("income.revenue"), set()).meta.output_name() == "income.revenue"
