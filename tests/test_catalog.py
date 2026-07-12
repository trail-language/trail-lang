from click.testing import CliRunner

from trail import ast
from trail.catalog import (
    CatalogResult,
    catalog,
    describe,
    evaluate_meta,
    fields,
    functions,
    namespaces,
    sources,
)
from trail.cli import main
from trail.config import DEFAULT_CONFIG
from trail.parser import parse_expr, parse_program, parse_repl_line


# --- discovery core ---

def test_namespaces_and_fields():
    assert set(namespaces()) == {"income", "balance", "cash", "price", "meta"}
    inc = fields("income")
    assert isinstance(inc, CatalogResult)
    assert "income.revenue" in inc.frame["field"].to_list()
    assert all(f.startswith("income.") for f in inc.frame["field"].to_list())


def test_functions_catalog_has_axis_and_arity():
    fr = functions().frame
    row = fr.filter(fr["function"] == "roll_mean").row(0, named=True)
    assert row["axis"] == "time-series" and row["args"] == "2"
    assert "zscore" in fr["function"].to_list()  # cross-sectional present too


def test_sources_from_config():
    fr = sources(DEFAULT_CONFIG).frame
    assert "fixture" in fr["source"].to_list()


def test_describe_resolves_field_namespace_function_source():
    assert describe(("income", "revenue")).frame.row(0, named=True) == {
        "property": "column", "value": "income.revenue"
    } or "flow" in describe(("income", "revenue")).frame["value"].to_list()
    assert describe(("income",)).title.startswith("Fields in 'income'")
    assert describe(("roll_mean",)).title == "Function roll_mean"
    assert describe(("fixture",)).title == "Source fixture"
    assert describe(("functions",)).frame.height == functions().frame.height
    assert "Unknown" in describe(("nope",)).title


def test_catalog_summary():
    c = catalog(DEFAULT_CONFIG)
    assert "namespace" in c.frame.columns and c.frame.height == len(namespaces())


# --- REPL dialect parsing (superset of the file grammar) ---

def test_meta_command_parsing():
    assert parse_repl_line("?") == ast.MetaCatalog()
    assert parse_repl_line("?income") == ast.MetaDescribe(("income",))
    assert parse_repl_line("?income.revenue") == ast.MetaDescribe(("income", "revenue"))
    assert parse_repl_line("?functions") == ast.MetaDescribe(("functions",))


def test_repl_line_is_a_superset():
    # a bare expression and a declaration both still parse in the repl dialect
    assert parse_repl_line("income.revenue / 2") == parse_expr("income.revenue / 2")
    decl = parse_repl_line("model m { a = income.revenue }")
    assert isinstance(decl, ast.ModelDecl)


def test_meta_commands_are_rejected_in_a_model_file():
    # the file grammar must NOT accept meta-commands
    import pytest
    from lark.exceptions import UnexpectedInput
    with pytest.raises(UnexpectedInput):
        parse_program("?income")


def test_evaluate_meta_routes_to_core():
    assert evaluate_meta(parse_repl_line("?")).frame.columns == ["namespace", "fields"]
    assert evaluate_meta(parse_repl_line("?income")).title.startswith("Fields in 'income'")


# --- CLI ---

def test_cli_catalog():
    r = CliRunner().invoke(main, ["catalog"])
    assert r.exit_code == 0 and "namespace" in r.output


def test_cli_catalog_describe_field():
    r = CliRunner().invoke(main, ["catalog", "income.revenue"])
    assert r.exit_code == 0 and "flow" in r.output


def test_functions_catalog_includes_derived_macros():
    fr = functions().frame
    assert "layer" in fr.columns
    layers = dict(zip(fr["function"].to_list(), fr["layer"].to_list()))
    assert layers["roll_mean"] == "primitive"
    assert layers["cagr"] == "derived"        # migrated builtin, now a stdlib macro
    assert layers["signed_log"] == "derived"


def test_describe_finds_migrated_macro():
    d = describe(("cagr",))
    assert d.title == "Function cagr" and "derived" in d.frame["value"].to_list()


def test_internal_helpers_hidden():
    # _grow_start etc. are internal to timeseries.trail and must not surface in discovery
    fr = functions().frame
    assert not any(n.startswith("_") for n in fr["function"].to_list())
