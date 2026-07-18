"""export NAME (surface an existing local as an export of the same name; no `= expr` boilerplate)."""
from click.testing import CliRunner

from trail import ast
from trail.compiler import compile_model
from trail.fixtures import load_panel
from trail.cli import main
from trail.parser import parse_program
from trail.validate import validate


def codes(src):
    return [i.code for i in validate(parse_program(src))]


def test_export_name_parses_to_assignment_with_no_expr():
    m = parse_program("model m { x = income.revenue\n export x }").decls[0]
    x, exp = m.statements
    assert isinstance(x, ast.Assignment) and x.name == "x" and not x.export
    assert isinstance(exp, ast.Assignment) and exp.export and exp.name == "x"
    assert exp.expr is None  # bare export carries no right-hand side


def test_export_name_rhs_form_still_parses():
    m = parse_program("model m { export y = income.revenue }").decls[0]
    (exp,) = m.statements
    assert exp.export and exp.name == "y" and exp.expr is not None


def test_export_name_validates_when_defined():
    assert codes("model m { x = income.revenue\n export x }") == []


def test_export_name_undefined_errors():
    cs = codes("model m { export nope }")
    assert "E-EXPORT-UNDEFINED" in cs


def test_export_name_does_not_trip_rebound():
    assert "E-NAME-REBOUND" not in codes("model m { x = income.revenue\n export x }")


def test_export_rhs_rebind_still_trips_rebound():
    # genuine rebind: z already bound to a different value -> the existing rule still fires
    cs = codes("model m { z = income.revenue\n a = income.cogs\n export z = a }")
    assert "E-NAME-REBOUND" in cs


def test_compile_export_name_yields_column():
    model = parse_program(
        "model m { x = income.operating_income / income.revenue\n export x }").decls[0]
    out = compile_model(model, {}).run(load_panel())
    assert "x" in out.columns and out["x"].null_count() == 0


def test_cli_run_export_name_prints_column(tmp_path):
    f = tmp_path / "m.trail"
    f.write_text("model m { x = income.operating_income / income.revenue\n export x }\n")
    res = CliRunner().invoke(main, ["run", str(f), "--model", "m"])
    assert res.exit_code == 0 and "x" in res.output


def test_cli_validate_export_undefined_errors(tmp_path):
    f = tmp_path / "m.trail"
    f.write_text("model m { export nope }\n")
    res = CliRunner().invoke(main, ["validate", str(f)])
    assert res.exit_code == 1 and "E-EXPORT-UNDEFINED" in res.output
