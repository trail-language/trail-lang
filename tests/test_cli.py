from click.testing import CliRunner

from trail.cli import main

GOOD = "model m { export margin = income.operating_income / income.revenue }\n"
BAD = "model m { a = income.bogus }\n"


def test_validate_ok(tmp_path):
    f = tmp_path / "good.trail"
    f.write_text(GOOD)
    res = CliRunner().invoke(main, ["validate", str(f)])
    assert res.exit_code == 0 and "OK" in res.output


def test_validate_error_exit_code(tmp_path):
    f = tmp_path / "bad.trail"
    f.write_text(BAD)
    res = CliRunner().invoke(main, ["validate", str(f)])
    assert res.exit_code == 1 and "E-FIELD-UNKNOWN" in res.output


def test_run_prints_exports(tmp_path):
    f = tmp_path / "good.trail"
    f.write_text(GOOD)
    res = CliRunner().invoke(main, ["run", str(f), "--model", "m"])
    assert res.exit_code == 0 and "margin" in res.output
