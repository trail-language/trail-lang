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


def test_run_ignores_unbound_universe_fields(tmp_path):
    # u2 references a frequency the fixture can't serve; model m binds u1 via `on`,
    # so u2's fields must not enter the load plan
    f = tmp_path / "unis.trail"
    f.write_text(
        "universe u1 = stocks where meta.is_active\n"
        "universe u2 = stocks where quarterly.income.revenue > 0\n"
        "model m on u1 { export margin = income.operating_income / income.revenue }\n"
    )
    res = CliRunner().invoke(main, ["run", str(f), "--model", "m"])
    assert res.exit_code == 0 and "margin" in res.output


def test_run_ignores_unserved_frequency_in_another_model(tmp_path):
    # model n references quarterly.* (unservable by the annual fixture); running m must not abort
    f = tmp_path / "multi.trail"
    f.write_text(
        "model m { export margin = income.operating_income / income.revenue }\n"
        "model n at quarterly { export q = quarterly.income.revenue }\n"
    )
    res = CliRunner().invoke(main, ["run", str(f), "--model", "m"])
    assert res.exit_code == 0 and "margin" in res.output


def test_signal_prints_series(tmp_path):
    f = tmp_path / "sig.trail"
    f.write_text("signal s = income.operating_income / income.revenue\n")
    res = CliRunner().invoke(main, ["signal", str(f), "--name", "s"])
    assert res.exit_code == 0 and "s" in res.output


def test_signal_unknown_name_errors(tmp_path):
    f = tmp_path / "sig.trail"
    f.write_text("signal s = income.operating_income / income.revenue\n")
    res = CliRunner().invoke(main, ["signal", str(f), "--name", "nope"])
    assert res.exit_code == 1 and "no signal named 'nope'" in res.output


def test_syntax_error_is_a_clean_message(tmp_path):
    f = tmp_path / "bad.trail"
    f.write_text("model m { a = income.revenue +++ }\n")
    res = CliRunner().invoke(main, ["validate", str(f)])
    assert res.exit_code == 1
    assert "Traceback" not in res.output   # no raw Python traceback
    assert "SYNTAX" in res.output


def test_syntax_error_line_is_relative_to_user_file(tmp_path):
    # error on line 3 of the user's file; stdlib is prepended internally but must not shift the number
    f = tmp_path / "bad.trail"
    f.write_text("model m {\n  a = income.revenue\n  b = = 2\n}\n")
    res = CliRunner().invoke(main, ["validate", str(f)])
    assert res.exit_code == 1 and "line 3" in res.output


def test_typed_marker_is_present():
    from pathlib import Path

    import trail
    assert (Path(trail.__file__).parent / "py.typed").exists()
