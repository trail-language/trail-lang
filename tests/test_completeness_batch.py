"""Regressions for the completeness review batch (iteration 3)."""
import datetime as dt

import polars as pl
from click.testing import CliRunner

from trail import ast
from trail.cli import main
from trail.compiler import compile_model
from trail.parser import parse_program
from trail.pipeline import prepare
from trail.validate import validate


def _codes(src):
    return [i.code for i in validate(parse_program(src))]


def _run_model(src):
    prog = prepare(src)
    assert not [i for i in validate(prog) if i.severity == "error"]
    model = next(d for d in prog.decls if isinstance(d, ast.ModelDecl))
    q = [dt.datetime(2023, m, 1) for m in (3, 6, 9, 12)]
    panel = pl.DataFrame({
        "entity": ["X"] * 4, "time": q,
        "income.revenue": [1.0, 2, 3, 4],
        "balance.total_assets": [100.0, 110, None, 130],
    }).with_columns(pl.col("time").cast(pl.Datetime("us")))
    return compile_model(model, {}).run(panel).sort("time")


def test_ttm_is_kind_aware():
    out = _run_model(
        "model m {\n"
        "  export rev_ttm = ttm(income.revenue)\n"
        "  export assets_ttm = ttm(balance.total_assets)\n"
        "}"
    )
    assert out["rev_ttm"].to_list()[-1] == 10.0        # flow: 1+2+3+4 summed
    # stock: LAST-KNOWN value, never summed; the None gap forward-fills
    assert out["assets_ttm"].to_list() == [100.0, 110.0, 110.0, 130.0]


def test_out_extension_dispatch(tmp_path):
    f = tmp_path / "m.trail"
    f.write_text("model m { export margin = income.operating_income / income.revenue }\n")
    out = tmp_path / "r.csv"
    res = CliRunner().invoke(main, ["run", str(f), "--model", "m", "--out", str(out)])
    assert res.exit_code == 0
    head = out.read_bytes()[:20]
    assert not head.startswith(b"PAR1")  # was: parquet bytes in a .csv file
    assert b"entity" in head


def test_strategy_run_gives_phase_error(tmp_path):
    f = tmp_path / "s.trail"
    f.write_text(
        "strategy s1 { universe u signal income.revenue rebalance quarterly }\n"
    )
    res = CliRunner().invoke(main, ["run", str(f), "--model", "s1"])
    assert res.exit_code == 1 and "E-PHASE-DEFERRED" in res.output


def test_import_path_does_not_collide_with_names():
    src = 'import "s1"\nstrategy s1 { universe u signal income.revenue rebalance quarterly }\n'
    assert "E-NAME-REBOUND" not in _codes(src)


def test_skew_and_kurtosis_aggregations_valid():
    assert "E-AGG-UNKNOWN" not in _codes('model m { a = resample(income.revenue, "annual", "skew") }')
    assert "E-AGG-UNKNOWN" not in _codes('model m { a = resample(income.revenue, "annual", "kurtosis") }')


def test_stock_flow_lint_covers_price_and_level_kinds():
    assert "W-KIND-STOCK-FLOW" in _codes("model m { a = income.revenue / price.adj_close }")
