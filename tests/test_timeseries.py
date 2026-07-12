"""Migrated derived time-series operators (stdlib/timeseries.trail), tested via prepare()."""
import polars as pl
import pytest

from trail import ast
from trail.compiler import compile_model
from trail.fixtures import load_panel
from trail.pipeline import prepare


def _run(exports: str):
    prog = prepare("model m {\n" + exports + "\n}\n")
    model = next(d for d in prog.decls if isinstance(d, ast.ModelDecl))
    return compile_model(model, {}).run(load_panel())


def _cell(df, col, sec, period):
    return df.filter((pl.col("security") == sec) & (pl.col("period") == period))[col][0]


def test_yoy_avg2_drawdown():
    df = _run(
        "export y = yoy(income.revenue)\n"          # AAA 2018: 110/100-1 = 0.10
        "export a = avg2(income.revenue)\n"         # AAA 2018: (100+110)/2 = 105
        "export d = drawdown(income.revenue)\n"     # AAA monotonic up -> 0
    )
    assert _cell(df, "y", "AAA", 2018) == pytest.approx(0.10)
    assert _cell(df, "a", "AAA", 2018) == pytest.approx(105.0)
    assert _cell(df, "d", "AAA", 2024) == pytest.approx(0.0)  # revenue only rises -> no drawdown


def test_cagr_negative_shift_rule():
    # EEE revenue shrinks 3%/yr -> cagr < 0; AAA grows 10%/yr -> cagr = 0.10 exactly (4y)
    df = _run("export c = cagr(income.revenue, 4)")
    assert _cell(df, "c", "AAA", 2024) == pytest.approx(0.10)
    assert _cell(df, "c", "EEE", 2024) < 0

    # explicit negative-straddle case via a synthetic series is covered by the engine
    # equivalence check; here we confirm the macro is wired and monotone-correct.


def test_pctile_and_beta_defined():
    df = _run(
        "export p = pctile(income.revenue)\n"                       # in [0,1]
        "export b = beta(income.net_income, income.revenue, 4)\n"   # net_income = 0.12*rev -> beta ~ 0.12
    )
    ps = [v for v in df["p"].to_list() if v is not None]
    assert ps and all(0.0 <= v <= 1.0 for v in ps)
    bs = [v for v in df["b"].to_list() if v is not None]
    # net_income = 0.12*revenue exactly. beta = roll_cov (population) / roll_var (sample, ddof=1)
    # = 0.12 * (n-1)/n = 0.09 at n=4. This population/sample convention is inherited verbatim
    # from the original engine builtin (migration preserved behavior; see FINDINGS note).
    assert bs and all(v == pytest.approx(0.12 * 3 / 4, abs=1e-9) for v in bs)
