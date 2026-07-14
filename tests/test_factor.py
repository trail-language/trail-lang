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


def test_ntile_buckets_are_balanced_and_ordered():
    df = _run("export b = ntile(income.revenue, 3)")
    assert set(df["b"].to_list()) <= {1.0, 2.0, 3.0}
    y = df.filter(pl.col("period") == 2024)
    by_sec = {r["entity"]: r["b"] for r in y.iter_rows(named=True)}
    # 6 securities, k=3 -> 2 per bucket; lowest revenue in bucket 1, highest in 3
    assert by_sec["FFF"] == 3.0  # FFF has the largest revenue
    assert min(by_sec.values()) == 1.0 and max(by_sec.values()) == 3.0


def test_scale_is_l1_normalized():
    df = _run("export s = scale(income.revenue)")
    per_period = df.with_columns(pl.col("s").abs().alias("a")).group_by("period").agg(pl.col("a").sum())
    for v in per_period["a"].to_list():
        assert v == pytest.approx(1.0)


def test_xs_corr_self_is_one():
    df = _run("export c = xs_corr(income.revenue, income.revenue)")
    for v in df["c"].to_list():
        assert v == pytest.approx(1.0)


def test_neutralize_residual_is_orthogonal():
    # residual of x on itself is ~0; residual is uncorrelated with the factor
    df = _run(
        "export self0 = neutralize(income.revenue, income.revenue)\n"
        "export cov = xs_cov(neutralize(income.net_income, income.revenue), income.revenue)"
    )
    assert all(abs(v) < 1e-6 for v in df["self0"].to_list())
    for v in df["cov"].to_list():
        assert v == pytest.approx(0.0, abs=1e-6)
