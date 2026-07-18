"""Quant stdlib defs (macros): pow, normal_cdf, and the risk-metric family
(max_drawdown, ann_sharpe, sortino, calmar) computed on a per-entity return series.
`ann_sharpe` is the whole-series annualized Sharpe (the name `sharpe` is already the
rolling variant in core.trail, left unchanged).
Exercised end-to-end through prepare() -> compile_model()."""
import datetime as dt
import math
from statistics import NormalDist

import polars as pl
import pytest

from trail import ast
from trail.compiler import compile_model
from trail.pipeline import prepare
from trail.validate import validate

_ND = NormalDist()
_Q = [dt.datetime(y, m, 1) for y in (2020, 2021, 2022) for m in (3, 6, 9, 12)]


def _run(exports: str, panel: pl.DataFrame) -> pl.DataFrame:
    """Compile a model over a synthetic panel; income.revenue carries the test series."""
    prog = prepare("model m {\n" + exports + "\n}\n")   # stdlib implicit
    assert not [i for i in validate(prog) if i.severity == "error"]
    model = next(d for d in prog.decls if isinstance(d, ast.ModelDecl))
    return compile_model(model, {}).run(panel)


def _panel(entity: str, series: list[float]) -> pl.DataFrame:
    return pl.DataFrame({
        "entity": [entity] * len(series),
        "time": _Q[: len(series)],
        "income.revenue": series,
    }).with_columns(pl.col("time").cast(pl.Datetime("us"))).sort(["entity", "time"])


# --- pow: composes the `^` operator ---

def test_pow_known_values():
    out = _run(
        "export a = pow(2, 10)\n"    # 1024
        "export b = pow(9, 0.5)\n",  # 3
        _panel("Z", [1.0, 2.0]),
    )
    assert out["a"][0] == pytest.approx(1024.0)
    assert out["b"][0] == pytest.approx(3.0)


# --- normal_cdf: 0.5 * (1 + erf(x / sqrt(2))) ---

def test_normal_cdf_known_values():
    out = _run(
        "export z = normal_cdf(income.revenue - income.revenue)\n"   # normal_cdf(0) = 0.5
        "export hi = normal_cdf(1.96)\n"
        "export lo = normal_cdf(-1.96)\n",
        _panel("Z", [5.0, 6.0]),
    )
    assert out["z"][0] == pytest.approx(0.5, abs=1e-9)
    assert out["hi"][0] == pytest.approx(_ND.cdf(1.96), abs=1e-6)
    assert out["lo"][0] == pytest.approx(_ND.cdf(-1.96), abs=1e-6)


# --- max_drawdown: leading-1.0 equity curve (inception drawdown not discarded) ---

def test_max_drawdown_seeds_peak_at_inception():
    # returns [-0.5, 1.0]: equity 0.5 then 1.0; peak seeded at 1.0 -> trough drawdown -0.5
    out = _run("export md = max_drawdown(income.revenue)", _panel("Z", [-0.5, 1.0]))
    assert out["md"][0] == pytest.approx(-0.5)


def test_max_drawdown_compounding_series():
    # returns [-0.1, -0.1, -0.1]: equity 0.9, 0.81, 0.729 -> mdd 0.729 - 1 = -0.271
    out = _run("export md = max_drawdown(income.revenue)", _panel("Z", [-0.1, -0.1, -0.1]))
    assert out["md"][0] == pytest.approx(-0.271, abs=1e-12)


def test_max_drawdown_is_nonpositive_and_per_entity():
    panel = pl.concat([_panel("A", [-0.2, 0.1, 0.05]), _panel("B", [0.3, -0.4, 0.2])])
    out = _run("export md = max_drawdown(income.revenue)", panel)
    assert all(v <= 0.0 for v in out["md"].to_list())
    a = out.filter(pl.col("entity") == "A")["md"][0]
    b = out.filter(pl.col("entity") == "B")["md"][0]
    assert a == pytest.approx(-0.2)     # first-period 20% loss from inception
    assert b == pytest.approx(-0.4)     # 40% loss after the initial gain


# --- ann_sharpe / sortino / calmar over a per-entity return series ---

def test_ann_sharpe_matches_manual_formula():
    rets = [0.01, -0.02, 0.03, 0.0, 0.015]
    out = _run("export s = ann_sharpe(income.revenue, 252)", _panel("Z", rets))
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)   # ddof=1
    expect = mean / math.sqrt(var) * math.sqrt(252)
    assert out["s"][0] == pytest.approx(expect, rel=1e-9)


def test_sortino_uses_downside_deviation():
    rets = [0.01, -0.02, 0.03, 0.0, 0.015]
    out = _run("export s = sortino(income.revenue, 252)", _panel("Z", rets))
    mean = sum(rets) / len(rets)
    dsd = math.sqrt(sum(min(r, 0.0) ** 2 for r in rets) / len(rets))
    expect = mean / dsd * math.sqrt(252)
    assert out["s"][0] == pytest.approx(expect, rel=1e-9)


def test_calmar_is_annualized_return_over_max_drawdown():
    rets = [-0.1, -0.1, -0.1]
    out = _run("export c = calmar(income.revenue, 4)", _panel("Z", rets))
    mean = sum(rets) / len(rets)
    mdd = -0.271                       # from the compounding series above
    expect = mean * 4 / (-mdd)
    assert out["c"][0] == pytest.approx(expect, rel=1e-6)


def test_preexisting_sharpe_and_drawdown_are_unchanged():
    # the new family did not redefine the rolling `sharpe(r, n)` or the level `drawdown(x)`;
    # both keep their original (window / level) semantics and coexist with the new metrics.
    levels = [1.0, 0.8, 1.2, 0.6]      # a price/equity LEVEL series
    out = _run(
        "export sh = sharpe(income.revenue, 2)\n"       # rolling: roll_mean/roll_std over 2
        "export dd = drawdown(income.revenue)\n",       # level: x/cummax(x) - 1
        _panel("Z", levels),
    )
    got = out.sort("time")["dd"].to_list()
    # drawdown of the LEVEL series against its running peak: 0, -0.2, 0, -0.5
    assert got == [pytest.approx(0.0), pytest.approx(-0.2), pytest.approx(0.0), pytest.approx(-0.5)]
    # rolling sharpe over a 2-window is defined (non-null once the window fills)
    assert out.sort("time")["sh"].to_list()[-1] is not None
