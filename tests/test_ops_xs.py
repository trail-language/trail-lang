import polars as pl
import pytest

from trail.ops import PERIOD, SEC, build

# one period, 4 securities, two sectors
_DF = pl.DataFrame({
    SEC: ["A", "B", "C", "D"],
    PERIOD: [2024] * 4,
    "meta.sector": ["Tech", "Tech", "Energy", "Energy"],
    "v": [1.0, 3.0, 10.0, 30.0],
}).sort([SEC, PERIOD])


def _col(expr):
    return _DF.with_columns(expr.alias("out"))["out"].to_list()


def test_zscore_within_period():
    out = _col(build("zscore", [pl.col("v")], {}, None))
    mean, std = 11.0, pl.Series([1.0, 3.0, 10.0, 30.0]).std()
    assert out[0] == pytest.approx((1.0 - mean) / std)


def test_zscore_by_sector():
    out = _col(build("zscore", [pl.col("v")], {}, ("meta", "sector")))
    tech_std = pl.Series([1.0, 3.0]).std()
    assert out[0] == pytest.approx((1.0 - 2.0) / tech_std)  # A vs Tech only


def test_rank():
    assert _col(build("rank", [pl.col("v")], {}, None)) == [1.0, 2.0, 3.0, 4.0]
    # pctile is now a derived macro (stdlib) — see tests/test_timeseries.py


def test_xs_frac_broadcasts():
    out = _col(build("xs_frac", [pl.col("v") > 5], {}, None))
    assert out == pytest.approx([0.5] * 4)  # 2 of 4 above 5, same value for every row
