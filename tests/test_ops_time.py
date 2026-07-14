import polars as pl
import pytest

from trail.ops import TIME, ENTITY, build

# 2 entities x 4 periods; X has a negative->positive transition to exercise the shift rule
_DF = pl.DataFrame({
    ENTITY: ["X"] * 4 + ["Y"] * 4,
    TIME: [1, 2, 3, 4] * 2,
    "v": [-10.0, 20.0, 30.0, 60.0, 100.0, 110.0, 121.0, 133.1],
}).sort([ENTITY, TIME])


def _col(expr):
    return _DF.with_columns(expr.alias("out"))["out"].to_list()


def test_lag_respects_entity_groups():
    out = _col(build("lag", [pl.col("v"), 1], {}, None))
    assert out[0] is None and out[1] == -10.0
    assert out[4] is None  # Y's first period must not see X's last value


def test_roll_mean_full_windows_only():
    out = _col(build("roll_mean", [pl.col("v"), 2], {}, None))
    assert out[0] is None and out[1] == pytest.approx(5.0)  # (-10+20)/2
    assert out[7] == pytest.approx((121.0 + 133.1) / 2)


def test_cummax_primitive():
    df = pl.DataFrame({ENTITY: ["X"] * 3, TIME: [1, 2, 3], "v": [10.0, 8.0, 12.0]})
    out = df.with_columns(build("cummax", [pl.col("v")], {}, None).alias("out"))["out"].to_list()
    assert out == pytest.approx([10.0, 10.0, 12.0])

# yoy/avg2/cagr/increase/drawdown are now DERIVED macros (stdlib/timeseries.trail),
# tested in tests/test_timeseries.py against known values.
