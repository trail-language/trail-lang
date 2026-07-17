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


# --- temporal operators (calendar extraction / arithmetic on datetimes) ---
import datetime as _dt  # noqa: E402

_TS = pl.DataFrame({
    "a": [_dt.datetime(2023, 2, 15, 10), _dt.datetime(2023, 11, 30)],
    "b": [_dt.datetime(2023, 2, 10), _dt.datetime(2022, 11, 30)],
})


def _tcol(name, args):
    return _TS.with_columns(build(name, args, {}, None).alias("o"))["o"].to_list()


def test_temporal_extractors():
    assert _tcol("year", [pl.col("a")]) == [2023, 2023]
    assert _tcol("month", [pl.col("a")]) == [2, 11]
    assert _tcol("quarter", [pl.col("a")]) == [1, 4]
    assert _tcol("day", [pl.col("a")]) == [15, 30]


def test_truncate_to_bucket():
    assert _tcol("truncate", [pl.col("a"), "1y"]) == [_dt.datetime(2023, 1, 1), _dt.datetime(2023, 1, 1)]
    assert _tcol("truncate", [pl.col("a"), "1mo"]) == [_dt.datetime(2023, 2, 1), _dt.datetime(2023, 11, 1)]


def test_datediff_days_default_and_hours():
    assert _tcol("datediff", [pl.col("a"), pl.col("b")]) == [5, 365]  # whole days (5d10h -> 5)
    assert _tcol("datediff", [pl.col("a"), pl.col("b"), "hours"])[0] == 5 * 24 + 10


def test_datediff_rejects_unknown_unit():
    with pytest.raises(ValueError, match="datediff unit"):
        build("datediff", [pl.col("a"), pl.col("b"), "fortnights"], {}, None)
