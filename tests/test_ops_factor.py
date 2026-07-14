import polars as pl
import pytest

from trail.ops import TIME, ENTITY, build

_DF = pl.DataFrame({
    ENTITY: ["A"] * 5 + ["B"] * 5,
    TIME: [1, 2, 3, 4, 5] * 2,
    "v": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
}).sort([ENTITY, TIME])


def _col(expr):
    return _DF.with_columns(expr.alias("o"))["o"].to_list()


def test_decay_linear_full_windows_and_weighting():
    out = _col(build("decay_linear", [pl.col("v"), 3], {}, None))
    assert out[0] is None and out[1] is None            # needs a full 3-window
    # window [1,2,3], weights [1,2,3] -> (1*1+2*2+3*3)/6 = 14/6
    assert out[2] == pytest.approx(14 / 6)
    assert out[5] is None  # B's first period must not borrow A's tail


def test_ewm_mean_is_causal_per_entity():
    out = _col(build("ewm_mean", [pl.col("v"), 3], {}, None))
    assert out[0] == pytest.approx(1.0)   # first value = itself
    assert out[5] == pytest.approx(10.0)  # B starts fresh
    assert out[1] == pytest.approx((2 + (1 - 0.5)) / (1 + 0.5))  # alpha=2/(span+1)=0.5


def test_ewm_std_exists():
    out = _col(build("ewm_std", [pl.col("v"), 3], {}, None))
    assert out[0] == pytest.approx(0.0)
