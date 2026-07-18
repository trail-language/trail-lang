"""Quant math/stats primitive ops: erf, norm_ppf, count_true, and the full-series
time-series reducers ts_mean/ts_std/ts_min. Tested directly through build() with
known input -> known output, plus registry/validate wiring."""
import datetime as dt
import math
from statistics import NormalDist

import polars as pl
import pytest

from trail.ops import ENTITY, TIME, build
from trail.validate import KNOWN_FUNCTIONS

_ND = NormalDist()


def _col(df, expr):
    return df.with_columns(expr.alias("o"))["o"].to_list()


# --- erf (Abramowitz-Stegun 7.1.26, |err| <= 1.5e-7) ---

def test_erf_known_values():
    df = pl.DataFrame({"x": [0.0, 0.5, 1.0, 2.0, 10.0, -10.0, -1.0]})
    out = _col(df, build("erf", [pl.col("x")], {}, None))
    assert out[0] == 0.0                                   # erf(0) is exactly 0
    assert out[1] == pytest.approx(math.erf(0.5), abs=1.5e-7)
    assert out[2] == pytest.approx(math.erf(1.0), abs=1.5e-7)
    assert out[3] == pytest.approx(math.erf(2.0), abs=1.5e-7)
    assert out[4] == pytest.approx(1.0, abs=1e-9)          # erf(large) -> 1
    assert out[5] == pytest.approx(-1.0, abs=1e-9)         # odd function
    assert out[6] == pytest.approx(-math.erf(1.0), abs=1.5e-7)


def test_erf_accuracy_bound_over_range():
    xs = [i / 50.0 for i in range(-200, 201)]
    out = _col(pl.DataFrame({"x": xs}), build("erf", [pl.col("x")], {}, None))
    assert max(abs(o - math.erf(x)) for o, x in zip(out, xs, strict=True)) <= 1.5e-7


def test_erf_null_propagates():
    out = _col(pl.DataFrame({"x": [None, 1.0]}), build("erf", [pl.col("x")], {}, None))
    assert out[0] is None


# --- norm_ppf (Acklam inverse-normal, ~1e-9 central) ---

def test_norm_ppf_known_values():
    df = pl.DataFrame({"p": [0.975, 0.5, 0.025]})
    out = _col(df, build("norm_ppf", [pl.col("p")], {}, None))
    assert out[0] == pytest.approx(1.959963984540054, abs=1e-6)
    assert out[1] == 0.0                                   # median is exactly 0
    assert out[2] == pytest.approx(-1.959963984540054, abs=1e-6)


def test_norm_ppf_accuracy_central_region():
    ps = [i / 500.0 for i in range(1, 500)]
    out = _col(pl.DataFrame({"p": ps}), build("norm_ppf", [pl.col("p")], {}, None))
    assert max(abs(o - _ND.inv_cdf(p)) for o, p in zip(out, ps, strict=True)) <= 1e-6


def test_norm_ppf_boundary_guards_to_infinity():
    out = _col(pl.DataFrame({"p": [0.0, 1.0, -0.5, 1.5]}), build("norm_ppf", [pl.col("p")], {}, None))
    assert out[0] == float("-inf")   # p <= 0 -> -inf
    assert out[1] == float("inf")    # p >= 1 -> +inf
    assert out[2] == float("-inf")
    assert out[3] == float("inf")


def test_norm_ppf_inverts_normal_cdf():
    # norm_ppf(normal_cdf(x)) ~ x for the standard-normal round trip
    for x in (-2.0, -0.3, 0.0, 0.7, 1.5):
        out = _col(pl.DataFrame({"p": [_ND.cdf(x)]}), build("norm_ppf", [pl.col("p")], {}, None))
        assert out[0] == pytest.approx(x, abs=1e-6)


# --- count_true: null-tolerant variadic boolean count ---

def test_count_true_treats_null_as_false():
    df = pl.DataFrame(
        {"a": [True], "b": [None], "c": [False], "d": [True]},
        schema={"a": pl.Boolean, "b": pl.Boolean, "c": pl.Boolean, "d": pl.Boolean},
    )
    args = [pl.col("a"), pl.col("b"), pl.col("c"), pl.col("d")]
    assert _col(df, build("count_true", args, {}, None))[0] == 2
    # contrast: the existing `count` null-propagates -> the whole count is null
    assert _col(df, build("count", args, {}, None))[0] is None


def test_count_true_single_and_all_false():
    df = pl.DataFrame({"a": [True, False, None]}, schema={"a": pl.Boolean})
    assert _col(df, build("count_true", [pl.col("a")], {}, None)) == [1, 0, 0]


# --- full-series time-series reducers (per entity, broadcast back) ---

_TS = [dt.datetime(2020, 1, i) for i in (1, 2, 3, 4)]


def _panel():
    return pl.DataFrame({
        ENTITY: ["X", "X", "X", "X", "Y", "Y"],
        TIME: _TS + _TS[:2],
        "v": [1.0, 2.0, 3.0, 4.0, 10.0, 30.0],
    }).with_columns(pl.col(TIME).cast(pl.Datetime("us"))).sort([ENTITY, TIME])


def test_ts_mean_std_min_are_per_entity_full_series():
    df = _panel()
    m = _col(df, build("ts_mean", [pl.col("v")], {}, None))
    s = _col(df, build("ts_std", [pl.col("v")], {}, None))
    mn = _col(df, build("ts_min", [pl.col("v")], {}, None))
    # X rows first (sorted): mean 2.5, sample std of [1,2,3,4], min 1; Y: mean 20, min 10
    assert m[:4] == [pytest.approx(2.5)] * 4
    assert m[4:] == [pytest.approx(20.0)] * 2
    assert s[0] == pytest.approx(pl.Series([1.0, 2, 3, 4]).std())     # ddof=1
    assert mn[:4] == [1.0] * 4 and mn[4:] == [10.0] * 2


# --- registry / validate wiring ---

def test_new_ops_registered_with_arity():
    assert KNOWN_FUNCTIONS["erf"] == (1, 1)
    assert KNOWN_FUNCTIONS["norm_ppf"] == (1, 1)
    assert KNOWN_FUNCTIONS["count_true"] == (1, 99)
    assert KNOWN_FUNCTIONS["ts_mean"] == (1, 1)
    assert KNOWN_FUNCTIONS["ts_std"] == (1, 1)
    assert KNOWN_FUNCTIONS["ts_min"] == (1, 1)
