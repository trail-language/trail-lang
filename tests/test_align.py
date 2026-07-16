import datetime as dt

import polars as pl

import pytest

import trail.schema as schema
from trail.align import AlignmentWarning, align_and_merge, finest
from trail.config import ConfigError
from trail.schema import FieldSpec
from trail.source import BROADCAST_ENTITY


@pytest.fixture
def macro_plugin(monkeypatch):
    monkeypatch.setattr(schema, "_plugin_fields",
                        lambda: {"macro.risk_free": FieldSpec("macro.risk_free", "rate")})

_Q = [
    dt.datetime(2022, 3, 31), dt.datetime(2022, 6, 30), dt.datetime(2022, 9, 30), dt.datetime(2022, 12, 31),
    dt.datetime(2023, 3, 31), dt.datetime(2023, 6, 30), dt.datetime(2023, 9, 30), dt.datetime(2023, 12, 31),
]
_A = [dt.datetime(2022, 12, 31), dt.datetime(2023, 12, 31)]

_QUARTERLY = pl.DataFrame({
    "entity": ["X"] * 8, "time": _Q, "income.revenue": [1.0, 2, 3, 4, 5, 6, 7, 8],
}).with_columns(pl.col("time").cast(pl.Datetime("us")))

_ANNUAL = pl.DataFrame({
    "entity": ["X"] * 2, "time": _A, "balance.total_assets": [100.0, 200.0],
}).with_columns(pl.col("time").cast(pl.Datetime("us")))


def test_finest():
    assert finest(["annual", "quarterly", "monthly"]) == "monthly"
    assert finest(["annual"]) == "annual"


def test_downsample_to_annual_sums_flow_and_keeps_stock():
    out = align_and_merge([(_QUARTERLY, "quarterly"), (_ANNUAL, "annual")], "annual")
    by_year = {r["time"].year: r for r in out.iter_rows(named=True)}
    assert out.height == 2
    assert by_year[2022]["income.revenue"] == 10.0   # flow -> sum of 4 quarters
    assert by_year[2023]["income.revenue"] == 26.0
    assert by_year[2022]["balance.total_assets"] == 100.0  # already annual
    assert by_year[2023]["balance.total_assets"] == 200.0


def test_upsample_annual_onto_quarterly_grid_by_asof():
    out = align_and_merge([(_QUARTERLY, "quarterly"), (_ANNUAL, "annual")], "quarterly").sort("time")
    assert out.height == 8
    assert out["income.revenue"].to_list() == [1, 2, 3, 4, 5, 6, 7, 8]  # native, unchanged
    ta = out["balance.total_assets"].to_list()
    # no annual known before FY2022 filing; FY2022 value carried through 2023 until FY2023
    assert ta[0] is None and ta[1] is None and ta[2] is None
    assert ta[3] == 100.0 and ta[4] == 100.0 and ta[6] == 100.0 and ta[7] == 200.0


def test_single_source_downsample_is_identity_when_target_equals_native():
    out = align_and_merge([(_QUARTERLY, "quarterly")], "quarterly").sort("time")
    assert out["income.revenue"].to_list() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert out.height == 8


_DAYS = [dt.datetime(2023, 1, d) for d in (3, 4, 5, 6, 9)]
_PX = pl.DataFrame({
    "entity": ["AAA"] * 5, "time": _DAYS, "price.adj_close": [10.0, 11, 12, 13, 14],
}).with_columns(pl.col("time").cast(pl.Datetime("us")))
_MACRO = pl.DataFrame({
    "entity": ["AAA"], "time": [dt.datetime(2022, 12, 31)], "balance.total_assets": [500.0],
}).with_columns(pl.col("time").cast(pl.Datetime("us")))


def test_daily_price_plus_annual_macro_no_phantom_rows():
    # the annual macro period-end (2022-12-31) must NOT stamp a row onto the daily grid;
    # its value is carried onto each trading day by backward as-of.
    out = align_and_merge([(_PX, "daily"), (_MACRO, "annual")], "daily").sort("time")
    assert out.height == 5  # exactly the trading days, no 2022-12-31 phantom
    assert out["price.adj_close"].to_list() == [10, 11, 12, 13, 14]
    assert out["balance.total_assets"].to_list() == [500.0] * 5


def test_all_sources_coarser_than_target_falls_back_to_union_grid():
    # user pins `at monthly` but only an annual source exists: still yield its rows
    out = align_and_merge([(_MACRO, "annual")], "monthly")
    assert out.height == 1
    assert out["balance.total_assets"].to_list() == [500.0]


_ANNUAL_FLOW = pl.DataFrame({
    "entity": ["AAA"], "time": [dt.datetime(2022, 12, 31)], "income.revenue": [400.0],
}).with_columns(pl.col("time").cast(pl.Datetime("us")))


def test_upsampling_a_flow_warns_w_upsample_flow():
    # an annual flow carried forward onto a finer grid repeats a total - mis-scales, so warn
    with pytest.warns(AlignmentWarning, match="W-UPSAMPLE-FLOW"):
        align_and_merge([(_PX, "daily"), (_ANNUAL_FLOW, "annual")], "daily")


def test_upsampling_a_stock_does_not_warn(recwarn):
    align_and_merge([(_PX, "daily"), (_MACRO, "annual")], "daily")  # total_assets is a stock
    assert not [w for w in recwarn.list if issubclass(w.category, AlignmentWarning)]


# a global (sentinel-entity) macro series that applies to every stock
_GLOBAL = pl.DataFrame({
    "entity": [BROADCAST_ENTITY, BROADCAST_ENTITY],
    "time": [dt.datetime(2022, 12, 31), dt.datetime(2023, 12, 31)],
    "macro.risk_free": [0.02, 0.03],
}).with_columns(pl.col("time").cast(pl.Datetime("us")))

# two stocks, daily
_PX2 = pl.DataFrame({
    "entity": ["AAA", "AAA", "BBB", "BBB"],
    "time": [dt.datetime(2023, 1, 3), dt.datetime(2023, 1, 4)] * 2,
    "price.adj_close": [10.0, 11.0, 20.0, 21.0],
}).with_columns(pl.col("time").cast(pl.Datetime("us")))


def test_global_series_broadcasts_across_every_grid_entity(macro_plugin, recwarn):
    out = align_and_merge([(_PX2, "daily"), (_GLOBAL, "annual")], "daily").sort(["entity", "time"])
    assert out.height == 4  # 2 stocks x 2 days; the sentinel "*" adds no rows of its own
    assert set(out["entity"].to_list()) == {"AAA", "BBB"}
    # every stock/day carries the FY2022 risk-free as-of (0.02 in effect across early 2023)
    assert out["macro.risk_free"].to_list() == [0.02, 0.02, 0.02, 0.02]
    assert out["price.adj_close"].to_list() == [10.0, 11.0, 20.0, 21.0]
    assert not [w for w in recwarn.list if issubclass(w.category, AlignmentWarning)]  # rate is safe


def test_broadcast_at_or_finer_than_target_aggregates_then_broadcasts(macro_plugin):
    # a daily global series at a daily target: per-day bucket (identity), broadcast to both stocks
    daily_global = pl.DataFrame({
        "entity": [BROADCAST_ENTITY, BROADCAST_ENTITY],
        "time": [dt.datetime(2023, 1, 3), dt.datetime(2023, 1, 4)],
        "macro.risk_free": [0.05, 0.06],
    }).with_columns(pl.col("time").cast(pl.Datetime("us")))
    out = align_and_merge([(_PX2, "daily"), (daily_global, "daily")], "daily").sort(["entity", "time"])
    assert out["macro.risk_free"].to_list() == [0.05, 0.06, 0.05, 0.06]  # by day, across both stocks


def test_all_broadcast_sources_raise_no_entity(macro_plugin):
    # a model backed only by global series has no entities to compute on - reject, do not leak "*"
    with pytest.raises(ConfigError, match="E-NO-ENTITY"):
        align_and_merge([(_GLOBAL, "annual")], "annual")


def test_panel_mixing_sentinel_and_real_entities_raises(macro_plugin):
    mixed = pl.DataFrame({
        "entity": [BROADCAST_ENTITY, "AAA"],
        "time": [dt.datetime(2022, 12, 31), dt.datetime(2022, 12, 31)],
        "macro.risk_free": [0.02, 0.03],
    }).with_columns(pl.col("time").cast(pl.Datetime("us")))
    with pytest.raises(ConfigError, match="E-BROADCAST-MIXED"):
        align_and_merge([(_PX2, "daily"), (mixed, "annual")], "daily")
