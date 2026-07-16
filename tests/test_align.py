import datetime as dt

import polars as pl

from trail.align import align_and_merge, finest

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
