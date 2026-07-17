"""Point-in-time alignment: a field with a `__date:*` coordinate is placed by knowability
(row-shift), while a naive field (no coordinate) keeps pre-PIT period-end placement."""
import datetime as dt

import polars as pl
import pytest

from trail.align import LoadedPanel, AlignmentWarning, align_and_merge
from trail.source import date_col

FILING = date_col("filing")  # "__date:filing"


def _d(y, m, d):
    return dt.datetime(y, m, d)


def _panel(cols):
    return pl.DataFrame(cols).with_columns(
        [pl.col(c).cast(pl.Datetime("us")) for c in cols if c == "time" or c.startswith("__date:")])


# two fiscal years of quarterly revenue, each quarter filed ~45 days after its quarter-end.
_QTR_TIME = [_d(2022, 3, 31), _d(2022, 6, 30), _d(2022, 9, 30), _d(2022, 12, 31),
             _d(2023, 3, 31), _d(2023, 6, 30), _d(2023, 9, 30), _d(2023, 12, 31)]
_QTR_FILING = [_d(2022, 5, 15), _d(2022, 8, 14), _d(2022, 11, 14), _d(2023, 2, 14),
               _d(2023, 5, 15), _d(2023, 8, 14), _d(2023, 11, 14), _d(2024, 2, 14)]
_QTR_REV = [1.0, 2, 3, 4, 5, 6, 7, 8]  # FY2022 sums to 10, FY2023 to 26


def test_downsample_row_shift_places_fiscal_year_at_its_filing_year():
    lp = LoadedPanel(
        _panel({"entity": ["X"] * 8, "time": _QTR_TIME, "income.revenue": _QTR_REV,
                FILING: _QTR_FILING}),
        "quarterly", "entity", {"income.revenue": FILING})
    out = align_and_merge([lp], "annual").sort("time")
    by_year = {r["time"].year: r["income.revenue"] for r in out.iter_rows(named=True)}
    # FY2022 (=10) is fully known only once Q4 is filed 2023-02 -> lands on the 2023 decision row;
    # FY2023 (=26) -> the 2024 row. The revenue label is one decision-year behind the fiscal year.
    assert by_year[2023] == 10.0
    assert by_year[2024] == 26.0
    assert by_year.get(2022) is None  # FY2022 not yet fully knowable at end-2022


def test_pit_emits_leading_not_yet_knowable_null_row():
    # PIT is honest about incompleteness: at end-2022 the FY2022 full-year figure is not yet
    # filed, so the 2022 decision row exists (seeded by the in-year quarterly filings) but is
    # null. This is an intended height/null divergence from naive placement - pin it here.
    lp = LoadedPanel(
        _panel({"entity": ["X"] * 8, "time": _QTR_TIME, "income.revenue": _QTR_REV,
                FILING: _QTR_FILING}),
        "quarterly", "entity", {"income.revenue": FILING})
    out = align_and_merge([lp], "annual").sort("time")
    assert out["time"].dt.year().to_list() == [2022, 2023, 2024]  # a 2022 row is present
    assert out["income.revenue"].to_list() == [None, 10.0, 26.0]  # but FY2022 unknowable at end-2022


def test_naive_same_data_places_fiscal_year_at_its_period_end():
    # identical values but NO coordinate: pre-PIT placement (FY at its own period-end year)
    lp = LoadedPanel(
        _panel({"entity": ["X"] * 8, "time": _QTR_TIME, "income.revenue": _QTR_REV}),
        "quarterly", "entity", {})
    out = align_and_merge([lp], "annual").sort("time")
    by_year = {r["time"].year: r["income.revenue"] for r in out.iter_rows(named=True)}
    assert by_year[2022] == 10.0 and by_year[2023] == 26.0  # no shift
    assert 2024 not in by_year


def test_upsample_asof_carries_value_from_filing_not_period_end():
    # annual FY2022 revenue filed 2023-02-14 onto a quarterly grid: invisible until filed
    grid_px = _panel({"entity": ["X"] * 3, "time": [_d(2022, 12, 31), _d(2023, 3, 31), _d(2023, 6, 30)],
                      "price.adj_close": [10.0, 11.0, 12.0]})
    stmt = LoadedPanel(
        _panel({"entity": ["X"], "time": [_d(2022, 12, 31)], "income.net_income": [100.0],
                FILING: [_d(2023, 2, 14)]}),
        "annual", "entity", {"income.net_income": FILING})
    out = align_and_merge([LoadedPanel(grid_px, "quarterly"), stmt], "quarterly").sort("time")
    ni = {r["time"].date(): r["income.net_income"] for r in out.iter_rows(named=True)}
    assert ni[dt.date(2022, 12, 31)] is None      # filed 2023-02, unknown at 2022 year-end
    assert ni[dt.date(2023, 3, 31)] == 100.0       # known by Q1-2023
    assert ni[dt.date(2023, 6, 30)] == 100.0


def test_mixed_coordinates_in_one_panel_align_independently():
    # one source, one fetch: statements align on filing, price aligns naive (period-end)
    lp = LoadedPanel(
        _panel({"entity": ["X"] * 2, "time": [_d(2022, 12, 31), _d(2023, 12, 31)],
                "income.net_income": [50.0, 60.0], "price.adj_close": [10.0, 11.0],
                FILING: [_d(2023, 2, 14), _d(2024, 2, 14)]}),
        "annual", "entity", {"income.net_income": FILING})  # price naive
    out = align_and_merge([lp], "annual").sort("time")
    rows = {r["time"].year: r for r in out.iter_rows(named=True)}
    # price (naive) stays at its period-end year; net_income (filing) shifts one year forward
    assert rows[2022]["price.adj_close"] == 10.0
    assert rows[2023]["price.adj_close"] == 11.0
    assert rows[2023]["income.net_income"] == 50.0   # FY2022 filed 2023
    assert rows[2024]["income.net_income"] == 60.0   # FY2023 filed 2024


def test_restatement_keeps_latest_filing_not_double_sum():
    # FY2022 filed twice (original + restatement); the latest filing's value wins, no double count
    lp = LoadedPanel(
        _panel({"entity": ["X", "X"], "time": [_d(2022, 12, 31), _d(2022, 12, 31)],
                "income.revenue": [100.0, 120.0], FILING: [_d(2023, 2, 14), _d(2023, 8, 20)]}),
        "annual", "entity", {"income.revenue": FILING})
    out = align_and_merge([lp], "annual").sort("time")
    revs = out["income.revenue"].to_list()
    assert 220.0 not in revs             # NOT summed
    assert out.filter(pl.col("income.revenue") == 120.0).height == 1  # restated value kept


def test_stock_last_under_coordinate_takes_year_end_balance():
    # quarterly balance (a stock) -> annual: last quarter's value, shifted by filing
    lp = LoadedPanel(
        _panel({"entity": ["X"] * 4, "time": _QTR_TIME[:4],
                "balance.total_assets": [10.0, 20, 30, 40], FILING: _QTR_FILING[:4]}),
        "quarterly", "entity", {"balance.total_assets": FILING})
    out = align_and_merge([lp], "annual").sort("time")
    row = {r["time"].year: r["balance.total_assets"] for r in out.iter_rows(named=True)}
    assert row[2023] == 40.0  # Q4 (last), FY2022 known once Q4 filed 2023-02


def test_null_coordinate_falls_back_to_period_end_with_warning():
    lp = LoadedPanel(
        _panel({"entity": ["X", "X"], "time": [_d(2022, 12, 31), _d(2023, 12, 31)],
                "income.net_income": [50.0, 60.0], FILING: [None, _d(2024, 2, 14)]}),
        "annual", "entity", {"income.net_income": FILING})
    with pytest.warns(AlignmentWarning, match="W-PIT-PARTIAL"):
        out = align_and_merge([lp], "annual").sort("time")
    by_year = {r["time"].year: r["income.net_income"] for r in out.iter_rows(named=True)}
    assert by_year[2022] == 50.0   # no filing date -> placed at its period-end (naive)
    assert by_year[2024] == 60.0   # filed 2024 -> shifted


def test_date_coordinate_never_reaches_output():
    lp = LoadedPanel(
        _panel({"entity": ["X"], "time": [_d(2022, 12, 31)], "income.revenue": [100.0],
                FILING: [_d(2023, 2, 14)]}),
        "annual", "entity", {"income.revenue": FILING})
    out = align_and_merge([lp], "annual")
    assert not [c for c in out.columns if c.startswith("__date:")]  # consumed + dropped
