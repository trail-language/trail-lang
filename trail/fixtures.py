"""Deterministic fixture panel: 6 entities x FY2017-2024, closed-form values."""
from __future__ import annotations

import datetime as dt

import polars as pl

_SECS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
_SECTOR = {"AAA": "Tech", "BBB": "Tech", "CCC": "Tech", "DDD": "Energy", "EEE": "Energy", "FFF": "Financials"}
_EXCH = {"AAA": "NASDAQ", "BBB": "NASDAQ", "CCC": "NYSE", "DDD": "NYSE", "EEE": "NYSE", "FFF": "NYSE"}
_COUNTRY = {"AAA": "USA", "BBB": "USA", "CCC": "USA", "DDD": "CAN", "EEE": "CAN", "FFF": "GBR"}
_BASE = {"AAA": 100.0, "BBB": 200.0, "CCC": 150.0, "DDD": 300.0, "EEE": 250.0, "FFF": 400.0}
_GROWTH = {"AAA": 0.10, "BBB": 0.05, "CCC": 0.08, "DDD": 0.02, "EEE": -0.03, "FFF": 0.04}
YEARS = list(range(2017, 2025))


def load_panel() -> pl.DataFrame:
    rows = []
    for i, sec in enumerate(_SECS):
        for t, year in enumerate(YEARS):
            rev = _BASE[sec] * (1 + _GROWTH[sec]) ** t
            row = {
                "entity": sec,
                "time": dt.datetime(year, 12, 31),
                "income.revenue": rev,
                "income.cogs": rev * 0.55,
                "income.gross_profit": rev * 0.45,
                "income.operating_income": rev * 0.20,
                "income.net_income": rev * 0.12,
                "income.interest_expense": None if sec == "FFF" else rev * 0.02,
                "income.income_tax_expense": rev * 0.03,
                "income.income_before_tax": rev * 0.15,
                "income.eps_diluted": rev * 0.12 / (10 + i),
                "income.weighted_average_shares_diluted": float(10 + i),
                "income.weighted_average_shares": float(10 + i) - 0.5,
                "income.depreciation_amortization": rev * 0.04,
                "income.ebitda": rev * 0.24,  # operating_income + d&a
                "income.sga": rev * 0.10,
                "balance.total_assets": rev * 2.0,
                "balance.current_assets": rev * 0.8,
                "balance.current_liabilities": rev * 0.5,
                "balance.total_liabilities": rev * 1.2,
                "balance.long_term_debt": rev * 0.6 * (0.98**t),
                "balance.total_debt": rev * 0.7,
                "balance.total_equity": rev * 0.8,
                "balance.retained_earnings": rev * 0.4,
                "balance.accounts_receivable": rev * 0.15,
                "balance.inventory": rev * 0.10,
                "balance.accounts_payable": rev * 0.12,
                "balance.net_fixed_assets": rev * 0.9,
                "balance.cash_and_equivalents": rev * 0.25,
                "balance.cash_and_short_term_investments": rev * 0.30,
                "balance.minority_interest": rev * 0.02,
                "balance.common_stock": rev * 0.05,
                "balance.goodwill": rev * 0.20,
                "cash.cfo": rev * 0.15,
                "cash.capex": rev * 0.05,
                "cash.free_cash_flow": rev * 0.10,
                "cash.stock_issued": None if sec == "AAA" else 0.0,
                "cash.cfi": rev * -0.06,
                "cash.cff": rev * -0.04,
                "cash.net_change_in_cash": rev * 0.05,  # cfo + cfi + cff
                "cash.dividends_paid": rev * 0.03,
                "price.adj_close": rev * 0.3 + 5 * i,
                "price.dividends": rev * 0.03 / (10 + i),  # dividends_paid / diluted shares
                "meta.sector": _SECTOR[sec],
                "meta.exchange": _EXCH[sec],
                "meta.market_cap": rev * 3.0e6,
                "meta.is_active": True,
                "meta.country": _COUNTRY[sec],
            }
            rows.append(row)
    df = pl.DataFrame(rows).with_columns(pl.col("time").cast(pl.Datetime("us")))
    return df.sort(["entity", "time"])
