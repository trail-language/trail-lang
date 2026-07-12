"""Canonical field registry (phase-1 subset of the language reference §4.2 namespaces)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpec:
    column: str
    kind: str  # flow | stock | ratio | per_share | price | meta


_FIELDS: list[tuple[str, str]] = [
    # income
    ("income.revenue", "flow"),
    ("income.cogs", "flow"),
    ("income.gross_profit", "flow"),
    ("income.operating_income", "flow"),
    ("income.net_income", "flow"),
    ("income.interest_expense", "flow"),
    ("income.income_tax_expense", "flow"),
    ("income.income_before_tax", "flow"),
    ("income.eps_diluted", "per_share"),
    ("income.weighted_average_shares_diluted", "stock"),
    # balance
    ("balance.total_assets", "stock"),
    ("balance.current_assets", "stock"),
    ("balance.current_liabilities", "stock"),
    ("balance.total_liabilities", "stock"),
    ("balance.long_term_debt", "stock"),
    ("balance.total_debt", "stock"),
    ("balance.total_equity", "stock"),
    ("balance.retained_earnings", "stock"),
    ("balance.accounts_receivable", "stock"),
    ("balance.inventory", "stock"),
    ("balance.accounts_payable", "stock"),
    # cash
    ("cash.cfo", "flow"),
    ("cash.capex", "flow"),
    ("cash.free_cash_flow", "flow"),
    ("cash.stock_issued", "flow"),
    # price & meta
    ("price.adj_close", "price"),
    ("meta.sector", "meta"),
    ("meta.exchange", "meta"),
    ("meta.market_cap", "meta"),
    ("meta.is_active", "meta"),
]

SCHEMA: dict[str, FieldSpec] = {c: FieldSpec(c, k) for c, k in _FIELDS}


def is_field(column: str) -> bool:
    return column in SCHEMA
