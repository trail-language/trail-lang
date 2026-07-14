"""Canonical field registry and the pluggable active schema.

Core fields are the language's built-in vocabulary. A data-source package may contribute
additional fields (e.g. gmd.*) via the `trail.schema` entry-point group; each entry point
resolves to a mapping of dotted column -> kind string. `active_schema()` merges core with all
installed contributions, and is what validation, catalog, and panel conformance read.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata

SCHEMA_ENTRY_POINT_GROUP = "trail.schema"


@dataclass(frozen=True)
class FieldSpec:
    column: str
    kind: str  # core: flow | stock | ratio | per_share | price | meta; plugins may add others


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

_CORE: dict[str, FieldSpec] = {c: FieldSpec(c, k) for c, k in _FIELDS}

#: core registry kept under a stable name; prefer active_schema() to include plugin fields
SCHEMA: dict[str, FieldSpec] = _CORE


@lru_cache(maxsize=1)
def _plugin_fields() -> dict[str, FieldSpec]:
    """Fields contributed by installed `trail.schema` entry points (column -> FieldSpec)."""
    out: dict[str, FieldSpec] = {}
    for ep in metadata.entry_points(group=SCHEMA_ENTRY_POINT_GROUP):
        try:
            contributed = ep.load()
        except Exception:
            continue
        for column, kind in dict(contributed).items():
            if column not in _CORE:
                out[str(column)] = FieldSpec(str(column), str(kind))
    return out


def active_schema() -> dict[str, FieldSpec]:
    """Core fields plus all installed `trail.schema` contributions (core wins on collision)."""
    plugins = _plugin_fields()
    return {**plugins, **_CORE} if plugins else dict(_CORE)


def is_field(column: str) -> bool:
    return column in _CORE or column in _plugin_fields()


def kind_of(column: str) -> str | None:
    spec = _CORE.get(column) or _plugin_fields().get(column)
    return spec.kind if spec else None
