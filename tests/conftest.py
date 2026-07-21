"""Test-only field vocabulary.

Approach X: the language ships no domain vocabulary - every source declares its own fields via the
`trail.schema` entry point. The unit tests, however, drive synthetic panels with the classic
income/balance/cash/price columns. This conftest registers that vocabulary (with kinds) into
`schema._CORE` for the test session - exactly as an installed data-source adapter would contribute a
schema - so the tests keep a stable field set without depending on any adapter being installed.
Nothing here ships in the package.
"""
from trail import schema

_FLOW_INCOME = ("revenue cogs gross_profit operating_income net_income interest_expense "
                "income_tax_expense income_before_tax ebitda depreciation_amortization sga").split()
_STOCK_INCOME = ("weighted_average_shares_diluted", "weighted_average_shares")
_BALANCE = ("total_assets current_assets other_current_assets current_liabilities total_liabilities "
            "long_term_debt total_debt total_equity retained_earnings accounts_receivable inventory "
            "accounts_payable net_fixed_assets cash_and_equivalents cash_and_short_term_investments "
            "minority_interest common_stock goodwill").split()
_CASH = "cfo capex free_cash_flow stock_issued cfi cff net_change_in_cash dividends_paid".split()

_TEST_VOCAB: dict[str, str] = {
    **{f"income.{f}": "flow" for f in _FLOW_INCOME},
    "income.eps_diluted": "per_share",
    **{f"income.{f}": "stock" for f in _STOCK_INCOME},
    **{f"balance.{f}": "stock" for f in _BALANCE},
    **{f"cash.{f}": "flow" for f in _CASH},
    "price.adj_close": "price",
    "price.dividends": "per_share",
}

# Patch at import (before collection/validation): mirrors adapter-contributed fields into the core map.
schema._CORE.update({c: schema.FieldSpec(c, k) for c, k in _TEST_VOCAB.items()})
