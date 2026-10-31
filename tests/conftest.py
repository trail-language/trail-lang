"""Test-only field vocabulary.

Approach X: the language ships no domain vocabulary - every source declares its own fields via the
`trail.schema` entry point. The unit tests, however, drive synthetic panels with the classic
income/balance/cash/price columns. This conftest registers that vocabulary (with kinds) into
`schema._CORE` for the test session - exactly as an installed data-source adapter would contribute a
schema - so the tests keep a stable field set without depending on any adapter being installed.
The vocabulary lists live in trail.fixtures (the shipped fixture's own schema declaration), so the
test session and the runtime can never drift. Nothing here ships in the package.
"""
from trail import fixtures, schema

_TEST_VOCAB: dict[str, str] = {
    **{f"income.{f}": "flow" for f in fixtures._FLOW_INCOME},
    "income.eps_diluted": "per_share",
    **{f"income.{f}": "stock" for f in fixtures._STOCK_INCOME},
    **{f"balance.{f}": "stock" for f in fixtures._BALANCE},
    **{f"cash.{f}": "flow" for f in fixtures._CASH},
    "price.adj_close": "price",
    "price.dividends": "per_share",
}

# Patch at import (before collection/validation): mirrors adapter-contributed fields into the core map.
schema._CORE.update({c: schema.FieldSpec(c, k) for c, k in _TEST_VOCAB.items()})
