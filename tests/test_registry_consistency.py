"""The function registry (ops.OPS) is the single source of truth - guard against drift."""
from trail.catalog import _FUNC_META
from trail.ops import OPS
from trail.validate import KNOWN_FUNCTIONS


def test_validate_arities_derive_from_registry():
    assert set(KNOWN_FUNCTIONS) == set(OPS)
    for name, spec in OPS.items():
        assert KNOWN_FUNCTIONS[name] == (spec.lo, spec.hi)


def test_catalog_meta_derives_from_registry():
    assert set(_FUNC_META) == set(OPS)
    for name, (axis, summary) in _FUNC_META.items():
        assert axis and summary, f"{name} has empty catalog metadata"


def test_previously_missing_functions_are_documented():
    # the completeness review found these five with blank catalog entries, and
    # ttm/trailing absent entirely
    for name in ("asof", "to_annual", "to_quarterly", "to_monthly", "to_daily", "ttm", "trailing"):
        axis, summary = _FUNC_META[name]
        assert axis == "time-series" and summary
