"""A reusable conformance check for data-source adapters.

Any provider imports :func:`assert_source_conforms` in its own test suite to prove its
adapter honors the panel contract, so ``trail-lang`` is the only dependency a provider
needs for contract testing.
"""
from __future__ import annotations

import polars as pl

from trail.schema import kind_of
from trail.source import (
    TIME_COL,
    ENTITY_COL,
    DataSource,
    SupportsCapabilities,
    SupportsDiscovery,
    SupportsUniverse,
)

_INT_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
}
_FLOAT_DTYPES = {pl.Float32, pl.Float64}
_NUMERIC_KINDS = {"flow", "stock", "ratio", "per_share", "price", "level", "rate", "index"}


def _is_temporal(dtype) -> bool:
    return dtype == pl.Date or isinstance(dtype, pl.Datetime)


def _is_numeric(dtype) -> bool:
    return dtype in _INT_DTYPES or dtype in _FLOAT_DTYPES


def assert_source_conforms(
    src: DataSource,
    fields: set[str],
    *,
    expect_rows: bool = True,
) -> None:
    """Assert that ``src`` honors the data-source contract for ``fields``.

    Checks the core panel contract (columns, dtypes, uniqueness) and, when the source
    advertises an extended capability, that capability's basic invariants. Raises
    :class:`AssertionError` on the first violation.
    """
    assert isinstance(src, DataSource), f"{src!r} is not a DataSource"

    if isinstance(src, SupportsDiscovery):
        avail = src.available_fields()
        assert isinstance(avail, set), "available_fields() must return a set"
        unknown = set(fields) - avail
        assert not unknown, f"requested fields absent from available_fields(): {sorted(unknown)}"
        for f in fields:
            info = src.describe_field(f)
            assert info is not None, f"describe_field({f!r}) is None for an available field"
            assert info.available, f"describe_field({f!r}).available is False for a requested field"

    panel = src.load(set(fields))
    assert isinstance(panel, pl.DataFrame), "load() must return a polars DataFrame"
    cols = set(panel.columns)
    for required in (ENTITY_COL, TIME_COL):
        assert required in cols, f"panel missing required column '{required}'"
    missing = set(fields) - cols
    assert not missing, f"panel missing requested field column(s): {sorted(missing)}"

    schema = panel.schema
    assert schema[ENTITY_COL] == pl.Utf8, f"'entity' must be Utf8, got {schema[ENTITY_COL]}"
    assert _is_temporal(schema[TIME_COL]), f"'time' must be temporal (Date/Datetime), got {schema[TIME_COL]}"
    for f in fields:
        if kind_of(f) in _NUMERIC_KINDS:
            assert _is_numeric(schema[f]), f"numeric field '{f}' has non-numeric dtype {schema[f]}"

    if panel.height:
        n_unique = panel.select([ENTITY_COL, TIME_COL]).unique().height
        assert n_unique == panel.height, "rows are not unique on (entity, time)"
    elif expect_rows:
        raise AssertionError("panel has no rows (expected data)")

    if isinstance(src, SupportsCapabilities):
        caps = src.capabilities()
        assert caps.frequency in {"annual", "quarterly", "mixed"}, f"bad frequency {caps.frequency!r}"

    if isinstance(src, SupportsUniverse):
        secs = src.entities()
        assert isinstance(secs, list), "entities() must return a list"

    src.close()
    src.close()  # idempotent
