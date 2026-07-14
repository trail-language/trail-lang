"""Source drivers, panel loading, and panel conformance.

Drivers are resolved via :mod:`trail.registry` (a registered ``trail.sources`` name or a
dotted import path) to a ``factory(options) -> DataSource``. :func:`load_panel_for` loads
the effective source's panel, checks it against the panel contract, and applies the
period filter. Conformance deviations are a hard error under ``panel.strict``; otherwise
they are warned and coerced.
"""
from __future__ import annotations

import warnings

import polars as pl

from trail.config import Config, ConfigError
from trail.registry import resolve_driver
from trail.schema import active_schema, kind_of
from trail.source import PERIOD_COL, ENTITY_COL, DataSource

__all__ = [
    "FixtureSource",
    "fixture",
    "conform_panel",
    "load_panel_for",
    "resolve_driver",
    "PanelConformanceWarning",
]


class PanelConformanceWarning(UserWarning):
    """A source returned a panel that deviates from the contract (lenient mode)."""


class FixtureSource(DataSource):
    """Deterministic in-memory panel; the zero-config default and test source."""

    name = "fixture"

    def load(self, fields: set[str], *, periods: tuple[int, int] | None = None) -> pl.DataFrame:
        from trail.fixtures import load_panel

        return load_panel()


def fixture(options: dict) -> FixtureSource:
    """Dotted-path factory kept for ``driver: trail.sources.fixture``."""
    return FixtureSource(options)


_INT_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
}


def _is_integer_dtype(dtype) -> bool:
    try:
        return bool(dtype.is_integer())
    except AttributeError:
        return dtype in _INT_DTYPES


def _null_series(name: str, height: int) -> pl.Series:
    kind = kind_of(name) or "flow"
    if kind == "meta":
        dtype = pl.Boolean if name == "meta.is_active" else pl.Utf8
    else:
        dtype = pl.Float64
    return pl.Series(name, [None] * height, dtype=dtype)


def conform_panel(
    panel: pl.DataFrame,
    fields: set[str],
    *,
    strict: bool,
    source_name: str = "",
) -> pl.DataFrame:
    """Check ``panel`` against the panel contract and return a conforming frame.

    Missing ``entity``/``period`` columns are always a hard :class:`ConfigError`
    (``E-SOURCE-PANEL``) - there is nothing to coerce to. Softer deviations (missing
    requested field columns, non-integer ``period``, columns outside the schema) raise
    under ``strict``; otherwise they emit :class:`PanelConformanceWarning`
    (``W-SOURCE-PANEL``) and are coerced: unknown columns dropped, ``period`` cast to an
    integer, and missing requested fields added as all-null columns.
    """
    src = f" '{source_name}'" if source_name else ""
    missing_index = [c for c in (ENTITY_COL, PERIOD_COL) if c not in panel.columns]
    if missing_index:
        raise ConfigError(
            f"E-SOURCE-PANEL source{src} returned a panel missing required "
            f"column(s) {missing_index}"
        )

    issues: list[str] = []
    provided = set(panel.columns)
    missing_fields = sorted(f for f in fields if f not in provided)
    if missing_fields:
        issues.append(f"missing requested field column(s) {missing_fields}")
    allowed = {ENTITY_COL, PERIOD_COL} | set(active_schema())
    extra = sorted(c for c in panel.columns if c not in allowed)
    if extra:
        issues.append(f"unexpected column(s) {extra}")
    if not _is_integer_dtype(panel.schema[PERIOD_COL]):
        issues.append(f"'period' has non-integer dtype {panel.schema[PERIOD_COL]}")

    if not issues:
        return panel
    if strict:
        raise ConfigError(f"E-SOURCE-PANEL source{src} " + "; ".join(issues))

    for msg in issues:
        warnings.warn(f"W-SOURCE-PANEL source{src} {msg}", PanelConformanceWarning, stacklevel=2)
    panel = panel.select([c for c in panel.columns if c in allowed])
    if not _is_integer_dtype(panel.schema[PERIOD_COL]):
        panel = panel.with_columns(pl.col(PERIOD_COL).cast(pl.Int64, strict=False))
    if missing_fields:
        panel = panel.with_columns([_null_series(f, panel.height) for f in missing_fields])
    return panel


def load_panel_for(config: Config, fields: set[str]) -> pl.DataFrame:
    primary = config.precedence["default"][0]  # phase 1: single effective source
    spec = config.sources[primary]
    source = resolve_driver(spec.driver)(spec.options)
    try:
        panel = source.load(fields, periods=config.periods)
    finally:
        source.close()
    panel = conform_panel(panel, fields, strict=config.strict, source_name=primary)
    if config.periods is not None:
        lo, hi = config.periods
        panel = panel.filter((pl.col(PERIOD_COL) >= lo) & (pl.col(PERIOD_COL) <= hi))
    return panel
