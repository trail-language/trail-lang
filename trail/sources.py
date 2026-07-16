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

from trail.align import AlignmentWarning, align_and_merge, finest, is_broadcast
from trail.config import Config, ConfigError
from trail.registry import resolve_driver
from trail.schema import active_schema, kind_of
from trail.source import TIME_COL, ENTITY_COL, DataSource, SupportsCapabilities, SupportsDiscovery

__all__ = [
    "FixtureSource",
    "fixture",
    "conform_panel",
    "load_panel_for",
    "resolve_driver",
    "PanelConformanceWarning",
    "AlignmentWarning",
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


#: canonical time dtype (a period-end instant); a Date time column is normalized to this
CANON_TIME = pl.Datetime("us")


def _is_temporal_dtype(dtype) -> bool:
    return dtype == pl.Date or isinstance(dtype, pl.Datetime)


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

    Missing ``entity``/``time`` columns are always a hard :class:`ConfigError`
    (``E-SOURCE-PANEL``) - there is nothing to coerce to. Softer deviations (missing
    requested field columns, a non-temporal ``time``, columns outside the schema) raise
    under ``strict``; otherwise they emit :class:`PanelConformanceWarning`
    (``W-SOURCE-PANEL``) and are coerced: unknown columns dropped and missing requested
    fields added as all-null columns. A ``Date`` ``time`` is normalized to the canonical
    period-end ``Datetime``.
    """
    src = f" '{source_name}'" if source_name else ""
    missing_index = [c for c in (ENTITY_COL, TIME_COL) if c not in panel.columns]
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
    allowed = {ENTITY_COL, TIME_COL} | set(active_schema())
    extra = sorted(c for c in panel.columns if c not in allowed)
    if extra:
        issues.append(f"unexpected column(s) {extra}")
    if not _is_temporal_dtype(panel.schema[TIME_COL]):
        issues.append(f"'time' has non-temporal dtype {panel.schema[TIME_COL]}")

    if strict and issues:
        raise ConfigError(f"E-SOURCE-PANEL source{src} " + "; ".join(issues))
    for msg in issues:
        warnings.warn(f"W-SOURCE-PANEL source{src} {msg}", PanelConformanceWarning, stacklevel=2)
    if issues:
        panel = panel.select([c for c in panel.columns if c in allowed])
        if missing_fields:
            panel = panel.with_columns([_null_series(f, panel.height) for f in missing_fields])
    # normalize a valid temporal time column to the canonical period-end Datetime
    if _is_temporal_dtype(panel.schema[TIME_COL]) and panel.schema[TIME_COL] != CANON_TIME:
        panel = panel.with_columns(pl.col(TIME_COL).cast(CANON_TIME))
    return panel


def _source_freq(src) -> str:
    return src.capabilities().frequency if isinstance(src, SupportsCapabilities) else "annual"


def _source_order(config: Config) -> list[str]:
    """Sources to try, precedence.default first, then any others (for field assignment)."""
    order = list(config.precedence.get("default", []))
    order += [s for s in config.sources if s not in order]
    return order


def load_panel_for(config: Config, fields: set[str], target_freq: str | None = None) -> pl.DataFrame:
    """Load the configured sources, assign each requested field to its first provider
    (precedence), align every source panel to the target frequency, and merge on
    ``(entity, time)``. `target_freq` is the model's ``at`` frequency (else the finest
    referenced). A lone source with no explicit target is used at its native frequency.
    """
    remaining = set(fields)
    loaded: list[tuple[pl.DataFrame, str]] = []
    for sname in _source_order(config):
        if not remaining:
            break
        spec = config.sources[sname]
        src = resolve_driver(spec.driver)(spec.options)
        try:
            take = (remaining & src.available_fields()) if isinstance(src, SupportsDiscovery) else set(remaining)
            if not take:
                continue
            panel = conform_panel(src.load(take, periods=config.periods), take,
                                  strict=config.strict, source_name=sname)
            panel = panel.select([ENTITY_COL, TIME_COL, *sorted(take)])  # disjoint field set
            loaded.append((panel, _source_freq(src)))
            remaining -= take
        finally:
            src.close()

    if not loaded:
        raise ConfigError("E-SOURCE-EMPTY no configured source provides the requested fields")

    if len(loaded) == 1 and target_freq is None and not is_broadcast(loaded[0][0]):
        panel = loaded[0][0]
    else:  # a lone broadcast source routes here too, so align_and_merge can reject it clearly
        panel = align_and_merge(loaded, target_freq or finest([f for _, f in loaded]))

    if config.periods is not None:
        lo, hi = config.periods
        yr = pl.col(TIME_COL).dt.year()
        panel = panel.filter((yr >= lo) & (yr <= hi))
    return panel
