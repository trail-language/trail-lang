"""Source drivers, panel loading, and panel conformance.

Drivers are resolved via :mod:`trail.registry` (a registered ``trail.sources`` name or a
dotted import path) to a ``factory(options) -> DataSource``. :func:`load_panel_for` loads
the effective source's panel, checks it against the panel contract, and applies the
period filter. Conformance deviations are a hard error under ``panel.strict``; otherwise
they are warned and coerced.
"""
from __future__ import annotations

import inspect
import warnings
from functools import lru_cache

import polars as pl

from trail.ast import _FREQUENCIES
from trail.align import _DIM_MAP_COL, AlignmentWarning, align_and_merge, finest, is_broadcast
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


def _source_dim(src) -> str:
    return src.capabilities().entity_dim if isinstance(src, SupportsCapabilities) else "entity"


def _foreign_dims_for(config: Config, requests: set[tuple[str | None, str]]) -> set[str]:
    """Entity dimensions (!= 'entity') required because a requested `(frequency, canonical)` is
    routed to a provider keyed by a coarser dimension (a country-keyed macro source). Uses the
    SAME claim predicate as the load loop (frequency-aware), so bridge detection matches routing;
    each such dimension needs its bridge meta field loaded so align can remap it onto entities."""
    dims: set[str] = set()
    pending = set(requests)
    for sname in _source_order(config):
        if not pending:
            break
        spec = config.sources[sname]
        src = resolve_driver(spec.driver)(spec.options)
        try:
            claimed = set(_claimable(src, pending))
            if not claimed:
                continue
            pending -= claimed
            if _source_dim(src) != "entity":
                dims.add(_source_dim(src))
        finally:
            src.close()
    return dims


@lru_cache(maxsize=None)
def _accepts_kwarg(func, name: str) -> bool:
    """Whether `func` opts into keyword `name` (a named param or **kwargs). Feature
    detection keeps optional seams (entities=, frequency=) off the base contracts: a
    source that has not opted in is never handed the kwarg and cannot raise TypeError."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return name in params


def _accepts_entities(load_func) -> bool:
    return _accepts_kwarg(load_func, "entities")


def _source_order(config: Config) -> list[str]:
    """Sources to try, precedence.default first, then any others (for field assignment)."""
    order = list(config.precedence.get("default", []))
    order += [s for s in config.sources if s not in order]
    return order


def _split_freq(qualified: str) -> tuple[str | None, str]:
    """(frequency | None, canonical) from a possibly frequency-qualified field string.
    Mirrors the parser: a known frequency leading a 3+-part path is the qualifier."""
    head, _, rest = qualified.partition(".")
    if head in _FREQUENCIES and rest.count(".") >= 1:
        return head, rest
    return None, qualified


def _source_freqs(src) -> tuple[str, ...]:
    """Every frequency a source can serve (its default when it declares no explicit set)."""
    if not isinstance(src, SupportsCapabilities):
        return ("annual",)
    caps = src.capabilities()
    return caps.frequencies or (caps.frequency,)


def _accepts_frequency(load_func) -> bool:
    return _accepts_kwarg(load_func, "frequency")


def _avail(src, fq: str | None):
    """Fields the source can serve at frequency `fq` (None = its default), or None when the
    source has no discovery. A source whose available_fields() opts into `frequency=` may
    serve different fields per frequency (e.g. statements at annual/quarterly, price at daily)."""
    if not isinstance(src, SupportsDiscovery):
        return None
    if _accepts_kwarg(type(src).available_fields, "frequency"):
        return src.available_fields(frequency=fq or _source_freq(src))
    return src.available_fields()


def _claimable(src, requests):
    """The subset of requests this source can serve. Each request is a tuple whose first two
    elements are (frequency | None, canonical); discovery- and frequency-aware."""
    sfreqs = _source_freqs(src)
    cache: dict = {}
    out = []
    for r in requests:
        fq, canon = r[0], r[1]
        if fq is not None and fq not in sfreqs:
            continue
        if fq not in cache:
            cache[fq] = _avail(src, fq)
        a = cache[fq]
        if a is None or canon in a:
            out.append(r)
    return out


def load_panel_for(config: Config, fields: set[str], target_freq: str | None = None,
                   entities: list[str] | None = None) -> pl.DataFrame:
    """Load the configured sources, assign each requested field to its first provider
    (precedence), align every source panel to the target frequency, and merge on
    ``(entity, time)``. `target_freq` is the model's ``at`` frequency (else the finest
    referenced). `entities` is the candidate entity universe to scope the fetch to; it is
    passed only to sources that opt in (see :func:`_accepts_entities`), else ignored. A lone
    source with no explicit target is used at its native frequency.
    """
    # a country-keyed (foreign-dimension) source needs its bridge meta field (meta.country)
    # loaded too, even though the model never names it - inject it (bare, canonical).
    bridges = {_DIM_MAP_COL[d] for d in _foreign_dims_for(config, {_split_freq(f) for f in fields})
               if d in _DIM_MAP_COL}
    # each request is (frequency | None, canonical, final_column); final_column is the
    # frequency-qualified name the compiler reads (bare == canonical). Deduped.
    pending = list({(*_split_freq(f), f) for f in fields} | {(None, b, b) for b in bridges})

    loaded: list[tuple[pl.DataFrame, str, str]] = []
    for sname in _source_order(config):
        if not pending:
            break
        spec = config.sources[sname]
        src = resolve_driver(spec.driver)(spec.options)
        try:
            claimed = _claimable(src, pending)
            if not claimed:
                continue
            pending = [r for r in pending if r not in claimed]
            # one fetch per distinct frequency; a bare request fetches the source's default.
            by_fetch: dict[str, list[tuple[str, str]]] = {}
            for fq, canon, final in claimed:
                by_fetch.setdefault(fq or _source_freq(src), []).append((canon, final))
            for fetch, aliases in by_fetch.items():
                take = {c for c, _ in aliases}
                kw = {"periods": config.periods}
                # scope by the entity universe only for entity-keyed sources that opt in
                if entities is not None and _source_dim(src) == "entity" and _accepts_entities(type(src).load):
                    kw["entities"] = entities
                if _accepts_frequency(type(src).load):
                    kw["frequency"] = fetch
                elif fetch != _source_freq(src):  # asked for a non-default freq it can't be told about
                    raise ConfigError(
                        f"E-FREQ-UNWIRED source '{sname}' advertises frequency '{fetch}' but its "
                        "load() takes no frequency= argument, so it cannot serve it"
                    )
                panel = conform_panel(src.load(take, **kw), take, strict=config.strict, source_name=sname)
                # project canonical -> final columns; one fetch feeds bare + qualified aliases
                proj = [pl.col(c).alias(fin) for c, fin in sorted(aliases, key=lambda a: a[1])]
                loaded.append((panel.select([ENTITY_COL, TIME_COL, *proj]), fetch, _source_dim(src)))
        finally:
            src.close()

    for fq, canon, final in pending:  # requests no configured source can serve
        if fq is not None:
            raise ConfigError(f"E-FREQ-UNAVAILABLE no configured source provides '{canon}' at frequency '{fq}'")
        if final not in bridges:  # an unserved injected bridge gets align's E-DIM-UNMAPPED instead
            raise ConfigError(
                f"E-FIELD-UNSERVED no configured source provides '{canon}' "
                "(the model references it, so the run would fail downstream)"
            )
    if not loaded:
        raise ConfigError("E-SOURCE-EMPTY no configured source provides the requested fields")

    if len(loaded) == 1 and target_freq is None and not is_broadcast(loaded[0][0]):
        panel = loaded[0][0]
    else:  # a lone broadcast source routes here too, so align_and_merge can reject it clearly
        panel = align_and_merge(loaded, target_freq or finest([f for _, f, _ in loaded]))

    if config.periods is not None:
        lo, hi = config.periods
        yr = pl.col(TIME_COL).dt.year()
        panel = panel.filter((yr >= lo) & (yr <= hi))
    return panel
