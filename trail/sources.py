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

from trail import fieldname
from trail.align import (
    _DIM_MAP_COL, AlignmentWarning, LoadedPanel, align_and_merge, finest, is_broadcast,
)
from trail.config import Config, ConfigError
from trail.registry import resolve_driver
from trail.schema import active_schema, kind_of
from trail.source import (
    BROADCAST_ENTITY, TIME_COL, ENTITY_COL, DataSource, LoadRequest, date_col, is_date_col,
)

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

    def load(self, request: LoadRequest) -> pl.DataFrame:
        from trail.fixtures import load_panel

        return load_panel()

    def available_fields(self, frequency: str | None = None) -> set[str]:
        from trail.fixtures import fixture_fields

        return set(fixture_fields())

    def capabilities(self):
        from trail.source import Capabilities

        return Capabilities(frequency="annual", provides_meta=True, provenance="in-memory fixture")


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
    # `__date:*` alignment coordinates are reserved (not schema fields); they pass through.
    allowed = {ENTITY_COL, TIME_COL} | set(active_schema())
    extra = sorted(c for c in panel.columns if c not in allowed and not is_date_col(c))
    if extra:
        issues.append(f"unexpected column(s) {extra}")
    if not _is_temporal_dtype(panel.schema[TIME_COL]):
        issues.append(f"'time' has non-temporal dtype {panel.schema[TIME_COL]}")
    # a reserved date coordinate must be temporal - it becomes an alignment key (phase 3);
    # a string/int `__date:*` would fail (or silently mis-compare) there, so flag it here.
    nontemporal_dates = sorted(
        c for c in panel.columns if is_date_col(c) and not _is_temporal_dtype(panel.schema[c]))
    if nontemporal_dates:
        issues.append(f"date coordinate(s) with non-temporal dtype {nontemporal_dates}")

    if strict and issues:
        raise ConfigError(f"E-SOURCE-PANEL source{src} " + "; ".join(issues))
    for msg in issues:
        warnings.warn(f"W-SOURCE-PANEL source{src} {msg}", PanelConformanceWarning, stacklevel=2)
    if issues:
        panel = panel.select([c for c in panel.columns if c in allowed or is_date_col(c)])
        if missing_fields:
            panel = panel.with_columns([_null_series(f, panel.height) for f in missing_fields])
    # normalize the time column and any date coordinates to the canonical period-end Datetime
    normalize = [
        pl.col(c).cast(CANON_TIME)
        for c in (TIME_COL, *(c for c in panel.columns if is_date_col(c)))
        if _is_temporal_dtype(panel.schema[c]) and panel.schema[c] != CANON_TIME
    ]
    if normalize:
        panel = panel.with_columns(normalize)
    return panel


def _source_freq(src) -> str:
    return src.capabilities().frequency


def _source_dim(src) -> str:
    return src.capabilities().entity_dim


def _foreign_dims_for(config: Config, requests: set[tuple[str | None, str]], get_src) -> set[str]:
    """Entity dimensions (!= 'entity') required because a requested `(frequency, canonical)` is
    routed to a provider keyed by a coarser dimension (a country-keyed macro source). Uses the
    SAME claim predicate as the load loop (frequency-aware), so bridge detection matches routing;
    each such dimension needs its bridge meta field loaded so align can remap it onto entities.
    `get_src` shares constructed source instances with the load loop (one construction per run)."""
    dims: set[str] = set()
    pending = set(requests)
    for sname in _source_order(config):
        if not pending:
            break
        src = get_src(sname)
        claimed = set(_claimable(src, pending))
        if not claimed:
            continue
        pending -= claimed
        if _source_dim(src) != "entity":
            dims.add(_source_dim(src))
    return dims


def _source_order(config: Config) -> list[str]:
    """Sources to try, precedence.default first, then any others (for field assignment)."""
    order = list(config.precedence.get("default", []))
    order += [s for s in config.sources if s not in order]
    return order


def _parse_field(field: str) -> tuple[str | None, str, str, str | None]:
    """(frequency | None, canonical, final_column, entity | None) for a requested field.
    `final_column` is the exact name the compiler reads (bare, freq-qualified, or pinned)."""
    base, ent = fieldname.split_pin(field)
    fq, canon = fieldname.split_frequency(base)
    return fq, canon, field, ent


def _source_freqs(src) -> tuple[str, ...]:
    """Every frequency a source can serve (its default when it declares no explicit set)."""
    caps = src.capabilities()
    return caps.frequencies or (caps.frequency,)


def _avail(src, fq: str | None) -> set[str]:
    """Fields the source can serve at frequency `fq` (None = its default). Discovery is core,
    so this is always a set; a source may serve different fields per frequency (e.g. statements
    at annual/quarterly, price at daily)."""
    return src.available_fields(frequency=fq or _source_freq(src))


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
        if canon in cache[fq]:
            out.append(r)
    return out


def _compile_align(node, date_cols: set[str]) -> pl.Expr:
    """Lower an `@ align(expr)` coordinate expression: a NAME binds to the source's `__date:<name>`
    column, a literal passes raw, a call lowers via the operator library. Yields a polars expr
    that must evaluate to a datetime (checked by the caller)."""
    from trail import ast
    from trail.ops import build as _op_build

    match node:
        case ast.NameRef():
            dc = date_col(node.name)
            if dc not in date_cols:
                have = sorted(c[len(date_col("")):] for c in date_cols)
                raise ConfigError(
                    f"E-ALIGN-UNKNOWN @align references date column '{node.name}', but the source "
                    f"provides {have or 'no date columns'}")
            return pl.col(dc)
        case ast.Literal():
            return node.value  # raw (e.g. the "1y" unit of truncate)
        case ast.Call():
            args = [_compile_align(a, date_cols) for a in node.args]
            return _op_build(node.name, args, {}, None)
        case _:
            raise ConfigError("E-ALIGN-EXPR @align supports only temporal functions over date columns")


def _coord_for(src, canon: str, panel_cols, pit_naive: bool) -> str | None:
    """The physical `__date:*` coordinate a field aligns on (from ``describe_field(canon).aligns_on``),
    or None when PIT is off, the field is naive, or the source didn't actually emit the column."""
    if pit_naive:
        return None
    info = src.describe_field(canon)
    aligns = info.aligns_on if info is not None else None
    if not aligns:
        return None
    dc = date_col(aligns)
    return dc if dc in panel_cols else None


def load_panel_for(config: Config, fields: set[str], target_freq: str | None = None,
                   entities: list[str] | None = None,
                   align_overrides: dict | None = None) -> pl.DataFrame:
    """Load the configured sources, assign each requested field to its first provider
    (precedence), align every source panel to the target frequency, and merge on
    ``(entity, time)``. `target_freq` is the model's ``at`` frequency (else the finest
    referenced). `entities` is the candidate entity universe to scope the fetch to; it reaches
    a source via ``LoadRequest.entities``, populated only for entity-keyed sources (a
    country-keyed source gets ``None`` and uses its own set). `align_overrides` maps a physical
    field column to an `@ align(expr)` AST that overrides that field's alignment coordinate
    (see :func:`_compile_align`). A lone source with no explicit target is used at its native
    frequency.
    """
    # sources are constructed once per run and shared by the bridge pre-scan and the load
    # loop (adapters may open sessions or set identities in __init__); all closed at the end.
    _srcs: dict[str, DataSource] = {}

    def _get_src(sname: str) -> DataSource:
        if sname not in _srcs:
            spec = config.sources[sname]
            _srcs[sname] = resolve_driver(spec.driver)(spec.options)
        return _srcs[sname]

    try:
        return _load_panel(config, fields, target_freq, entities, _get_src, align_overrides or {})
    finally:
        for s in _srcs.values():
            s.close()


def _load_panel(config: Config, fields: set[str], target_freq: str | None,
                entities: list[str] | None, _get_src, align_overrides: dict) -> pl.DataFrame:
    # a country-keyed (foreign-dimension) source needs its bridge meta field (meta.country)
    # loaded too, even though the model never names it - inject it (bare, canonical).
    bases = {fieldname.split_pin(f)[0] for f in fields}
    bridges = {_DIM_MAP_COL[d]
               for d in _foreign_dims_for(config, {fieldname.split_frequency(b) for b in bases}, _get_src)
               if d in _DIM_MAP_COL}
    # each request is (frequency | None, canonical, final_column, pin_entity | None);
    # final_column is the exact name the compiler reads. Deduped.
    requests = {_parse_field(f) for f in fields}
    # deterministic order: fetch grouping drives join order drives OUTPUT COLUMN order,
    # which must not vary with set-iteration order across processes
    pending = sorted(requests | {(None, b, b, None) for b in bridges}, key=lambda r: r[2])
    # an entity pin may reference an entity outside the model universe (a benchmark index,
    # another country): widen an explicit fetch scope to include it.
    pin_entities = {r[3] for r in requests if r[3]}
    if entities is not None and pin_entities:
        entities = sorted(set(entities) | pin_entities)

    loaded: list[LoadedPanel] = []
    for sname in _source_order(config):
        if not pending:
            break
        src = _get_src(sname)
        claimed = _claimable(src, pending)
        if not claimed:
            continue
        pending = [r for r in pending if r not in claimed]
        # the entity universe scopes only entity-keyed sources (a country-keyed source keys on
        # its own dimension, so it uses its configured set, not stock tickers) - loop-invariant.
        req_entities = (tuple(entities)
                        if entities is not None and _source_dim(src) == "entity" else None)
        # PIT off (globally or per-source) -> no coordinates resolved, every field naive.
        pit_naive = config.pit == "naive" or src.options.get("pit") == "naive"
        # one fetch per distinct frequency; a bare request fetches the source's default.
        by_fetch: dict[str, list[tuple[str, str, str | None]]] = {}
        for fq, canon, final, ent in claimed:
            by_fetch.setdefault(fq or _source_freq(src), []).append((canon, final, ent))
        for fetch, items in by_fetch.items():
            take = {c for c, _, _ in items}
            request = LoadRequest(fields=frozenset(take), frequency=fetch,
                                  periods=config.periods, entities=req_entities)
            panel = conform_panel(src.load(request), take, strict=config.strict, source_name=sname)
            # project canonical -> final columns (one fetch feeds bare + qualified aliases), and
            # carry each field's alignment coordinate (__date:*) so align can place by knowability.
            regular = sorted(((c, fin) for c, fin, ent in items if ent is None), key=lambda a: a[1])
            if regular:
                proj = [pl.col(c).alias(fin) for c, fin in regular]
                coord_map = {}
                for c, fin in regular:
                    dc = _coord_for(src, c, panel.columns, pit_naive)
                    if dc:
                        coord_map[fin] = dc
                # carried date columns: the resolved coordinates; plus (only when an @align expr
                # needs them) every source date column, so the expr may reference any of them.
                has_override = not pit_naive and any(fin in align_overrides for _, fin in regular)
                date_cols = ([col for col in panel.columns if is_date_col(col)] if has_override
                             else list(dict.fromkeys(coord_map[fin] for _, fin in regular if fin in coord_map)))
                keep = panel.select([ENTITY_COL, TIME_COL, *proj, *[pl.col(d) for d in date_cols]])
                if has_override:  # materialize a derived coordinate per @align-overridden field
                    avail = {c for c in date_cols}
                    keep = keep.with_columns(
                        [_compile_align(align_overrides[fin], avail).alias(date_col(f"__align__{fin}"))
                         for _, fin in regular if fin in align_overrides])
                    for _, fin in regular:
                        if fin not in align_overrides:
                            continue
                        dcol = date_col(f"__align__{fin}")
                        if not _is_temporal_dtype(keep.schema[dcol]):
                            raise ConfigError(
                                f"E-ALIGN-DTYPE @align for '{fin}' must yield a datetime coordinate, "
                                f"got {keep.schema[dcol]}")
                        coord_map[fin] = dcol
                    keep = keep.with_columns(
                        [pl.col(coord_map[fin]).cast(CANON_TIME) for _, fin in regular
                         if fin in align_overrides and keep.schema[coord_map[fin]] != CANON_TIME])
                loaded.append(LoadedPanel(keep, fetch, _source_dim(src), coord_map))
            # an entity pin becomes a synthetic broadcast panel: the pinned entity's series,
            # keyed by the '*' sentinel, so align's broadcast pass replicates it onto the
            # grid (as-of, kind-aware, PIT-safe) exactly like a global series.
            for canon, final, ent in items:
                if ent is None:
                    continue
                sl = panel.filter(pl.col(ENTITY_COL) == ent)
                if sl.height == 0:
                    raise ConfigError(
                        f"E-ENTITY-UNKNOWN entity '{ent}' has no rows for '{canon}' from "
                        f"source '{sname}'; add it to the source's fetch scope"
                    )
                dc = _coord_for(src, canon, panel.columns, pit_naive)
                sel = [TIME_COL, pl.col(canon).alias(final)] + ([pl.col(dc)] if dc else [])
                pin_panel = sl.select(sel).with_columns(pl.lit(BROADCAST_ENTITY).alias(ENTITY_COL))
                loaded.append(LoadedPanel(pin_panel, fetch, "entity", {final: dc} if dc else {}))

    for fq, canon, final, _ent in pending:  # requests no configured source can serve
        if fq is not None:
            raise ConfigError(f"E-FREQ-UNAVAILABLE no configured source provides '{canon}' at frequency '{fq}'")
        if final not in bridges:  # an unserved injected bridge gets align's E-DIM-UNMAPPED instead
            raise ConfigError(
                f"E-FIELD-UNSERVED no configured source provides '{canon}' "
                "(the model references it, so the run would fail downstream)"
            )
    if not loaded:
        raise ConfigError("E-SOURCE-EMPTY no configured source provides the requested fields")

    if (len(loaded) == 1 and target_freq is None and not is_broadcast(loaded[0].panel)
            and not loaded[0].coord_map):  # naive single source: pass through unchanged
        panel = loaded[0].panel
    else:  # PIT single source self-aligns (row-shift); a lone broadcast routes here to be rejected
        panel = align_and_merge(loaded, target_freq or finest([lp.freq for lp in loaded]))

    if config.periods is not None:
        lo, hi = config.periods
        yr = pl.col(TIME_COL).dt.year()
        panel = panel.filter((yr >= lo) & (yr <= hi))
    return panel
