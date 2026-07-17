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


def _foreign_dims_for(config: Config, reqs, get_src) -> set[str]:
    """Entity dimensions (!= 'entity') required because a requested field routes to a provider
    keyed by a coarser dimension (a country-keyed macro source). Chain-aware: each `(frequency,
    canonical, pin_source)` routes through :func:`_chain_for`, and the dims of the sources that
    actually SERVE it (chain sources that can claim it) are unioned - so bridge detection matches
    the load loop's routing exactly. Each such dimension needs its bridge meta field loaded so
    align can remap it onto entities. `get_src` shares constructed instances with the load loop."""
    reqs = list(reqs)
    chain_of = {r: _chain_for(config, r[1], r[2]) for r in reqs}
    dims: set[str] = set()
    for sname in _all_source_order(config):
        routed = [r for r in reqs if sname in chain_of[r]]
        if not routed:
            continue
        src = get_src(sname)
        if _claimable(src, routed) and _source_dim(src) != "entity":
            dims.add(_source_dim(src))
    return dims


def _all_source_order(config: Config) -> list[str]:
    """Deterministic source-visitation order: `precedence.default`, then every other namespace
    chain in sorted key order, then any remaining declared sources - deduped, order-preserving.
    Drives fetch grouping -> join order -> output column order, so it must be stable across runs."""
    order = list(config.precedence.get("default", []))
    for ns in sorted(config.precedence):
        if ns != "default":
            order += config.precedence[ns]
    order += list(config.sources)
    return list(dict.fromkeys(order))


def _namespace(canon: str) -> str:
    """The precedence namespace of a canonical field: its first dotted segment
    (income.revenue -> income)."""
    return canon.split(".", 1)[0]


def _chain_for(config: Config, canon: str, pin_source: str | None) -> list[str]:
    """The precedence chain a field routes through. A `@ source` pin forces exactly that source
    (E-PIN-SOURCE-UNKNOWN if it is not configured); otherwise the field's namespace chain, falling
    back to `precedence.default`."""
    if pin_source is not None:
        if pin_source not in config.sources:
            raise ConfigError(
                f"E-PIN-SOURCE-UNKNOWN '@ {pin_source}' pins an undeclared source; "
                f"configured sources are {sorted(config.sources)}")
        return [pin_source]
    return config.precedence.get(_namespace(canon), config.precedence["default"])


def _single_dim(srcs) -> bool:
    """Whether every source shares one entity dimension (coalescing across dimensions is illegal)."""
    return len({_source_dim(s) for s in srcs}) <= 1


def _parse_field(field: str) -> tuple[str | None, str, str, str | None, str | None]:
    """(frequency | None, canonical, final_column, entity | None, source | None) for a requested
    field. `final_column` is the exact name the compiler reads (bare, freq-qualified, entity- or
    source-pinned). Entity (`@`) and source (`#`) pins are mutually exclusive in the codec."""
    base, ent = fieldname.split_pin(field)
    base, src = fieldname.split_source(base)
    fq, canon = fieldname.split_frequency(base)
    return fq, canon, field, ent, src


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


def _align_date_refs(node) -> set[str]:
    """The `__date:*` columns an `@ align(expr)` references (its NAMEs), for carrying just those."""
    from trail import ast

    if isinstance(node, ast.NameRef):
        return {date_col(node.name)}
    if isinstance(node, ast.Call):
        out: set[str] = set()
        for a in node.args:
            out |= _align_date_refs(a)
        return out
    return set()


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
    # each request is (frequency | None, canonical, final_column, pin_entity | None,
    # pin_source | None); final_column is the exact name the compiler reads. Deduped.
    requests = {_parse_field(f) for f in fields}
    # a country-keyed (foreign-dimension) source needs its bridge meta field (meta.country)
    # loaded too, even though the model never names it - inject it (bare, canonical). Chain-aware.
    base_reqs = {(fq, canon, src) for fq, canon, final, ent, src in requests}
    bridges = {_DIM_MAP_COL[d]
               for d in _foreign_dims_for(config, base_reqs, _get_src) if d in _DIM_MAP_COL}
    # deterministic order: fetch grouping drives join order drives OUTPUT COLUMN order,
    # which must not vary with set-iteration order across processes
    pending = sorted(requests | {(None, b, b, None, None) for b in bridges}, key=lambda r: r[2])
    # an entity pin may reference an entity outside the model universe (a benchmark index,
    # another country): widen an explicit fetch scope to include it.
    pin_entities = {r[3] for r in requests if r[3]}
    if entities is not None and pin_entities:
        entities = sorted(set(entities) | pin_entities)

    # --- routing: resolve each request's precedence chain to the sources that actually serve it ---
    # serving[r] = chain sources (in chain order) that can claim r; batch _claimable per source.
    chain_of = {r: _chain_for(config, r[1], r[4]) for r in pending}
    claimable_by_source: dict[str, set] = {}
    for sname in _all_source_order(config):
        routed = [r for r in pending if sname in chain_of[r]]
        if routed:
            claimable_by_source[sname] = set(_claimable(_get_src(sname), routed))
    serving = {r: [s for s in chain_of[r] if r in claimable_by_source.get(s, ())] for r in pending}

    requested_finals = {r[2] for r in requests}  # what the compiler reads (excludes injected bridges)
    for r in pending:  # a request no source in its chain can serve is an error (a bridge is skipped)
        fq, canon, final, ent, src = r
        if serving[r]:
            continue
        if src is not None:
            raise ConfigError(f"E-PIN-UNSERVED '@ {src}' pins a source that does not provide "
                              f"'{canon}'" + (f" at frequency '{fq}'" if fq else ""))
        if fq is not None:
            raise ConfigError(
                f"E-FREQ-UNAVAILABLE no configured source provides '{canon}' at frequency '{fq}'")
        if final not in bridges:  # an unserved injected bridge gets align's E-DIM-UNMAPPED instead
            raise ConfigError(
                f"E-FIELD-UNSERVED no configured source provides '{canon}' "
                "(the model references it, so the run would fail downstream)")

    def _coalesced(r) -> bool:
        """A field is coalesced when >1 source in its chain serves it; entity/source pins and
        bridges are always first-provider. A mixed-dimension coalesce is illegal (E-COALESCE-DIM-MIXED)."""
        fq, canon, final, ent, src = r
        if ent is not None or src is not None or final in bridges or len(serving[r]) < 2:
            return False
        if not _single_dim([_get_src(s) for s in serving[r]]):
            raise ConfigError(
                f"E-COALESCE-DIM-MIXED field '{canon}' coalesces across sources of different entity "
                f"dimensions {sorted({_source_dim(_get_src(s)) for s in serving[r]})}; pin one "
                "source (@ source) or split its precedence chain by dimension")
        return True

    coalesced = {r: _coalesced(r) for r in pending if serving[r]}

    def _fetch_sources(r):
        return serving[r] if coalesced[r] else serving[r][:1]

    # coalesce_plan: untagged final -> its per-source tag columns in chain order (post-merge coalesce)
    coalesce_plan: dict[str, list[str]] = {}
    for r in pending:
        if serving[r] and coalesced[r]:
            fq, canon, final, ent, src = r
            coalesce_plan[final] = [fieldname.qualified(canon, frequency=fq, source=s)
                                    for s in serving[r]]

    # per_source[sname][fetch_freq] = list of (canon, target_col, entity | None, untagged_final).
    # A coalesced field projects to a `#source` tag column per serving source; every other field
    # (pinned, single-source, entity-pin) projects to its plain final.
    per_source: dict[str, dict[str, list]] = {}
    for r in pending:
        if not serving[r]:
            continue
        fq, canon, final, ent, src = r
        for s in _fetch_sources(r):
            fetch = fq or _source_freq(_get_src(s))
            target = fieldname.qualified(canon, frequency=fq, source=s) if coalesced[r] else final
            per_source.setdefault(s, {}).setdefault(fetch, []).append((canon, target, ent, final))

    loaded: list[LoadedPanel] = []
    for sname in _all_source_order(config):
        if sname not in per_source:
            continue
        src = _get_src(sname)
        # the entity universe scopes only entity-keyed sources (a country-keyed source keys on
        # its own dimension, so it uses its configured set, not stock tickers) - loop-invariant.
        req_entities = (tuple(entities)
                        if entities is not None and _source_dim(src) == "entity" else None)
        # PIT off (globally or per-source) -> no coordinates resolved, every field naive.
        pit_naive = config.pit == "naive" or src.options.get("pit") == "naive"
        for fetch in sorted(per_source[sname]):
            items = per_source[sname][fetch]
            # dedup projection targets per source (a field pinned-to-S and coalesced-through-S share
            # one tag - polars rejects a duplicate alias). Stable (target, final) order keeps the
            # plain (untagged) final's @align override over a pinned final's.
            by_target: dict[str, tuple] = {}
            for canon, target, ent, final in sorted(items, key=lambda it: (it[1], it[3])):
                by_target.setdefault(target, (canon, target, ent, final))
            deduped = list(by_target.values())
            take = {canon for canon, _t, _e, _f in deduped}
            request = LoadRequest(fields=frozenset(take), frequency=fetch,
                                  periods=config.periods, entities=req_entities)
            panel = conform_panel(src.load(request), take, strict=config.strict, source_name=sname)
            # project canonical -> target columns, and carry each field's alignment coordinate
            # (__date:*) so align can place by knowability. coord_map is keyed by the TARGET column.
            regular = [(canon, target, final) for canon, target, ent, final in deduped if ent is None]
            if regular:
                proj = [pl.col(canon).alias(target) for canon, target, _f in regular]
                coord_map = {}
                for canon, target, _f in regular:
                    dc = _coord_for(src, canon, panel.columns, pit_naive)
                    if dc:
                        coord_map[target] = dc
                # carried date columns: the resolved coordinates, plus (only for an @align expr,
                # keyed by the UNTAGGED final) the source date columns that expr references -
                # carrying unused date columns would emit spurious W-PIT-PARTIAL noise.
                has_override = not pit_naive and any(f in align_overrides for _c, _t, f in regular)
                resolved = list(dict.fromkeys(
                    coord_map[target] for _c, target, _f in regular if target in coord_map))
                src_dates = {c for c in panel.columns if is_date_col(c)}
                referenced = set()
                if has_override:
                    for _c, _t, final in regular:
                        if final in align_overrides:
                            referenced |= _align_date_refs(align_overrides[final])
                date_cols = list(dict.fromkeys([*resolved, *sorted(referenced & src_dates)]))
                keep = panel.select([ENTITY_COL, TIME_COL, *proj, *[pl.col(d) for d in date_cols]])
                if has_override:  # materialize a derived coordinate per @align-overridden TAG
                    keep = keep.with_columns(
                        [_compile_align(align_overrides[final], src_dates).alias(
                            date_col(f"__align__{target}"))
                         for _c, target, final in regular if final in align_overrides])
                    for _c, target, final in regular:
                        if final not in align_overrides:
                            continue
                        dcol = date_col(f"__align__{target}")
                        if not _is_temporal_dtype(keep.schema[dcol]):
                            raise ConfigError(
                                f"E-ALIGN-DTYPE @align for '{final}' must yield a datetime "
                                f"coordinate, got {keep.schema[dcol]}")
                        coord_map[target] = dcol
                    keep = keep.with_columns(
                        [pl.col(coord_map[target]).cast(CANON_TIME) for _c, target, final in regular
                         if final in align_overrides and keep.schema[coord_map[target]] != CANON_TIME])
                loaded.append(LoadedPanel(keep, fetch, _source_dim(src), coord_map))
            # an entity pin becomes a synthetic broadcast panel: the pinned entity's series,
            # keyed by the '*' sentinel, so align's broadcast pass replicates it onto the
            # grid (as-of, kind-aware, PIT-safe) exactly like a global series.
            for canon, target, ent, final in deduped:
                if ent is None:
                    continue
                sl = panel.filter(pl.col(ENTITY_COL) == ent)
                if sl.height == 0:
                    raise ConfigError(
                        f"E-ENTITY-UNKNOWN entity '{ent}' has no rows for '{canon}' from "
                        f"source '{sname}'; add it to the source's fetch scope"
                    )
                dc = _coord_for(src, canon, panel.columns, pit_naive)
                sel = [TIME_COL, pl.col(canon).alias(target)] + ([pl.col(dc)] if dc else [])
                pin_panel = sl.select(sel).with_columns(pl.lit(BROADCAST_ENTITY).alias(ENTITY_COL))
                loaded.append(LoadedPanel(pin_panel, fetch, "entity", {target: dc} if dc else {}))

    if not loaded:
        raise ConfigError("E-SOURCE-EMPTY no configured source provides the requested fields")

    if (len(loaded) == 1 and target_freq is None and not is_broadcast(loaded[0].panel)
            and not loaded[0].coord_map and not coalesce_plan):  # naive single source: passthrough
        panel = loaded[0].panel
    else:  # PIT single source self-aligns (row-shift); a lone broadcast routes here to be rejected
        panel = align_and_merge(loaded, target_freq or finest([lp.freq for lp in loaded]))

    # per-cell coalescing (spec §5.1): each (entity, period) cell takes the first NON-NULL tag down
    # the chain. Coalesce before the period filter, then drop pure intermediates (a tag that is
    # itself a requested/pinned final stays). Deterministic order throughout.
    if coalesce_plan:
        for final in sorted(coalesce_plan):
            tags = [t for t in coalesce_plan[final] if t in panel.columns]
            if tags:
                panel = panel.with_columns(pl.coalesce([pl.col(t) for t in tags]).alias(final))
        drop = sorted({t for tags in coalesce_plan.values() for t in tags
                       if t not in requested_finals and t in panel.columns})
        if drop:
            panel = panel.drop(drop)

    if config.periods is not None:
        lo, hi = config.periods
        yr = pl.col(TIME_COL).dt.year()
        panel = panel.filter((yr >= lo) & (yr <= hi))
    return panel
