"""Tracked views: materialize `track`ed declarations into a ViewStore, serving the stored frame when
the declaration's AST and its sources' freshness tokens are unchanged, else recomputing and
re-persisting. Lazy and pull-based — nothing runs until a reference materializes it.

Staleness = declaration hash + panel-config fingerprint + per-source freshness. When a dep source
exposes a changefeed (`changed_since`), a reference recomputes only the affected footprint (the dirty
entities' cells); without one, it falls back to a whole-view rebuild. See docs/tracked-views design.
"""
from __future__ import annotations

import datetime as dt
import functools
import hashlib

import polars as pl

from trail import ast
from trail.compiler import compile_model, compile_signal, universe_chain
from trail.deps import extract
from trail.fieldname import canonical
from trail.footprint import build_index, model_footprint, replace_rows
from trail.registry import resolve_driver
from trail.schema import VIEW_NAMESPACE
from trail.source import ENTITY_COL, TIME_COL, Capabilities, DataSource, LoadRequest
from trail.store import LocalDiskViewStore, Manifest, ViewStore, view_columns


def _group_signature(panel, by_cols) -> dict:
    """Per by-column hash of the (entity, period)->group mapping. A grouping change (e.g. a sector
    reclassification) is invisible to a fresh panel's group membership, so it must invalidate the
    incremental path — a mismatch against the stored hash forces a full rebuild. Keyed by (entity,
    time, group) so a time-varying grouping that permutes assignments can't hash-match by accident."""
    out: dict = {}
    for col in by_cols:
        if col in panel.columns:
            rows = sorted(set(zip(panel.get_column(ENTITY_COL).to_list(),
                                  panel.get_column(TIME_COL).to_list(),
                                  panel.get_column(col).to_list())))
            out[col] = hashlib.sha256(repr(rows).encode()).hexdigest()[:16]
    return out


def _decl_body(decl) -> tuple:
    """The name-independent computational identity of a decl (its hash ignores the view name so a
    rename is not read as a computation change)."""
    if isinstance(decl, ast.SignalDecl):
        return ("signal", decl.universe, decl.frequency, repr(decl.expr))
    return ("model", decl.universe, decl.frequency, decl.on_missing, repr(decl.statements))


def _bound_universe(decl, universes):
    """Mirror resolve_config_panel's binding: explicit `on`, else the sole universe, else None."""
    if decl.universe is not None:
        return universes.get(decl.universe)
    if len(universes) == 1:
        return next(iter(universes.values()))
    return None


def _panel_key(config) -> str:
    """Fingerprint of the config that changes the loaded frame: the panel window / PIT / strict knobs,
    plus the routing (`precedence`) and every source's `(driver, options)` — a per-source `options.pit`
    or a `precedence` edit changes which/how data loads just as `periods` does. A view built under one
    config must not be served under another, so all of it is part of the view's identity. Over-scoped
    on purpose (over-invalidation is the safe direction); repr is stable cross-process because the
    dicts are insertion-ordered from a deterministic YAML parse (same argument as expr_hash)."""
    sources = {n: (s.driver, s.options) for n, s in config.sources.items()}
    return repr((config.periods, config.pit, config.strict, config.precedence, sources))


def expr_hash(decl, universes) -> str:
    """A stable hash of the scoped computation (universe chain + decl body), independent of the view
    name. A mismatch against a stored manifest forces a full rebuild (handles edits + schema drift)."""
    bound = _bound_universe(decl, universes)
    chain = tuple(repr(u) for u in universe_chain(bound, universes))
    payload = repr((chain, _decl_body(decl)))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _by_fields(decl) -> set[tuple]:
    """Every `by`-grouping field tuple used anywhere in a decl's expressions (post-expansion), so the
    footprint index can precompute those groups' membership."""
    found: set = set()

    def walk(e):
        match e:
            case ast.Call():
                if e.by:
                    found.add(e.by)
                for a in e.args:
                    walk(a)
                for _, v in e.kwargs:
                    walk(v)
            case ast.BinOp() | ast.Compare() | ast.BoolOp() | ast.Coalesce():
                walk(e.left)
                walk(e.right)
            case ast.In():
                walk(e.item)
                for o in e.options:
                    walk(o)
            case ast.Not() | ast.Neg():
                walk(e.operand)
            case ast.Ternary():
                walk(e.value)
                walk(e.cond)
                walk(e.orelse)

    exprs: list = []
    if isinstance(decl, ast.SignalDecl):
        exprs = [decl.expr]
    else:
        for s in decl.statements:
            if isinstance(s, ast.Assignment) and s.expr is not None:
                exprs.append(s.expr)
            elif isinstance(s, ast.ScoreDecl):
                for c in s.cases:
                    exprs += [c.value, c.cond]
                exprs.append(s.default)
    for e in exprs:
        walk(e)
    return found


class ViewManager:
    """Owns the persist/serve/recompute decision for tracked views under one config + store."""

    def __init__(self, store: ViewStore, config, config_path: str) -> None:
        self.store = store
        self.config = config
        self.config_path = config_path

    def tracked(self, program) -> list:
        return [d for d in program.decls
                if isinstance(d, (ast.ModelDecl, ast.SignalDecl)) and getattr(d, "track", False)]

    def _source_tokens(self) -> dict[str, str | None]:
        toks: dict[str, str | None] = {}
        for name, spec in self.config.sources.items():
            src = None
            try:
                src = resolve_driver(spec.driver)(spec.options)
                toks[name] = src.freshness_token()
            except Exception:
                toks[name] = None
            finally:
                if src is not None:
                    try:
                        src.close()
                    except Exception:
                        pass
        return toks

    # --- view-of-view: dependencies on other tracked views (P3 injection does the compute) ----------
    def _view_deps(self, decl, universes) -> set[str]:
        """Names of the tracked views `decl` references, from its `views.*` fields. `views.rating.score`
        and the signal-view `views.momentum` both yield the segment after `views` (the view name)."""
        bound = _bound_universe(decl, universes)
        scoped = ast.Program(tuple(universe_chain(bound, universes)) + (decl,))
        out: set[str] = set()
        for f in extract(scoped).fields:
            parts = canonical(f).split(".")             # strips any freq prefix / pin qualifier
            if len(parts) >= 2 and parts[0] == VIEW_NAMESPACE:
                out.add(parts[1])
        return out

    def _view_dep_fingerprints(self, deps) -> dict[str, str]:
        """Per-dep fingerprint (`expr_hash:built_at` of its stored manifest, `""` when not built). A
        change in any recorded fingerprint is what makes a dependent stale (coarse invalidation)."""
        out: dict[str, str] = {}
        for name in deps:
            mf = self.store.manifest(name)
            out[name] = f"{mf.expr_hash}:{mf.built_at}" if mf is not None else ""
        return out

    def _view_deps_stale(self, mf) -> bool:
        """True when any view-dep recorded in `mf` has a different current fingerprint (rebuilt/vanished)."""
        return self._view_dep_fingerprints(set(mf.view_deps)) != mf.view_deps

    def _in_dep_closure(self, target, deps, universes, decls) -> bool:
        """Whether `target` is reachable through the view-dep graph starting from `deps` — a same-program
        dep expands via its decl, a cross-run dep via its stored manifest's `view_deps`. `target` already
        appearing in `deps` is a direct self-reference. Used to reject cycles before materializing."""
        seen: set[str] = set()
        stack = list(deps)
        while stack:
            d = stack.pop()
            if d == target:
                return True
            if d in seen:
                continue
            seen.add(d)
            dd = (decls or {}).get(d)
            if dd is not None:
                stack.extend(self._view_deps(dd, universes))
            else:
                mf = self.store.manifest(d)
                if mf is not None:
                    stack.extend(mf.view_deps.keys())
        return False

    def is_stale(self, decl, universes) -> bool:
        mf = self.store.manifest(decl.name)
        if mf is None or mf.expr_hash != expr_hash(decl, universes):
            return True
        if mf.panel_key != _panel_key(self.config):   # period window / pit / strict changed
            return True
        if self._view_deps_stale(mf):                 # a referenced view was rebuilt -> rebuild (coarse)
            return True
        return self._source_tokens() != mf.freshness

    def _manifest(self, decl, universes, kind, exports, cols, group_hash=None) -> Manifest:
        return Manifest(
            name=decl.name, kind=kind, exports=tuple(exports),
            expr_hash=expr_hash(decl, universes),
            sources=tuple(self.config.sources.keys()),
            freshness=self._source_tokens(),
            built_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            columns=tuple(cols), panel_key=_panel_key(self.config),
            group_hash=group_hash or {}, frequency=decl.frequency,
            view_deps=self._view_dep_fingerprints(self._view_deps(decl, universes)),
        )

    def _compiled(self, decl, universes):
        if isinstance(decl, ast.ModelDecl):
            plan = compile_model(decl, universes)
            return plan, plan.exports, "model"
        return compile_signal(decl, universes), (decl.name,), "signal"

    def build_frame(self, decl, universes, entities):
        from trail.mcp._config_data import resolve_config_panel  # heavy import, kept lazy
        # fresh=True bypasses the process-level panel memo: a rebuild is exactly when we must observe
        # the current config/data, not a panel cached under an older period window or data version.
        panel, _ = resolve_config_panel(self.config_path, decl, universes, entities=entities, fresh=True)
        plan, exports, kind = self._compiled(decl, universes)
        cols = view_columns(kind, decl.name, exports, self.store.namespace)
        result = plan.run(panel).rename({e: c for e, c in zip(exports, cols)})
        gh = _group_signature(panel, {".".join(b) for b in _by_fields(decl)})
        return result, self._manifest(decl, universes, kind, exports, cols, gh)

    def _full_build(self, decl, universes, entities):
        frame, mf = self.build_frame(decl, universes, entities)
        self.store.write(decl.name, frame, mf)
        return frame

    def materialize(self, program, universes, entities=None) -> list[str]:
        """Build every stale tracked view in `program`, writing each to the store. Returns the names
        that were (re)computed; an empty list means all tracked views were served from the store.
        `entities` is intentionally ignored for the persisted artifact: a tracked view is always the
        full universe, so a scoped request can never overwrite it with a subset.

        Eager pre-build helper (declaration order); NOT view-of-view-aware — it neither orders by
        view-deps nor rejects cycles. The live request path is `serve`, which does both; a program
        mixing view-of-view deps should be driven through `serve` (as run_tool does), not here."""
        built = []
        for decl in self.tracked(program):
            if self.is_stale(decl, universes):
                self._full_build(decl, universes, None)
                built.append(decl.name)
        return built

    # --- incremental recompute (P2) ---------------------------------------------------------------
    def _detect_dirty(self, mf):
        """Dirty inputs since the manifest's cursors, as ``{field_col: {(entity, time), …}}``.
        ``None`` = nothing changed; the string ``"COARSE"`` = a dep source changed but offers no
        changefeed (or errored), so the caller must rebuild the whole view."""
        dirty: dict = {}
        changed_any = False
        for name in mf.sources:
            spec = self.config.sources.get(name)
            if spec is None:
                continue
            src = None
            try:
                src = resolve_driver(spec.driver)(spec.options)
                cur = mf.freshness.get(name)
                cells = src.changed_since(cur)
                if cells is None:                       # no changefeed -> coarse
                    if src.freshness_token() != cur:
                        return "COARSE"
                    continue
                if cells:
                    changed_any = True
                    fields = set(src.available_fields())  # a filing dirties all of that entity's fields
                    try:                                  # union fields served only at other frequencies
                        caps = src.capabilities()
                        for fq in (caps.frequencies or ([caps.frequency] if caps.frequency else [])):
                            fields |= set(src.available_fields(fq))
                    except Exception:
                        pass
                    for f in fields:
                        dirty.setdefault(f, set()).update(cells)
            except Exception:
                return "COARSE"
            finally:
                if src is not None:
                    try:
                        src.close()
                    except Exception:
                        pass
        return dirty if changed_any else None

    def _recompute_merge(self, decl, universes, mf, dirty):
        from trail.mcp._config_data import resolve_config_panel
        by_cols = {".".join(b) for b in _by_fields(decl)}
        if len(by_cols) >= 2:  # composed distinct groupings: entities(F) can't guarantee inner groups
            return self._full_build(decl, universes, None)  # -> full rebuild (exact, just not scoped)
        # cheap universe-wide panel for group membership + period grids (by-fields carry the grouping)
        prop_panel, _ = resolve_config_panel(self.config_path, decl, universes, fresh=True)
        cur_gh = _group_signature(prop_panel, by_cols)
        if cur_gh != mf.group_hash:  # a group reassignment invalidates footprint scoping -> full rebuild
            return self._full_build(decl, universes, None)
        index = build_index(prop_panel, by_cols)
        footprint = model_footprint(decl, dirty, index)
        if not footprint:
            return self.store.read(decl.name)
        e_f = sorted({e for e, _ in footprint})
        # heavy recompute, scoped to just the footprint entities (complete groups at the dirty periods)
        sub, _ = resolve_config_panel(self.config_path, decl, universes, entities=e_f, fresh=True)
        plan, exports, kind = self._compiled(decl, universes)
        cols = view_columns(kind, decl.name, exports, self.store.namespace)
        res = plan.run(sub).rename({e: c for e, c in zip(exports, cols)})
        merged = replace_rows(self.store.read(decl.name), res, footprint)
        self.store.write(decl.name, merged, self._manifest(decl, universes, kind, exports, cols, cur_gh))
        return merged

    def serve(self, decl, universes, decls=None, entities=None, _visiting=()):
        """Return `decl`'s frame. First materialize any referenced tracked views (view-of-view —
        same-program deps in `decls` are built recursively in topological order as full persisted views;
        cross-run deps are used from the store), rejecting reference cycles (E-VIEW-CYCLE). An
        entity-scoped request then computes + returns WITHOUT persisting the dependent (a scoped rescore
        must never overwrite the full stored view). Otherwise rebuild fully on a recipe change OR a
        changed view-dep; else consult the source changefeed — recompute only the dirty footprint when
        one exists, else a full rebuild; serve stored unchanged when nothing changed."""
        if decl.name in _visiting:                  # recursion guard (belt-and-suspenders vs the closure)
            raise ValueError(f"E-VIEW-CYCLE view '{decl.name}' is part of a reference cycle")
        deps = self._view_deps(decl, universes)
        if self._in_dep_closure(decl.name, deps, universes, decls):
            raise ValueError(f"E-VIEW-CYCLE view '{decl.name}' is part of a reference cycle")
        # Materialize same-program deps first (always as FULL persisted views, even for a scoped
        # request) so the build below reads real dep values, never silent nulls from an absent dep.
        for dep in sorted(deps):
            dd = (decls or {}).get(dep)
            if dd is not None:
                self.serve(dd, universes, decls=decls, _visiting=_visiting + (decl.name,))
        if entities is not None:                    # scoped rescore: compute + return, never persist
            frame, _ = self.build_frame(decl, universes, entities)
            return frame
        mf = self.store.manifest(decl.name)
        if (mf is None or mf.expr_hash != expr_hash(decl, universes)
                or mf.panel_key != _panel_key(self.config)
                or self._view_deps_stale(mf)):      # a referenced view was rebuilt -> full rebuild (coarse)
            return self._full_build(decl, universes, None)
        dirty = self._detect_dirty(mf)
        if dirty is None:                           # nothing changed -> serve stored (rebuild if gone)
            cached = self.store.read(decl.name)
            return cached if cached is not None else self._full_build(decl, universes, None)
        if dirty == "COARSE":                       # a changefeed-less source moved -> full rebuild
            return self._full_build(decl, universes, None)
        return self._recompute_merge(decl, universes, mf, dirty)


_FREQ_ORDER = ("annual", "quarterly", "monthly", "weekly", "daily", "hourly", "minute")  # coarse -> fine


class ViewSource(DataSource):
    """Read-only source surfacing stored tracked-view frames as `<namespace>.*` fields, so views compose
    with other sources (`views.rating.score × fmp.pe`). Naive: a view's `time` is already the PIT decision
    grid, so it emits no `__date:*` coordinate and does not override `describe_field`. Auto-injected into
    the `{config}` load path by resolve_config_panel when a load references `views.*`."""

    name = "views"

    def __init__(self, options=None) -> None:
        super().__init__(options)
        # Resolve the store the SAME WAY the write path does, instead of assuming a
        # local directory. The ViewStore interface is pluggable through
        # `providers.views`, but this source used to hardcode LocalDiskViewStore and
        # read `options["dir"]` -- an attribute no other ViewStore implementation has
        # any reason to expose. The result was that a project configuring a custom
        # store wrote its views there correctly and then could not compose them at
        # all: `views.*` resolved to nothing, and every reference failed with
        # E-FIELD-UNSERVED naming a column that was sitting in the store.
        #
        # `config_path` is the supported way in; `dir` still works for a direct
        # local-disk construction, so existing callers are unaffected.
        self.store = self._resolve_store()
        self.name = self.store.namespace

    def _resolve_store(self):
        store = self.options.get("store")
        if store is not None:                       # already-constructed store
            return store
        config_path = self.options.get("config_path")
        if config_path is not None:
            from trail.config import load_config
            from trail.providers import store_for_config
            return store_for_config(load_config(config_path), config_path)
        return LocalDiskViewStore(self.options)     # requires options['dir']

    def _views(self):
        for n in self.store.list():
            mf = self.store.manifest(n)
            if mf is not None:
                yield n, mf

    def available_fields(self, frequency: str | None = None) -> set[str]:
        # The store serves EVERY stored column regardless of the requested grid - per-view frequency is
        # not a routing gate. A bare `views.x` reference resolves the source's default frequency (which
        # needn't match a given view), so gating on exact frequency would make any non-default-frequency
        # or `at`-less view unservable-yet-discoverable. Alignment places each view's frame by its own
        # `time` values (naive), so same-frequency joins are correct; cross-frequency is out of v1 scope.
        out: set[str] = set()
        for _, mf in self._views():
            out.update(mf.columns)
        return out

    def capabilities(self) -> Capabilities:
        freqs = {mf.frequency for _, mf in self._views() if mf.frequency}
        ordered = tuple(f for f in _FREQ_ORDER if f in freqs)   # canonical (coarse->fine), not lexicographic
        return Capabilities(frequency=(ordered[0] if ordered else "annual"),
                            frequencies=ordered, provenance="tracked view store")

    def load(self, request: LoadRequest) -> pl.DataFrame:
        want = set(request.fields)
        frames = []
        for name, mf in self._views():
            cols = [c for c in mf.columns if c in want]
            if not cols:
                continue
            df = self.store.read(name).select([ENTITY_COL, TIME_COL, *cols])
            if request.entities:
                df = df.filter(pl.col(ENTITY_COL).is_in(list(request.entities)))
            frames.append(df)
        if not frames:
            return pl.DataFrame({ENTITY_COL: [], TIME_COL: []})
        return functools.reduce(
            lambda a, b: a.join(b, on=[ENTITY_COL, TIME_COL], how="full", coalesce=True), frames)

    def freshness_token(self) -> str | None:
        parts = sorted((n, mf.expr_hash, mf.built_at) for n, mf in self._views())
        return hashlib.sha256(repr(parts).encode()).hexdigest()[:16] if parts else None
