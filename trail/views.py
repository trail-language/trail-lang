"""Tracked views: materialize `track`ed declarations into a ViewStore, serving the stored frame when
the declaration's AST and its sources' freshness tokens are unchanged, else recomputing and
re-persisting. Lazy and pull-based — nothing runs until a reference materializes it.

Staleness = declaration hash + panel-config fingerprint + per-source freshness. When a dep source
exposes a changefeed (`changed_since`), a reference recomputes only the affected footprint (the dirty
entities' cells); without one, it falls back to a whole-view rebuild. See docs/tracked-views design.
"""
from __future__ import annotations

import datetime as dt
import hashlib

from trail import ast
from trail.compiler import compile_model, compile_signal, universe_chain
from trail.footprint import build_index, model_footprint, replace_rows
from trail.registry import resolve_driver
from trail.source import ENTITY_COL, TIME_COL
from trail.store import Manifest, ViewStore, view_columns


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

    def is_stale(self, decl, universes) -> bool:
        mf = self.store.manifest(decl.name)
        if mf is None or mf.expr_hash != expr_hash(decl, universes):
            return True
        if mf.panel_key != _panel_key(self.config):   # period window / pit / strict changed
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
        full universe, so a scoped request can never overwrite it with a subset."""
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

    def serve(self, decl, universes, entities=None):
        """Return `decl`'s persisted frame. An entity-scoped request computes + returns WITHOUT
        persisting (a scoped rescore must never overwrite the full stored view). Otherwise: rebuild
        fully on a recipe change; else consult the source changefeed — recompute only the dirty
        footprint when one exists, else a full rebuild. Serve stored unchanged when nothing changed."""
        if entities is not None:                    # scoped rescore: compute + return, never persist
            frame, _ = self.build_frame(decl, universes, entities)
            return frame
        mf = self.store.manifest(decl.name)
        if (mf is None or mf.expr_hash != expr_hash(decl, universes)
                or mf.panel_key != _panel_key(self.config)):
            return self._full_build(decl, universes, None)
        dirty = self._detect_dirty(mf)
        if dirty is None:                           # nothing changed -> serve stored (rebuild if gone)
            cached = self.store.read(decl.name)
            return cached if cached is not None else self._full_build(decl, universes, None)
        if dirty == "COARSE":                       # a changefeed-less source moved -> full rebuild
            return self._full_build(decl, universes, None)
        return self._recompute_merge(decl, universes, mf, dirty)
