"""Tracked views: materialize `track`ed declarations into a ViewStore, serving the stored frame when
the declaration's AST and its sources' freshness tokens are unchanged, else recomputing and
re-persisting. Lazy and pull-based — nothing runs until a reference materializes it.

P1 scope: staleness is coarse (declaration hash + per-source freshness token); a whole view is
recomputed when any configured source's token changes. Per-cell change detection and footprint-scoped
recompute are a later phase. See docs/tracked-views design.
"""
from __future__ import annotations

import datetime as dt
import hashlib

from trail import ast
from trail.compiler import compile_model, compile_signal, universe_chain
from trail.registry import resolve_driver
from trail.store import Manifest, ViewStore, view_columns


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


def expr_hash(decl, universes) -> str:
    """A stable hash of the scoped computation (universe chain + decl body), independent of the view
    name. A mismatch against a stored manifest forces a full rebuild (handles edits + schema drift)."""
    bound = _bound_universe(decl, universes)
    chain = tuple(repr(u) for u in universe_chain(bound, universes))
    payload = repr((chain, _decl_body(decl)))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


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
            try:
                toks[name] = resolve_driver(spec.driver)(spec.options).freshness_token()
            except Exception:
                toks[name] = None
        return toks

    def is_stale(self, decl, universes) -> bool:
        mf = self.store.manifest(decl.name)
        if mf is None or mf.expr_hash != expr_hash(decl, universes):
            return True
        return self._source_tokens() != mf.freshness

    def build_frame(self, decl, universes, entities):
        from trail.mcp._config_data import resolve_config_panel  # heavy import, kept lazy
        panel, _ = resolve_config_panel(self.config_path, decl, universes, entities=entities)
        if isinstance(decl, ast.ModelDecl):
            plan = compile_model(decl, universes)
            exports, kind = plan.exports, "model"
        else:
            plan = compile_signal(decl, universes)
            exports, kind = (decl.name,), "signal"
        result = plan.run(panel)
        cols = view_columns(kind, decl.name, exports, self.store.namespace)
        result = result.rename({e: c for e, c in zip(exports, cols)})
        mf = Manifest(
            name=decl.name, kind=kind, exports=tuple(exports),
            expr_hash=expr_hash(decl, universes),
            sources=tuple(self.config.sources.keys()),
            freshness=self._source_tokens(),
            built_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            columns=tuple(cols),
        )
        return result, mf

    def materialize(self, program, universes, entities=None) -> list[str]:
        """Build every stale tracked view in `program`, writing each to the store. Returns the names
        that were (re)computed; an empty list means all tracked views were served from the store."""
        built = []
        for decl in self.tracked(program):
            if self.is_stale(decl, universes):
                frame, mf = self.build_frame(decl, universes, entities)
                self.store.write(decl.name, frame, mf)
                built.append(decl.name)
        return built

    def serve(self, decl, universes, entities=None):
        """Return `decl`'s persisted frame, recomputing + repersisting first only if it is stale.
        The lazy pull: a reference to a tracked view triggers recompute-on-change, then reads back."""
        if self.is_stale(decl, universes):
            frame, mf = self.build_frame(decl, universes, entities)
            self.store.write(decl.name, frame, mf)
        return self.store.read(decl.name)
