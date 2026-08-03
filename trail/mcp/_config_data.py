"""The {config} branch of resolve_panel: load referenced fields from configured sources, scoped to a
model/signal's universe root chain when given (mirrors cli._scoped_panel but returns warnings)."""
from __future__ import annotations

import os
import warnings
from dataclasses import replace

import polars as pl

from trail import ast
from trail.compiler import universe_chain
from trail.config import SourceSpec, load_config
from trail.deps import extract
from trail.registry import resolve_driver
from trail.schema import VIEW_NAMESPACE
from trail.source import ENTITY_COL
from trail.sources import AlignmentWarning, PanelConformanceWarning, load_panel_for

_CACHE: dict[tuple, pl.DataFrame] = {}


def _all_fields(config) -> set[str]:
    out: set[str] = set()
    for spec in config.sources.values():
        try:
            src = resolve_driver(spec.driver)(spec.options)
        except Exception:
            continue
        try:
            out |= set(src.available_fields())
        finally:
            try:
                src.close()
            except Exception:
                pass
    return out


def _load(config, fields: frozenset[str], freq, align_overrides,
          entities=None) -> tuple[pl.DataFrame, list[str]]:
    warns: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PanelConformanceWarning)
        warnings.simplefilter("always", AlignmentWarning)
        panel = load_panel_for(config, set(fields), target_freq=freq, entities=entities,
                               align_overrides=align_overrides)
    for w in caught:
        if issubclass(w.category, (PanelConformanceWarning, AlignmentWarning)):
            warns.append(str(w.message))
    # align_and_merge returns the panel sorted [entity, time]; flag entity sorted so per-entity
    # window ops skip re-sorting (mirrors mcp.data._finalize for the {config} path).
    return panel.with_columns(pl.col(ENTITY_COL).set_sorted()), warns


def _references_views(fields) -> bool:
    return any(f.split(".", 1)[0] == VIEW_NAMESPACE for f in fields)


def _with_view_source(config, config_path):
    """A copy of `config` with a synthetic `views` source (the tracked-view store, read-only) added +
    a `precedence.views` chain, so `views.*` fields route to it. Kept local to the load - never handed
    to ViewManager - so it can't perturb a tracked view's panel_key or make a view reference itself."""
    from trail.providers import store_for_config  # lazy: avoids importing the store at module load
    store = store_for_config(config, config_path)
    ns = store.namespace
    src = SourceSpec(ns, "trail.views.ViewSource", {"dir": str(store.dir), "namespace": ns})
    return replace(config, sources={**config.sources, ns: src},
                   precedence={**config.precedence, ns: [ns]})


def resolve_config_panel(config_path, decl, universes,
                         entities=None, fresh=False) -> tuple[pl.DataFrame, list[str]]:
    config = load_config(config_path)
    if decl is None:
        fields = frozenset(_all_fields(config))
        freq, aligns = None, {}
    else:
        if decl.universe is not None:
            bound = universes.get(decl.universe)
        elif len(universes) == 1:
            bound = next(iter(universes.values()))
        else:
            bound = None
        scoped = ast.Program(tuple(universe_chain(bound, universes)) + (decl,))
        dep = extract(scoped)
        fields, freq, aligns = frozenset(dep.fields), decl.frequency, dep.align_overrides
    # Make stored views queryable: when a load references `views.*` (or on discovery), inject the
    # read-only view source so those fields resolve. Guarded - no reachable/writable store just leaves
    # `views.*` unserved (pre-P3 behaviour), never breaks a plain source load.
    if decl is None or _references_views(fields):
        try:
            config = _with_view_source(config, config_path)
            if decl is None:
                fields = frozenset(_all_fields(config))
        except Exception:
            pass
    # `entities` MUST be part of the key: an entity-scoped load produces a narrower panel, and
    # serving it to a later unscoped request would silently answer from a subset of the universe.
    scope = tuple(sorted(entities)) if entities else None
    key = (os.path.abspath(config_path), fields, freq, scope)
    # `fresh` forces a reload (and refreshes the memo): callers that must observe current config/data
    # rather than a panel cached under an older period window or data version pass fresh=True.
    if fresh or key not in _CACHE:
        panel, warns = _load(config, fields, freq, aligns, entities=entities)
        _CACHE[key] = panel
        return panel, warns
    return _CACHE[key], []
