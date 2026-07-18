"""The {config} branch of resolve_panel: load referenced fields from configured sources, scoped to a
model/signal's universe root chain when given (mirrors cli._scoped_panel but returns warnings)."""
from __future__ import annotations

import os
import warnings

import polars as pl

from trail import ast
from trail.compiler import universe_chain
from trail.config import load_config
from trail.deps import extract
from trail.registry import resolve_driver
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


def _load(config, fields: frozenset[str], freq, align_overrides) -> tuple[pl.DataFrame, list[str]]:
    warns: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PanelConformanceWarning)
        warnings.simplefilter("always", AlignmentWarning)
        panel = load_panel_for(config, set(fields), target_freq=freq, align_overrides=align_overrides)
    for w in caught:
        if issubclass(w.category, (PanelConformanceWarning, AlignmentWarning)):
            warns.append(str(w.message))
    return panel, warns


def resolve_config_panel(config_path, decl, universes) -> tuple[pl.DataFrame, list[str]]:
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
    key = (os.path.abspath(config_path), fields, freq)
    if key not in _CACHE:
        panel, warns = _load(config, fields, freq, aligns)
        _CACHE[key] = panel
        return panel, warns
    return _CACHE[key], []
