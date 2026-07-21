"""Canonical field registry and the pluggable active schema.

Core fields are the language's built-in vocabulary. A data-source package may contribute
additional fields (e.g. gmd.*) via the `trail.schema` entry-point group; each entry point
resolves to a mapping of dotted column -> kind string. `active_schema()` merges core with all
installed contributions, and is what validation, catalog, and panel conformance read.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata

SCHEMA_ENTRY_POINT_GROUP = "trail.schema"


@dataclass(frozen=True)
class FieldSpec:
    column: str
    kind: str  # core: flow | stock | ratio | per_share | price | meta; plugins may add others


# Approach X: the language ships NO domain vocabulary - every source declares its own fields
# (edgar.*, fmp.*, gmd.*, ...) via the `trail.schema` entry point. `_CORE` retains only the shared
# `meta.*` coordination fields the engine needs by name: `meta.country` is the cross-source bridge key
# (Capabilities.bridge_field), and sector/exchange/market_cap/is_active are shared entity metadata
# that multiple sources may provide (routed by precedence like any shared field).
_FIELDS: list[tuple[str, str]] = [
    ("meta.sector", "meta"),
    ("meta.exchange", "meta"),
    ("meta.market_cap", "meta"),
    ("meta.is_active", "meta"),
    ("meta.country", "meta"),  # ISO3; the bridge that maps an entity to a country-keyed source
]

_CORE: dict[str, FieldSpec] = {c: FieldSpec(c, k) for c, k in _FIELDS}

#: core registry kept under a stable name; prefer active_schema() to include plugin fields
SCHEMA: dict[str, FieldSpec] = _CORE


@lru_cache(maxsize=1)
def _plugin_fields() -> dict[str, FieldSpec]:
    """Fields contributed by installed `trail.schema` entry points (column -> FieldSpec)."""
    out: dict[str, FieldSpec] = {}
    for ep in metadata.entry_points(group=SCHEMA_ENTRY_POINT_GROUP):
        try:
            contributed = ep.load()
        except Exception:
            continue
        for column, kind in dict(contributed).items():
            if column not in _CORE:
                out[str(column)] = FieldSpec(str(column), str(kind))
    return out


def active_schema() -> dict[str, FieldSpec]:
    """Core fields plus all installed `trail.schema` contributions (core wins on collision)."""
    plugins = _plugin_fields()
    return {**plugins, **_CORE} if plugins else dict(_CORE)


def is_field(column: str) -> bool:
    return column in _CORE or column in _plugin_fields()


def kind_of(column: str) -> str | None:
    spec = _CORE.get(column) or _plugin_fields().get(column)
    return spec.kind if spec else None
