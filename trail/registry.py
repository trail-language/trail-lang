"""Driver resolution for data sources.

A `trail.yaml` source `driver` is either the short name of a registered plugin
(entry-point group ``trail.sources``) or a dotted import path. Registered names win;
dotted paths are the fallback, so any callable or class is usable without packaging.

A resolved driver is a callable ``factory(options) -> DataSource``. A ``DataSource``
subclass satisfies this directly, since calling the class constructs an instance.
"""
from __future__ import annotations

import importlib
from importlib import metadata

from trail.config import ConfigError

ENTRY_POINT_GROUP = "trail.sources"


def _entry_points() -> dict[str, metadata.EntryPoint]:
    return {ep.name: ep for ep in metadata.entry_points(group=ENTRY_POINT_GROUP)}


def registered_drivers() -> list[str]:
    """Names registered under the ``trail.sources`` entry-point group."""
    return sorted(_entry_points())


def resolve_driver(ref: str):
    """Resolve a driver reference to a ``factory(options) -> DataSource`` callable.

    Tries a registered entry-point name first, then a dotted import path. Raises
    :class:`ConfigError` (code ``E-SOURCE-DRIVER``) if neither resolves.
    """
    registered = _entry_points()
    if ref in registered:
        try:
            return registered[ref].load()
        except Exception as e:  # any import/attr failure inside the plugin
            raise ConfigError(f"E-SOURCE-DRIVER cannot load registered driver '{ref}': {e}") from e
    mod_path, _, attr = ref.rpartition(".")
    if not mod_path:
        raise ConfigError(
            f"E-SOURCE-DRIVER cannot resolve driver '{ref}': "
            f"not a registered name (have: {registered_drivers()}) or a dotted path"
        )
    try:
        return getattr(importlib.import_module(mod_path), attr)
    except (ImportError, AttributeError, ValueError) as e:
        raise ConfigError(f"E-SOURCE-DRIVER cannot resolve driver '{ref}': {e}") from e
