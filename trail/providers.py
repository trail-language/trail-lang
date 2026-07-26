"""Driver resolution for writeable view-store providers (parallel to trail/registry.py).

A `trail.yaml` provider `driver` is either the short name of a registered plugin (entry-point group
``trail.providers``) or a dotted import path. Registered names win; dotted paths are the fallback.

A resolved driver is a callable ``factory(options) -> ViewStore``. A ``ViewStore`` subclass satisfies
this directly, since calling the class constructs an instance.
"""
from __future__ import annotations

import importlib
import os
from importlib import metadata

from trail.config import ConfigError

ENTRY_POINT_GROUP = "trail.providers"
DEFAULT_PROVIDER = "views_local"  # the built-in local-disk store, registered in pyproject.toml


def _entry_points() -> dict[str, metadata.EntryPoint]:
    return {ep.name: ep for ep in metadata.entry_points(group=ENTRY_POINT_GROUP)}


def registered_drivers() -> list[str]:
    """Names registered under the ``trail.providers`` entry-point group."""
    return sorted(_entry_points())


def resolve_provider(ref: str):
    """Resolve a provider reference to a ``factory(options) -> ViewStore`` callable.

    Tries a registered entry-point name first, then a dotted import path. Raises
    :class:`ConfigError` (code ``E-PROVIDER-DRIVER``) if neither resolves.
    """
    registered = _entry_points()
    if ref in registered:
        try:
            return registered[ref].load()
        except Exception as e:  # any import/attr failure inside the plugin
            raise ConfigError(f"E-PROVIDER-DRIVER cannot load registered driver '{ref}': {e}") from e
    mod_path, _, attr = ref.rpartition(".")
    if not mod_path:
        raise ConfigError(
            f"E-PROVIDER-DRIVER cannot resolve driver '{ref}': "
            f"not a registered name (have: {registered_drivers()}) or a dotted path"
        )
    try:
        return getattr(importlib.import_module(mod_path), attr)
    except (ImportError, AttributeError, ValueError) as e:
        raise ConfigError(f"E-PROVIDER-DRIVER cannot resolve driver '{ref}': {e}") from e


def store_for_config(config, config_path: str):
    """The configured `views` provider, or a local-disk store rooted at `<config dir>/.trail/views`
    when none is configured — so `track` works out of the box."""
    spec = config.providers.get("views") if config.providers else None
    if spec is not None:
        return resolve_provider(spec.driver)(spec.options)
    default_dir = os.path.join(os.path.dirname(os.path.abspath(config_path)), ".trail", "views")
    return resolve_provider(DEFAULT_PROVIDER)({"dir": default_dir})
