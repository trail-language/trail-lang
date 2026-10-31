"""Generic `trail init <driver>` scaffolding.

Each driver package registers an :class:`InitSpec` under the ``trail.init`` entry-point
group (one entry named after the driver, exactly like the ``trail.sources`` and
``trail.schema`` groups). The spec carries the prompt fields (label + default) and the
YAML template with ``{field}`` placeholders; the CLI does the prompting and the file
write, so adding a driver to ``trail init`` is a pure driver-side contribution
(a spec + an entry point) with no engine change.
"""
from __future__ import annotations

from dataclasses import dataclass

import importlib.metadata


@dataclass(frozen=True)
class InitField:
    """One prompted value. ``name`` is the template placeholder and the matching CLI flag
    (``--<name>``); ``prompt`` is the exact prompt label; pressing Enter accepts ``default``
    (``None`` means the value is required)."""
    name: str
    prompt: str
    default: str | None = None


@dataclass(frozen=True)
class InitSpec:
    """A driver's ``trail init`` registration: what to prompt for and what to write."""
    driver: str
    package: str
    install_hint: str
    fields: tuple[InitField, ...]
    template: str


def normalize_list(raw: str) -> str:
    """Canonical flow-list content: ``"a,b"`` / ``"a, b"`` -> ``"a, b"`` (and strips
    whitespace around a scalar value)."""
    return ", ".join(part.strip() for part in raw.split(",") if part.strip())


def resolve_init_spec(driver: str) -> InitSpec | None:
    """Load the registered ``InitSpec`` for ``driver``, or ``None`` when the driver is not
    installed (its entry point is absent) or registers no template."""
    for ep in importlib.metadata.entry_points(group="trail.init"):
        if ep.name == driver:
            spec = ep.load()
            return spec if isinstance(spec, InitSpec) else None
    return None
