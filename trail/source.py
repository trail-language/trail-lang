"""The data-source adapter contract.

A data source turns a set of requested canonical fields into a `(entity x time)`
panel. The contract is two tiers:

- Core tier: :class:`DataSource` - the minimum a provider must implement (``load``).
- Extended tier: the ``Supports*`` capability protocols and the :class:`ExtendedDataSource`
  convenience base, which add discovery, universe enumeration, and a capabilities descriptor.

The runtime never requires the extended tier; it uses ``isinstance(src, SupportsDiscovery)``
and friends to light up optional features, so a core-only source keeps working unchanged.

Providers register under the ``trail.sources`` entry-point group (see :mod:`trail.registry`),
so ``pip install trail-<name>`` makes ``driver: <name>`` usable by name in ``trail.yaml``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import polars as pl

# Panel column contract: every panel a source returns has these two index columns,
# plus one column per provided field (named by its canonical dotted path).
ENTITY_COL = "entity"
TIME_COL = "time"

#: reserved entity id: a panel whose entity axis is entirely this value is a broadcast
#: series - one value per period, replicated across every grid entity at align time.
#: This is how a global macro series (e.g. a risk-free rate) meets a per-stock panel; the
#: signal lives in the data plane (a source emits it), so the language stays symbol-free.
BROADCAST_ENTITY = "*"


class DataSource(ABC):
    """Minimum contract: turn requested canonical fields into a panel.

    Everything beyond ``load`` is an optional capability (see the extended tier).
    A source is constructed with its ``trail.yaml`` ``options`` dict.
    """

    #: stable short name; equals the entry-point name when registered as a plugin.
    name: str = ""

    def __init__(self, options: dict | None = None) -> None:
        self.options = options or {}

    @abstractmethod
    def load(self, fields: set[str], *, periods: tuple[int, int] | None = None) -> pl.DataFrame:
        """Return a panel for ``fields``.

        The frame has columns ``entity`` (Utf8), ``time`` (Datetime), and one column
        per provided field. It may return a superset of ``fields``. It may ignore
        ``periods`` (the runtime re-filters), but honoring it as a fetch bound is a
        performance win. Rows must be unique on ``(entity, period)``.
        """

    def close(self) -> None:
        """Release resources (network sessions, file handles). Idempotent; default no-op."""


@dataclass(frozen=True)
class FieldInfo:
    """How a source supplies (or cannot supply) a single canonical field."""

    field: str
    available: bool
    strategy: str  # direct | derived | raw | unavailable
    note: str = ""


@dataclass(frozen=True)
class Capabilities:
    """A source's self-description: what it serves and where it comes from."""

    frequency: str  # annual | quarterly | mixed
    period_range: tuple[int, int] | None = None
    forms: tuple[str, ...] = field(default_factory=tuple)
    provides_meta: bool = False
    provenance: str = ""


@runtime_checkable
class SupportsDiscovery(Protocol):
    """A source that can report which canonical fields it provides."""

    def available_fields(self) -> set[str]: ...

    def describe_field(self, field: str) -> FieldInfo | None: ...


@runtime_checkable
class SupportsUniverse(Protocol):
    """A source that can enumerate the entities it can serve."""

    def entities(self, universe: str | None = None) -> list[str]: ...


@runtime_checkable
class SupportsCapabilities(Protocol):
    """A source that can describe its frequency, period range, and provenance."""

    def capabilities(self) -> Capabilities: ...


class ExtendedDataSource(DataSource, ABC):
    """Convenience base for a full provider: implement the whole extended surface here.

    Subclassing this is equivalent to implementing :class:`SupportsDiscovery`,
    :class:`SupportsUniverse`, and :class:`SupportsCapabilities` together.
    """

    @abstractmethod
    def available_fields(self) -> set[str]: ...

    @abstractmethod
    def describe_field(self, field: str) -> FieldInfo | None: ...

    @abstractmethod
    def entities(self, universe: str | None = None) -> list[str]: ...

    @abstractmethod
    def capabilities(self) -> Capabilities: ...
