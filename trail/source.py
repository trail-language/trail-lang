"""The data-source adapter contract (v2).

A data source turns a :class:`LoadRequest` into a `(entity x time)` panel. The contract
is one tier: every source declares what it serves (``available_fields``), describes itself
(``capabilities``), and loads (``load``). Discovery is mandatory - a source cannot participate
without saying what it provides, which is what makes multi-source routing and coalescing
well-defined (no "discovery-less source silently claims everything").

Optional refinements have safe defaults: ``describe_field`` (per-field detail, incl. the
alignment coordinate), ``entities`` (universe enumeration), and ``close`` (resource release).

Providers register under the ``trail.sources`` entry-point group (see :mod:`trail.registry`),
so ``pip install trail-<name>`` makes ``driver: <name>`` usable by name in ``trail.yaml``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field

import polars as pl

# Panel column contract: every panel a source returns has these two index columns,
# plus one column per provided field (named by its canonical dotted path).
ENTITY_COL = "entity"
TIME_COL = "time"

#: Reserved namespace for a source's date columns - the per-field alignment coordinates
#: (filing date, trade date, ...). A field's `describe_field(f).aligns_on` names the date
#: column it aligns on. `__date:` keeps them out of the field vocabulary and the
#: `entity`/`time` index. As of this phase the namespace is reserved and conformed
#: (`conform_panel` passes it through and normalizes its dtype); phase 3 (per-field alignment)
#: will consume these during alignment and drop them so they never reach the compiler.
#: See trail.align and the PIT model.
DATE_COL_PREFIX = "__date:"


def date_col(name: str) -> str:
    """Physical column name for a source date coordinate (``filing_date`` -> ``__date:filing_date``)."""
    return f"{DATE_COL_PREFIX}{name}"


def is_date_col(column: str) -> bool:
    """Whether ``column`` is a reserved source date column."""
    return column.startswith(DATE_COL_PREFIX)


#: reserved entity id: a panel whose entity axis is entirely this value is a broadcast
#: series - one value per period, replicated across every grid entity at align time.
#: This is how a global macro series (e.g. a risk-free rate) meets a per-stock panel; the
#: signal lives in the data plane (a source emits it), so the language stays symbol-free.
BROADCAST_ENTITY = "*"


@dataclass(frozen=True)
class LoadRequest:
    """A resolved request for one fetch against one source.

    The load seam grows by adding fields here - versioned and explicit - rather than by
    feature-detected keyword accretion on ``load``.
    """

    #: canonical fields to serve (no frequency prefix, no ``@`` qualifier). A source may
    #: return a superset.
    fields: frozenset[str]
    #: native frequency to fetch at; ``None`` means the source's default frequency.
    frequency: str | None = None
    #: inclusive ``(lo, hi)`` year bounds - a fetch hint; the runtime re-filters.
    periods: tuple[int, int] | None = None
    #: candidate entity universe to scope the fetch to; ``None`` means the source's own set.
    #: Only populated for entity-keyed sources (a country-keyed source gets ``None``).
    entities: tuple[str, ...] | None = None
    #: ``@ params(...)`` fetch parameters for this request (endpoint knobs).
    params: Mapping[str, str] = field(default_factory=dict)
    #: ``@ asof`` requested a historical series rather than a current-snapshot read.
    asof: bool = False


class DataSource(ABC):
    """Turn a :class:`LoadRequest` into a panel, and declare what you serve.

    A source is constructed with its ``trail.yaml`` ``options`` dict. ``load``,
    ``available_fields``, and ``capabilities`` are mandatory; the rest have defaults.
    """

    #: stable short name; equals the entry-point name when registered as a plugin.
    name: str = ""

    def __init__(self, options: dict | None = None) -> None:
        self.options = options or {}

    @abstractmethod
    def load(self, request: LoadRequest) -> pl.DataFrame:
        """Return a panel for ``request.fields``.

        The frame has columns ``entity`` (Utf8), ``time`` (Datetime), and one column per
        provided field. It may return a superset of the requested fields, and may ignore
        ``request.periods`` (the runtime re-filters). It MAY also carry reserved
        ``__date:*`` columns (see :data:`DATE_COL_PREFIX`) that fields align on; a source
        that emits none is treated as naive (period-end = known instantly).

        Rows must be unique on ``(entity, period)`` - or, once a source carries a date
        coordinate for restatements, on ``(entity, period, coordinate)``.
        """

    @abstractmethod
    def available_fields(self, frequency: str | None = None) -> set[str]:
        """Canonical fields this source can serve at ``frequency`` (``None`` = its default).

        A source that serves different fields per frequency (e.g. statements at
        annual/quarterly but only price at daily) branches on ``frequency``.
        """

    @abstractmethod
    def capabilities(self) -> Capabilities:
        """The source's self-description: frequencies, provenance, entity dimension, PIT."""

    def describe_field(self, field: str) -> FieldInfo | None:
        """Per-field detail (availability, strategy, alignment coordinate). Default ``None``."""
        return None

    def entities(self, universe: str | None = None) -> list[str]:
        """Entities this source can serve (optionally within a named universe). Default ``[]``."""
        return []

    def close(self) -> None:
        """Release resources (network sessions, file handles). Idempotent; default no-op."""


@dataclass(frozen=True)
class FieldInfo:
    """How a source supplies (or cannot supply) a single canonical field."""

    field: str
    available: bool
    strategy: str  # direct | derived | raw | unavailable
    note: str = ""
    #: name of the source date column this field aligns on (a ``__date:<name>`` coordinate,
    #: given here WITHOUT the prefix, e.g. ``"filing_date"``). ``None`` = naive (period-end).
    aligns_on: str | None = None


@dataclass(frozen=True)
class Capabilities:
    """A source's self-description: what it serves and where it comes from."""

    frequency: str  # the default/native frequency
    #: every frequency this source can serve (for a frequency-qualified field like
    #: quarterly.income.revenue); empty means single-frequency (just `frequency`).
    frequencies: tuple[str, ...] = ()
    period_range: tuple[int, int] | None = None
    forms: tuple[str, ...] = field(default_factory=tuple)
    provides_meta: bool = False
    provenance: str = ""
    #: the dimension the source's `entity` column denotes. "entity" (default) = the canonical
    #: grid entity; a coarser dimension (e.g. "country") is remapped onto entities at align time
    #: via a bridge meta field. See trail.align.
    entity_dim: str = "entity"
    #: advisory: whether this source supplies real known-dates (emits ``__date:*`` coordinates).
    #: Catalog/validate warn when a source is naive; the engine keys off the actual columns.
    pit: bool = False
