"""The physical column-name codec - the single seam between a FieldRef and its polars column.

A qualified field reference (frequency prefix, entity pin, ...) is encoded into one
column-name string that flows through deps, the compiler (`pl.col`), the loader, and the
alignment engine. Encode, decode, and canonicalization live here so the column micro-grammar
has ONE definition rather than the three hand-rolled copies it grew (parser, loader, align).

Also the home of the frequency ladder, since "is this leading token a frequency?" is a
column-name question; `trail.ast` re-exports it for existing importers.
"""
from __future__ import annotations

# frequency ladder, coarse -> fine
FREQUENCIES: tuple[str, ...] = ("annual", "quarterly", "monthly", "weekly", "daily", "hourly", "minute")
_FREQUENCIES = frozenset(FREQUENCIES)


def qualified(canonical: str, *, frequency: str | None = None, entity: str | None = None,
              source: str | None = None) -> str:
    """Encode a canonical field plus qualifiers into the physical column name
    (daily.price.adj_close@SPY, annual.income.revenue#edgar). The entity pin (`@`) and the
    source pin (`#`) are mutually exclusive - a reference pins one axis, never both."""
    if entity is not None and source is not None:
        raise ValueError("a field reference cannot pin both an entity (@) and a source (#)")
    base = f"{frequency}.{canonical}" if frequency else canonical
    if entity:
        return f"{base}@{entity}"
    if source:
        return f"{base}#{source}"
    return base


def parse_ref(names: tuple[str, ...]) -> tuple[str | None, tuple[str, ...]]:
    """(frequency | None, canonical path) from parsed dotted NAME parts. A known frequency
    leading a 3+-part path is the qualifier; a 2-part daily.price is never mis-split."""
    if len(names) >= 3 and names[0] in _FREQUENCIES:
        return names[0], names[1:]
    return None, names


def split_frequency(column: str) -> tuple[str | None, str]:
    """(frequency | None, rest) - mirrors parse_ref over an already-joined string."""
    head, _, rest = column.partition(".")
    if head in _FREQUENCIES and rest.count(".") >= 1:
        return head, rest
    return None, column


def split_pin(column: str) -> tuple[str, str | None]:
    """(base, entity | None) from a possibly entity-pinned column (x@SPY -> x, SPY)."""
    base, sep, ent = column.partition("@")
    return base, (ent if sep else None)


def split_source(column: str) -> tuple[str, str | None]:
    """(base, source | None) from a possibly source-pinned/tagged column (x#edgar -> x, edgar).

    A `#source` qualifier tags the physical column of a field served from one specific source -
    both an explicit `@ source` pin and a coalescing intermediate share this one encoding."""
    base, sep, src = column.partition("#")
    return base, (src if sep else None)


def canonical(column: str) -> str:
    """Strip every qualifier (pin/source suffix + frequency prefix) to the canonical field, for
    schema/kind lookup: daily.price.adj_close@SPY -> price.adj_close, income.revenue#edgar ->
    income.revenue. HIGHEST-severity for align: a `#source` tag that leaks here would resolve the
    wrong kind (summing a stock/ratio tag instead of taking its last)."""
    base = split_source(split_pin(column)[0])[0]
    return split_frequency(base)[1]
