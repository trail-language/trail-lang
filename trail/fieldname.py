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


def qualified(canonical: str, *, frequency: str | None = None, entity: str | None = None) -> str:
    """Encode a canonical field plus qualifiers into the physical column name
    (daily.price.adj_close@SPY)."""
    base = f"{frequency}.{canonical}" if frequency else canonical
    return f"{base}@{entity}" if entity else base


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


def canonical(column: str) -> str:
    """Strip every qualifier (pin suffix + frequency prefix) to the canonical field, for
    schema/kind lookup: daily.price.adj_close@SPY -> price.adj_close."""
    base = split_pin(column)[0]
    return split_frequency(base)[1]
