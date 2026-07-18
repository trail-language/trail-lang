"""Panel discoverability: summarize a loaded panel's fields and its categorical value distributions.

Surfaces what a provider actually publishes - e.g. which ``meta.sector`` strings a source emits -
so a user writes a correct, provider-specific expression instead of guessing (the language stays
schema-agnostic, so the remedy for a mismatched literal is discoverability). Purely descriptive:
values are reported verbatim, never normalized or canonicalized.
"""
from __future__ import annotations

import polars as pl

from trail.source import ENTITY_COL, TIME_COL, is_date_col

#: a string field with at most this many distinct values is auto-detected as categorical
CATEGORICAL_MAX_CARDINALITY = 50
#: meta fields always summarized when present (regardless of cardinality)
ALWAYS_CATEGORICAL = ("meta.sector", "meta.exchange", "meta.country")
#: default cap on a single field's reported distinct values (`--field` high-cardinality path)
FIELD_VALUE_CAP = 50


def panel_fields(panel: pl.DataFrame) -> list[str]:
    """Value fields on the panel (columns minus the index and reserved ``__date:*`` coordinates)."""
    return sorted(c for c in panel.columns
                  if c not in (ENTITY_COL, TIME_COL) and not is_date_col(c))


def fields_by_namespace(panel: pl.DataFrame) -> dict[str, list[str]]:
    """Value fields grouped by their leading namespace (``income``/``meta``/...), each list sorted."""
    out: dict[str, list[str]] = {}
    for f in panel_fields(panel):
        out.setdefault(f.split(".", 1)[0], []).append(f)
    return out


def value_counts(panel: pl.DataFrame, field: str, cap: int | None = None
                 ) -> tuple[list[tuple[object, int]], int]:
    """Distinct observed (non-null) values of ``field`` with row counts.

    Returns ``(rows, total_distinct)`` where ``rows`` is sorted by count descending then value
    ascending (a stable, greppable order). ``cap`` truncates ``rows`` to that many entries while
    ``total_distinct`` still reports the full count, so a caller can flag the truncation.
    """
    s = panel.get_column(field).drop_nulls()
    vc = s.value_counts()
    count_col = "count" if "count" in vc.columns else "counts"
    val_col = next(c for c in vc.columns if c != count_col)
    vc = vc.sort([count_col, val_col], descending=[True, False])
    total = vc.height
    rows = [(r[val_col], int(r[count_col])) for r in vc.iter_rows(named=True)]
    if cap is not None and len(rows) > cap:
        rows = rows[:cap]
    return rows, total


def categorical_fields(panel: pl.DataFrame) -> list[str]:
    """String fields worth summarizing: the always-on meta fields, plus any string field whose
    distinct-value count is at most :data:`CATEGORICAL_MAX_CARDINALITY`. Sorted by field name."""
    out: list[str] = []
    for f in panel_fields(panel):
        is_string = panel.schema[f] == pl.Utf8
        if f in ALWAYS_CATEGORICAL and is_string:
            out.append(f)
        elif is_string and panel.get_column(f).n_unique() <= CATEGORICAL_MAX_CARDINALITY:
            out.append(f)
    return out


def _format_counts(rows: list[tuple[object, int]], indent: str = "    ") -> list[str]:
    if not rows:
        return [f"{indent}(no observed values)"]
    width = max(len(str(v)) for v, _ in rows)
    return [f"{indent}{str(v):<{width}}  {c}" for v, c in rows]


def _period(panel: pl.DataFrame) -> str:
    if TIME_COL not in panel.columns or panel.height == 0:
        return ""
    lo = panel.get_column(TIME_COL).min()
    hi = panel.get_column(TIME_COL).max()
    return f" | period {lo:%Y-%m-%d}..{hi:%Y-%m-%d}"


def render_field(panel: pl.DataFrame, field: str, cap: int | None = FIELD_VALUE_CAP) -> str:
    """Render just one field's distinct values + counts. Any field is allowed; a high-cardinality
    field is capped to ``cap`` values with an explicit truncation note (never silently hidden)."""
    if field not in panel.columns:
        return (f"field '{field}' is not in the loaded panel; available fields: "
                f"{', '.join(panel_fields(panel)) or '(none)'}")
    rows, total = value_counts(panel, field, cap=cap)
    lines = [f"{field} ({total} distinct value(s), {panel.height} rows):"]
    lines += _format_counts(rows)
    if len(rows) < total:
        lines.append(f"    ... showing top {len(rows)} of {total} distinct values (truncated)")
    return "\n".join(lines)


def render_describe(panel: pl.DataFrame, field: str | None = None) -> str:
    """Render the full panel summary, or just one field when ``field`` is given.

    Full summary: a one-line panel header, the available fields grouped by namespace, and - for
    every categorical (low-cardinality string) field - its distinct observed values with counts.
    """
    if field is not None:
        return render_field(panel, field)

    fields = panel_fields(panel)
    entities = panel.get_column(ENTITY_COL).n_unique() if ENTITY_COL in panel.columns else 0
    lines = [f"Panel: {panel.height} rows | {entities} entities | {len(fields)} fields"
             f"{_period(panel)}", ""]

    lines.append("Fields by namespace:")
    ns = fields_by_namespace(panel)
    for name in sorted(ns):
        cols = ns[name]
        lines.append(f"  {name} ({len(cols)}): {', '.join(cols)}")
    lines.append("")

    cats = categorical_fields(panel)
    lines.append(f"Categorical fields (string, <= {CATEGORICAL_MAX_CARDINALITY} distinct):")
    if not cats:
        lines.append("  (none)")
    for f in cats:
        rows, total = value_counts(panel, f)
        lines.append(f"  {f} ({total} distinct):")
        lines += _format_counts(rows, indent="      ")
    return "\n".join(lines)
