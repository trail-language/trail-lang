"""Resolve a `data` spec ({config}/{file}/{rows}) to a Polars panel. {file}/{rows} are complete
provided panels (used as-is); {config} loads the referenced fields from configured sources (see
_config_data). A process-level cache memoizes on the normalized spec so repeated calls are cheap."""
from __future__ import annotations

import polars as pl

from trail.source import ENTITY_COL, TIME_COL


class DataSpecError(ValueError):
    pass


def _finalize(panel: pl.DataFrame) -> pl.DataFrame:
    for col in (ENTITY_COL, TIME_COL):
        if col not in panel.columns:
            raise DataSpecError(f"E-DATA panel is missing the required '{col}' column")
    if not panel.schema[TIME_COL].is_temporal():           # accept ISO strings / dates for `time`
        panel = panel.with_columns(
            pl.col(TIME_COL).str.to_datetime(strict=False)
            if panel.schema[TIME_COL] == pl.Utf8
            else pl.col(TIME_COL).cast(pl.Datetime("us")))
    # Mark entity sorted so per-entity window ops (.over(entity)) skip re-sorting the panel
    # (~20% on time-series ops). entity is the leading sort key, so it is globally non-decreasing;
    # time is only sorted *within* entity, so it is deliberately not flagged.
    return panel.sort([ENTITY_COL, TIME_COL]).with_columns(pl.col(ENTITY_COL).set_sorted())


def _finalize_lazy(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Lazy counterpart of _finalize for the scan/{file} path: keeps the plan lazy from the scan
    down to collect, so Polars' projection pushdown reads only the columns an expression references
    (and predicate pushdown can drop rows) at the parquet scan itself, instead of loading every field."""
    schema = lf.collect_schema()   # metadata only - no data is read
    for col in (ENTITY_COL, TIME_COL):
        if col not in schema.names():
            raise DataSpecError(f"E-DATA panel is missing the required '{col}' column")
    if not schema[TIME_COL].is_temporal():
        lf = lf.with_columns(
            pl.col(TIME_COL).str.to_datetime(strict=False)
            if schema[TIME_COL] == pl.Utf8 else pl.col(TIME_COL).cast(pl.Datetime("us")))
    return lf.sort([ENTITY_COL, TIME_COL]).with_columns(pl.col(ENTITY_COL).set_sorted())


def resolve_panel(data: dict, decl=None, universes=None,
                  lazy: bool = False) -> tuple[pl.DataFrame | pl.LazyFrame, list[str]]:
    if not isinstance(data, dict) or len(data) == 0:
        raise DataSpecError("E-DATA `data` must be one of {config}, {file}, {rows}")
    if "rows" in data:
        return _finalize(pl.from_dicts(data["rows"], infer_schema_length=None)), []
    if "file" in data:  # lazy scan keeps projection pushdown alive (read only referenced fields)
        path = data["file"]
        if lazy:
            scan = pl.scan_csv(path) if path.endswith(".csv") else pl.scan_parquet(path)
            return _finalize_lazy(scan), []
        df = pl.read_csv(path) if path.endswith(".csv") else pl.read_parquet(path)
        return _finalize(df), []
    if "config" in data:
        from trail.mcp._config_data import resolve_config_panel   # optional-heavy; imported lazily
        return resolve_config_panel(data["config"], decl, universes)
    raise DataSpecError(f"E-DATA unknown data spec keys: {sorted(data)}")
