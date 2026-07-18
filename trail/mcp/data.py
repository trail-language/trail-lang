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
    return panel.sort([ENTITY_COL, TIME_COL])


def resolve_panel(data: dict, decl=None, universes=None) -> tuple[pl.DataFrame, list[str]]:
    if not isinstance(data, dict) or len(data) == 0:
        raise DataSpecError("E-DATA `data` must be one of {config}, {file}, {rows}")
    if "rows" in data:
        return _finalize(pl.from_dicts(data["rows"], infer_schema_length=None)), []
    if "file" in data:
        path = data["file"]
        df = pl.read_csv(path) if path.endswith(".csv") else pl.read_parquet(path)
        return _finalize(df), []
    if "config" in data:
        from trail.mcp._config_data import resolve_config_panel   # optional-heavy; imported lazily
        return resolve_config_panel(data["config"], decl, universes)
    raise DataSpecError(f"E-DATA unknown data spec keys: {sorted(data)}")
