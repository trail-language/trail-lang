"""Turn a result panel into a paginated, agent-friendly payload. offset/limit omitted => full data;
`format` picks the representation (compact columnar by default). `to_file` writes instead of inlining."""
from __future__ import annotations

import polars as pl

_VALID = ("compact", "records", "markdown", "csv")


def _isoize(df: pl.DataFrame) -> pl.DataFrame:
    # JSON has no datetime; render temporal columns as ISO strings so payloads serialize cleanly.
    temporal = [c for c, t in df.schema.items() if t.is_temporal()]
    if not temporal:
        return df
    return df.with_columns([pl.col(c).dt.to_string("%Y-%m-%dT%H:%M:%S").alias(c) for c in temporal])


def format_result(df: pl.DataFrame, *, offset: int | None = None, limit: int | None = None,
                  fmt: str = "compact", to_file: str | None = None, extra: dict | None = None) -> dict:
    if fmt not in _VALID:
        raise ValueError(f"E-FORMAT unknown format '{fmt}'; use one of {_VALID}")
    if to_file:
        if to_file.endswith(".csv"):
            df.write_csv(to_file)
        else:
            df.write_parquet(to_file)
        return {"path": to_file, "shape": [df.height, df.width]}

    total = df.height
    off = offset or 0
    page = df.slice(off, limit) if (offset is not None or limit is not None) else df
    page = _isoize(page)
    out: dict = {"total_rows": total, "returned_rows": page.height, "offset": off, "format": fmt}
    if fmt == "compact":
        out |= {"columns": page.columns, "data": {c: page.get_column(c).to_list() for c in page.columns}}
    elif fmt == "records":
        out["records"] = page.to_dicts()
    elif fmt == "markdown":
        with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_hide_dataframe_shape=True,
                       tbl_rows=page.height, tbl_cols=page.width):
            out["table"] = str(page)
    else:  # csv
        out["csv"] = page.write_csv()
    if extra:
        out |= extra
    return out
