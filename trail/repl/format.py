"""Formatting helpers for REPL output."""
from __future__ import annotations


def truncate_table(
    df,  # noqa: ANN001 — polars DataFrame
    max_rows: int = 20,
    max_cols: int = 12,
    col_width: int = 14,
) -> str:
    """Format a Polars DataFrame for terminal display.

    Truncates rows and columns with a summary row when limits are exceeded.
    Uses Polars' internal formatting but controls truncation.
    """
    import polars as pl

    # Use Polars' built-in formatting
    with pl.Config() as cfg:
        cfg.set_tbl_width_chars(200)
        cfg.set_tbl_rows(50)
        cfg.set_tbl_cols(20)

        if len(df.columns) > max_cols:
            # Keep first half and last half of columns
            n_keep = max_cols // 2
            cols = list(df.columns[:n_keep]) + list(df.columns[-n_keep:])
            df = df.select(cols)

        if len(df) > max_rows:
            head = df.head(max_rows // 2)
            tail = df.tail(max_rows // 2)
            head_str = str(head)
            tail_str = str(tail)
            omitted = len(df) - max_rows
            middle = f"\n  ... {omitted} rows omitted ...\n"
            return f"{head_str}{middle}{tail_str}"
        return str(df)
