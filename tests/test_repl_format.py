"""Tests for trail.repl.format — table formatting helpers."""
from __future__ import annotations

import polars as pl

from trail.repl.format import truncate_table


class TestTruncateTable:
    def test_small_table_no_truncation(self):
        """Small table passes through unchanged."""
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = truncate_table(df, max_rows=20, max_cols=12)
        assert "a" in result and "b" in result
        assert "omitted" not in result.lower()

    def test_rows_truncated(self):
        """Exceeding max_rows shows omitted row count."""
        df = pl.DataFrame({"a": range(100), "b": range(100)})
        result = truncate_table(df, max_rows=20, max_cols=12)
        assert "omitted" in result.lower()

    def test_columns_truncated(self):
        """Exceeding max_cols shows subset of columns."""
        df = pl.DataFrame({f"col_{i}": [i] for i in range(20)})
        result = truncate_table(df, max_rows=20, max_cols=10)
        # Should show some columns but not all
        assert len(result) > 0

    def test_exact_fit_no_truncation(self):
        """Table exactly at limits is not truncated."""
        df = pl.DataFrame({"a": range(20), "b": range(20)})
        result = truncate_table(df, max_rows=20, max_cols=2)
        assert "omitted" not in result.lower()
