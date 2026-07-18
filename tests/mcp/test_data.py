import datetime as dt

import polars as pl
import pytest

from trail.mcp.data import DataSpecError, resolve_panel


def test_rows_spec_builds_panel():
    data = {"rows": [{"entity": "A", "time": "2020-12-31", "income.revenue": 10.0},
                     {"entity": "B", "time": "2020-12-31", "income.revenue": 20.0}]}
    panel, warns = resolve_panel(data)
    assert panel.height == 2 and "income.revenue" in panel.columns
    assert panel.schema["time"].is_temporal()


def test_rows_missing_index_errors():
    with pytest.raises(DataSpecError):
        resolve_panel({"rows": [{"entity": "A", "income.revenue": 1.0}]})  # no time


def test_file_spec_reads_parquet(tmp_path):
    p = tmp_path / "panel.parquet"
    pl.DataFrame({"entity": ["A"], "time": [dt.datetime(2020, 12, 31)], "x": [1.0]}).write_parquet(p)
    panel, _ = resolve_panel({"file": str(p)})
    assert panel.height == 1 and "x" in panel.columns


def test_unknown_spec_errors():
    with pytest.raises(DataSpecError):
        resolve_panel({"nope": 1})
