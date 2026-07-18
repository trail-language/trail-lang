import datetime as dt

import polars as pl

from trail.mcp.format import format_result


def _df():
    return pl.DataFrame({"entity": ["A", "B", "C"],
                         "time": [dt.datetime(2020, 12, 31)] * 3,
                         "value": [1.0, 2.0, 3.0]})


def test_full_when_no_pagination_compact():
    r = format_result(_df(), fmt="compact")
    assert r["total_rows"] == 3 and r["returned_rows"] == 3
    assert r["data"]["value"] == [1.0, 2.0, 3.0]
    assert r["data"]["time"] == ["2020-12-31T00:00:00"] * 3   # datetimes ISO-serialized


def test_offset_limit_slices_but_reports_total():
    r = format_result(_df(), offset=1, limit=1, fmt="records")
    assert r["total_rows"] == 3 and r["returned_rows"] == 1
    assert r["records"] == [{"entity": "B", "time": "2020-12-31T00:00:00", "value": 2.0}]


def test_markdown_and_csv():
    assert "| entity" in format_result(_df(), fmt="markdown")["table"]
    assert format_result(_df(), fmt="csv")["csv"].splitlines()[0] == "entity,time,value"


def test_to_file_writes_and_returns_path(tmp_path):
    p = tmp_path / "out.parquet"
    r = format_result(_df(), to_file=str(p))
    assert r["path"] == str(p) and r["shape"] == [3, 3] and p.exists()
