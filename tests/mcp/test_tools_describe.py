from trail.mcp.tools import describe_tool

DATA = {"rows": [{"entity": "A", "time": "2020-12-31", "meta.sector": "Tech"},
                 {"entity": "B", "time": "2020-12-31", "meta.sector": "Tech"},
                 {"entity": "C", "time": "2020-12-31", "meta.sector": "Energy"}]}


def test_describe_categorical_values():
    r = describe_tool(DATA)
    sect = next(c for c in r["categorical"] if c["field"] == "meta.sector")
    assert {"value": "Tech", "count": 2} in sect["distinct"]


def test_describe_single_field():
    r = describe_tool(DATA, field="meta.sector")
    assert r["field"] == "meta.sector" and r["total_distinct"] == 2


def test_describe_bad_data_returns_error():
    assert "error" in describe_tool({"rows": [{"entity": "A"}]})   # missing time
