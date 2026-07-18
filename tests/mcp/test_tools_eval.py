from trail.mcp.tools import eval_tool

DATA = {"rows": [{"entity": "A", "time": "2020-12-31", "income.revenue": 10.0, "income.cogs": 6.0},
                 {"entity": "B", "time": "2020-12-31", "income.revenue": 20.0, "income.cogs": 8.0}]}


def test_eval_expression_over_rows():
    r = eval_tool("income.revenue - income.cogs", DATA, format="records")
    vals = {rec["entity"]: rec["value"] for rec in r["records"]}
    assert vals == {"A": 4.0, "B": 12.0}


def test_eval_where_filter():
    r = eval_tool("income.revenue", DATA, where='entity == "A"', format="records")
    assert r["total_rows"] == 1 and r["records"][0]["entity"] == "A"


def test_eval_bad_field_returns_error():
    assert "error" in eval_tool("nope.field", DATA)


def test_eval_pagination_full_by_default():
    assert eval_tool("income.revenue", DATA)["returned_rows"] == 2
