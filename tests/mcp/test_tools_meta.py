from trail.mcp.tools import functions_tool, schema_tool, validate_tool


def test_functions_lists_and_filters():
    allf = functions_tool()
    assert any(f["function"] == "zscore" for f in allf["functions"])
    names = [f["function"] for f in functions_tool(query="zscore")["functions"]]
    assert "zscore" in names and all("zscore" in n for n in names)   # substring filter


def test_functions_axis_filter():
    xs = functions_tool(axis="cross-sectional")
    assert xs["functions"] and all(f["axis"] == "cross-sectional" for f in xs["functions"])


def test_schema_lists_and_filters():
    assert any(f["field"] == "income.revenue" for f in schema_tool()["fields"])
    inc = schema_tool(namespace="income")
    assert all(f["field"].startswith("income.") for f in inc["fields"])


def test_validate_ok_and_error():
    assert validate_tool("model m at annual { on_missing skip export r = income.revenue }")["valid"] is True
    bad = validate_tool("model m at annual { on_missing skip export r = nope.field }")
    assert bad["valid"] is False
    assert any(i["code"] == "E-FIELD-UNKNOWN" for i in bad["issues"])


def test_validate_syntax_error_is_issue_not_crash():
    r = validate_tool("model m at annual {")
    assert r["valid"] is False and r["issues"]
