"""fetch_tool: project several expressions into one wide [entity, time, <cols>] frame (retrieval,
not a single computed value). Reuses the model/export machinery, so universe binding, validation,
and alignment behave exactly as eval/run."""
from trail.mcp.tools import fetch_tool


def _rows(n_ent=4, n_per=5):
    rows = []
    for e in range(n_ent):
        for t in range(n_per):
            rows.append({"entity": f"E{e}", "time": f"20{10 + t:02d}-01-01",
                         "income.revenue": float((e + 1) * (t + 1) * 100),
                         "income.net_income": float((e + 1) * (t + 1) * 10),
                         "meta.market_cap": float((e + 1) * 1000)})
    return rows


def test_fetch_multiple_fields_wide_frame():
    out = fetch_tool(["income.revenue", "meta.market_cap"], {"rows": _rows()}, format="records")
    assert "records" in out, out
    assert out["total_rows"] == 20
    r0 = out["records"][0]
    assert {"entity", "time", "income.revenue", "meta.market_cap"} <= set(r0)


def test_fetch_single_field():
    out = fetch_tool(["income.revenue"], {"rows": _rows()}, format="records", limit=3)
    assert "income.revenue" in out["records"][0]


def test_fetch_computed_expression_becomes_a_column():
    col = "income.net_income / income.revenue"
    out = fetch_tool([col], {"rows": _rows()}, format="records", limit=2)
    rec = out["records"][0]
    assert col in rec and abs(rec[col] - 0.1) < 1e-9  # net_income is 1/10 of revenue in the fixture


def test_fetch_where_filters_the_universe():
    out = fetch_tool(["income.revenue"], {"rows": _rows()}, where='entity == "E1"', format="records")
    assert {r["entity"] for r in out["records"]} == {"E1"}


def test_fetch_empty_list_errors():
    assert fetch_tool([], {"rows": _rows()}).get("error", {}).get("code") == "E-ARGS"


def test_fetch_duplicate_expressions_keep_columns_map():
    # ambiguous column naming -> keep f0..fn and hand back an explicit map rather than collapse
    out = fetch_tool(["income.revenue", "income.revenue"], {"rows": _rows()}, format="records", limit=1)
    assert out.get("columns") == {"f0": "income.revenue", "f1": "income.revenue"}
    assert {"f0", "f1"} <= set(out["records"][0])


def test_fetch_unknown_field_errors():
    out = fetch_tool(["income.revenue", "not.a_field"], {"rows": _rows()})
    assert "error" in out
