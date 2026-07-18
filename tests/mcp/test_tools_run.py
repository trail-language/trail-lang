from trail.mcp.tools import run_tool

DATA = {"rows": [{"entity": "A", "time": "2020-12-31", "income.revenue": 10.0, "income.cogs": 6.0},
                 {"entity": "B", "time": "2020-12-31", "income.revenue": 20.0, "income.cogs": 8.0}]}
PROGRAM = ("model m at annual { on_missing skip\n"
           "  export gross = income.revenue - income.cogs\n"
           "  export rev = income.revenue }")


def test_run_model_multi_export():
    r = run_tool("m", DATA, program=PROGRAM, format="records")
    recs = {rec["entity"]: rec for rec in r["records"]}
    assert recs["A"]["gross"] == 4.0 and recs["A"]["rev"] == 10.0


def test_run_unknown_name_errors():
    assert run_tool("nope", DATA, program=PROGRAM)["error"]["code"] == "E-NAME-UNKNOWN"


def test_run_requires_exactly_one_source():
    assert "error" in run_tool("m", DATA)                       # neither program nor path
