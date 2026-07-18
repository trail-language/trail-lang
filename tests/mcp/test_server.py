import click.testing

import trail.cli


def test_mcp_command_registered():
    res = click.testing.CliRunner().invoke(trail.cli.main, ["--help"])
    assert "mcp" in res.output


def test_mcp_without_sdk_errors_cleanly(monkeypatch):
    import trail.mcp.server as srv

    def _boom():
        raise ImportError("no mcp")

    monkeypatch.setattr(srv, "_import_fastmcp", _boom)
    res = click.testing.CliRunner().invoke(trail.cli.main, ["mcp"])
    assert res.exit_code == 1
    assert "pip install trail-lang[mcp]" in res.output


import pytest  # noqa: E402

pytest.importorskip("mcp")   # the integration test needs the optional SDK


async def test_server_lists_and_calls_eval():
    from mcp.server.fastmcp import FastMCP

    from trail.mcp.server import _register
    server = FastMCP("trail")
    _register(server)
    tools = await server.list_tools()
    names = {t.name for t in tools}
    assert {"functions", "schema", "validate", "describe", "eval", "run"} <= names
    res = await server.call_tool("eval", {
        "expression": "income.revenue",
        "data": {"rows": [{"entity": "A", "time": "2020-12-31", "income.revenue": 5.0}]},
        "format": "records"})
    assert "5.0" in str(res) or 5.0 in str(res)
