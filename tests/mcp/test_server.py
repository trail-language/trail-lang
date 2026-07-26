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


def test_cors_app_allows_browser_preflight_and_exposes_session_id():
    """A browser calling /mcp cross-origin must get an allowed preflight AND be able to READ the
    session id - without `Mcp-Session-Id` in expose-headers every follow-up call is sessionless."""
    from mcp.server.fastmcp import FastMCP
    from starlette.testclient import TestClient

    from trail.mcp.server import _cors_http_app, _register
    # host mirrors the deployed container (`--host 0.0.0.0`). It matters: the SDK auto-enables
    # DNS-rebinding protection for 127.0.0.1/localhost/::1 only, which would 421 a browser Host.
    server = FastMCP("trail", host="0.0.0.0", port=3000)
    _register(server)
    app = _cors_http_app(server)
    origin = "http://primer.ws.local"

    pre = TestClient(app).options(
        "/mcp", headers={"Origin": origin, "Access-Control-Request-Method": "POST",
                         "Access-Control-Request-Headers": "content-type,mcp-session-id"})
    assert pre.status_code == 200
    assert pre.headers["access-control-allow-origin"] == "*"

    # expose-headers rides the ACTUAL response (never the preflight), so drive a real `initialize`
    with TestClient(app) as client:  # the `with` runs the lifespan that starts the session manager
        res = client.post("/mcp", headers={
            "Origin": origin, "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"}, json={
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                           "clientInfo": {"name": "browser", "version": "1.0"}}})
    assert res.status_code == 200
    assert "mcp-session-id" in res.headers["access-control-expose-headers"].lower()
    assert res.headers.get("mcp-session-id")  # the id the browser must be able to read back


async def test_server_lists_and_calls_eval():
    from mcp.server.fastmcp import FastMCP

    from trail.mcp.server import _register
    server = FastMCP("trail")
    _register(server)
    tools = await server.list_tools()
    names = {t.name for t in tools}
    assert {"functions", "schema", "validate", "describe", "eval", "run",
            "fetch", "drop", "views", "refresh"} <= names
    res = await server.call_tool("eval", {
        "expression": "income.revenue",
        "data": {"rows": [{"entity": "A", "time": "2020-12-31", "income.revenue": 5.0}]},
        "format": "records"})
    assert "5.0" in str(res) or 5.0 in str(res)
