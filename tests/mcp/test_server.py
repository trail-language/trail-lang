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
