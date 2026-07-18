"""FastMCP server exposing the trail tools. The only module that imports the `mcp` SDK, and only
when serving - so `import trail` never pulls in the optional dependency."""
from __future__ import annotations


def _import_fastmcp():
    from mcp.server.fastmcp import FastMCP  # optional extra
    return FastMCP


def serve(transport: str = "stdio") -> None:
    try:
        FastMCP = _import_fastmcp()
    except ImportError as e:
        raise RuntimeError(
            "the MCP server needs the optional 'mcp' dependency: pip install trail-lang[mcp]"
        ) from e
    server = FastMCP("trail")
    _register(server)
    server.run(transport=transport)


def _register(server) -> None:  # populated in Task 10
    pass
