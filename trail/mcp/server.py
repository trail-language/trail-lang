"""FastMCP server exposing the trail tools. The only module that imports the `mcp` SDK, and only
when serving - so `import trail` never pulls in the optional dependency."""
from __future__ import annotations


def _import_fastmcp():
    from mcp.server.fastmcp import FastMCP  # optional extra
    return FastMCP


def serve(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8000) -> None:
    try:
        FastMCP = _import_fastmcp()
    except ImportError as e:
        raise RuntimeError(
            "the MCP server needs the optional 'mcp' dependency: pip install trail-lang[mcp]"
        ) from e
    server = FastMCP("trail", host=host, port=port)  # host/port apply to the streamable-http transport
    _register(server)
    server.run(transport=transport)


def _register(server) -> None:
    from trail.mcp import tools

    @server.tool()
    def functions(query: str | None = None, axis: str | None = None) -> dict:
        """Search trail's function/operator catalog. `query` filters by name/summary; `axis` is one of
        elementwise|time-series|cross-sectional|model."""
        return tools.functions_tool(query=query, axis=axis)

    @server.tool()
    def schema(namespace: str | None = None) -> dict:
        """List the core field vocabulary (field + kind), optionally filtered to one namespace
        (income/balance/cash/price/meta/...)."""
        return tools.schema_tool(namespace=namespace)

    @server.tool()
    def validate(source: str, no_stdlib: bool = False) -> dict:
        """Parse + validate trail source (an expression, model, or full program). Returns
        {valid, issues:[{severity,code,message}]}."""
        return tools.validate_tool(source, no_stdlib=no_stdlib)

    @server.tool()
    def describe(data: dict, field: str | None = None) -> dict:
        """Explore a dataset: fields by namespace + categorical fields' distinct values (verbatim).
        `data` is one of {"config":path} | {"file":path} | {"rows":[...]}. `field` narrows to one field."""
        return tools.describe_tool(data, field=field)

    @server.tool()
    def eval(expression: str, data: dict, where: str | None = None, at: str | None = None,
             offset: int | None = None, limit: int | None = None, format: str = "compact",
             to_file: str | None = None, no_stdlib: bool = False) -> dict:
        """Evaluate a trail EXPRESSION over `data` -> a [entity, time, value] panel. `where` filters the
        universe; `at` sets frequency. offset/limit omitted => full data; `format` is
        compact|records|markdown|csv; `to_file` writes instead of inlining."""
        return tools.eval_tool(expression, data, where=where, at=at, offset=offset, limit=limit,
                               format=format, to_file=to_file, no_stdlib=no_stdlib)

    @server.tool()
    def run(name: str, data: dict, program: str | None = None, path: str | None = None,
            offset: int | None = None, limit: int | None = None, format: str = "compact",
            to_file: str | None = None, no_stdlib: bool = False) -> dict:
        """Run a named model/signal from a full trail program. Pass exactly one of `program` (inline
        source) or `path` (a .trail file, so `import` resolves). Result panel paginated + formatted."""
        return tools.run_tool(name, data, program=program, path=path, offset=offset, limit=limit,
                              format=format, to_file=to_file, no_stdlib=no_stdlib)
