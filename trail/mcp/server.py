"""FastMCP server exposing the trail tools. The only module that imports the `mcp` SDK, and only
when serving - so `import trail` never pulls in the optional dependency."""
from __future__ import annotations


def _import_fastmcp():
    from mcp.server.fastmcp import FastMCP  # optional extra
    return FastMCP


def _cors_http_app(server):
    """The streamable-http Starlette app wrapped with CORS so a browser can call `/mcp` cross-origin
    (an HTML report opened in Primer Studio). `expose_headers` is the linchpin: without
    `Mcp-Session-Id` exposed the browser cannot read the session id from the `initialize` response
    and every follow-up call would be rejected as sessionless. Origins are unrestricted because this
    is an internal-only service holding no credentials, and a sandboxed iframe presents a `null`
    origin an allow-list would reject. DNS-rebinding protection stays off (the SDK default), so the
    browser `Origin`/`Host` are not rejected upstream of CORS."""
    from starlette.middleware.cors import CORSMiddleware

    app = server.streamable_http_app()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Mcp-Session-Id"],
    )
    return app


def serve(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8000) -> None:
    try:
        FastMCP = _import_fastmcp()
    except ImportError as e:
        raise RuntimeError(
            "the MCP server needs the optional 'mcp' dependency: pip install trail-lang[mcp]"
        ) from e
    server = FastMCP("trail", host=host, port=port)  # host/port apply to the streamable-http transport
    _register(server)
    if transport == "streamable-http":  # serve the CORS-wrapped app ourselves; stdio is unchanged
        import uvicorn

        uvicorn.run(_cors_http_app(server), host=host, port=port, log_level="info")
    else:
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
             to_file: str | None = None, no_stdlib: bool = False, streaming: bool = False,
             entities: list[str] | None = None) -> dict:
        """Evaluate a trail EXPRESSION over `data` -> a [entity, time, value] panel. `where` filters the
        universe; `at` sets frequency. offset/limit omitted => full data; `format` is
        compact|records|markdown|csv; `to_file` writes instead of inlining. `streaming` runs the
        bounded-memory out-of-core engine (for panels larger than RAM; slower per-row, so leave off
        when the panel fits). `entities` scopes the FETCH itself to those symbols (a `{config}` source
        then requests only them) - use it to refresh a few names without paying for the whole universe;
        `where` filters after loading and does not reduce fetch cost."""
        return tools.eval_tool(expression, data, where=where, at=at, offset=offset, limit=limit,
                               format=format, to_file=to_file, no_stdlib=no_stdlib,
                               streaming=streaming, entities=entities)

    @server.tool()
    def run(name: str, data: dict, program: str | None = None, path: str | None = None,
            offset: int | None = None, limit: int | None = None, format: str = "compact",
            to_file: str | None = None, no_stdlib: bool = False, streaming: bool = False,
            entities: list[str] | None = None) -> dict:
        """Run a named model/signal from a full trail program. Pass exactly one of `program` (inline
        source) or `path` (a .trail file, so `import` resolves). Result panel paginated + formatted.
        `streaming` runs the bounded-memory out-of-core engine (for panels larger than RAM).
        `entities` scopes the FETCH to those symbols - rescoring a handful of names costs a handful
        of fetches rather than the whole configured universe."""
        return tools.run_tool(name, data, program=program, path=path, offset=offset, limit=limit,
                              format=format, to_file=to_file, no_stdlib=no_stdlib,
                              streaming=streaming, entities=entities)

    @server.tool()
    def fetch(expressions: list[str], data: dict, where: str | None = None, at: str | None = None,
              offset: int | None = None, limit: int | None = None, format: str = "compact",
              to_file: str | None = None, no_stdlib: bool = False, streaming: bool = False,
              entities: list[str] | None = None) -> dict:
        """Fetch a WIDE frame: project several trail EXPRESSIONS into one [entity, time, <cols>] panel
        (retrieval, not a single computed value like `eval`). Each expression becomes a column. `where`
        filters the universe, `at` sets frequency; `data`/`format`/`streaming`/`entities` behave as in
        `eval` - `entities` scoping the fetch, `where` filtering what was already loaded."""
        return tools.fetch_tool(expressions, data, where=where, at=at, offset=offset, limit=limit,
                                format=format, to_file=to_file, no_stdlib=no_stdlib,
                                streaming=streaming, entities=entities)
