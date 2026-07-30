# Trail MCP Tools — Detailed Contracts

The ten tools (`functions`, `schema`, `validate`, `describe`, `eval`, `run`, `fetch`,
`drop`, `views`, `refresh`) are pure importable functions in `trail.mcp.tools`
(`<name>_tool`) with identical signatures to the MCP tools, so you can verify any
call offline. **Every tool returns structured JSON — a trail-level failure comes
back as `{"error": {"code": "E-...", "message": "..."}}`, never an exception or a
crash.** `validate` returns `{valid, issues}` instead (a syntax error is an issue).

## Connecting

- **stdio** (local MCP clients): `trail mcp` (needs `pip install "trail-lang[mcp]"`).
- **HTTP** (network): `trail mcp --transport streamable-http --host 0.0.0.0 --port 3000`
  → MCP endpoint at `/mcp`. A deployed instance is reachable at
  **`http://trail-mcp.ws.local/mcp`**. Point any MCP client's streamable-HTTP
  transport at that URL.

---

## `functions(query=None, axis=None) -> {"functions": [...]}`
Search the function/operator catalog (primitives from the op registry + derived
stdlib macros).
- `query` — case-insensitive substring over function name **and** summary.
- `axis` — exact filter: `elementwise` | `time-series` | `cross-sectional` | `model`.
- Row shape: `{function, layer("primitive"|"derived"), axis, args("1"|"2..3"|...), summary}`.
```python
functions_tool(query="drawdown")
functions_tool(axis="cross-sectional")   # zscore, rank, winsorize, xs_*, ...
```

## `schema(namespace=None) -> {"fields": [{"field","kind"}]}`
The active field vocabulary (core + any installed source packages), optionally one
namespace (`income`/`balance`/`cash`/`price`/`meta`/…).
```python
schema_tool()             # everything
schema_tool("balance")    # balance.* only
```

## `validate(source, no_stdlib=False, base_dir=None) -> {"valid", "issues"}`
Static parse + check of an expression, model, or full program. **No data, no
config** — your first, fastest gate. `issues` is `[{severity,code,message}]`;
`valid` is false only if some issue has `severity == "error"` (warnings pass).
`base_dir` (a file path whose directory holds `import` targets) resolves imports.
```python
validate_tool("model m at annual { on_missing skip export r = income.revenue }")
# {"valid": true, "issues": []}
validate_tool("export r = nope.field")           # -> valid:false, E-FIELD-UNKNOWN
```

## `describe(data, field=None) -> shape + fields_by_namespace + categorical[...]`
Explore a dataset. Without `field`: `{shape:{rows,entities,fields},
fields_by_namespace, categorical:[{field, distinct:[{value,count}], truncated}],
warnings}`. With `field`: `{field, total_distinct, distinct:[{value,count}]}`.
Use it to find the exact categorical strings a source emits (e.g. `meta.sector` =
"Financial Services", not "Financials") before writing a filter.
```python
describe_tool({"config":"trail.yaml"}, field="meta.sector")
```

## `eval(expression, data, where=None, at=None, offset=None, limit=None, format="compact", to_file=None, no_stdlib=False) -> panel`
Evaluate ONE expression over `data` → an `(entity, time, value)` panel.
- `where` — a boolean filter; wraps the panel in `stocks where <where>`.
- `at` — target frequency (`annual`/`quarterly`/`monthly`/`daily`/…); default =
  finest referenced.
- `offset`/`limit` — pagination (omit both = full panel).
- `format` — `compact`(default)|`records`|`markdown`|`csv`.
- `to_file` — write parquet/csv, return `{path, shape}` instead of inlining.
```python
eval_tool("zscore(income.net_income/income.revenue) by meta.sector", {"rows":rows}, format="records")
eval_tool("roll_mean(price.adj_close, 3)", {"rows":ts_rows}, at="daily")
```

## `run(name, data, program=None, path=None, offset=None, limit=None, format="compact", to_file=None, no_stdlib=False) -> panel`
Run a named `model`/`signal`. Pass **exactly one** of `program` (inline source) or
`path` (a `.trail` file — needed for `import` to resolve). Output carries every
`export` column. This is where `score`/`weighted_score()` and multi-export models run.
```python
run_tool("quality", {"rows":rows}, program=full_source, format="records")
run_tool("composite", {"config":"trail.yaml"}, path="models/composite.trail")
```
A `track`ed model/signal persists its result and serves it back incrementally — see
**Tracked views** below.

## `fetch(expressions, data, where=None, at=None, offset=None, limit=None, format="compact", to_file=None, entities=None) -> wide panel`
Project SEVERAL expressions into one `(entity, time, <cols>)` frame — retrieval, not a single
computed value. Each expression becomes a column (named by its text when unambiguous, else
`f0..fn` with a `columns` map). `where`/`at`/`format` behave as in `eval`; `entities` scopes
the *fetch* to those symbols (a `{config}` source then requests only them — pull a few names
without paying for the whole universe; `where` filters after loading and does not reduce fetch cost).
```python
fetch_tool(["income.revenue", "meta.sector", "zscore(income.net_income) by meta.sector"],
           {"config":"trail.yaml"}, entities=["AAPL","MSFT"], format="records")
```

---

## Tracked views: persist + incremental recompute
A `track model` / `track signal` declaration persists its result frame to a **view store** (a
writeable provider; a local-disk store is auto-configured under `<config-dir>/.trail/views` when
none is set). Running a tracked decl **serves the stored frame back** instead of recomputing —
until the program or panel config changes, or a dependency source reports new data. When a source
exposes a changefeed (e.g. FMP statement filing dates), only the affected cells recompute: a
reporter dirties its own time-series tail and its whole `by <group>` cross-section, nothing else;
the rest of the frame is reused unchanged. The result is identical to a full recompute — tracking is
a transparent optimization. Persistence needs a `{config}` data spec (the store lives beside it).

### `drop(name, data) -> {"dropped": bool}`
Delete a stored view so its next run recomputes fully from the expression (a force-refresh).
`dropped` is false if the view was absent. `data` is a `{config}` spec.

### `views(data) -> {"views":[{name, kind, columns, built_at, sources}]}`
List stored views with a manifest summary.

### `refresh(name, data, program=None, path=None) -> {"refreshed", "shape"}`
Drop then eagerly rebuild a tracked view. Pass the same `program`/`path` that declares it.
```python
run_tool("scores", {"config":"trail.yaml"}, path="models/scores.trail")   # build, then serve back
views_tool({"config":"trail.yaml"})                                        # what is stored
drop_tool("scores", {"config":"trail.yaml"})                               # force a full recompute next run
```
Configure a store explicitly (else the local-disk default is used) under `providers:` in `trail.yaml`:
```yaml
providers:
  views: { driver: views_local, options: { dir: /var/trail/views } }
```

---

## Result envelope (eval / run)
Every panel result: `{total_rows, returned_rows, offset, format, <payload>, warnings?}`.
Payload by format: `compact` → `{columns, data:{col:[...]}}`; `records` →
`{records:[{...}]}`; `markdown` → `{table:"..."}`; `csv` → `{csv:"..."}`.
`warnings` appears only when alignment/conformance warned (e.g. `W-UPSAMPLE-FLOW`).

## Error codes you will see
`E-SYNTAX` (parse; often a stray `;` or `a<b<c`), `E-FIELD-UNKNOWN`,
`E-FUNC-UNKNOWN`, `E-FUNC-ARITY`, `E-NAME-UNDEFINED`, `E-NAME-REBOUND`,
`E-UNIVERSE-UNKNOWN`, `E-MODEL-CONTEXT` (`weighted_score()` misuse), `E-IMPORT-*`,
`E-DATA` (bad `data` spec / missing `entity`/`time`), `E-CONFIG` (config/credential),
`E-ARGS` (`run` without exactly one of program/path; a view tool without `{config}`),
`E-VIEW-NAME` (invalid/traversing view name in `drop`/`refresh`), `E-PROVIDER-DRIVER` /
`E-PROVIDER-OPTIONS` (view-store config). Warnings (`W-KIND-STOCK-FLOW`,
`W-UPSAMPLE-FLOW`) never block.

## The universal loop
`schema()`/`functions()` to confirm names → `validate(source)` for a data-free
yes/no → build a tiny `{"rows":[...]}` panel → `eval`/`run` and read the value.
Never surface a Trail snippet you have not run through `eval`/`validate`.
