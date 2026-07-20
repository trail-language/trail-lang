# Trail over MCP

Trail is a small, total, declarative language for computing financial
indicators, scores, and screens over **panels** of entities. Every field and
every expression denotes a value per **`(entity, time)` cell** — there is no
scalar context and no per-entity loop. Programs are pure data: they always
terminate, have statically knowable data needs, and cannot look ahead in time.
The compiler lowers every construct to vectorized **Polars** columnar ops.

```trail
universe us = stocks where meta.exchange in ("NYSE", "NASDAQ") and meta.is_active
model quality on us at annual {
    on_missing skip
    gross_profitability = (income.revenue - income.cogs) / balance.total_assets
    export quality_z = zscore(gross_profitability) by meta.sector
    score gp weight 7 {
        2 if gross_profitability > 0.33
        1 if gross_profitability > 0.15
        else 0
    }
    export composite = weighted_score()
}
```

## Golden rule: validate everything before you trust it

The tools are the source of truth, not your memory of the syntax. **Never hand a
user a Trail expression you have not run through `validate` or `eval`.** Build a
tiny inline panel, `eval` the expression, confirm there is no `error`/`issues`
key and the value is sensible. If it errors, fix it against
`references/expressions.md` + `references/functions.md` and re-run.

## The MCP surface (six tools)

MCP tool names (from `trail.mcp.server`) — each is also a pure importable Python
function `trail.mcp.tools.<name>_tool` with the same signature, so you can verify
offline:

```python
from trail.mcp.tools import functions_tool, schema_tool, validate_tool, \
                             describe_tool, eval_tool, run_tool
```

| Tool | Signature (key args) | Returns | Use for |
|---|---|---|---|
| `functions` | `functions(query?, axis?)` | `{functions:[{function,layer,axis,args,summary}]}` | discover / confirm a function name, arity, axis, and whether it is primitive vs derived |
| `schema` | `schema(namespace?)` | `{fields:[{field,kind}]}` | list the field vocabulary and each field's kind |
| `validate` | `validate(source, no_stdlib?, base_dir?)` | `{valid, issues:[{severity,code,message}]}` | static parse+check of an expression / model / program — **config-free, no data** |
| `describe` | `describe(data, field?)` | shape + fields-by-namespace + categorical distinct values | explore an unknown dataset; find the exact `meta.sector` strings a source emits |
| `eval` | `eval(expression, data, where?, at?, offset?, limit?, format?, to_file?, no_stdlib?)` | `(entity, time, value)` panel | evaluate ONE expression over data |
| `run` | `run(name, data, program? | path?, offset?, limit?, format?, to_file?, no_stdlib?)` | model/signal output panel | run a named model or signal from a full program |

`axis` filters `functions` to one of `elementwise | time-series | cross-sectional | model`.

### The shared `data` spec

`describe`, `eval`, and `run` all take a `data` dict — exactly one of three keys:

| Form | Meaning |
|---|---|
| `{"rows": [ {...}, ... ]}` | an inline panel (list of cell dicts). **The fastest way to test.** |
| `{"file": "panel.parquet"}` \| `{"file":"panel.csv"}` | a complete panel read from disk |
| `{"config": "trail.yaml"}` | load fields from configured data sources (FMP, EDGAR, …) |

Each row MUST have `entity` (string) and `time` (ISO date/datetime string is
accepted and auto-parsed), plus one column per field named by its **canonical
dotted path** (`income.revenue`, `meta.sector`, …). Only real schema fields are
allowed — call `schema()` to see them. For cross-sectional / `by` examples give
**≥2 entities** in the same period plus the group field (e.g. `meta.sector`);
for time-series give **one entity across ≥3 periods**.

### Result formats (`format=`)

`compact` (default) = `{columns, data:{col:[...]}}`; `records` = list of row
dicts; `markdown` = an ASCII table string; `csv` = a CSV string. `offset`/`limit`
paginate (omit both = full panel). `to_file` writes parquet/csv and returns a
path instead of inlining. Every panel result also carries `total_rows`,
`returned_rows`, `offset`, `format`, and (when alignment warned) `warnings`.

## Worked tool calls (all verified)

**functions** — confirm a name/arity/axis before using it:
```python
functions_tool(query="zscore")
# {"functions":[{"function":"zscore","layer":"primitive","axis":"cross-sectional","args":"1","summary":"standardize within (period[, group])"}]}
functions_tool(axis="time-series")   # every time-series op
```

**schema** — the field vocabulary of one namespace:
```python
schema_tool("cash")
# {"fields":[{"field":"cash.capex","kind":"flow"}, {"field":"cash.cfo","kind":"flow"}, ...]}
```

**validate** — static check; no data needed, so this is your first gate for any
model. Returns `valid:true` even with warnings (only `severity:"error"` blocks):
```python
validate_tool('model m at annual { on_missing skip export gm = income.gross_profit/income.revenue }')
# {"valid": true, "issues": []}
```

**describe** — learn an unfamiliar panel (namespaces present + categorical values):
```python
describe_tool({"rows":[{"entity":"A","time":"2023-12-31","income.revenue":100.0,"meta.sector":"Tech"}]})
# {"shape":{"rows":1,"entities":1,"fields":2}, "fields_by_namespace":{"income":["income.revenue"],"meta":["meta.sector"]},
#  "categorical":[{"field":"meta.sector","distinct":[{"value":"Tech","count":1}], "truncated":false}], "warnings":[]}
```

**eval** — one expression over an inline panel (the core verify loop). Cross
-sectional example needs ≥2 entities + the group field:
```python
rows=[{"entity":"A","time":"2023-12-31","income.net_income":10.0,"income.revenue":100.0,"meta.sector":"Tech"},
      {"entity":"B","time":"2023-12-31","income.net_income":30.0,"income.revenue":200.0,"meta.sector":"Tech"}]
eval_tool("zscore(income.net_income / income.revenue) by meta.sector", {"rows":rows}, format="records")
# records: A -> -0.7071, B -> 0.7071
```
`where` wraps the panel in a filter universe; `at` sets the target frequency:
```python
eval_tool("yoy(income.revenue)", {"rows":ts_rows}, at="annual")     # one entity, ≥3 periods
eval_tool("zscore(income.revenue)", {"rows":rows}, where='meta.sector == "Tech"')
```

**run** — a named model/signal from a full program (inline `program=` OR a
`.trail` `path=` so `import` resolves). This is where `weighted_score()` and
multi-export models run:
```python
run_tool("quality", {"rows":rows}, program=full_source, format="records")
# records carry entity, time, and every `export` column (checklist, composite, ...)
run_tool("quality", {"config":"trail.yaml"}, path="models/quality.trail")   # real sources
```

## Choosing a tool

- Confirm a function exists / its arity → **functions**. Confirm a field exists → **schema**.
- Author a model, want a fast yes/no on syntax + names → **validate** (no data).
- Handed an unknown dataset → **describe** first, then write expressions.
- Test a single expression's *values* → **eval** with a `{"rows":[...]}` panel.
- Run a full model with `score`/`weighted_score()`/multiple exports → **run**.

## References (read these for depth)

- **`references/expressions.md`** — the complete expression notation: operators &
  precedence, field references & namespaces, the `@` qualifiers, null semantics,
  the `by` clause, the negative-value shift rule, and every declaration.
- **`references/functions.md`** — the full function catalog by category
  (time-series, cross-sectional, elementwise, temporal, risk, frequency,
  model-context) with signatures + a verified one-line example each.
- **`references/tools.md`** — detailed per-tool contracts, every argument, error
  shapes, and more worked calls.

## Gotchas confirmed against the reference implementation

- **Score cases and statements are NEWLINE-separated, not `;`-separated.** The
  spec prose sometimes writes `{ 2 if x>0.1; else 0 }` inline for readability, but
  the grammar rejects `;` (`E-SYNTAX`). Put each case on its own line.
- **`eval`/`run` only accept real schema fields.** A made-up field (`price.return`)
  is `E-FIELD-UNKNOWN`. The core price field is `price.adj_close`. Call `schema()`.
- **`normal_pdf` is spec-listed but NOT implemented** → `E-FUNC-UNKNOWN`. So are
  the post-1.0 names `sply`, `roll_tail_mean`, and registered functions.
- **Entity pins `@ entity("SPY")` validate statically but need source-backed
  data to execute** — they resolve at the alignment layer, so a plain `{"rows"}`
  panel raises an internal error. Use `{"config":...}` to run them.
- **`count` propagates null** (any null flag → null result); **`count_true`** folds
  null to false. Guard field-dependent flags: `(cash.stock_issued ?? 0) == 0`.
- Window/quantile/period args (`lag(x, n)`, `roll_mean(x, n)`, `cagr(x, n)`) MUST
  be numeric literals — never computed.
