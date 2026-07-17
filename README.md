<p align="center">
  <img src="https://raw.githubusercontent.com/trail-language/spec/main/brand/icon-color-240.png" alt="Trail" width="120">
</p>

# Trail

> `pip install trail-lang` &middot; `import trail` &middot; CLI `trail`

The reference implementation of **Trail** - a small, total, declarative language for computing
financial indicators, scores, and screening strategies over panels of securities. Trail
expressions compile to vectorized [Polars](https://pola.rs) operations.

The language specification (grammar, reference, standard library) lives in
[**trail-language/spec**](https://github.com/trail-language/spec).

```trail
model quality on us_main at annual {
    operating_margin = income.operating_income / income.revenue
    score om_score weight 7 {
        2 if operating_margin > 0.12
        1 if operating_margin > 0.05
        else 0
    }
    export composite = weighted_score()
}
```

## Install

```bash
pip install trail-lang        # or: uv add trail-lang
```

## CLI

```bash
trail validate models/quality.trail                 # parse, kind-check, dry-run
trail run models/quality.trail --model quality      # evaluate a model
trail run models/quality.trail --model quality --config trail.yaml
trail catalog                                       # discover fields, functions, sources
trail catalog income                                # fields in a namespace
trail catalog cagr                                  # describe a function
```

The standard library is loaded implicitly; pass `--no-stdlib` to opt out.

## Library

```python
from trail.pipeline import prepare
from trail.compiler import compile_model
from trail.fixtures import load_panel  # a bundled demo panel

program = prepare("model m { export margin = income.operating_income / income.revenue }")
model = next(d for d in program.decls if type(d).__name__ == "ModelDecl")
result = compile_model(model, {}).run(load_panel())
```

## Data sources

Trail evaluates over a `(entity × time)` panel supplied by a **data source**. With no config a
bundled in-memory `fixture` source is used (a demo panel), so `trail run` works out of the box.
Real data comes from **provider packages** — install one and it registers a driver you name in
`trail.yaml`:

```bash
pip install trail-fmp      # Financial Modeling Prep   → driver: fmp
pip install trail-edgar    # SEC EDGAR (10-K / 10-Q)    → driver: edgar
pip install trail-gmd      # Global Macro Database      → driver: gmd
```

Each provider registers under the `trail.sources` entry-point group, so `pip install trail-<name>`
makes `driver: <name>` usable by name — no import wiring. Run `trail catalog` to list the sources
and fields available in your environment.

`trail.yaml` (auto-loaded from the working directory, or pass `--config`) wires sources to models:

```yaml
sources:                          # name → driver + provider-specific options (see each provider's README)
  edgar:
    driver: edgar
    options: { identity: "Jane Quant jane@example.com", tickers: [AAPL, MSFT, NVDA] }
  fmp:
    driver: fmp
    options: { api_key: ${FMP_API_KEY}, tickers: [AAPL, MSFT, NVDA] }
  gmd:
    driver: gmd                   # country-keyed macro; bridges onto stocks via meta.country

precedence:                       # which source serves each field, keyed by namespace
  income:  [edgar, fmp]           # income.* → EDGAR first, FMP fills the gaps
  balance: [edgar, fmp]
  cash:    [edgar, fmp]
  default: [fmp, gmd]             # every other namespace

panel:
  periods: [2015, 2024]           # inclusive year bounds (fetch hint + filter)
  pit: auto                       # "auto" (default, lookahead-safe) | "naive" (period-end placement)
  strict: false                   # true → a non-conforming source panel is a hard error
```

- **Precedence is per-namespace** — a field's first dotted segment (`income`, `price`, `fmp`, `gmd`, …),
  falling back to `default`. A chain with more than one source **coalesces per `(entity, period)` cell**
  (first non-null down the chain), so you can layer a precise primary over a broad fallback.
- **Pin one source** inline with `income.revenue @ edgar` (skips coalescing).
- **Point-in-time is on by default** (`pit: auto`): each value is placed by when it became knowable
  (its filing date), so a backtest never sees a statement before it was filed. Use `pit: naive`
  (globally, or `options.pit: naive` per source) for pure period-end fundamental analysis.

Provider-specific options live in each provider's README:
[trail-fmp](https://github.com/trail-language/trail-fmp) ·
[trail-edgar](https://github.com/trail-language/trail-edgar) ·
[trail-gmd](https://github.com/trail-language/trail-gmd).

## Writing a data source

A provider is a class implementing the `DataSource` contract, registered under `trail.sources`.
Three methods are mandatory:

```python
import polars as pl
from trail.source import DataSource, LoadRequest, Capabilities

class MySource(DataSource):
    name = "mine"

    def load(self, request: LoadRequest) -> pl.DataFrame:
        # Return a panel: columns `entity` (Utf8), `time` (Datetime, period-end), and one
        # column per requested canonical field. request.fields/frequency/periods/entities
        # scope the fetch; a superset is fine (the runtime re-filters).
        ...

    def available_fields(self, frequency: str | None = None) -> set[str]:
        return {"income.revenue", "income.net_income"}      # what you serve

    def capabilities(self) -> Capabilities:
        return Capabilities(frequency="annual", frequencies=("annual", "quarterly"))
```

Optional refinements:

- **Point-in-time** — emit a reserved `__date:<name>` column (Datetime, e.g. `__date:filing_date`)
  and point a field at it via `describe_field(f) → FieldInfo(..., aligns_on="filing_date")`. The
  engine then places that field's values by their known-date. A field with no coordinate is naive
  (placed at period-end).
- **A field vocabulary** — contribute new fields (e.g. `mine.*`) with a `trail.schema` entry point
  resolving to a `{column: kind}` mapping; they then reference like any built-in field.
- **A coarse dimension** — if your `entity` axis is a coarser key (e.g. country), set
  `Capabilities(entity_dim="country", bridge_field="meta.country")` and the engine remaps it onto
  entities through that bridge meta field.

Register in your package's `pyproject.toml`:

```toml
[project.entry-points."trail.sources"]
mine = "my_pkg.source:MySource"

[project.entry-points."trail.schema"]      # optional — only if you add a mine.* vocabulary
mine = "my_pkg.schema:FIELDS"              # a {column: kind} dict
```

`trail.testing.assert_source_conforms(src, fields)` checks your adapter against the panel
contract. The full contract (`LoadRequest`, `Capabilities`, `FieldInfo`, the `__date:*`
convention, coalescing) is in the [spec](https://github.com/trail-language/spec) §5 and §11.

## Architecture

Parser (Lark LALR) → typed AST → macro expansion (`def` inlining) → kind-checked validation →
Polars compiler. The engine carries only irreducible **primitives**; the large **derived**
function library is written in Trail itself and shipped as `trail/stdlib/*.trail` (canonical
copy in the [spec](https://github.com/trail-language/spec)).

## Development

```bash
uv sync
uv run pytest -q
uv run ruff check .
```

## License

[MIT](LICENSE).
