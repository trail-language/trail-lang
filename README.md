<p align="center">
  <img src="https://raw.githubusercontent.com/trail-language/spec/main/brand/icon-color-240.png" alt="Trail" width="120">
</p>

# trail-py

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
