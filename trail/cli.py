"""Trail command line: validate and run .trail files (fixture-backed in this phase)."""
from __future__ import annotations

import sys
import warnings

import click
import polars as pl
from lark.exceptions import UnexpectedInput, VisitError

from trail import ast, catalog as catalog_core
from trail.compiler import compile_model, compile_signal, universe_chain
from trail.config import Config, ConfigError, load_config
from trail.deps import extract
from trail.describe import render_describe
from trail.macro import TrailFunctionError
from trail.pipeline import prepare
from trail.sources import AlignmentWarning, PanelConformanceWarning, load_panel_for, resolve_driver
from trail.validate import validate


@click.group()
def main() -> None:
    """Trail - financial expression language."""


def _load_and_validate(path: str, with_stdlib: bool = True) -> ast.Program:
    with open(path) as fh:
        src = fh.read()
    try:
        program = prepare(src, stdlib=with_stdlib)  # prepend stdlib, parse, inline defs
    except UnexpectedInput as e:
        tok = getattr(e, "token", None)
        detail = f": unexpected {str(tok)!r}" if tok else ""
        click.echo(f"ERROR SYNTAX at line {e.line}, column {e.column}{detail}")
        sys.exit(1)
    except VisitError as e:  # a transformer-level rejection (e.g. a malformed @ qualifier arg)
        click.echo(f"ERROR SYNTAX {e.orig_exc}")
        sys.exit(1)
    except TrailFunctionError as e:
        click.echo(f"ERROR FUNC {e}")
        sys.exit(1)
    issues = validate(program)
    for i in issues:
        click.echo(f"{'ERROR' if i.severity == 'error' else 'WARN '} {i.code} {i.message}")
    if any(i.severity == "error" for i in issues):
        sys.exit(1)
    return program


@main.command("validate")
@click.argument("path", type=click.Path(exists=True))
@click.option("--no-stdlib", is_flag=True, help="Do not load the bundled standard library.")
def validate_cmd(path: str, no_stdlib: bool) -> None:
    _load_and_validate(path, with_stdlib=not no_stdlib)
    click.echo("OK")


@main.command("catalog")
@click.argument("target", required=False)
@click.option("--config", "config_path", default=None, type=click.Path(exists=True))
def catalog_cmd(target: str | None, config_path: str | None) -> None:
    """Discover fields, functions, and sources. TARGET is a namespace, field,
    function, source, or one of: fields, functions, sources. Same discovery core
    as the REPL `?` meta-command."""
    try:
        config = load_config(config_path)
    except ConfigError as e:
        click.echo(f"ERROR CONFIG {e}")
        sys.exit(1)
    if target is None:
        result = catalog_core.catalog(config)
    else:
        result = catalog_core.describe(tuple(target.split(".")), config)
    click.echo(str(result))


def _all_available_fields(config: Config) -> set[str]:
    """Every field the configured sources can serve (union of ``available_fields``). A source that
    fails to construct (e.g. a missing credential) is skipped, so discovery stays usable offline."""
    out: set[str] = set()
    for spec in config.sources.values():
        try:
            src = resolve_driver(spec.driver)(spec.options)
        except Exception:  # instantiation/credential failure: skip, don't abort discovery
            continue
        try:
            out |= set(src.available_fields())
        finally:
            try:
                src.close()
            except Exception:
                pass
    return out


def _scoped_panel(decl: ast.ModelDecl | ast.SignalDecl,
                  universes: dict[str, ast.UniverseDecl], config: Config) -> pl.DataFrame:
    """Load exactly the panel a model/signal needs: bind its universe (compile_model's rule -
    explicit `on` wins, a sole declared universe auto-binds, else the whole panel), then load the
    decl PLUS its bound universe root chain so only the referenced fields are fetched (a stray
    field in another decl - or in a universe this one never binds - must not abort the load).
    Ancestor `where` fields load too since universes compose. Emits conformance/alignment warnings
    as they surface. Shared by `run`, `describe`, and `signal`."""
    if decl.universe is not None:
        bound = universes.get(decl.universe)
    elif len(universes) == 1:
        bound = next(iter(universes.values()))
    else:
        bound = None
    scoped = ast.Program(tuple(universe_chain(bound, universes)) + (decl,))
    dep = extract(scoped)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PanelConformanceWarning)
        warnings.simplefilter("always", AlignmentWarning)
        panel = load_panel_for(config, set(dep.fields), target_freq=decl.frequency,
                               align_overrides=dep.align_overrides)
    for w in caught:
        if issubclass(w.category, (PanelConformanceWarning, AlignmentWarning)):
            click.echo(f"WARN  {w.message}")
    return panel


def _emit_result(result: pl.DataFrame, out_path: str | None) -> None:
    """Write the result frame to --out (csv/json/ndjson/parquet by extension) or print it."""
    if out_path:
        if out_path.endswith(".csv"):
            result.write_csv(out_path)
        elif out_path.endswith((".json", ".ndjson")):
            result.write_ndjson(out_path)
        else:
            result.write_parquet(out_path)
        click.echo(f"wrote {out_path}")
    else:
        click.echo(str(result))


@main.command("describe")
@click.argument("path", type=click.Path(exists=True))
@click.option("--config", "config_path", default=None, type=click.Path(exists=True))
@click.option("--model", "model_name", default=None,
              help="Scope the panel to this model + its bound universe (else the whole config).")
@click.option("--field", "field", default=None, help="Report just this field's distinct values.")
@click.option("--no-stdlib", is_flag=True, help="Do not load the bundled standard library.")
def describe_cmd(path: str, config_path: str | None, model_name: str | None,
                 field: str | None, no_stdlib: bool) -> None:
    """Discover what a provider actually publishes: the available fields grouped by namespace, and
    - for low-cardinality categorical fields - their distinct observed values with row counts. This
    answers "which sector/exchange strings does this source emit?" so you write a correct,
    provider-specific expression. Values are surfaced verbatim (no normalization)."""
    program = _load_and_validate(path, with_stdlib=not no_stdlib)
    models = {d.name: d for d in program.decls if isinstance(d, ast.ModelDecl)}
    universes = {d.name: d for d in program.decls if isinstance(d, ast.UniverseDecl)}
    if model_name is not None and model_name not in models:
        click.echo(f"ERROR no model named '{model_name}'")
        sys.exit(1)
    try:
        config = load_config(config_path)
        if model_name is not None:
            # scope the panel to the model + its bound universe root chain (shared with `run`)
            panel = _scoped_panel(models[model_name], universes, config)
        else:  # whole config: describe every field the sources can serve
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", PanelConformanceWarning)
                warnings.simplefilter("always", AlignmentWarning)
                panel = load_panel_for(config, _all_available_fields(config))
            for w in caught:
                if issubclass(w.category, (PanelConformanceWarning, AlignmentWarning)):
                    click.echo(f"WARN  {w.message}")
    except ConfigError as e:
        click.echo(f"ERROR CONFIG {e}")
        sys.exit(1)
    if field is not None and field not in panel.columns:
        click.echo(f"ERROR no field '{field}' in the loaded panel")
        sys.exit(1)
    click.echo(render_describe(panel, field=field))


@main.command("run")
@click.argument("path", type=click.Path(exists=True))
@click.option("--model", "model_name", required=True)
@click.option("--config", "config_path", default=None, type=click.Path(exists=True))
@click.option("--no-stdlib", is_flag=True, help="Do not load the bundled standard library.")
@click.option("--out", "out_path", default=None, type=click.Path())
def run_cmd(path: str, model_name: str, config_path: str | None, no_stdlib: bool, out_path: str | None) -> None:
    program = _load_and_validate(path, with_stdlib=not no_stdlib)
    models = {d.name: d for d in program.decls if isinstance(d, ast.ModelDecl)}
    if model_name not in models:
        strategies = {d.name for d in program.decls
                      if isinstance(d, ast.OpaqueDecl) and d.kind == "strategy"}
        if model_name in strategies:
            click.echo(f"ERROR E-PHASE-DEFERRED '{model_name}' is a strategy; "
                       "strategy/backtest execution lands in a later phase - run a model")
        else:
            click.echo(f"ERROR no model named '{model_name}'")
        sys.exit(1)
    universes = {d.name: d for d in program.decls if isinstance(d, ast.UniverseDecl)}
    try:
        config = load_config(config_path)
        panel = _scoped_panel(models[model_name], universes, config)
    except ConfigError as e:
        click.echo(f"ERROR CONFIG {e}")
        sys.exit(1)
    result = compile_model(models[model_name], universes).run(panel)
    _emit_result(result, out_path)


@main.command("signal")
@click.argument("path", type=click.Path(exists=True))
@click.option("--name", "signal_name", required=True)
@click.option("--config", "config_path", default=None, type=click.Path(exists=True))
@click.option("--no-stdlib", is_flag=True, help="Do not load the bundled standard library.")
@click.option("--out", "out_path", default=None, type=click.Path())
def signal_cmd(path: str, signal_name: str, config_path: str | None, no_stdlib: bool,
               out_path: str | None) -> None:
    """Run a `signal` declaration: a one-export, no-score model. Prints (or writes via --out)
    the resulting [entity, time, NAME] series."""
    program = _load_and_validate(path, with_stdlib=not no_stdlib)
    signals = {d.name: d for d in program.decls if isinstance(d, ast.SignalDecl)}
    if signal_name not in signals:
        click.echo(f"ERROR no signal named '{signal_name}'")
        sys.exit(1)
    universes = {d.name: d for d in program.decls if isinstance(d, ast.UniverseDecl)}
    try:
        config = load_config(config_path)
        panel = _scoped_panel(signals[signal_name], universes, config)
    except ConfigError as e:
        click.echo(f"ERROR CONFIG {e}")
        sys.exit(1)
    result = compile_signal(signals[signal_name], universes).run(panel)
    _emit_result(result, out_path)
