"""Trail command line: validate and run .trail files (fixture-backed in this phase)."""
from __future__ import annotations

import sys
import warnings

import click
from lark.exceptions import UnexpectedInput

from trail import ast, catalog as catalog_core
from trail.compiler import compile_model, universe_chain
from trail.config import ConfigError, load_config
from trail.deps import extract
from trail.macro import TrailFunctionError
from trail.pipeline import prepare
from trail.sources import AlignmentWarning, PanelConformanceWarning, load_panel_for
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
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", PanelConformanceWarning)
            warnings.simplefilter("always", AlignmentWarning)
            # scope loading to the run model + its BOUND universe (compile_model's binding
            # rule: explicit `on` wins, a sole universe auto-binds). A stray field in another
            # model - or in a universe this model never binds - must not abort this run.
            model = models[model_name]
            if model.universe is not None:
                bound = universes.get(model.universe)
            elif len(universes) == 1:
                bound = next(iter(universes.values()))
            else:
                bound = None
            # the whole root chain: ancestor `where` fields must load too (universes compose)
            scoped = ast.Program(tuple(universe_chain(bound, universes)) + (model,))
            dep = extract(scoped)
            panel = load_panel_for(config, set(dep.fields),
                                   target_freq=models[model_name].frequency,
                                   align_overrides=dep.align_overrides)
        for w in caught:
            if issubclass(w.category, (PanelConformanceWarning, AlignmentWarning)):
                click.echo(f"WARN  {w.message}")
    except ConfigError as e:
        click.echo(f"ERROR CONFIG {e}")
        sys.exit(1)
    result = compile_model(models[model_name], universes).run(panel)
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
