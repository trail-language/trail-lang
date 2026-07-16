"""Trail command line: validate and run .trail files (fixture-backed in this phase)."""
from __future__ import annotations

import sys
import warnings

import click
from lark.exceptions import UnexpectedInput

from trail import ast, catalog as catalog_core
from trail.compiler import compile_model
from trail.config import ConfigError, load_config
from trail.deps import extract
from trail.macro import TrailFunctionError
from trail.pipeline import prepare
from trail.sources import PanelConformanceWarning, load_panel_for
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
        click.echo(f"ERROR no model named '{model_name}'")
        sys.exit(1)
    universes = {d.name: d for d in program.decls if isinstance(d, ast.UniverseDecl)}
    try:
        config = load_config(config_path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", PanelConformanceWarning)
            panel = load_panel_for(config, set(extract(program).fields),
                                   target_freq=models[model_name].frequency)
        for w in caught:
            if issubclass(w.category, PanelConformanceWarning):
                click.echo(f"WARN  {w.message}")
    except ConfigError as e:
        click.echo(f"ERROR CONFIG {e}")
        sys.exit(1)
    result = compile_model(models[model_name], universes).run(panel)
    if out_path:
        result.write_parquet(out_path)
        click.echo(f"wrote {out_path}")
    else:
        click.echo(str(result))
