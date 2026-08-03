"""The six MCP tool functions. Pure Python (JSON-serializable dicts) over trail's public API - no
FastMCP here, so they unit-test offline. server.py registers them with the SDK."""
from __future__ import annotations

from lark.exceptions import UnexpectedInput, VisitError

from trail import ast, catalog as catalog_core
from trail.compiler import compile_model, compile_signal
from trail.describe import categorical_fields, fields_by_namespace, panel_fields, value_counts
from trail.macro import TrailFunctionError
from trail.mcp.data import resolve_panel
from trail.mcp.errors import to_error
from trail.mcp.format import format_result
from trail.pipeline import TrailImportError, prepare
from trail.source import ENTITY_COL
from trail.validate import validate

_PARSE_ERRORS = (UnexpectedInput, VisitError, TrailImportError, TrailFunctionError)


def _frame_records(cat) -> list[dict]:
    return cat.frame.to_dicts()


def functions_tool(query: str | None = None, axis: str | None = None) -> dict:
    rows = _frame_records(catalog_core.functions())
    if axis:
        rows = [r for r in rows if r.get("axis") == axis]
    if query:
        q = query.lower()
        rows = [r for r in rows if q in r["function"].lower() or q in str(r.get("summary", "")).lower()]
    return {"functions": rows}


def schema_tool(namespace: str | None = None) -> dict:
    return {"fields": _frame_records(catalog_core.fields(namespace))}


def validate_tool(source: str, no_stdlib: bool = False, base_dir: str | None = None) -> dict:
    try:
        program = prepare(source, stdlib=not no_stdlib, path=base_dir)
    except _PARSE_ERRORS as e:
        return {"valid": False, "issues": [{"severity": "error", **to_error(e)["error"]}]}
    issues = [{"severity": i.severity, "code": i.code, "message": i.message} for i in validate(program)]
    return {"valid": not any(i["severity"] == "error" for i in issues), "issues": issues}


def _validate_or_error(program):
    errs = [i for i in validate(program) if i.severity == "error"]
    if errs:
        return {"error": {"code": errs[0].code, "message": errs[0].message}}
    return None


def _serve_view(decl, universes, config_path, entities, prog=None):
    """Materialize (if stale) and read back a tracked view's persisted frame. `prog` supplies the
    other tracked decls so a view-of-view can build its same-program dependencies (view-of-view)."""
    from trail.config import load_config
    from trail.providers import store_for_config
    from trail.views import ViewManager
    cfg = load_config(config_path)
    store = store_for_config(cfg, config_path)
    mgr = ViewManager(store, cfg, config_path)
    decls = {d.name: d for d in mgr.tracked(prog)} if prog is not None else None
    return mgr.serve(decl, universes, decls=decls, entities=entities)


def describe_tool(data: dict, field: str | None = None) -> dict:
    try:
        panel, warns = resolve_panel(data)
    except Exception as e:  # any load failure -> structured error the agent can fix
        return to_error(e)
    if field is not None:
        if field not in panel.columns:
            return {"error": {"code": "E-FIELD-UNKNOWN",
                              "message": f"'{field}' not in panel; have: {panel_fields(panel)}"}}
        rows, total = value_counts(panel, field, cap=50)
        return {"field": field, "total_distinct": total,
                "distinct": [{"value": v, "count": c} for v, c in rows]}
    cats = []
    for f in categorical_fields(panel):
        rows, total = value_counts(panel, f)
        cats.append({"field": f, "distinct": [{"value": v, "count": c} for v, c in rows],
                     "truncated": len(rows) < total})
    ents = panel.get_column(ENTITY_COL).n_unique() if ENTITY_COL in panel.columns else 0
    return {"shape": {"rows": panel.height, "entities": ents, "fields": len(panel_fields(panel))},
            "fields_by_namespace": fields_by_namespace(panel), "categorical": cats, "warnings": warns}


def eval_tool(expression: str, data: dict, where: str | None = None, at: str | None = None,
              offset: int | None = None, limit: int | None = None, format: str = "compact",
              to_file: str | None = None, no_stdlib: bool = False, streaming: bool = False,
              entities: list[str] | None = None) -> dict:
    parts = []
    on = ""
    if where:
        parts.append(f"universe __eval_u = stocks where {where}")
        on = " on __eval_u"
    at_clause = f" at {at}" if at else ""
    parts.append(f"signal value{on}{at_clause} = {expression}")
    source = "\n".join(parts)
    try:
        program = prepare(source, stdlib=not no_stdlib)
    except _PARSE_ERRORS as e:
        return to_error(e)
    if (err := _validate_or_error(program)) is not None:
        return err
    universes = {d.name: d for d in program.decls if isinstance(d, ast.UniverseDecl)}
    signal = next(d for d in program.decls if isinstance(d, ast.SignalDecl) and d.name == "value")
    try:
        panel, warns = resolve_panel(data, signal, universes, lazy=True, entities=entities)
        result = compile_signal(signal, universes).run(panel, engine="streaming" if streaming else None)
    except Exception as e:
        return to_error(e)
    return format_result(result, offset=offset, limit=limit, fmt=format, to_file=to_file,
                         extra={"warnings": warns} if warns else None)


def run_tool(name: str, data: dict, program: str | None = None, path: str | None = None,
             offset: int | None = None, limit: int | None = None, format: str = "compact",
             to_file: str | None = None, no_stdlib: bool = False, streaming: bool = False,
             entities: list[str] | None = None) -> dict:
    if (program is None) == (path is None):
        return {"error": {"code": "E-ARGS", "message": "pass exactly one of `program` or `path`"}}
    try:
        if path is not None:
            with open(path) as fh:
                src = fh.read()
            prog = prepare(src, stdlib=not no_stdlib, path=path)
        else:
            prog = prepare(program, stdlib=not no_stdlib)
    except (*_PARSE_ERRORS, OSError) as e:
        return to_error(e)
    if (err := _validate_or_error(prog)) is not None:
        return err
    models = {d.name: d for d in prog.decls if isinstance(d, ast.ModelDecl)}
    signals = {d.name: d for d in prog.decls if isinstance(d, ast.SignalDecl)}
    universes = {d.name: d for d in prog.decls if isinstance(d, ast.UniverseDecl)}
    if name in models:
        decl = models[name]
    elif name in signals:
        decl = signals[name]
    else:
        return {"error": {"code": "E-NAME-UNKNOWN", "message": f"no model or signal named '{name}'"}}
    # A `track`ed decl is served from the view store (built on first use, recomputed when stale),
    # so a re-run skips the fetch+compute entirely. A view-of-view (its expr references `views.*`) has
    # its dependency views materialized first, in dependency order; reference cycles raise E-VIEW-CYCLE.
    # Requires a {config} data spec (the store lives beside it); without one, compute normally.
    if getattr(decl, "track", False) and "config" in data:
        try:
            result = _serve_view(decl, universes, data["config"], entities, prog)
        except Exception as e:
            return to_error(e)
        return format_result(result, offset=offset, limit=limit, fmt=format, to_file=to_file)
    plan = compile_model(decl, universes) if isinstance(decl, ast.ModelDecl) \
        else compile_signal(decl, universes)
    try:
        panel, warns = resolve_panel(data, decl, universes, lazy=True, entities=entities)
        result = plan.run(panel, engine="streaming" if streaming else None)
    except Exception as e:
        return to_error(e)
    return format_result(result, offset=offset, limit=limit, fmt=format, to_file=to_file,
                         extra={"warnings": warns} if warns else None)


def fetch_tool(expressions: list[str], data: dict, where: str | None = None, at: str | None = None,
               offset: int | None = None, limit: int | None = None, format: str = "compact",
               to_file: str | None = None, no_stdlib: bool = False, streaming: bool = False,
               entities: list[str] | None = None) -> dict:
    """Project several trail EXPRESSIONS into one wide [entity, time, <cols>] frame - retrieval, not a
    single computed value. Each expression becomes a column (named by the expression when unambiguous,
    else `f0..fn` with a `columns` map). Compiles as a throwaway multi-export model so it reuses the
    same universe binding, validation, alignment, and lazy/streaming execution as `eval`/`run`."""
    if not isinstance(expressions, list) or not expressions:
        return {"error": {"code": "E-ARGS", "message": "`expressions` must be a non-empty list of trail expressions"}}
    if not all(isinstance(e, str) for e in expressions):
        return {"error": {"code": "E-ARGS", "message": "`expressions` must be a list of strings"}}
    parts: list[str] = []
    on = ""
    if where:
        parts.append(f"universe __fetch_u = stocks where {where}")
        on = " on __fetch_u"
    at_clause = f" at {at}" if at else ""
    names = [f"f{i}" for i in range(len(expressions))]
    body = "\n".join(f"  export {n} = {e}" for n, e in zip(names, expressions))
    parts.append(f"model __fetch{on}{at_clause} {{\n{body}\n}}")
    try:
        program = prepare("\n".join(parts), stdlib=not no_stdlib)
    except _PARSE_ERRORS as e:
        return to_error(e)
    if (err := _validate_or_error(program)) is not None:
        return err
    universes = {d.name: d for d in program.decls if isinstance(d, ast.UniverseDecl)}
    model = next(d for d in program.decls if isinstance(d, ast.ModelDecl) and d.name == "__fetch")
    try:
        panel, warns = resolve_panel(data, model, universes, lazy=True, entities=entities)
        result = compile_model(model, universes).run(panel, engine="streaming" if streaming else None)
    except Exception as e:
        return to_error(e)
    # Present columns by their expression text when that is unambiguous; otherwise keep f0..fn and
    # hand back an explicit column->expression map so nothing is silently collapsed.
    extra: dict = {}
    if warns:
        extra["warnings"] = warns
    if len(set(expressions)) == len(expressions) and not ({"entity", "time"} & set(expressions)):
        result = result.rename(dict(zip(names, expressions)))
    else:
        extra["columns"] = dict(zip(names, expressions))
    return format_result(result, offset=offset, limit=limit, fmt=format, to_file=to_file,
                         extra=extra or None)


def _view_store(data: dict):
    """(store, None) for a {config} data spec, or (None, error-dict) otherwise."""
    if "config" not in data:
        return None, {"error": {"code": "E-ARGS", "message": "view tools require a {config} data spec"}}
    from trail.config import ConfigError, load_config
    from trail.providers import store_for_config
    cfg_path = data["config"]
    try:
        return store_for_config(load_config(cfg_path), cfg_path), None
    except (ConfigError, OSError) as e:
        return None, to_error(e)


def drop_tool(name: str, data: dict) -> dict:
    """Delete a stored tracked view; its next run recomputes fully. `dropped` is False if absent."""
    store, err = _view_store(data)
    if err:
        return err
    try:
        return {"dropped": store.delete(name)}
    except ValueError as e:  # invalid/traversing view name
        return {"error": {"code": "E-VIEW-NAME", "message": str(e)}}


def views_tool(data: dict) -> dict:
    """List stored tracked views with a manifest summary."""
    store, err = _view_store(data)
    if err:
        return err
    out = []
    for n in store.list():
        mf = store.manifest(n)
        if mf is None:
            continue
        out.append({"name": n, "kind": mf.kind, "columns": list(mf.columns),
                    "built_at": mf.built_at, "sources": list(mf.sources)})
    return {"views": out}


def refresh_tool(name: str, data: dict, program: str | None = None,
                 path: str | None = None) -> dict:
    """Drop then eagerly rebuild a tracked view. Pass the program/path that declares it."""
    store, err = _view_store(data)
    if err:
        return err
    try:
        store.delete(name)  # force a full rebuild on the run below
    except ValueError as e:  # invalid/traversing view name
        return {"error": {"code": "E-VIEW-NAME", "message": str(e)}}
    r = run_tool(name, data, program=program, path=path, format="compact")
    if "error" in r:
        return r
    return {"refreshed": name, "shape": [r.get("total_rows"), len(r.get("columns", []))]}
