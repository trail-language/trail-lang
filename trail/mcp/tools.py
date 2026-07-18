"""The six MCP tool functions. Pure Python (JSON-serializable dicts) over trail's public API - no
FastMCP here, so they unit-test offline. server.py registers them with the SDK."""
from __future__ import annotations

from lark.exceptions import UnexpectedInput, VisitError

from trail import ast, catalog as catalog_core
from trail.compiler import compile_model, compile_signal
from trail.describe import categorical_fields, fields_by_namespace, panel_fields, value_counts
from trail.macro import TrailFunctionError
from trail.mcp.data import DataSpecError, resolve_panel
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
              to_file: str | None = None, no_stdlib: bool = False) -> dict:
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
        panel, warns = resolve_panel(data, signal, universes)
        result = compile_signal(signal, universes).run(panel)
    except Exception as e:
        return to_error(e)
    return format_result(result, offset=offset, limit=limit, fmt=format, to_file=to_file,
                         extra={"warnings": warns} if warns else None)


def run_tool(name: str, data: dict, program: str | None = None, path: str | None = None,
             offset: int | None = None, limit: int | None = None, format: str = "compact",
             to_file: str | None = None, no_stdlib: bool = False) -> dict:
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
        decl, plan = models[name], compile_model(models[name], universes)
    elif name in signals:
        decl, plan = signals[name], compile_signal(signals[name], universes)
    else:
        return {"error": {"code": "E-NAME-UNKNOWN", "message": f"no model or signal named '{name}'"}}
    try:
        panel, warns = resolve_panel(data, decl, universes)
        result = plan.run(panel)
    except Exception as e:
        return to_error(e)
    return format_result(result, offset=offset, limit=limit, fmt=format, to_file=to_file,
                         extra={"warnings": warns} if warns else None)
