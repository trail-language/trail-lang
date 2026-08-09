"""Interactive REPL session for Trail.

A session evaluates a line at a time, keeping definitions in scope so that
names from previous lines are available in later expressions.

Usage:   trail repl
Input:   income.revenue / balance.total_assets   (bare expression)
         margin = income.operating_income / income.revenue  (assignment)
         model m { ... }   (full model declaration)
         ?                 (full catalog)
         ? income.revenue  (describe a field)
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from trail import ast
from trail.catalog import catalog as _catalog, describe as _describe
from trail.compiler import compile_expr as _compile_expr, compile_model, compile_signal
from trail.config import DEFAULT_CONFIG
from trail.parser import parse_repl_line
from trail.source import LoadRequest
from trail.sources import FixtureSource


@dataclass(frozen=True)
class Result:
    """An REPL result — either a computed value or an error."""
    is_result: bool = False
    is_error: bool = False
    value: pl.DataFrame | None = None
    message: str = ""

    @classmethod
    def ok(cls, value: pl.DataFrame | None = None) -> "Result":
        return cls(is_result=True, value=value)

    @classmethod
    def err(cls, message: str) -> "Result":
        return cls(is_error=True, message=message)


@dataclass
class ReplSession:
    """Interactive REPL session state.

    Keeps a fixture panel and a mapping of user-defined names → AST expressions
    so that later lines can reference names defined earlier.
    """
    panel: pl.DataFrame = field(default=None)  # type: ignore[assignment]
    definitions: dict[str, ast.Expr] = field(default_factory=dict)

    # ------------------------------------------------------------------ session
    def process_input(self, text: str) -> Result:
        """Process a single line of REPL input.

        Dispatch order:
        1. Assignment  (``name = expr``)         – compile + run.
        2. Meta-command (``?``, ``? field``, …) – catalogue.
        3. Declaration (model / signal / def)   – compile and execute.
        4. Bare expression  (``field / field``) – compile + run.
        """
        text = text.strip()
        if not text:
            return Result.ok(None)

        # ── 1. Assignment (NAME = expr) ────────────────────────────────
        # Must come before parsing: ``=`` is invalid at the expr level.
        assign_result = self._try_assignment(text)
        if assign_result is not None:
            return assign_result

        # ── 2. Meta-command (already wired in grammar / parser) ─────────
        try:
            parsed = parse_repl_line(text)
        except Exception as e:
            return Result.err(f"PARSE ERROR: {e}")

        if isinstance(parsed, ast.MetaCatalog):
            return Result.ok(self._run_catalog())
        if isinstance(parsed, ast.MetaDescribe):
            return Result.ok(self._run_describe(parsed.target))

        # ── 3. Declaration (model / signal / def) ──────────────────────
        decl_result = self._run_declaration(parsed)
        if decl_result is not None:
            return decl_result

        # ── 4. Bare expression ─────────────────────────────────────────
        return self._try_expression(parsed)

    # ---------------------------------------------------------------- helpers
    def _ensure_panel(self) -> pl.DataFrame:
        """Lazy-load the deterministic fixture panel (6 entities x 8 years)."""
        if self.panel is None:
            src = FixtureSource()
            self.panel = src.load(LoadRequest(fields=frozenset()))
        return self.panel

    def _run_declaration(self, parsed: object) -> Result | None:
        """Try to run a parsed declaration (model / signal / def).

        Returns ``Result`` if the node is a model/signal, ``None`` if it's
        something we don't execute (e.g. a ``def`` — handled implicitly).
        """
        # ``def`` — store body with param names for substitution
        if isinstance(parsed, ast.FuncDef):
            # Store as (body, [param_names]) tuple
            self.definitions[parsed.name] = (parsed.body, list(parsed.params))
            return Result.ok(None)

        if isinstance(parsed, ast.ModelDecl):
            panel = self._ensure_panel()
            result = compile_model(parsed, {}).run(panel)
            return Result.ok(result)

        if isinstance(parsed, ast.SignalDecl):
            panel = self._ensure_panel()
            result = compile_signal(parsed, {}).run(panel)
            return Result.ok(result)

        # universe_decl, import_decl, etc. → silently accepted
        return None

    def _try_assignment(self, text: str) -> Result | None:
        """Try to parse as assignment: ``name = expr``.

        Returns ``None`` if the text doesn't look like an assignment so the
        caller can try a bare expression fallback.
        """
        # ``repl_line``'s ``expr`` already handles ``NAME=expr``
        # when the NAME is part of a ``ref`` (e.g. ``foo.bar = 1`` would be
        # parsed as a field ref ``foo.bar`` which won't match a simple NAME).
        # But ``margin = revenue`` is NOT a valid ``expr`` (it has an ``=``
        # outside parentheses), so we need this explicit path.
        if '=' not in text:
            return None
        eq_idx = text.index('=')
        name = text[:eq_idx].strip()
        if not name:
            return None
        # Bare identifier check (no dots, no special chars)
        if not all(c.isalnum() or c in ('_', '$') for c in name):
            return None

        expr_text = text[eq_idx + 1:].strip()
        if not expr_text:
            return None

        # Parse the RHS as an expression
        try:
            expr = parse_repl_line(expr_text)
            if not isinstance(expr, ast.Expr):
                return Result.err(f"RHS is not an expression: {type(expr).__name__}")
        except Exception as e:
            return Result.err(f"PARSE ERROR on RHS: {e}")

        panel = self._ensure_panel()
        try:
            col_name = f"__repl_{name}"
            lf = panel.lazy()
            lf = lf.with_columns(
                _compile_expr(expr, set(self.definitions.keys())).alias(col_name)
            )
            result = lf.select(["entity", "time", col_name])
            result = result.rename({col_name: name}).collect()
            self.definitions[name] = expr
            # Persist the column so later lines can reference it
            self.panel = panel.join(result, on=["entity", "time"], how="left")
            return Result.ok(result)
        except Exception as e:
            return Result.err(str(e))

    def _try_expression(self, expr: object) -> Result:
        """Evaluate a bare expression AST node."""
        if not isinstance(expr, ast.Expr):
            return Result.err(f"Expected expression, got {type(expr).__name__}")

        # Expand user-defined functions inline
        expanded = _expand_user_funcs(expr, self.definitions)

        panel = self._ensure_panel()
        try:
            col_name = "__repl_result"
            lf = panel.lazy()
            lf = lf.with_columns(
                _compile_expr(expanded, set(self.definitions.keys())).alias(col_name)
            )
            result = lf.select(["entity", "time", col_name]).collect()
            return Result.ok(result)
        except Exception as e:
            return Result.err(str(e))

    # ── Catalog helpers ────────────────────────────────────────────────
    def _run_catalog(self) -> pl.DataFrame:
        """Return a summary of available fields / functions / sources."""
        result = _catalog(DEFAULT_CONFIG)
        return result.frame

    def _run_describe(self, target: tuple[str, ...]) -> pl.DataFrame:
        """Describe a field, function, or source."""
        result = _describe(target, DEFAULT_CONFIG)
        return result.frame


# ── User-defined function expansion ────────────────────────────────────

def _substitute(name: str, arg: ast.Expr, expr: ast.Expr) -> ast.Expr:
    """Replace all occurrences of a NameRef(name) in expr with arg."""
    def _replace(e: ast.Expr) -> ast.Expr:
        if isinstance(e, ast.NameRef) and e.name == name:
            return deepcopy(arg)
        if isinstance(e, ast.Call):
            new_args = tuple(_replace(a) for a in e.args)
            new_kwargs = tuple((k, _replace(v)) for k, v in e.kwargs)
            return ast.Call(e.name, new_args, new_kwargs, e.by)
        if isinstance(e, ast.BinOp):
            return ast.BinOp(e.op, _replace(e.left), _replace(e.right))
        if isinstance(e, ast.Compare):
            return ast.Compare(e.op, _replace(e.left), _replace(e.right))
        if isinstance(e, ast.BoolOp):
            return ast.BoolOp(e.op, _replace(e.left), _replace(e.right))
        if isinstance(e, ast.Not):
            return ast.Not(_replace(e.operand))
        if isinstance(e, ast.Neg):
            return ast.Neg(_replace(e.operand))
        if isinstance(e, ast.Coalesce):
            return ast.Coalesce(_replace(e.left), _replace(e.right))
        if isinstance(e, ast.Ternary):
            return ast.Ternary(_replace(e.value), _replace(e.cond), _replace(e.orelse))
        if isinstance(e, ast.In):
            return ast.In(_replace(e.item), tuple(deepcopy(o) for o in e.options))
        return deepcopy(e)
    return _replace(expr)


def _expand_user_funcs(expr: ast.Expr, definitions: dict[str, Any]) -> ast.Expr:
    """Expand user-defined function calls inline.

    Walks the AST, finds ``ast.Call`` nodes whose name is in ``definitions``,
    and substitutes argument values into the stored body expression.

    ``definitions`` stores:
    - ``Expr`` from assignments (e.g. ``margin = revenue / balance``) — these
      are *values*, not functions, so they are never expanded as calls.
    - ``(body, params)`` tuples from ``def`` declarations — these are expanded.
    """
    if not definitions:
        return expr

    def _expand(e: ast.Expr) -> ast.Expr:
        if isinstance(e, ast.Call) and e.name in definitions:
            entry = definitions[e.name]
            # Only expand if it's a def'd function (tuple), not a value (Expr)
            if not isinstance(entry, tuple) or len(entry) != 2:
                return deepcopy(e)
            body, params = entry
            args = list(e.args)
            # Map each arg to its corresponding param name and substitute
            result: ast.Expr = deepcopy(body)
            for i, arg in enumerate(args):
                if i < len(params):
                    result = _substitute(params[i], arg, result)
                else:
                    # Extra args beyond params — skip (not used in body)
                    pass
            return result
        # Recurse into children
        if isinstance(e, ast.Call):
            new_args = tuple(_expand(a) for a in e.args)
            new_kwargs = tuple((k, _expand(v)) for k, v in e.kwargs)
            return ast.Call(e.name, new_args, new_kwargs, e.by)
        if isinstance(e, ast.BinOp):
            return ast.BinOp(e.op, _expand(e.left), _expand(e.right))
        if isinstance(e, ast.Compare):
            return ast.Compare(e.op, _expand(e.left), _expand(e.right))
        if isinstance(e, ast.BoolOp):
            return ast.BoolOp(e.op, _expand(e.left), _expand(e.right))
        if isinstance(e, ast.Not):
            return ast.Not(_expand(e.operand))
        if isinstance(e, ast.Neg):
            return ast.Neg(_expand(e.operand))
        if isinstance(e, ast.Coalesce):
            return ast.Coalesce(_expand(e.left), _expand(e.right))
        if isinstance(e, ast.Ternary):
            return ast.Ternary(_expand(e.value), _expand(e.cond), _expand(e.orelse))
        if isinstance(e, ast.In):
            return ast.In(_expand(e.item), tuple(_expand(o) for o in e.options))
        return deepcopy(e)

    return _expand(expr)
