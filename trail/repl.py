"""Interactive REPL for Trail.

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

from dataclasses import dataclass, field

import polars as pl

from trail import ast
from trail.compiler import compile_expr as _compile_expr, compile_model, compile_signal
from trail.config import DEFAULT_CONFIG
from trail.parser import parse_repl_line, parse_program
from trail.pipeline import prepare
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
        1. Meta-catalog (``?``, ``? field``, …)  – handled by the parser/AST.
        2. Declaration (model / signal / def)   – compiled and executed.
        3. Assignment  (``name = expr``)         – compile + run.
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
        # ``def`` — store in definitions, return a no-op result (not an error)
        if isinstance(parsed, ast.FuncDef):
            # ``def avg2(x) = (x + lag(x)) / 2`` → store body
            self.definitions[parsed.name] = parsed.body
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
        # Quick reject: ``repl_line``'s ``expr`` already handles ``NAME=expr``
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
            panel = self._ensure_panel()
            self.panel = panel.join(result, on=["entity", "time"], how="left")
            return Result.ok(result)
        except Exception as e:
            return Result.err(str(e))

    def _try_expression(self, expr: object) -> Result:
        """Evaluate a bare expression AST node."""
        if not isinstance(expr, ast.Expr):
            return Result.err(f"Expected expression, got {type(expr).__name__}")

        panel = self._ensure_panel()
        try:
            col_name = "__repl_result"
            lf = panel.lazy()
            lf = lf.with_columns(
                _compile_expr(expr, set(self.definitions.keys())).alias(col_name)
            )
            result = lf.select(["entity", "time", col_name]).collect()
            return Result.ok(result)
        except Exception as e:
            return Result.err(str(e))

    # ── Catalog helpers ────────────────────────────────────────────────
    def _run_catalog(self) -> pl.DataFrame:
        """Return a summary of available fields / functions / sources."""
        from trail.catalog import catalog as _catalog

        result = _catalog(DEFAULT_CONFIG)
        return result.frame

    def _run_describe(self, target: tuple[str, ...]) -> pl.DataFrame:
        """Describe a field, function, or source."""
        from trail.catalog import describe as _describe

        result = _describe(target, DEFAULT_CONFIG)
        return result.frame
