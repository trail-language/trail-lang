"""Interactive REPL session for Trail.

A session evaluates a line at a time, keeping definitions in scope so that
names from previous lines are available in later expressions.

Usage:   trail repl [--config trail.yaml] [--model NAME]
Input:   edgar.revenue                                (bare expression)
         margin = edgar.net_income / edgar.revenue    (assignment)
         universe u = stocks where edgar.revenue > 0  (universe declaration)
         model m on u { export r = edgar.revenue }    (model declaration)
         track model t { export r = edgar.revenue }   (tracked view, needs --config)
         def dbl(x) = x * 2                           (user function)
         import "defs.trail"                          (merge a file's defs + universes)
         ?                                            (full catalog)
         ? edgar.revenue                              (describe a field)

With `--config` the session runs on the configured sources (trail.yaml), exactly like
`trail run`: model/signal lines bind the session's universes and load a scoped panel;
`track`ed decls persist through the view store. Without `--config` the session runs on
the deterministic bundled fixture (back-compat).
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field, replace

import polars as pl

from trail import ast, macro
from trail.align import AlignmentWarning
from trail.catalog import catalog as _catalog, describe as _describe
from trail.compiler import compile_expr as _compile_expr, compile_model, compile_signal
from trail.config import Config, DEFAULT_CONFIG
from trail.library import stdlib_source
from trail.macro import TrailFunctionError, collect_functions
from trail.parser import parse_program, parse_repl_line
from trail.pipeline import TrailImportError, _import_defs
from trail.source import LoadRequest
from trail.sources import (
    FixtureSource, PanelConformanceWarning, all_available_fields, load_panel_for,
)
from trail.validate import validate

_STDLIB_FUNCS: dict[str, ast.FuncDef] | None = None


def _stdlib_funcs() -> dict[str, ast.FuncDef]:
    """The standard library's defs, parsed once. They are inlined exactly as `prepare`
    does for a model file, so a REPL line has the full derived-function surface (yoy,
    cagr, gross_margin, ...) - not just the primitive operator registry."""
    global _STDLIB_FUNCS
    if _STDLIB_FUNCS is None:
        _STDLIB_FUNCS = collect_functions(parse_program(stdlib_source()))
    return _STDLIB_FUNCS


class _ReplError(Exception):
    """A defined error (macro expansion failed with a pipeline diagnostic) surfaced
    as a Result.err - never a raw traceback."""


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

    Keeps the data panel (the bundled fixture by default; the configured sources when
    built with `config`) and a mapping of user-defined names → AST (value assignments
    as ``Expr``, user functions as ``FuncDef``) so that later lines can reference
    names defined earlier. Universe declarations accumulate in ``universes`` and bind
    into later model/signal lines exactly like a model file's universes do.
    """
    panel: pl.DataFrame = field(default=None)  # type: ignore[assignment]
    definitions: dict[str, object] = field(default_factory=dict)
    config: Config | None = None
    config_path: str | None = None
    universes: dict[str, ast.UniverseDecl] = field(default_factory=dict)
    model_scope: str | None = None
    tracked: dict[str, ast.ModelDecl | ast.SignalDecl] = field(default_factory=dict)
    imported: set[str] = field(default_factory=set)

    # ------------------------------------------------------------------ session
    def process_input(self, text: str) -> Result:
        """Process a single line of REPL input.

        Dispatch order:
        1. Assignment  (``name = expr``)         – compile + run.
        2. Meta-command (``?``, ``? field``, …) – catalogue.
        3. Declaration (model / signal / def / universe / import / strategy / …)
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

        # ── 3. Declaration (model / signal / def / universe / import / …)
        decl_result = self._run_declaration(parsed)
        if decl_result is not None:
            return decl_result

        # ── 4. Bare expression ─────────────────────────────────────────
        return self._try_expression(parsed)

    # ---------------------------------------------------------------- helpers
    def _ensure_panel(self) -> pl.DataFrame:
        """Lazy-load the session panel: every field the configured sources can serve
        when built with `config`, else the deterministic fixture panel (6 entities
        x 8 years). Load warnings surface with the CLI's `WARN` prefix."""
        if self.panel is None:
            if self.config is None:
                self.panel = FixtureSource().load(LoadRequest(fields=frozenset()))
            else:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", PanelConformanceWarning)
                    warnings.simplefilter("always", AlignmentWarning)
                    self.panel = load_panel_for(self.config, all_available_fields(self.config))
                for w in caught:
                    if issubclass(w.category, (PanelConformanceWarning, AlignmentWarning)):
                        print(f"WARN  {w.message}")
        return self.panel

    def _funcs(self) -> dict[str, ast.FuncDef]:
        """Every function name in scope: the standard library plus the session's user
        defs. Collisions with stdlib names are rejected at def/import time, so no
        shadowing can occur here."""
        funcs = dict(_stdlib_funcs())
        for name, d in self.definitions.items():
            if isinstance(d, ast.FuncDef):
                funcs[name] = d
        return funcs

    def _expand(self, expr: ast.Expr) -> ast.Expr:
        """Inline stdlib + user function calls (the pipeline's macro pass, session-
        scoped). Arity and recursion violations raise with the pipeline's E-FUNC-*
        diagnostics."""
        try:
            return macro.expand(expr, self._funcs())
        except TrailFunctionError as e:
            raise _ReplError(str(e)) from e

    def _expand_decl(
        self, decl: ast.ModelDecl | ast.SignalDecl
    ) -> ast.ModelDecl | ast.SignalDecl:
        """Macro-expand a declaration's body (the pipeline inlines before validate and
        compile, so all three stages see the same inlined form). Score values are
        numeric literals per the grammar - only cases' conditions expand."""
        if isinstance(decl, ast.SignalDecl):
            return replace(decl, expr=self._expand(decl.expr))
        stmts = []
        for st in decl.statements:
            if isinstance(st, ast.Assignment):
                stmts.append(st if st.expr is None else replace(st, expr=self._expand(st.expr)))
            else:
                cases = tuple(ast.ScoreCase(c.value, self._expand(c.cond)) for c in st.cases)
                stmts.append(replace(st, cases=cases))
        return replace(decl, statements=tuple(stmts))

    def _expanded_universes(self) -> dict[str, ast.UniverseDecl]:
        """Session universes with `where` clauses macro-expanded: a def call inside a
        where must inline before dependency extraction (scoped load) and compilation."""
        out: dict[str, ast.UniverseDecl] = {}
        for name, u in self.universes.items():
            out[name] = u if u.where is None else replace(u, where=self._expand(u.where))
        return out

    def _first_error(self, expr: ast.Expr) -> str | None:
        """Run the shared validator (the `trail validate` pass) over an already-expanded
        expression with the session's names in scope; return the first error message
        ("CODE message") or None. The line sits in a synthetic model whose preceding
        statements are dummies binding the session's assignment names, so NameRef
        scoping matches the interactive history."""
        dummies = tuple(
            ast.Assignment(n, ast.Literal(0.0))
            for n, v in self.definitions.items() if isinstance(v, ast.Expr)
        )
        model = ast.ModelDecl(
            name="__repl__", universe=None, frequency=None, desc=None, on_missing="skip",
            statements=dummies + (ast.Assignment("__repl__", expr),),
        )
        errs = [i for i in validate(ast.Program((model,))) if i.severity == "error"]
        return f"{errs[0].code} {errs[0].message}" if errs else None

    def _run_declaration(self, parsed: object) -> Result | None:
        """Try to run a parsed declaration.

        Returns a ``Result`` for declarations the session executes, stores, or rejects,
        and ``None`` when there is nothing to show. No declaration is silently dropped:
        each kind has a defined outcome.
        """
        # ``def`` — store as a FuncDef; macro.expand resolves calls (arity/recursion
        # checked at the call site with the pipeline's E-FUNC-* diagnostics)
        if isinstance(parsed, ast.FuncDef):
            if parsed.name in _stdlib_funcs():
                return Result.err(
                    f"E-FUNC-DUP duplicate function definition '{parsed.name}' "
                    "(it shadows a standard-library function)")
            if parsed.name in self.definitions:
                return Result.err(f"E-FUNC-DUP duplicate function definition '{parsed.name}'")
            self.definitions[parsed.name] = parsed
            return Result.ok(None)

        # ``universe`` — store so later model/signal lines can bind it
        if isinstance(parsed, ast.UniverseDecl):
            if parsed.name in self.universes or parsed.name in self.definitions:
                return Result.err(f"E-NAME-REBOUND duplicate top-level declaration '{parsed.name}'")
            self.universes[parsed.name] = parsed
            return Result.ok(None)

        if isinstance(parsed, ast.ImportDecl):
            return self._run_import(parsed)

        if isinstance(parsed, (ast.ModelDecl, ast.SignalDecl)):
            return self._run_model_or_signal(parsed)

        # strategy / backtest / learn — parse-only in 1.0 (mirrors `trail run`)
        if isinstance(parsed, ast.OpaqueDecl):
            return Result.err(
                f"E-PHASE-DEFERRED '{parsed.kind}' execution lands in a later phase "
                "(parse-only in 1.0); the declaration parsed OK")

        return None

    def _run_model_or_signal(self, decl: ast.ModelDecl | ast.SignalDecl) -> Result:
        """Run a model/signal line with the CLI `run` semantics: macro-expand (defs),
        validate, bind the session's universes, load a scoped panel when config-backed,
        and - for `track`ed decls - serve through the view store."""
        if self.model_scope is not None and decl.name != self.model_scope:
            return Result.err(
                f"session is scoped to model '{self.model_scope}' (started with --model); "
                f"'{decl.name}' will not run - restart without --model to run it")
        try:
            unis = self._expanded_universes()
            decl = self._expand_decl(decl)
        except _ReplError as e:
            return Result.err(str(e))
        errs = [i for i in validate(ast.Program(tuple(unis.values()) + (decl,)))
                if i.severity == "error"]
        if errs:
            return Result.err(f"{errs[0].code} {errs[0].message}")

        if decl.track:
            # View persistence needs a trail.yaml path (the store lives beside it) - the
            # fixture session has no file context, so refuse clearly instead of silently
            # running without persisting.
            if self.config is None or self.config_path is None:
                return Result.err(
                    "'track' needs --config: views persist beside trail.yaml "
                    "(restart: trail repl --config trail.yaml)")
            from trail.providers import store_for_config
            from trail.views import ViewManager
            store = store_for_config(self.config, self.config_path)
            mgr = ViewManager(store, self.config, self.config_path)
            try:
                result = mgr.serve(decl, unis,
                                   decls={d.name: d for d in self.tracked.values()})
            except Exception as e:
                return Result.err(str(e))
            self.tracked[decl.name] = decl
            return Result.ok(result)

        if self.config is None:
            panel = self._ensure_panel()  # fixture back-compat: whole panel, no scoping
        else:
            from trail.cli import _scoped_panel  # the CLI's shared scoped-load (binding + deps)
            panel = _scoped_panel(decl, unis, self.config)
        plan = compile_model(decl, unis) if isinstance(decl, ast.ModelDecl) \
            else compile_signal(decl, unis)
        try:
            result = plan.run(panel)
        except Exception as e:
            return Result.err(str(e))
        return Result.ok(result)

    def _run_import(self, decl: ast.ImportDecl) -> Result:
        """``import "path"`` - merge the file's reusable top-level `def`s and
        `universe`s into the session (source-level inclusion, as `prepare` does for a
        model file: model/signal decls are skipped). Paths resolve relative to the CWD;
        transitive imports and cycles go through the pipeline's resolver. All names are
        checked before anything is applied (an E-IMPORT-DUP leaves the session intact).
        """
        resolved = os.path.realpath(os.path.join(os.getcwd(), decl.path))
        if resolved in self.imported:
            return Result.ok(None)  # same file already included (dedup, as in prepare)
        if not os.path.isfile(resolved):
            return Result.err(
                f"E-IMPORT-NOT-FOUND cannot resolve import '{decl.path}' (looked for '{resolved}')")
        try:
            with open(resolved) as fh:
                program = parse_program(fh.read())
            imported = _import_defs(program, os.path.dirname(resolved),
                                    seen=set(), stack=(resolved,))
        except TrailImportError as e:
            return Result.err(str(e))
        except Exception as e:
            return Result.err(f"PARSE ERROR in import '{decl.path}': {e}")
        own = [d for d in program.decls if isinstance(d, (ast.FuncDef, ast.UniverseDecl))]
        decls = [*imported, *own]
        names_seen: set[str] = set()
        for d in decls:
            kind = "function" if isinstance(d, ast.FuncDef) else "universe"
            if d.name in names_seen:
                return Result.err(
                    f"E-IMPORT-DUP imported {kind} '{d.name}' is defined by more than one imported file")
            if d.name in _stdlib_funcs():
                return Result.err(
                    f"E-IMPORT-DUP imported {kind} '{d.name}' collides with a standard-library definition")
            if d.name in self.definitions or d.name in self.universes:
                return Result.err(
                    f"E-IMPORT-DUP imported {kind} '{d.name}' collides with a definition in the session")
            names_seen.add(d.name)
        self.imported.add(resolved)
        for d in decls:
            if isinstance(d, ast.FuncDef):
                self.definitions[d.name] = d
            else:
                self.universes[d.name] = d
        return Result.ok(None)

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

        # Same pipeline as a model-file line: macro-expand, then shared validation
        try:
            expanded = self._expand(expr)
        except _ReplError as e:
            return Result.err(str(e))
        err = self._first_error(expanded)
        if err is not None:
            return Result.err(err)

        panel = self._ensure_panel()
        try:
            col_name = f"__repl_{name}"
            lf = panel.lazy()
            lf = lf.with_columns(
                _compile_expr(expanded, set(self.definitions.keys())).alias(col_name)
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

        # Same pipeline as a model-file line: macro-expand, then shared validation
        try:
            expanded = self._expand(expr)
        except _ReplError as e:
            return Result.err(str(e))
        err = self._first_error(expanded)
        if err is not None:
            return Result.err(err)

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
    def _active_config(self) -> Config:
        """The config discovery runs against: the session's when loaded, else the
        default (fixture) - discovery always reflects the active data."""
        return self.config if self.config is not None else DEFAULT_CONFIG

    def _run_catalog(self) -> pl.DataFrame:
        """Return a summary of available fields / functions / sources."""
        return _catalog(self._active_config()).frame

    def _run_describe(self, target: tuple[str, ...]) -> pl.DataFrame:
        """Describe a field, function, or source."""
        return _describe(target, self._active_config()).frame
