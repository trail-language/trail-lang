"""Front-door pipeline: source text -> a compile-ready Program.

Resolves `import`s (source-level inclusion), prepends the standard library (so derived
functions are implicitly available), parses, and inlines all function definitions. Every
entry point (CLI, tests, future REPL) should go through `prepare` so the primitive/derived
split and the import layer are invisible to callers.
"""
from __future__ import annotations

import os

from trail import ast
from trail.library import stdlib_source
from trail.macro import expand_program
from trail.parser import parse_program


class TrailImportError(Exception):
    """An `import` could not be resolved: missing file, cycle, or a name collision."""


def _import_defs(program: ast.Program, base_dir: str, *,
                 seen: set[str], stack: tuple[str, ...]) -> list:
    """Recursively gather the reusable top-level decls (`def`s and `universe`s) contributed by
    `program`'s imports. Paths resolve relative to `base_dir` (the importing file's directory).
    `seen` dedups files loaded via more than one path; `stack` is the current import chain for
    cycle detection. A file's `model`/`signal`/`strategy`/`backtest`/`learn` decls are execution
    units, not library decls, so they are skipped on import (a file can be both runnable and
    importable)."""
    out: list = []
    for decl in program.decls:
        if not isinstance(decl, ast.ImportDecl):
            continue
        resolved = os.path.realpath(os.path.join(base_dir, decl.path))
        if resolved in stack:
            cycle = " -> ".join([*stack, resolved])
            raise TrailImportError(f"E-IMPORT-CYCLE import cycle: {cycle}")
        if resolved in seen:
            continue  # same file already included via another import path (dedup)
        if not os.path.isfile(resolved):
            raise TrailImportError(
                f"E-IMPORT-NOT-FOUND cannot resolve import '{decl.path}' (looked for '{resolved}')")
        seen.add(resolved)
        with open(resolved) as fh:
            imported = parse_program(fh.read())
        child_dir = os.path.dirname(resolved)
        # transitive imports first (an imported file may itself import), then this file's own defs
        out.extend(_import_defs(imported, child_dir, seen=seen, stack=(*stack, resolved)))
        out.extend(d for d in imported.decls if isinstance(d, (ast.FuncDef, ast.UniverseDecl)))
    return out


def _check_import_dups(imported: list, stdlib_decls, user_decls) -> None:
    """An imported `def`/`universe` may not shadow an existing name (in the standard library, the
    importing file, or another imported file). Erroring keeps inclusion unambiguous - E-IMPORT-DUP."""
    def names(decls):
        return {d.name for d in decls if isinstance(d, (ast.FuncDef, ast.UniverseDecl))}

    stdlib_names = names(stdlib_decls)
    user_names = names(user_decls)
    from_imports: set[str] = set()
    for d in imported:
        n = d.name
        kind = "function" if isinstance(d, ast.FuncDef) else "universe"
        if n in stdlib_names:
            raise TrailImportError(
                f"E-IMPORT-DUP imported {kind} '{n}' collides with a standard-library definition")
        if n in user_names:
            raise TrailImportError(
                f"E-IMPORT-DUP imported {kind} '{n}' collides with a definition in the importing file")
        if n in from_imports:
            raise TrailImportError(
                f"E-IMPORT-DUP imported {kind} '{n}' is defined by more than one imported file")
        from_imports.add(n)


def prepare(source: str, *, stdlib: bool = True, path: str | None = None) -> ast.Program:
    # Parse the caller's source on its own so any syntax error carries line/column
    # numbers relative to *their* file, not an internally prepended standard library.
    user = parse_program(source)

    # Resolve imports before macro-expansion so imported `def`s/`universe`s are visible: paths
    # resolve relative to the importing file's directory (falling back to the CWD for raw source),
    # and `path` (when known) seeds the cycle-detection chain so a file importing itself is caught.
    base_dir = os.path.dirname(os.path.abspath(path)) if path else os.getcwd()
    stack: tuple[str, ...] = (os.path.realpath(path),) if path else ()
    imported = _import_defs(user, base_dir, seen=set(), stack=stack)
    user_decls = tuple(d for d in user.decls if not isinstance(d, ast.ImportDecl))

    stdlib_decls = parse_program(stdlib_source()).decls if stdlib else ()
    _check_import_dups(imported, stdlib_decls, user_decls)

    program = ast.Program(tuple(stdlib_decls) + tuple(imported) + user_decls)
    return expand_program(program)
