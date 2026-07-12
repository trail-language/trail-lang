"""Front-door pipeline: source text -> a compile-ready Program.

Prepends the standard library (so derived functions are implicitly available), parses,
and inlines all function definitions. Every entry point (CLI, tests, future REPL) should
go through `prepare` so the primitive/derived split is invisible to callers.
"""
from __future__ import annotations

from trail import ast
from trail.library import stdlib_source
from trail.macro import expand_program
from trail.parser import parse_program


def prepare(source: str, *, stdlib: bool = True) -> ast.Program:
    # Parse the caller's source on its own so any syntax error carries line/column
    # numbers relative to *their* file, not an internally prepended standard library.
    user = parse_program(source)
    if stdlib:
        library = parse_program(stdlib_source())
        program = ast.Program(library.decls + user.decls)
    else:
        program = user
    return expand_program(program)
