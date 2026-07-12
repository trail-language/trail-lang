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
    full = f"{stdlib_source()}\n{source}" if stdlib else source
    return expand_program(parse_program(full))
