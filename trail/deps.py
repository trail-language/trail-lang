"""Static dependency extraction: which schema fields, functions, and pins a program needs."""
from __future__ import annotations

from dataclasses import dataclass, field as dfield

from trail import ast


@dataclass
class _Acc:
    fields: set[str] = dfield(default_factory=set)
    functions: set[str] = dfield(default_factory=set)
    locals_used: set[str] = dfield(default_factory=set)
    pins: set[tuple[str, str]] = dfield(default_factory=set)
    #: physical column -> `@ align(expr)` override AST (the field's alignment coordinate)
    align_overrides: dict[str, object] = dfield(default_factory=dict)


@dataclass(frozen=True)
class DepReport:
    fields: frozenset[str]
    functions: frozenset[str]
    locals_used: frozenset[str]
    pins: frozenset[tuple[str, str]]
    align_overrides: dict = dfield(default_factory=dict)


def _walk(node, acc: _Acc) -> None:
    match node:
        case ast.FieldRef():
            acc.fields.add(node.qualified_column)  # frequency-qualified so the loader sees the freq
            if node.source:
                acc.pins.add((node.column, node.source))
            if node.align is not None:  # names in the align expr are source DATE columns, not fields
                acc.align_overrides[node.qualified_column] = node.align
        case ast.NameRef():
            acc.locals_used.add(node.name)
        case ast.Call():
            acc.functions.add(node.name)
            for a in node.args:
                _walk(a, acc)
            for _, v in node.kwargs:
                _walk(v, acc)
            if node.by:
                acc.fields.add(".".join(node.by))
        case ast.BinOp() | ast.Compare() | ast.BoolOp() | ast.Coalesce():
            _walk(node.left, acc)
            _walk(node.right, acc)
        case ast.In():
            _walk(node.item, acc)
            for opt in node.options:
                _walk(opt, acc)
        case ast.Not() | ast.Neg():
            _walk(node.operand, acc)
        case ast.Ternary():
            _walk(node.value, acc)
            _walk(node.cond, acc)
            _walk(node.orelse, acc)
        case ast.Literal():
            pass


def extract(node) -> DepReport:
    acc = _Acc()
    if isinstance(node, ast.Program):
        for decl in node.decls:
            match decl:
                case ast.UniverseDecl() if decl.where is not None:
                    _walk(decl.where, acc)
                case ast.ModelDecl():
                    for st in decl.statements:
                        if isinstance(st, ast.Assignment):
                            _walk(st.expr, acc)
                        else:  # ScoreDecl
                            for c in st.cases:
                                _walk(c.value, acc)
                                _walk(c.cond, acc)
                            _walk(st.default, acc)
                case ast.SignalDecl():
                    _walk(decl.expr, acc)
    else:
        _walk(node, acc)
    return DepReport(
        frozenset(acc.fields), frozenset(acc.functions), frozenset(acc.locals_used),
        frozenset(acc.pins), dict(acc.align_overrides),
    )
