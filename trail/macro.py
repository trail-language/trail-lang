"""Function expansion: user/stdlib functions are non-recursive expression macros.

A `def` is inlined at each call site by substituting argument expressions into its body,
then compiling the result normally. This preserves totality (no recursion), vectorized
execution (the body composes ops that already lower to Polars), and static analyzability
(the expanded AST contains only builtins + fields). Primitives cannot be defined this way -
they stay in the operator library; a `def` may only compose existing functions.
"""
from __future__ import annotations

from dataclasses import replace

from trail import ast
from trail.ops import OPS

# The cross-sectional ops are the only builtins that consume a `by` group (they lower via
# ops._group(by)); time-series/elementwise ops ignore grouping. Derived from the op registry
# so this stays the single source of truth for "which ops take a group".
_CROSS_SECTIONAL = frozenset(name for name, spec in OPS.items() if spec.axis == "cross-sectional")


class TrailFunctionError(Exception):
    pass


def _rebuild(e, recur):
    """Rebuild expression `e`, applying `recur` to each immediate sub-expression."""
    match e:
        case ast.BinOp() | ast.Compare() | ast.BoolOp() | ast.Coalesce():
            return replace(e, left=recur(e.left), right=recur(e.right))
        case ast.Not() | ast.Neg():
            return replace(e, operand=recur(e.operand))
        case ast.In():
            return replace(e, item=recur(e.item))  # options are literals
        case ast.Ternary():
            return replace(e, value=recur(e.value), cond=recur(e.cond), orelse=recur(e.orelse))
        case ast.Call():
            return replace(
                e,
                args=tuple(recur(a) for a in e.args),
                kwargs=tuple((k, recur(v)) for k, v in e.kwargs),
            )
        case _:  # Literal, NameRef, FieldRef
            return e


def substitute(e, mapping: dict):
    """Replace parameter NameRefs with their argument expressions (hygienic: params only)."""
    if isinstance(e, ast.NameRef):
        return mapping.get(e.name, e)
    return _rebuild(e, lambda c: substitute(c, mapping))


def _propagate_by(e, by: tuple[str, ...]):
    """Fill a call-site `by` into every cross-sectional op in `e` that carries no `by` of its
    own. Ops that already specify a `by` (an explicit inner-group override) keep it; ops on
    other axes ignore grouping. Recurses the whole tree so a def body's several cross-sectional
    reducers are all grouped consistently."""
    if isinstance(e, ast.Call) and e.name in _CROSS_SECTIONAL and e.by is None:
        e = replace(e, by=by)
    return _rebuild(e, lambda c: _propagate_by(c, by))


def expand(e, funcs: dict[str, ast.FuncDef], stack: frozenset = frozenset()):
    """Inline every user-function call in `e`; raise on recursion or arity mismatch."""
    if isinstance(e, ast.Call) and e.name in funcs:
        if e.name in stack:
            raise TrailFunctionError(f"E-FUNC-RECURSION recursive function '{e.name}' is not allowed")
        fd = funcs[e.name]
        if len(e.args) != len(fd.params):
            raise TrailFunctionError(
                f"E-FUNC-ARITY function '{e.name}' takes {len(fd.params)} argument(s), got {len(e.args)}"
            )
        if e.kwargs:
            raise TrailFunctionError(f"E-FUNC-ARITY function '{e.name}' does not take keyword arguments")
        args = tuple(expand(a, funcs, stack) for a in e.args)
        # Expand the body's own nested defs first, thread the call-site `by` into the
        # cross-sectional ops it introduces, THEN substitute arguments. Doing this before
        # substitution scopes the `by` to the def's own reducers: it never leaks into a
        # caller's argument expression (mirroring how `by` binds to one builtin op, not its
        # operands). Innermost expansion runs first, so an inner call-site `by` is already
        # in place and _propagate_by (fill-if-None) leaves it untouched.
        inlined = expand(fd.body, funcs, stack | {e.name})
        if e.by is not None:
            inlined = _propagate_by(inlined, e.by)
        return substitute(inlined, dict(zip(fd.params, args, strict=True)))
    return _rebuild(e, lambda c: expand(c, funcs, stack))


def collect_functions(program: ast.Program) -> dict[str, ast.FuncDef]:
    funcs: dict[str, ast.FuncDef] = {}
    for d in program.decls:
        if isinstance(d, ast.FuncDef):
            if d.name in funcs:
                raise TrailFunctionError(f"E-FUNC-DUP duplicate function definition '{d.name}'")
            funcs[d.name] = d
    return funcs


def _expand_decl(d, funcs):
    def ex(e):
        return expand(e, funcs)

    match d:
        case ast.UniverseDecl() if d.where is not None:
            return replace(d, where=ex(d.where))
        case ast.ModelDecl():
            stmts = []
            for st in d.statements:
                if isinstance(st, ast.Assignment):
                    stmts.append(replace(st, expr=ex(st.expr)))
                else:  # ScoreDecl
                    cases = tuple(ast.ScoreCase(c.value, ex(c.cond)) for c in st.cases)
                    stmts.append(replace(st, cases=cases))
            return replace(d, statements=tuple(stmts))
        case ast.SignalDecl():
            return replace(d, expr=ex(d.expr))
        case _:
            return d


def expand_program(program: ast.Program) -> ast.Program:
    """Return a new program with functions collected, calls inlined, and defs stripped."""
    funcs = collect_functions(program)
    decls = [_expand_decl(d, funcs) for d in program.decls if not isinstance(d, ast.FuncDef)]
    return ast.Program(tuple(decls))
