"""Lark parser + transformer producing trail.ast nodes."""
from __future__ import annotations

from importlib.resources import files

from lark import Lark, Token, Transformer

from trail import ast, fieldname

_GRAMMAR = files("trail").joinpath("grammar.lark").read_text()

_ARITH = {"+": "add", "-": "sub", "*": "mul", "/": "div", "%": "mod"}
_CMP = {"==": "eq", "!=": "ne", ">": "gt", "<": "lt", ">=": "ge", "<=": "le"}


def _reject_requalified(node, what: str) -> None:
    """A field reference takes at most one `@` qualifier (source/entity/align). Chaining a second
    would silently drop the first, so reject it. Frequency is a prefix, not an `@` qualifier."""
    if isinstance(node, ast.FieldRef) and (node.source or node.entity or node.align is not None):
        raise ValueError(f"{what} chains a second @ qualifier onto an already-qualified field")


def _num(text: str) -> int | float:
    text = text.replace("_", "")
    val = float(text)
    keep_int = val.is_integer() and "e" not in text.lower() and "." not in text
    return int(val) if keep_int else val


class _T(Transformer):
    # --- literals ---
    def number(self, s):
        return ast.Literal(_num(s[0].value))

    def string(self, s):
        return ast.Literal(s[0].value[1:-1])

    def true(self, s):
        return ast.Literal(True)

    def false(self, s):
        return ast.Literal(False)

    # --- references ---
    def ref(self, parts):
        names = tuple(p.value for p in parts)
        if len(names) == 1:
            return ast.NameRef(names[0])
        freq, path = fieldname.parse_ref(names)
        return ast.FieldRef(path, frequency=freq)

    def dotted(self, parts):
        return tuple(p.value for p in parts)

    def kwarg(self, s):
        return (s[0].value, s[1])

    def call(self, s):
        name = s[0].value
        by = None
        rest = list(s[1:])
        # trailing `by dotted` clause arrives as a tuple[str, ...]
        if rest and isinstance(rest[-1], tuple) and rest[-1] and all(isinstance(x, str) for x in rest[-1]):
            by = rest.pop()
        args, kwargs = [], []
        for item in rest:
            if item is None:
                continue
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                kwargs.append(item)
            else:
                args.append(item)
        return ast.Call(name, tuple(args), tuple(kwargs), by)

    # --- operators ---
    def pinned(self, s):
        node, src = s
        _reject_requalified(node, f"@ {src.value}")
        return ast.FieldRef(node.path, source=src.value, frequency=node.frequency)

    def selector_pinned(self, s):
        # atom "@" NAME "(" expr ")" - the general selector form: entity("SPY") | align(<expr>).
        node, selector, arg = s
        sel = selector.value
        if not isinstance(node, ast.FieldRef):
            raise ValueError(f"@ {sel}(...) qualifies a schema field reference, not an expression")
        _reject_requalified(node, f"@ {sel}(...)")
        if sel == "entity":
            if not (isinstance(arg, ast.Literal) and isinstance(arg.value, str)):
                raise ValueError('@ entity(...) needs a quoted symbol, e.g. entity("SPY")')
            return ast.FieldRef(node.path, frequency=node.frequency, entity=arg.value)
        if sel == "align":
            # `arg` is an expression over the source's date columns (e.g. truncate(filing_date,"1y"));
            # it overrides the field's alignment coordinate. Materialized in the loader.
            return ast.FieldRef(node.path, frequency=node.frequency, align=arg)
        raise ValueError(f"unknown pin selector '{sel}(...)'; expected entity(\"...\") or align(...)")

    def neg(self, s):
        return ast.Neg(s[0])

    def power(self, s):
        return ast.BinOp("pow", s[0], s[1])

    def arith(self, s):
        return ast.BinOp(_ARITH[s[1].value], s[0], s[2])

    def coalesce(self, s):
        return ast.Coalesce(s[0], s[1])

    def compare(self, s):
        return ast.Compare(_CMP[s[1].value], s[0], s[2])

    def in_(self, s):
        return ast.In(s[0], tuple(s[1:]))

    def not_(self, s):
        return ast.Not(s[0])

    def bool_and(self, s):
        return ast.BoolOp("and", s[0], s[1])

    def bool_or(self, s):
        return ast.BoolOp("or", s[0], s[1])

    def ternary(self, s):
        return ast.Ternary(s[0], s[1], s[2])

    # --- declarations ---
    def universe_decl(self, s):
        name = s[0].value
        root = s[1]  # tuple[str, ...] from `dotted`
        where = s[2] if len(s) > 2 else None
        return ast.UniverseDecl(name, root, where)

    def desc_stmt(self, s):
        return ("desc", s[0].value[1:-1])

    def policy_stmt(self, s):
        return ("on_missing", s[0].value)

    def assign_stmt(self, s):
        return ast.Assignment(s[0].value, s[1], export=False)

    def export_stmt(self, s):
        # `export NAME` (no RHS) surfaces the existing local NAME as an export of the same
        # name; expr=None marks the bare form (validated/compiled against the local binding).
        expr = s[1] if len(s) > 1 else None
        return ast.Assignment(s[0].value, expr, export=True)

    def score_case(self, s):
        return ast.ScoreCase(ast.Literal(_num(s[0].value)), s[1])

    def score_stmt(self, s):
        name = s[0].value
        weight = float(s[1].value)
        cases = [x for x in s[2:] if isinstance(x, ast.ScoreCase)]
        default = ast.Literal(_num(s[-1].value))  # trailing NUMBER = else value
        return ast.ScoreDecl(name, weight, tuple(cases), default)

    def model_decl(self, s):
        name = s[0].value
        idx = 1
        universe = None
        frequency = None  # omitted `at` -> finest referenced (resolved at load)
        if idx < len(s) and isinstance(s[idx], Token) and s[idx].type == "NAME":
            universe = s[idx].value
            idx += 1
        if idx < len(s) and isinstance(s[idx], Token) and s[idx].type == "FREQ":
            frequency = s[idx].value
            idx += 1
        desc, on_missing, stmts = None, "skip", []
        for item in s[idx:]:
            if isinstance(item, tuple) and item[0] == "desc":
                desc = item[1]
            elif isinstance(item, tuple) and item[0] == "on_missing":
                on_missing = item[1]
            else:
                stmts.append(item)
        return ast.ModelDecl(name, universe, frequency, desc, on_missing, tuple(stmts))

    def signal_decl(self, s):
        name = s[0].value
        idx = 1
        universe = None
        frequency = None  # omitted `at` -> finest referenced (resolved at load)
        if idx < len(s) - 1 and isinstance(s[idx], Token) and s[idx].type == "NAME":
            universe = s[idx].value
            idx += 1
        if idx < len(s) - 1 and isinstance(s[idx], Token) and s[idx].type == "FREQ":
            frequency = s[idx].value
            idx += 1
        return ast.SignalDecl(name, universe, frequency, s[-1])

    def func_def(self, s):
        # s = [NAME(name), NAME(param)*, body_expr]; params are the NAME tokens
        name = s[0].value
        params = tuple(t.value for t in s[1:-1])
        return ast.FuncDef(name, params, s[-1])

    def import_decl(self, s):
        return ast.ImportDecl(s[0].value[1:-1])

    def strategy_decl(self, s):
        return ast.OpaqueDecl("strategy", s[0].value)

    def backtest_decl(self, s):
        return ast.OpaqueDecl("backtest", s[0].value)

    def learn_decl(self, s):
        return ast.OpaqueDecl("learn", s[0].value)

    def start(self, s):
        return ast.Program(tuple(s))

    # --- REPL dialect ---
    def meta_catalog(self, s):
        return ast.MetaCatalog()

    def meta_describe(self, s):
        return ast.MetaDescribe(s[0])  # s[0] is the dotted tuple

    def repl_line(self, s):
        return s[0]


_transformer = _T()
_expr_parser = Lark(_GRAMMAR, start="expr", parser="lalr", maybe_placeholders=False)
_program_parser = Lark(_GRAMMAR, start="start", parser="lalr", maybe_placeholders=False)
_repl_parser = Lark(_GRAMMAR, start="repl_line", parser="lalr", maybe_placeholders=False)


def parse_expr(text: str) -> ast.Expr:
    return _transformer.transform(_expr_parser.parse(text))


def parse_program(text: str) -> ast.Program:
    return _transformer.transform(_program_parser.parse(text))


def parse_repl_line(text: str):
    """Parse one interactive line: a meta-command, a declaration, or a bare expression."""
    return _transformer.transform(_repl_parser.parse(text))
