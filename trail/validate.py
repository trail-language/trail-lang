"""Semantic validation over the AST. Returns a flat list of Issues; errors block compilation."""
from __future__ import annotations

from dataclasses import dataclass

from trail import ast
from trail.ops import _AGG, FREQ_DUR
from trail.schema import is_field, kind_of

_AGG_NAMES = frozenset(_AGG)
_FREQ_NAMES = frozenset(FREQ_DUR)
_TO_FUNCS = frozenset({"to_annual", "to_quarterly", "to_monthly", "to_daily"})

# name -> (min_args, max_args) counting positional args only; kwargs checked by name.
KNOWN_FUNCTIONS: dict[str, tuple[int, int]] = {
    "lag": (2, 2), "roll_mean": (2, 2), "roll_sum": (2, 2), "roll_std": (2, 2),
    "roll_var": (2, 2), "roll_max": (2, 2), "roll_min": (2, 2), "roll_quantile": (3, 3),
    "roll_median": (2, 2), "roll_skew": (2, 2),
    "ewm_mean": (2, 2), "ewm_std": (2, 2), "decay_linear": (2, 2), "resample": (3, 3),
    "asof": (1, 1),
    "to_annual": (1, 2), "to_quarterly": (1, 2), "to_monthly": (1, 2), "to_daily": (1, 2),
    "cummax": (1, 1), "cumsum": (1, 1), "cumprod": (1, 1), "cummin": (1, 1),
    "zscore": (1, 1), "rank": (1, 1), "winsorize": (2, 2),
    "xs_mean": (1, 1), "xs_median": (1, 1), "xs_sum": (1, 1), "xs_frac": (1, 1),
    "xs_std": (1, 1), "xs_var": (1, 1), "xs_min": (1, 1), "xs_max": (1, 1),
    "xs_count": (1, 1), "xs_quantile": (2, 2),
    "count": (1, 99), "sqrt": (1, 1), "abs": (1, 1), "log": (1, 1), "exp": (1, 1),
    "sin": (1, 1), "cos": (1, 1), "tan": (1, 1), "asin": (1, 1), "acos": (1, 1), "atan": (1, 1),
    "floor": (1, 1), "ceil": (1, 1), "round": (1, 1),
    "clamp": (3, 3), "min": (2, 2), "max": (2, 2), "weighted_score": (0, 0),
}


@dataclass(frozen=True)
class Issue:
    severity: str  # error | warning
    code: str
    message: str


def _kind(e) -> str | None:
    return kind_of(e.column) if isinstance(e, ast.FieldRef) else None


def _is_ws_call(e) -> bool:
    return isinstance(e, ast.Call) and e.name == "weighted_score"


def _contains_ws(e) -> bool:
    """weighted_score anywhere in the expression tree (deps.extract sees all calls)."""
    from trail.deps import extract

    return "weighted_score" in extract(e).functions


def _check_agg(agg: ast.Expr, out: list[Issue]) -> None:
    """A literal aggregation name must be one the engine knows (catches typos and non-strings)."""
    if isinstance(agg, ast.Literal) and agg.value not in _AGG_NAMES:
        out.append(Issue("error", "E-AGG-UNKNOWN",
                         f"unknown aggregation {agg.value!r}; expected one of {sorted(_AGG_NAMES)}"))


def _check_freq_agg(e: ast.Call, out: list[Issue]) -> None:
    """resample(x, freq, agg) / to_<freq>(x[, agg]): validate literal freq and agg names."""
    if e.name == "resample" and len(e.args) == 3:
        freq = e.args[1]
        if isinstance(freq, ast.Literal) and freq.value not in _FREQ_NAMES:
            out.append(Issue("error", "E-FREQ-UNKNOWN",
                             f"unknown frequency {freq.value!r}; expected one of {sorted(_FREQ_NAMES)}"))
        _check_agg(e.args[2], out)
    elif e.name in _TO_FUNCS and len(e.args) == 2:
        _check_agg(e.args[1], out)


def _lint_stock_flow(e: ast.BinOp, out: list[Issue]) -> None:
    if e.op == "div" and {_kind(e.left), _kind(e.right)} == {"flow", "stock"}:
        out.append(Issue("warning", "W-KIND-STOCK-FLOW",
                         "flow/stock ratio uses a point-in-time balance value; consider avg2(...)"))


def _check_expr(e, defined: set[str], out: list[Issue]) -> None:
    match e:
        case ast.FieldRef():
            if not is_field(e.column):
                out.append(Issue("error", "E-FIELD-UNKNOWN", f"unknown field '{e.column}'"))
            if e.source is not None:
                out.append(Issue("error", "E-PIN-UNSUPPORTED",
                                 f"source pin '@ {e.source}' is not supported in this phase"))
        case ast.NameRef():
            if e.name not in defined:
                out.append(Issue("error", "E-NAME-UNDEFINED", f"name '{e.name}' is not defined here"))
        case ast.Call():
            if e.name == "fwd_return":
                out.append(Issue("error", "E-FWD-CONTEXT", "fwd_return is only legal in learn.target"))
            elif e.name not in KNOWN_FUNCTIONS:
                out.append(Issue("error", "E-FUNC-UNKNOWN", f"unknown function '{e.name}'"))
            else:
                lo, hi = KNOWN_FUNCTIONS[e.name]
                if not (lo <= len(e.args) <= hi):
                    out.append(Issue("error", "E-FUNC-ARITY",
                                     f"{e.name} takes {lo}..{hi} args, got {len(e.args)}"))
                _check_freq_agg(e, out)
            for a in e.args:
                _check_expr(a, defined, out)
            for _, v in e.kwargs:
                _check_expr(v, defined, out)
            if e.by is not None and not is_field(".".join(e.by)):
                out.append(Issue("error", "E-FIELD-UNKNOWN", f"unknown 'by' field {'.'.join(e.by)}"))
        case ast.BinOp():
            _lint_stock_flow(e, out)
            _check_expr(e.left, defined, out)
            _check_expr(e.right, defined, out)
        case ast.Compare() | ast.BoolOp() | ast.Coalesce():
            _check_expr(e.left, defined, out)
            _check_expr(e.right, defined, out)
        case ast.In():
            _check_expr(e.item, defined, out)
        case ast.Not() | ast.Neg():
            _check_expr(e.operand, defined, out)
        case ast.Ternary():
            _check_expr(e.value, defined, out)
            _check_expr(e.cond, defined, out)
            _check_expr(e.orelse, defined, out)


def _phase_warnings(decl, out: list[Issue]) -> None:
    """Constructs that parse but whose full semantics land in a later phase."""
    if getattr(decl, "on_missing", None) == "median":
        out.append(Issue("warning", "W-MEDIAN-DEFERRED",
                         "on_missing median is treated as skip until phase 2"))


def validate(program: ast.Program) -> list[Issue]:
    out: list[Issue] = []
    universes = {d.name for d in program.decls if isinstance(d, ast.UniverseDecl)}

    seen_top: set[str] = set()
    for decl in program.decls:
        # backtest/learn REFERENCE a strategy/model name - they bind nothing, so
        # `strategy s {...}` followed by `backtest s ...` is not a rebind (spec App. C).
        if isinstance(decl, ast.OpaqueDecl) and decl.kind in ("backtest", "learn"):
            continue
        name = getattr(decl, "name", None)
        if name is not None:
            if name in seen_top:
                out.append(Issue("error", "E-NAME-REBOUND",
                                 f"duplicate top-level declaration '{name}'"))
            seen_top.add(name)

    # universe roots must resolve (stocks or a declared universe) and must not cycle
    for decl in program.decls:
        if isinstance(decl, ast.UniverseDecl):
            root = ".".join(decl.root)
            if root != "stocks" and root not in universes:
                out.append(Issue("error", "E-UNIVERSE-UNKNOWN",
                                 f"universe '{decl.name}' has unknown root '{root}'"))
    by_name = {d.name: d for d in program.decls if isinstance(d, ast.UniverseDecl)}
    for name, decl in by_name.items():
        seen_chain = {name}
        cur = by_name.get(".".join(decl.root))
        while cur is not None:
            if cur.name in seen_chain:
                out.append(Issue("error", "E-UNIVERSE-CYCLE",
                                 f"universe '{name}' participates in a root cycle"))
                break
            seen_chain.add(cur.name)
            cur = by_name.get(".".join(cur.root))

    for decl in program.decls:
        match decl:
            case ast.UniverseDecl() if decl.where is not None:
                _check_expr(decl.where, set(), out)
            case ast.ModelDecl():
                _phase_warnings(decl, out)
                if decl.universe is not None and decl.universe not in universes:
                    out.append(Issue("error", "E-UNIVERSE-UNKNOWN",
                                     f"model '{decl.name}' references unknown universe '{decl.universe}'"))
                elif decl.universe is None and len(universes) > 1:
                    out.append(Issue("error", "E-UNIVERSE-UNKNOWN",
                                     f"model '{decl.name}' must declare 'on': multiple universes exist"))
                defined: set[str] = set()
                for st in decl.statements:
                    if isinstance(st, ast.Assignment):
                        _check_expr(st.expr, defined, out)
                        if (not _is_ws_call(st.expr)) and _contains_ws(st.expr):
                            out.append(Issue("error", "E-MODEL-CONTEXT",
                                             "weighted_score() is legal only as the complete "
                                             "right-hand side of a model assignment"))
                    else:  # ScoreDecl
                        for c in st.cases:
                            _check_expr(c.cond, defined, out)
                    if st.name in defined:
                        out.append(Issue("error", "E-NAME-REBOUND",
                                         f"'{st.name}' is already defined in model '{decl.name}'"))
                    defined.add(st.name)
            case ast.SignalDecl():
                _phase_warnings(decl, out)
                if decl.universe is not None and decl.universe not in universes:
                    out.append(Issue("error", "E-UNIVERSE-UNKNOWN",
                                     f"signal '{decl.name}' references unknown universe '{decl.universe}'"))
                _check_expr(decl.expr, set(), out)
    return out
