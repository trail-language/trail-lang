"""Lower AST to Polars expressions and executable model plans."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

from trail import ast
from trail.ops import AGG_FOR_KIND, TIME, ENTITY, build, safe_div
from trail.schema import kind_of

# to_<freq>(x[, agg]) sugar -> resample(x, freq, agg); agg defaults by the field's kind (§4.4).
_TO_FREQ = {"to_annual": "annual", "to_quarterly": "quarterly",
            "to_monthly": "monthly", "to_daily": "daily"}


def _to_agg(args: list[ast.Expr]) -> str:
    """Aggregation for a to_<freq> call: an explicit second arg wins; else the kind
    default of a bare field argument; else last (a safe point-in-time snapshot)."""
    if len(args) >= 2 and isinstance(args[1], ast.Literal) and isinstance(args[1].value, str):
        return args[1].value
    if isinstance(args[0], ast.FieldRef):
        return AGG_FOR_KIND.get(kind_of(args[0].column) or "", "last")
    return "last"

# mod guards divide-by-zero (NaN would MATCH comparisons: polars NaN > 0 is true); pow can
# produce NaN for negative base ^ fractional exponent - normalized to null per §4.3.
_BIN = {
    "add": lambda x, y: x + y, "sub": lambda x, y: x - y, "mul": lambda x, y: x * y,
    "mod": lambda x, y: pl.when(y.is_null() | (y == 0)).then(None).otherwise(x % y),
    "pow": lambda x, y: x.pow(y).fill_nan(None),
}
_CMP = {
    "eq": lambda x, y: x == y, "ne": lambda x, y: x != y, "gt": lambda x, y: x > y,
    "lt": lambda x, y: x < y, "ge": lambda x, y: x >= y, "le": lambda x, y: x <= y,
}


def compile_expr(e: ast.Expr, defined: set[str]) -> pl.Expr:
    match e:
        case ast.Literal():
            return pl.lit(e.value)
        case ast.NameRef():
            return pl.col(e.name)
        case ast.FieldRef():
            return pl.col(e.qualified_column)
        case ast.BinOp() if e.op == "div":
            return safe_div(compile_expr(e.left, defined), compile_expr(e.right, defined))
        case ast.BinOp():
            return _BIN[e.op](compile_expr(e.left, defined), compile_expr(e.right, defined))
        case ast.Compare():
            return _CMP[e.op](compile_expr(e.left, defined), compile_expr(e.right, defined))
        case ast.In():
            return compile_expr(e.item, defined).is_in([o.value for o in e.options])
        case ast.BoolOp() if e.op == "and":
            return compile_expr(e.left, defined) & compile_expr(e.right, defined)
        case ast.BoolOp():
            return compile_expr(e.left, defined) | compile_expr(e.right, defined)
        case ast.Not():
            return ~compile_expr(e.operand, defined)
        case ast.Neg():
            return -compile_expr(e.operand, defined)
        case ast.Coalesce():
            return pl.coalesce([compile_expr(e.left, defined), compile_expr(e.right, defined)])
        case ast.Ternary():
            return (pl.when(compile_expr(e.cond, defined))
                    .then(compile_expr(e.value, defined))
                    .otherwise(compile_expr(e.orelse, defined)))
        case ast.Call() if e.name in _TO_FREQ:  # desugar to resample with a kind-aware default agg
            return build("resample", [compile_expr(e.args[0], defined), _TO_FREQ[e.name], _to_agg(e.args)], {}, e.by)
        case ast.Call():
            args = [_call_arg(a, defined) for a in e.args]
            kwargs = {k: _call_arg(v, defined) for k, v in e.kwargs}
            return build(e.name, args, kwargs, e.by)
    raise TypeError(f"cannot compile {type(e).__name__}")


def _call_arg(a: ast.Expr, defined: set[str]):
    # numeric/string literals pass through raw (window sizes, quantiles, freq/agg names)
    if isinstance(a, ast.Literal) and isinstance(a.value, (int, float, str)) and not isinstance(a.value, bool):
        return a.value
    return compile_expr(a, defined)


def _score_expr(sd: ast.ScoreDecl, defined: set[str]) -> pl.Expr:
    """First-match-wins cases. Null iff every condition is null (all inputs missing),
    which is what makes `on_missing skip` renormalization meaningful (reference §4.3/§7.5)."""
    conds = [compile_expr(c.cond, defined) for c in sd.cases]
    node = pl.when(conds[0]).then(compile_expr(sd.cases[0].value, defined))
    for cond, case in zip(conds[1:], sd.cases[1:], strict=True):
        node = node.when(cond).then(compile_expr(case.value, defined))
    normal = node.otherwise(compile_expr(sd.default, defined)).cast(pl.Float64)
    all_null = conds[0].is_null()
    for cond in conds[1:]:
        all_null = all_null & cond.is_null()
    return pl.when(all_null).then(pl.lit(None, dtype=pl.Float64)).otherwise(normal)


def _score_max(sd: ast.ScoreDecl) -> float:
    return float(max([c.value.value for c in sd.cases] + [sd.default.value]))


def _weighted_score(scores: list[ast.ScoreDecl], on_missing: str) -> pl.Expr:
    num = pl.lit(0.0)
    den = pl.lit(0.0)
    for sd in scores:
        s = pl.col(sd.name)
        w, mx = sd.weight, _score_max(sd)
        num = num + pl.coalesce([s * w, pl.lit(0.0)])
        if on_missing == "zero":
            den = den + pl.lit(w * mx)
        else:  # skip (median compiles like skip this phase)
            den = den + pl.when(s.is_not_null()).then(w * mx).otherwise(0.0)
    return safe_div(num, den)


def _is_weighted_score(expr: ast.Expr) -> bool:
    return isinstance(expr, ast.Call) and expr.name == "weighted_score"


@dataclass
class ModelPlan:
    _lf_builder: Callable[[pl.DataFrame], pl.LazyFrame]
    exports: tuple[str, ...]

    def run(self, panel: pl.DataFrame) -> pl.DataFrame:
        return self._lf_builder(panel).select([ENTITY, TIME, *self.exports]).collect()


def universe_chain(uni: ast.UniverseDecl | None,
                   universes: dict[str, ast.UniverseDecl]) -> list[ast.UniverseDecl]:
    """The universe and its ancestors (sub -> base -> ... -> stocks). Universes COMPOSE
    (reference §8.2): every `where` down the root chain applies. Cycles are a validation
    error; walked defensively here with a seen-set."""
    chain: list[ast.UniverseDecl] = []
    seen: set[str] = set()
    while uni is not None and uni.name not in seen:
        seen.add(uni.name)
        chain.append(uni)
        uni = universes.get(".".join(uni.root))
    return chain


def compile_model(model: ast.ModelDecl, universes: dict[str, ast.UniverseDecl]) -> ModelPlan:
    # Universe binding per reference §8.3: explicit `on` wins; a sole declared
    # universe auto-binds; zero universes = full panel.
    if model.universe is not None:
        uni = universes.get(model.universe)
    elif len(universes) == 1:
        uni = next(iter(universes.values()))
    else:
        uni = None
    chain = universe_chain(uni, universes)
    scores = [s for s in model.statements if isinstance(s, ast.ScoreDecl)]

    def builder(panel: pl.DataFrame) -> pl.LazyFrame:
        lf = panel.lazy()
        for u in chain:  # ancestor filters compose (AND)
            if u.where is not None:
                lf = lf.filter(compile_expr(u.where, set()))
        defined: set[str] = set()
        for st in model.statements:
            if isinstance(st, ast.ScoreDecl):
                lf = lf.with_columns(_score_expr(st, defined).alias(st.name))
                defined.add(st.name)
            elif _is_weighted_score(st.expr):
                lf = lf.with_columns(_weighted_score(scores, model.on_missing).alias(st.name))
                defined.add(st.name)
            else:
                lf = lf.with_columns(compile_expr(st.expr, defined).alias(st.name))
                defined.add(st.name)
        return lf

    exports = tuple(s.name for s in model.statements if isinstance(s, ast.Assignment) and s.export)
    return ModelPlan(builder, exports)
