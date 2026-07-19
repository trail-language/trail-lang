"""Lower AST to Polars expressions and executable model plans."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

from trail import ast
from trail.ops import AGG_FOR_KIND, OPS, TIME, ENTITY, build, safe_div
from trail.schema import kind_of

# Ops whose builder lowers to a Polars window (`.over(...)`). Staging hoists these into their own
# columns so a compound expression becomes a chain of with_columns instead of one nested expression.
_WINDOW_AXES = ("time-series", "cross-sectional")


class _Stager:
    """Hoist each window-producing subexpression (one that lowers to a Polars `.over(...)`) into its
    own intermediate column, turning a nested/compound expression into a chain of `with_columns`.
    Polars then computes each window once and parallelizes independent columns, instead of
    re-partitioning the whole panel for every nested `.over()`. Measured ~5x on multi-window
    composites. Semantics-preserving: pure let-binding of subexpressions. Intermediates use a
    reserved `__stage_*` prefix and are dropped by the plan's final `.select([entity, time, ...])`."""
    __slots__ = ("stages", "emitted")

    def __init__(self) -> None:
        self.stages: list[tuple[str, pl.Expr]] = []
        self.emitted = 0

    def hoist(self, expr: pl.Expr) -> pl.Expr:
        alias = f"__stage_{len(self.stages)}"
        self.stages.append((alias, expr))
        return pl.col(alias)

    def drain(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Emit a `with_columns` for every stage created since the last drain, in dependency order
        (inner windows first). One `with_columns` per stage so a later stage may reference an
        earlier one (columns within a single `with_columns` cannot)."""
        while self.emitted < len(self.stages):
            alias, ex = self.stages[self.emitted]
            lf = lf.with_columns(ex.alias(alias))
            self.emitted += 1
        return lf

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


def compile_expr(e: ast.Expr, defined: set[str], stager: _Stager | None = None) -> pl.Expr:
    def c(x: ast.Expr) -> pl.Expr:
        return compile_expr(x, defined, stager)

    def _hoist(built: pl.Expr) -> pl.Expr:  # a window op: hoist to its own column when staging
        return stager.hoist(built) if stager is not None else built

    match e:
        case ast.Literal():
            return pl.lit(e.value)
        case ast.NameRef():
            return pl.col(e.name)
        case ast.FieldRef():
            return pl.col(e.qualified_column)
        case ast.BinOp() if e.op == "div":
            return safe_div(c(e.left), c(e.right))
        case ast.BinOp():
            return _BIN[e.op](c(e.left), c(e.right))
        case ast.Compare():
            return _CMP[e.op](c(e.left), c(e.right))
        case ast.In():
            return c(e.item).is_in([o.value for o in e.options])
        case ast.BoolOp() if e.op == "and":
            return c(e.left) & c(e.right)
        case ast.BoolOp():
            return c(e.left) | c(e.right)
        case ast.Not():
            return ~c(e.operand)
        case ast.Neg():
            return -c(e.operand)
        case ast.Coalesce():
            return pl.coalesce([c(e.left), c(e.right)])
        case ast.Ternary():
            return (pl.when(c(e.cond)).then(c(e.value)).otherwise(c(e.orelse)))
        case ast.Call() if e.name in _TO_FREQ:  # desugar to resample with a kind-aware default agg
            return _hoist(build("resample", [c(e.args[0]), _TO_FREQ[e.name], _to_agg(e.args)], {}, e.by))
        case ast.Call() if e.name in ("ttm", "trailing"):
            # kind-aware trailing window (§4.4): a flow accumulates, a rate/ratio averages,
            # a return compounds, a stock/level/price/index is the last-known value (a
            # balance sheet must not be summed). Computed expressions default to flow.
            arg = e.args[0]
            k = kind_of(arg.column) if isinstance(arg, ast.FieldRef) else None
            window = "1y" if e.name == "ttm" else _call_arg(e.args[1], defined, stager)
            x = c(arg)
            if k in ("rate", "ratio"):
                built = build("roll_mean", [x, window], {}, e.by)
            elif k == "return":  # exact compounding: exp(sum(log(1+r))) - 1
                built = build("roll_sum", [(x + 1).log(), window], {}, e.by).exp() - 1
            elif k in (None, "flow", "per_share"):
                built = build("roll_sum", [x, window], {}, e.by)
            else:
                built = build("asof", [x], {}, e.by)
            return _hoist(built)
        case ast.Call():
            args = [_call_arg(a, defined, stager) for a in e.args]
            kwargs = {k: _call_arg(v, defined, stager) for k, v in e.kwargs}
            built = build(e.name, args, kwargs, e.by)
            if OPS[e.name].axis in _WINDOW_AXES:
                return _hoist(built)
            return built
    raise TypeError(f"cannot compile {type(e).__name__}")


def _call_arg(a: ast.Expr, defined: set[str], stager: _Stager | None = None):
    # numeric/string literals pass through raw (window sizes, quantiles, freq/agg names)
    if isinstance(a, ast.Literal) and isinstance(a.value, (int, float, str)) and not isinstance(a.value, bool):
        return a.value
    return compile_expr(a, defined, stager)


def _score_expr(sd: ast.ScoreDecl, defined: set[str], stager: _Stager | None = None) -> pl.Expr:
    """First-match-wins cases. Null iff every condition is null (all inputs missing),
    which is what makes `on_missing skip` renormalization meaningful (reference §4.3/§7.5)."""
    conds = [compile_expr(c.cond, defined, stager) for c in sd.cases]
    node = pl.when(conds[0]).then(compile_expr(sd.cases[0].value, defined, stager))
    for cond, case in zip(conds[1:], sd.cases[1:], strict=True):
        node = node.when(cond).then(compile_expr(case.value, defined, stager))
    normal = node.otherwise(compile_expr(sd.default, defined, stager)).cast(pl.Float64)
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


def _as_lazy(panel: "pl.DataFrame | pl.LazyFrame") -> pl.LazyFrame:
    """Accept either an eager panel (the common case) or a LazyFrame (the lazy `{file}` scan path,
    which keeps projection/predicate pushdown alive all the way down to the parquet scan)."""
    return panel if isinstance(panel, pl.LazyFrame) else panel.lazy()


def _collect(lf: pl.LazyFrame, engine: str | None) -> pl.DataFrame:
    # engine=None -> Polars' default in-memory engine; "streaming" -> bounded-memory out-of-core.
    return lf.collect(engine=engine) if engine is not None else lf.collect()


@dataclass
class ModelPlan:
    _lf_builder: Callable[[pl.DataFrame | pl.LazyFrame], pl.LazyFrame]
    exports: tuple[str, ...]

    def run(self, panel: "pl.DataFrame | pl.LazyFrame", engine: str | None = None) -> pl.DataFrame:
        return _collect(self._lf_builder(panel).select([ENTITY, TIME, *self.exports]), engine)


@dataclass
class SignalPlan:
    _lf_builder: Callable[[pl.DataFrame | pl.LazyFrame], pl.LazyFrame]
    name: str

    def run(self, panel: "pl.DataFrame | pl.LazyFrame", engine: str | None = None) -> pl.DataFrame:
        return _collect(self._lf_builder(panel).select([ENTITY, TIME, self.name]), engine)


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

    def builder(panel: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
        lf = _as_lazy(panel)
        for u in chain:  # ancestor filters compose (AND)
            if u.where is not None:
                lf = lf.filter(compile_expr(u.where, set()))
        defined: set[str] = set()
        stager = _Stager()  # one counter across the model so __stage_* names stay unique
        for st in model.statements:
            if isinstance(st, ast.Assignment) and st.expr is None:
                continue  # bare `export NAME`: the local is already a column; nothing to compute
            if isinstance(st, ast.ScoreDecl):
                col = _score_expr(st, defined, stager)
            elif _is_weighted_score(st.expr):
                col = _weighted_score(scores, model.on_missing)
            else:
                col = compile_expr(st.expr, defined, stager)
            lf = stager.drain(lf).with_columns(col.alias(st.name))  # hoisted windows first, then the value
            defined.add(st.name)
        return lf

    exports = tuple(s.name for s in model.statements if isinstance(s, ast.Assignment) and s.export)
    return ModelPlan(builder, exports)


def compile_signal(signal: ast.SignalDecl, universes: dict[str, ast.UniverseDecl]) -> SignalPlan:
    # A signal is a one-export, no-score model: same universe binding as compile_model
    # (reference §8.3) - explicit `on` wins, a sole declared universe auto-binds, zero
    # universes = full panel. The single expr compiles with defined=set() (a stray NameRef
    # is already E-NAME-UNDEFINED in validate).
    if signal.universe is not None:
        uni = universes.get(signal.universe)
    elif len(universes) == 1:
        uni = next(iter(universes.values()))
    else:
        uni = None
    chain = universe_chain(uni, universes)

    def builder(panel: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
        lf = _as_lazy(panel)
        for u in chain:  # ancestor filters compose (AND)
            if u.where is not None:
                lf = lf.filter(compile_expr(u.where, set()))
        stager = _Stager()
        col = compile_expr(signal.expr, set(), stager)
        return stager.drain(lf).with_columns(col.alias(signal.name))

    return SignalPlan(builder, signal.name)
