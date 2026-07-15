"""Operator library: every Trail function lowered to a vectorized Polars expression.

The panel is always sorted [ENTITY, TIME]; per-entity ops close over that ordering.
"""
from __future__ import annotations

import polars as pl

ENTITY = "entity"
TIME = "time"


def _x(v) -> pl.Expr:
    """Coerce an operand to an expression. Literal args arrive as raw numbers
    (so window positions can `int()` them); operand positions need an Expr."""
    return v if isinstance(v, pl.Expr) else pl.lit(v)


def safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den.is_null() | (den == 0)).then(None).otherwise(num / den)


def _group(by: tuple[str, ...] | None) -> list[str]:
    return [TIME] + ([".".join(by)] if by else [])


def build(name: str, args: list, kwargs: dict, by: tuple[str, ...] | None) -> pl.Expr:
    a = args
    match name:
        # --- time-series (per entity) ---
        case "lag":
            return a[0].shift(int(a[1])).over(ENTITY)
        case "roll_mean" | "roll_sum" | "roll_std" | "roll_var" | "roll_max" | "roll_min":
            n = int(a[1])
            method = {
                "roll_mean": "rolling_mean", "roll_sum": "rolling_sum", "roll_std": "rolling_std",
                "roll_var": "rolling_var", "roll_max": "rolling_max", "roll_min": "rolling_min",
            }[name]
            return getattr(a[0], method)(window_size=n, min_samples=n).over(ENTITY)
        case "roll_quantile":
            n = int(a[1])
            return a[0].rolling_quantile(quantile=float(a[2]), window_size=n, min_samples=n).over(ENTITY)
        case "cummax":
            return a[0].cum_max().over(ENTITY)
        case "cumsum":
            return a[0].cum_sum().over(ENTITY)
        case "cumprod":
            return a[0].cum_prod().over(ENTITY)
        case "cummin":
            return a[0].cum_min().over(ENTITY)
        case "roll_median":
            n = int(a[1])
            return a[0].rolling_median(window_size=n, min_samples=n).over(ENTITY)
        case "roll_skew":
            return a[0].rolling_skew(window_size=int(a[1])).over(ENTITY)
        case "ewm_mean":
            return a[0].ewm_mean(span=float(a[1])).over(ENTITY)
        case "ewm_std":
            return a[0].ewm_std(span=float(a[1])).over(ENTITY)
        case "decay_linear":
            n = int(a[1])
            weights = [float(i) for i in range(1, n + 1)]  # most recent period weighted highest
            return a[0].rolling_mean(window_size=n, weights=weights, min_samples=n).over(ENTITY)
        # --- cross-sectional (per period[, group]) ---
        case "zscore":
            g = _group(by)
            return safe_div(a[0] - a[0].mean().over(g), a[0].std().over(g))
        case "rank":
            return a[0].rank("average").over(_group(by))
        case "winsorize":
            g = _group(by)
            p = float(a[1])
            return a[0].clip(a[0].quantile(p).over(g), a[0].quantile(1 - p).over(g))
        case "xs_mean":
            return a[0].mean().over(_group(by))
        case "xs_median":
            return a[0].median().over(_group(by))
        case "xs_sum":
            return a[0].sum().over(_group(by))
        case "xs_frac":
            return a[0].cast(pl.Float64).mean().over(_group(by))
        case "xs_std":
            return a[0].std().over(_group(by))
        case "xs_var":
            return a[0].var().over(_group(by))
        case "xs_min":
            return a[0].min().over(_group(by))
        case "xs_max":
            return a[0].max().over(_group(by))
        case "xs_count":
            return a[0].count().over(_group(by))
        case "xs_quantile":
            return a[0].quantile(float(a[1])).over(_group(by))
        # --- scalar / elementwise ---
        case "count":
            out = a[0].cast(pl.Int32)
            for extra in a[1:]:
                out = out + extra.cast(pl.Int32)
            return out
        case "sqrt":
            return pl.when(_x(a[0]) < 0).then(None).otherwise(_x(a[0]).sqrt())
        case "abs":
            return _x(a[0]).abs()
        case "log":
            return pl.when(_x(a[0]) <= 0).then(None).otherwise(_x(a[0]).log())
        case "exp":
            return _x(a[0]).exp()
        case "sin":
            return _x(a[0]).sin()
        case "cos":
            return _x(a[0]).cos()
        case "tan":
            return _x(a[0]).tan()
        case "asin":
            return _x(a[0]).arcsin()
        case "acos":
            return _x(a[0]).arccos()
        case "atan":
            return _x(a[0]).arctan()
        case "floor":
            return _x(a[0]).floor()
        case "ceil":
            return _x(a[0]).ceil()
        case "round":
            return _x(a[0]).round(0)
        case "clamp":
            return _x(a[0]).clip(float(a[1]), float(a[2]))
        case "min":
            return pl.min_horizontal(_x(a[0]), _x(a[1]))
        case "max":
            return pl.max_horizontal(_x(a[0]), _x(a[1]))
        case _:
            raise KeyError(f"no builder for function '{name}'")
