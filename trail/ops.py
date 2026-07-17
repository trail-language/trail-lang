"""Operator library: every Trail function lowered to a vectorized Polars expression.

The panel is always sorted [ENTITY, TIME]; per-entity ops close over that ordering.
"""
from __future__ import annotations

from typing import NamedTuple

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


# frequency name -> polars truncate/duration string (the target bucket for resample)
FREQ_DUR = {
    "annual": "1y", "quarterly": "3mo", "monthly": "1mo",
    "weekly": "1w", "daily": "1d", "hourly": "1h", "minute": "1m",
}

# default downsample aggregation per field kind - the automatic rule of reference §4.4.
# The single source of truth shared by cross-source alignment and the to_* sugar.
AGG_FOR_KIND = {
    "flow": "sum", "per_share": "sum",
    "rate": "mean", "ratio": "mean",
    "return": "compound",
    "stock": "last", "level": "last", "price": "last", "index": "last", "meta": "last",
}

# the aggregation library: a bucket reduction (list of values -> one value)
_AGG = {
    "sum": lambda e: e.sum(),
    "mean": lambda e: e.mean(),
    "last": lambda e: e.last(),
    "first": lambda e: e.first(),
    "min": lambda e: e.min(),
    "max": lambda e: e.max(),
    "count": lambda e: e.count(),
    "median": lambda e: e.median(),
    "std": lambda e: e.std(),
    "var": lambda e: e.var(),
    "prod": lambda e: e.product(),
    "compound": lambda e: (e + 1).product() - 1,
    "geomean": lambda e: e.log().mean().exp(),
    "skew": lambda e: e.skew(),
    "kurtosis": lambda e: e.kurtosis(),
    "range": lambda e: e.max() - e.min(),
    "change": lambda e: e.last() - e.first(),
}

class OpSpec(NamedTuple):
    lo: int
    hi: int
    axis: str  # time-series | cross-sectional | elementwise | model
    summary: str


# THE function registry - single source of truth. validate derives arities from it,
# catalog derives axis/summary. Names marked (desugar) lower in the compiler, not build().
OPS: dict[str, OpSpec] = {
    # --- time-series (per entity) ---
    "lag": OpSpec(2, 2, "time-series", "value n periods earlier (per entity)"),
    "roll_mean": OpSpec(2, 2, "time-series", "rolling mean over n periods or a duration"),
    "roll_sum": OpSpec(2, 2, "time-series", "rolling sum over n periods or a duration"),
    "roll_std": OpSpec(2, 2, "time-series", "rolling sample std (ddof=1) over n periods"),
    "roll_var": OpSpec(2, 2, "time-series", "rolling sample variance over n periods"),
    "roll_max": OpSpec(2, 2, "time-series", "rolling max over n periods"),
    "roll_min": OpSpec(2, 2, "time-series", "rolling min over n periods"),
    "roll_quantile": OpSpec(3, 3, "time-series", "rolling q-quantile (historical VaR)"),
    "roll_median": OpSpec(2, 2, "time-series", "rolling median over n periods"),
    "roll_skew": OpSpec(2, 2, "time-series", "rolling skewness over n periods"),
    "ewm_mean": OpSpec(2, 2, "time-series", "exponentially-weighted mean (span)"),
    "ewm_std": OpSpec(2, 2, "time-series", "exponentially-weighted std (span)"),
    "decay_linear": OpSpec(2, 2, "time-series", "linearly-decayed weighted mean over n periods"),
    "resample": OpSpec(3, 3, "time-series", "downsample to a frequency by an aggregation, broadcast back"),
    "asof": OpSpec(1, 1, "time-series", "carry the last known value forward over gaps (per entity)"),
    "ttm": OpSpec(1, 1, "time-series", "trailing twelve months, kind-aware (flow sums, stock is last-known) (desugar)"),
    "trailing": OpSpec(2, 2, "time-series", "trailing duration window, kind-aware (desugar)"),
    "to_annual": OpSpec(1, 2, "time-series", "resample to annual; aggregation defaults by kind (desugar)"),
    "to_quarterly": OpSpec(1, 2, "time-series", "resample to quarterly; aggregation defaults by kind (desugar)"),
    "to_monthly": OpSpec(1, 2, "time-series", "resample to monthly; aggregation defaults by kind (desugar)"),
    "to_daily": OpSpec(1, 2, "time-series", "resample to daily; aggregation defaults by kind (desugar)"),
    "cummax": OpSpec(1, 1, "time-series", "expanding maximum"),
    "cumsum": OpSpec(1, 1, "time-series", "expanding sum (discrete integral)"),
    "cumprod": OpSpec(1, 1, "time-series", "expanding product (compounding)"),
    "cummin": OpSpec(1, 1, "time-series", "expanding minimum"),
    # --- cross-sectional (per period[, group]) ---
    "zscore": OpSpec(1, 1, "cross-sectional", "standardize within (period[, group])"),
    "rank": OpSpec(1, 1, "cross-sectional", "average-tie rank, ascending, within group"),
    "winsorize": OpSpec(2, 2, "cross-sectional", "clip to [p, 1-p] group quantiles"),
    "xs_mean": OpSpec(1, 1, "cross-sectional", "group mean, broadcast back to members"),
    "xs_median": OpSpec(1, 1, "cross-sectional", "group median, broadcast back"),
    "xs_sum": OpSpec(1, 1, "cross-sectional", "group sum, broadcast back"),
    "xs_frac": OpSpec(1, 1, "cross-sectional", "fraction of group where cond is true"),
    "xs_std": OpSpec(1, 1, "cross-sectional", "group sample std (ddof=1)"),
    "xs_var": OpSpec(1, 1, "cross-sectional", "group sample variance"),
    "xs_min": OpSpec(1, 1, "cross-sectional", "group minimum, broadcast back"),
    "xs_max": OpSpec(1, 1, "cross-sectional", "group maximum, broadcast back"),
    "xs_count": OpSpec(1, 1, "cross-sectional", "non-null count in group"),
    "xs_quantile": OpSpec(2, 2, "cross-sectional", "group q-quantile, broadcast back"),
    # --- elementwise / scalar ---
    "count": OpSpec(1, 99, "elementwise", "sum of boolean flags as integers"),
    "sqrt": OpSpec(1, 1, "elementwise", "square root (null for x<0)"),
    "abs": OpSpec(1, 1, "elementwise", "absolute value"),
    "log": OpSpec(1, 1, "elementwise", "natural log (null for x<=0)"),
    "exp": OpSpec(1, 1, "elementwise", "e ** x"),
    "sin": OpSpec(1, 1, "elementwise", "sine (radians)"),
    "cos": OpSpec(1, 1, "elementwise", "cosine (radians)"),
    "tan": OpSpec(1, 1, "elementwise", "tangent (radians)"),
    "asin": OpSpec(1, 1, "elementwise", "arcsine"),
    "acos": OpSpec(1, 1, "elementwise", "arccosine"),
    "atan": OpSpec(1, 1, "elementwise", "arctangent"),
    "floor": OpSpec(1, 1, "elementwise", "round down to integer"),
    "ceil": OpSpec(1, 1, "elementwise", "round up to integer"),
    "round": OpSpec(1, 1, "elementwise", "round to nearest integer"),
    "clamp": OpSpec(3, 3, "elementwise", "clip x to [lo, hi]"),
    "min": OpSpec(2, 2, "elementwise", "cell-wise min of two panels"),
    "max": OpSpec(2, 2, "elementwise", "cell-wise max of two panels"),
    # --- temporal (calendar extraction / arithmetic on a datetime, e.g. time or a date column) ---
    "year": OpSpec(1, 1, "elementwise", "calendar year of a datetime"),
    "month": OpSpec(1, 1, "elementwise", "calendar month (1-12) of a datetime"),
    "quarter": OpSpec(1, 1, "elementwise", "calendar quarter (1-4) of a datetime"),
    "day": OpSpec(1, 1, "elementwise", "day of month (1-31) of a datetime"),
    "truncate": OpSpec(2, 2, "elementwise", 'truncate a datetime to a duration bucket (e.g. "1y", "1mo")'),
    "datediff": OpSpec(2, 3, "elementwise", "whole units between two datetimes (unit days|hours|minutes|seconds, default days)"),
    # --- model axis ---
    "weighted_score": OpSpec(0, 0, "model", "weighted rollup of the model's score blocks (desugar)"),
}

_ROLL = {
    "roll_mean": "rolling_mean", "roll_sum": "rolling_sum", "roll_std": "rolling_std",
    "roll_var": "rolling_var", "roll_max": "rolling_max", "roll_min": "rolling_min",
}


def build(name: str, args: list, kwargs: dict, by: tuple[str, ...] | None) -> pl.Expr:
    a = args
    match name:
        # --- time-series (per entity) ---
        case "lag":
            return a[0].shift(int(a[1])).over(ENTITY)
        case "roll_mean" | "roll_sum" | "roll_std" | "roll_var" | "roll_max" | "roll_min":
            base = _ROLL[name]
            if isinstance(a[1], str):  # duration window over the time axis (e.g. "1y", "90d")
                return getattr(a[0], base + "_by")(pl.col(TIME), window_size=a[1]).over(ENTITY)
            n = int(a[1])
            return getattr(a[0], base)(window_size=n, min_samples=n).over(ENTITY)
        case "resample":  # downsample to `freq`, reduce each bucket by `agg`, broadcast back to the grid
            return _AGG[a[2]](a[0]).over([pl.col(ENTITY), pl.col(TIME).dt.truncate(FREQ_DUR[a[1]])])
        case "asof":  # force last-known alignment: carry each value forward over gaps, per entity
            return a[0].forward_fill().over(ENTITY)
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
        # --- temporal ---
        case "year":
            return _x(a[0]).dt.year()
        case "month":
            return _x(a[0]).dt.month()
        case "quarter":
            return _x(a[0]).dt.quarter()
        case "day":
            return _x(a[0]).dt.day()
        case "truncate":
            return _x(a[0]).dt.truncate(a[1])
        case "datediff":
            delta = _x(a[0]) - _x(a[1])
            unit = a[2] if len(a) > 2 else "days"
            total = {"days": "total_days", "hours": "total_hours",
                     "minutes": "total_minutes", "seconds": "total_seconds"}.get(unit)
            if total is None:
                raise ValueError(f"datediff unit must be days|hours|minutes|seconds, got {unit!r}")
            return getattr(delta.dt, total)()
        case _:
            raise KeyError(f"no builder for function '{name}'")
