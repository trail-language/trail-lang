# Trail Function Catalog

Every function below is cross-checked against the op registry
(`trail/ops.py` — primitives) and the bundled stdlib (`trail/stdlib/*.trail` —
derived macros). **Layer**: `P` = primitive (engine op), `D` = derived (stdlib
`def`, auto-loaded unless `no_stdlib`). **Axis**: `T` time-series (per entity
along periods), `X` cross-sectional (within a period, over the universe; accepts
trailing `by <field>`), `E` elementwise, `M` model-context. Every example was run
through `eval_tool`; results are shown as `-> values`.

Window / quantile / period args (`n`, `q`, `p`) MUST be numeric **literals**.
`by <field>` may trail any `X`-axis call.

**The two panels used in examples below:**
- `XS` — one period `2023-12-31`, entities A/B/C/D with `income.net_income` =
  10/30/15/5, `income.revenue` = 100/200/300/150, `meta.sector` = Tech/Tech/Health/Health.
- `TS` — one entity A across 2019–2023, `income.revenue` = 100/110/121/133.1/146.41,
  `price.adj_close` = 50/55/60/52/70.

---

## 1. Time-series (T) — per entity, along the period axis

Require the panel sorted by time (engine guarantee). Windowed ops return null
until the window is full (`min_samples = n`).

### Primitives (P)
| Function | Arity | Summary | Verified example (TS) |
|---|---|---|---|
| `lag(x, n)` | 2 | value `n` periods earlier | `lag(income.revenue,1)` -> None,100,110,121,133.1 |
| `roll_mean(x, n)` | 2 | rolling mean over `n` periods **or a duration string** | `roll_mean(income.revenue,3)` -> …,110.33,121.37,133.50 |
| `roll_sum(x, n)` | 2 | rolling sum | `roll_sum(income.net_income,2)` on TS-ni -> …,17,16,19,27 |
| `roll_std(x, n)` | 2 | rolling sample std (ddof=1) | `roll_std(income.revenue,3)` -> …,10.504,11.554,12.710 |
| `roll_var(x, n)` | 2 | rolling sample variance | (see `roll_std`) |
| `roll_max(x, n)` | 2 | rolling max | `roll_max(price.adj_close,3)` -> …,60,60,70 |
| `roll_min(x, n)` | 2 | rolling min | `roll_min(price.adj_close,3)` -> …,50,52,52 |
| `roll_median(x, n)` | 2 | rolling median | `roll_median(income.revenue,3)` -> …,110,121,133.1 |
| `roll_skew(x, n)` | 2 | rolling skewness | validated |
| `roll_quantile(x, n, q)` | 3 | rolling q-quantile (historical VaR) | `roll_quantile(price.adj_close,3,0.5)` -> …,55,55,60 |
| `ewm_mean(x, span)` | 2 | exponentially-weighted mean | `ewm_mean(price.adj_close,3)` -> 50,53.33,57.14,54.4,62.45 |
| `ewm_std(x, span)` | 2 | exponentially-weighted std | validated |
| `decay_linear(x, n)` | 2 | linearly-decayed weighted mean (recent weighted highest) | `decay_linear(price.adj_close,3)` -> …,56.67,55.17,62.33 |
| `cummax(x)` | 1 | expanding max | `cummax(price.adj_close)` -> 50,55,60,60,70 |
| `cumsum(x)` | 1 | expanding sum (discrete integral) | `cumsum(income.net_income)` -> 8,17,24,36,51 |
| `cumprod(x)` | 1 | expanding product (compounding) | `cumprod(1+income.net_income/100)` -> 1.08,1.177,1.26,1.41,1.62 |
| `cummin(x)` | 1 | expanding min | `cummin(price.adj_close)` -> 50,50,50,50,50 |
| `ts_mean(x)` | 1 | whole-series mean, broadcast back (NOT PIT-causal) | `ts_mean(price.adj_close)` -> 57.4 (all rows) |
| `ts_std(x)` | 1 | whole-series sample std | `ts_std(price.adj_close)` -> 7.9875 |
| `ts_min(x)` | 1 | whole-series min | `ts_min(price.adj_close)` -> 50 |
| `asof(x)` | 1 | carry last known value forward over gaps | `asof(income.revenue)` -> 100,110,121,133.1,146.41 |

`ts_mean`/`ts_std`/`ts_min` see the entire per-entity series at once, so they are
retrospective summaries — not look-ahead-safe features for a live decision.

### Derived (D) — stdlib `timeseries.trail`, `stats.trail`, `calculus.trail`
| Function | Def | Verified (TS) |
|---|---|---|
| `yoy(x)` | `x/lag(x,1) - 1` | `yoy(income.revenue)` -> None,0.1,0.1,0.1,0.1 |
| `avg2(x)` | `(x + lag(x,1))/2` | `avg2(balance.total_assets)` -> None,205,215,225,235 |
| `cagr(x, n)` | `(end'/start')^(1/n) - 1`, shift rule §4 | `cagr(income.revenue,4)` -> …,0.1 |
| `increase(x, n)` | `(end'-start')/start'`, shift rule | `increase(income.revenue,4)` -> …,0.4641 |
| `drawdown(x)` | `x/cummax(x) - 1` (LEVEL series) | `drawdown(price.adj_close)` -> 0,0,0,-0.1333,0 |
| `diff(x)` | `x - lag(x,1)` | `diff(income.revenue)` -> None,10,11,12.1,13.31 |
| `diff2(x)` | `x - 2·lag(x,1) + lag(x,2)` | `diff2(income.revenue)` -> …,1,1.1,1.21 |
| `deriv(x, n)` | `(x - lag(x,n))/n` | `deriv(income.revenue,2)` -> …,10.5,11.55,12.705 |
| `pct_change(x, n)` | `x/lag(x,n) - 1` | `pct_change(price.adj_close,1)` -> None,0.1,0.0909,-0.1333,0.3462 |
| `momentum(x, n)` | `x - lag(x,n)` | `momentum(price.adj_close,2)` -> …,10,-3,10 |
| `log_return(x)` | `log(x) - log(lag(x,1))` | `log_return(price.adj_close)` -> None,0.0953,0.087,-0.1431,0.2973 |
| `cum_return(r)` | `cumprod(1+r) - 1` | `cum_return(yoy(price.adj_close))` -> None,0.1,0.2,0.04,0.4 |
| `integral(x)` | `cumsum(x)` | `integral(income.net_income)` -> 8,17,24,36,51 |
| `trapz(x)` | `cumsum((x+lag(x,1))/2)` | `trapz(income.net_income)` -> None,8.5,16.5,26,39.5 |
| `roll_cov(x, y, n)` | `roll_mean(xy,n) - roll_mean(x,n)·roll_mean(y,n)` | `roll_cov(income.revenue,price.adj_close,3)` -> …,35,-12.34,44.10 |
| `roll_corr(x, y, n)` | `roll_cov/(roll_std·roll_std)` | `roll_corr(income.revenue,price.adj_close,3)` -> …,0.666,-0.264,0.385 |
| `beta(x, b, n)` | `roll_cov(x,b,n)/roll_var(b,n)` | `beta(price.adj_close,income.revenue,3)` -> …,0.317,-0.093,0.273 |
| `roll_zscore(x, n)` | `(x-roll_mean)/roll_std` | `roll_zscore(price.adj_close,3)` -> …,1.0,-0.907,1.035 |
| `roll_range(x, n)` | `roll_max - roll_min` | validated |
| `roll_cv(x, n)` | `roll_std/roll_mean` | validated |
| `roll_geomean(x, n)` | `exp(roll_mean(log(x),n))` | `roll_geomean(price.adj_close,3)` -> …,54.85,55.57,60.22 |
| `autocorr(x, k, n)` | `roll_corr(x, lag(x,k), n)` | `autocorr(price.adj_close,1,3)` -> …,-0.247,-0.652 |
| `sharpe(r, n)` | `roll_mean(r,n)/roll_std(r,n)` (ROLLING) | `sharpe(yoy(price.adj_close),3)` -> …,0.145,0.422 |

---

## 2. Cross-sectional (X) — within a period, over the universe

Compute within the group `(time × universe)`, refined by a trailing
`by <field>`. Null cells are excluded (the group is the non-null members).

### Primitives (P)
| Function | Arity | Verified (XS) |
|---|---|---|
| `zscore(x)` | 1 | `zscore(income.revenue)` -> -1.026,0.146,1.317,-0.439 (sample std) |
| `rank(x)` | 1 | ascending, 1-based, avg ties; `rank(income.net_income) by meta.sector` -> 1,2,2,1 |
| `winsorize(x, p)` | 2 | clip to group `[p,1-p]` quantiles; `winsorize(income.revenue,0.25)` -> 150,200,200,150 |
| `xs_mean(x)` | 1 | `xs_mean(income.revenue)` -> 187.5 (all) |
| `xs_median(x)` | 1 | `xs_median(income.revenue)` -> 175 |
| `xs_sum(x)` | 1 | `xs_sum(income.revenue)` -> 750 |
| `xs_std(x)` | 1 | sample std; `xs_std(income.revenue)` -> 85.3913 |
| `xs_var(x)` | 1 | sample variance |
| `xs_min(x)` | 1 | `xs_min(income.revenue)` -> 100 |
| `xs_max(x)` | 1 | `xs_max(income.revenue)` -> 300 |
| `xs_count(x)` | 1 | non-null count; `xs_count(income.revenue)` -> 4 |
| `xs_quantile(x, q)` | 2 | `xs_quantile(income.revenue,0.5)` -> 200 |
| `xs_frac(cond)` | 1 | fraction true (market breadth); `xs_frac(income.net_income>12)` -> 0.5 |

### Derived (D) — stdlib `stats.trail`, `factor.trail`, `timeseries.trail`
| Function | Def | Verified (XS) |
|---|---|---|
| `pctile(x)` | `(rank(x)-1)/(xs_count(x)-1)` ∈ [0,1] | `pctile(income.revenue)` -> 0,0.667,1,0.333 |
| `demean(x)` | `x - xs_mean(x)` | `demean(income.revenue)` -> -87.5,12.5,112.5,-37.5 |
| `minmax(x)` | `(x-xs_min)/(xs_max-xs_min)` | `minmax(income.revenue)` -> 0,0.5,1,0.25 |
| `xs_range(x)` | `xs_max - xs_min` | `xs_range(income.revenue)` -> 200 |
| `xs_cv(x)` | `xs_std/xs_mean` | validated |
| `xs_mad(x)` | `xs_median(abs(x - xs_median(x)))` | `xs_mad(income.revenue)` -> 50 |
| `robust_zscore(x)` | `(x-xs_median)/(1.4826·xs_mad)` | `robust_zscore(income.revenue)` -> -1.012,0.337,1.686,-0.337 |
| `ntile(x, k)` | quantile bucket 1..k | `ntile(income.revenue,2)` -> 1,2,2,1 |
| `scale(x)` | `x/xs_sum(abs(x))` (L1-normalize) | `scale(income.net_income)` -> 0.167,0.5,0.25,0.083 |
| `neutralize(x, f)` | single-factor residual | `neutralize(income.net_income,income.revenue)` -> -1,14.43,-5.14,-8.29 |
| `xs_pvar(x)` | population variance | validated |
| `xs_cov(a, b)` | population covariance | validated |
| `xs_corr(a, b)` | `xs_cov/sqrt(xs_pvar·xs_pvar)` | `xs_corr(income.net_income,income.revenue)` -> 0.3614 |
| `ic(signal, fwd)` | `xs_corr(signal, fwd)` (information coefficient) | validated |

Sector-neutral idiom: `zscore(gross_profitability) by meta.sector`.

---

## 3. Elementwise (E) — cell-wise scalar math

### Primitives (P)
| Function | Arity | Note | Verified (XS) |
|---|---|---|---|
| `sqrt(x)` | 1 | null for x<0 | `sqrt(income.revenue)` -> 10,14.14,17.32,12.25 |
| `abs(x)` | 1 | | `abs(income.net_income-100)` -> 90,70,85,95 |
| `log(x)` | 1 | natural; null for x≤0 | `log(income.revenue)` -> 4.605,5.298,5.704,5.011 |
| `exp(x)` | 1 | eˣ | `exp(income.net_income/100)` -> 1.105,1.350,1.162,1.051 |
| `sin/cos/tan(x)` | 1 | radians (transcendental primitives) | validated |
| `asin/acos/atan(x)` | 1 | inverse trig | validated |
| `floor(x)` | 1 | round down | `floor(income.revenue/7)` -> 14,28,42,21 |
| `ceil(x)` | 1 | round up | `ceil(income.revenue/7)` -> 15,29,43,22 |
| `round(x)` | 1 | nearest integer | `round(income.revenue/7)` -> 14,29,43,21 |
| `clamp(x, lo, hi)` | 3 | clip to [lo,hi] (literals) | `clamp(income.net_income,8,20)` -> 10,20,15,8 |
| `min(a, b)` | 2 | cell-wise pair min | `min(income.net_income,income.operating_income)` -> 10,30,15,5 |
| `max(a, b)` | 2 | cell-wise pair max | `max(...)` -> 18,50,33,12 |
| `erf(x)` | 1 | Gauss error function (|err|≤1.5e-7) | `erf(income.net_income/20)` -> 0.521,0.966,0.711,0.276 |
| `norm_ppf(p)` | 1 | inverse-normal / probit; p≤0→-inf, p≥1→+inf | `norm_ppf(income.net_income/40)` -> -0.675,0.675,-0.319,-1.150 |
| `count(b1..bk)` | 1..99 | sum of bool flags; **null-propagating** | `count(a>1,b>1)` — any null → null |
| `count_true(b1..bk)` | 1..99 | count of true flags, **null→false** | `count_true(a>1,b>1)` — null-tolerant |

Bare literals lift to constants: `log(10)` -> 2.3026 (verified).

### Derived (D) — stdlib `math.trail`, `transform.trail`
| Function | Def | Verified (XS) |
|---|---|---|
| `sign(x)` | `1 if x>0 else -1 if x<0 else 0` | `sign(income.net_income-12)` -> -1,1,1,-1 |
| `square(x)` / `cube(x)` | `x*x` / `x*x*x` | `square(income.net_income)` -> 100,900,225,25 |
| `reciprocal(x)` | `1/x` | `reciprocal(income.revenue)` -> 0.01,0.005,0.0033,0.0067 |
| `cbrt(x)` | `sign(x)·abs(x)^(1/3)` (handles neg) | `cbrt(income.net_income-12)` -> -1.26,2.62,1.44,-1.91 |
| `hypot(a, b)` | `sqrt(a²+b²)` | `hypot(income.net_income,income.operating_income)` -> 20.59,58.31,36.25,13 |
| `log10/log2(x)` | `log(x)/log(10 or 2)` | `log10(income.revenue)` -> 2,2.301,2.477,2.176 |
| `logb(x, b)` | `log(x)/log(b)` | validated |
| `log1p(x)` / `expm1(x)` | `log(1+x)` / `exp(x)-1` | `log1p(income.net_income)` -> 2.398,3.434,2.773,1.792 |
| `pow10(x)` | `10^x` | validated |
| `pow(x, y)` | `x^y` (named power) | `pow(income.net_income,2)` -> 100,900,225,25 |
| `sigmoid(x)` | `1/(1+exp(-x))` | `sigmoid(income.net_income-12)` -> 0.119,1.0,0.953,0.001 |
| `logit(p)` | `log(p/(1-p))` | `logit(income.net_income/40)` -> -1.099,1.099,-0.511,-1.946 |
| `softplus(x)` | `log(1+exp(x))` | `softplus(income.net_income/10)` -> 1.313,3.049,1.701,0.974 |
| `normal_cdf(x)` | `0.5·(1+erf(x/√2))` | `normal_cdf(income.net_income/20)` -> 0.691,0.933,0.773,0.599 |
| `sinh/cosh/tanh(x)` | via `exp` | `tanh(income.net_income/20)` -> 0.462,0.905,0.635,0.245 |
| `asinh/acosh/atanh(x)` | via `log`/`sqrt` | validated |
| `deg2rad/rad2deg(x)` | `x·π/180` / `x·180/π` | validated |
| `signed_log(x)` | `sign(x)·log1p(abs(x))` | `signed_log(income.net_income-12)` -> -1.099,2.944,1.386,-2.079 |
| `signed_power(x, a)` | `sign(x)·abs(x)^a` | `signed_power(income.net_income-12,0.5)` -> -1.41,4.24,1.73,-2.65 |
| `clip_lower(x, lo)` / `clip_upper(x, hi)` | `max(x,lo)` / `min(x,hi)` | validated |
| `to_bps(x)` / `to_pct(x)` | `x·10000` / `x·100` | `to_bps(income.net_income/income.revenue)` -> 1000,1500,500,333.3 |
| `indicator(c)` | `1 if c else 0` | `indicator(income.net_income>12)` -> 0,1,1,0 |
| `between(x, lo, hi)` | `x>=lo and x<=hi` | `indicator(between(income.net_income,8,20))` -> 1,0,1,0 |
| `pi()` / `tau()` / `euler()` | zero-arg constant macros | validated |

---

## 4. Temporal / calendar (E) — over a datetime

Operate on a datetime value: the panel atom `time`, a source date column inside an
`@ align(...)`, or any datetime field. Double as calendar factors and alignment
reductions.

| Function | Arity | Verified |
|---|---|---|
| `year(t)` | 1 | `year(time)` -> 2023,… |
| `month(t)` | 1 | `month(time)` -> 12,… |
| `quarter(t)` | 1 | `quarter(time)` -> 1,2,3,4 on quarterly time |
| `day(t)` | 1 | `day(time)` -> 31 |
| `truncate(t, unit)` | 2 | `year(truncate(time,"1y"))` -> 2023 (unit = duration string) |
| `datediff(a, b [, unit])` | 2..3 | whole units; unit ∈ days\|hours\|minutes\|seconds (default days) |

Seasonality flag: `indicator(quarter(time) == 4)` -> 0,0,0,1 (verified).

---

## 5. Frequency alignment & aggregation (T)

Move a field between native and target frequencies (target set by `at <freq>`).

| Function | Layer | Note | Verified |
|---|---|---|---|
| `resample(x, freq, agg)` | P | re-bucket to `freq`, reduce each bucket by `agg` | `resample(price.adj_close,"annual","max")` |
| `asof(x)` | P | upsample: carry last known value forward | see §1 |
| `to_annual/quarterly/monthly/daily(x)` | D | resample to that freq; `agg` = kind default unless 2nd arg overrides | `to_annual(income.revenue)` on quarterly -> yearly sums |
| `ttm(x)` | P(desugar) | trailing 12 months, kind-aware (flow=sum, stock=last) | `ttm(income.revenue)` at quarterly -> trailing-4Q sums |
| `trailing(x, "1y")` | P(desugar) | trailing duration window, kind-aware | `trailing(income.revenue,"1y")` = same as ttm at quarterly |
| `roll_*(x, "1y")` | P | duration-string window on any rolling reducer | `roll_sum(income.revenue,"1y")` |

`freq` ∈ `annual quarterly monthly weekly daily hourly minute`. `agg` (the
aggregation library): basic `sum mean last first min max count`; distribution
`median std var skew kurtosis quantile range`; multiplicative `prod compound
geomean`; change `change`. An unknown agg is `E-AGG-UNKNOWN`, an unknown freq
`E-FREQ-UNKNOWN`.

---

## 6. Model-context (M)

| Function | Legal position | Verified |
|---|---|---|
| `weighted_score()` | complete RHS of a model statement only | run of the model in `expressions.md §8` -> composite 1.0 / 0.588 |

Anywhere else → `E-MODEL-CONTEXT`.

---

## 7. Risk / performance (D) — per-entity, whole-series over a RETURN series

`risk.trail`; build the equity curve internally as `cumprod(1+r)` with the peak
seeded at 1.0. Take a periodic **return** series `r` (e.g. `yoy(price.adj_close)`).
`ppy` = periods per year (annualization). All broadcast back to every row.

| Function | Def | Verified (TS, r=yoy(price.adj_close), ppy=1) |
|---|---|---|
| `max_drawdown(r)` | `ts_min(_dd(r))` — worst peak-to-trough (≤0) | -0.1333 |
| `ann_sharpe(r, ppy)` | `ts_mean(r)/ts_std(r)·sqrt(ppy)` | 0.5153 |
| `sortino(r, ppy)` | downside-deviation denominator | 1.6927 |
| `calmar(r, ppy)` | `ts_mean(r)·ppy / (-max_drawdown(r))` | 0.757 |

---

## 8. Spec-listed but NOT in the reference implementation

Do **not** hand these to `eval`/`run` — they raise `E-FUNC-UNKNOWN`. Note them as
"spec-listed, not yet implemented" if a user asks.

| Name | Status |
|---|---|
| `normal_pdf(x)` | in `function-catalog.md` (derived `exp(-x²/2)/√(2π)`) but **not shipped in stdlib** → `E-FUNC-UNKNOWN`. Use the composition inline if needed: `exp(-x*x/2)/sqrt(2*pi())`. |
| `sply(x)` | post-1.0 (same period last year) |
| `roll_tail_mean(x, n, q)` | post-1.0 (historical CVaR) |
| `atan2 / arctan2` | listed R/P "add when needed"; not in `OPS` |
| Registered functions (`ols_residual`, `ff3_residual`, `dcf_two_stage`, `peter_lynch_fair_value`, …) | post-1.0 §7.6 ABI; not in 1.0 |

**Discovery tip:** never guess — call `functions_tool(query="name")` (empty result
= not implemented) or `functions_tool(axis="time-series")` to list a whole axis.
`layer` in the result tells you primitive vs derived.
