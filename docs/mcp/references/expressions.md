# Trail Expression Notation — Full Reference

Every construct here was verified with `eval_tool`/`validate_tool` against the
reference implementation. Verified results are shown inline as
`expr  ->  values`. Field vocabulary and per-function detail live in
`functions.md`; tool contracts in `tools.md`.

The evaluation domain is the **panel**: rows keyed by `(entity, time)`. Writing a
field name denotes the whole grid. All arithmetic is **cell-aligned** on
`(entity, time)`, never positional. A scalar literal broadcasts to every cell.

---

## 1. Lexical basics

| Token | Pattern | Examples |
|---|---|---|
| NUMBER | `[0-9][0-9_]*(.[0-9]+)?([eE][+-]?[0-9]+)?` | `0`, `2.5`, `200e6`, `1_000_000`, `4.5e-2` |
| STRING | `"[^"]*"` (double quotes only, no escapes) | `"NYSE"`, `"Total Debt"` |
| Boolean | `true` / `false` (lowercase only) | `true` |
| NAME | `[a-zA-Z_][a-zA-Z0-9_]*`, minus reserved words | `roa`, `fcf_per_share`, `_tmp` |
| DATE | `\d{4}-\d{2}(-\d{2})?` | `2010-01`, `2025-12-31` |
| DURATION | `\d+[dmy]` | `45d`, `3m`, `1y` |

- **Comments** run from `#` to end of line, anywhere whitespace may appear.
- `-3` is **not** a literal — unary minus is an operator. `.5` and `1.` are invalid.
- Whitespace/newlines are insignificant to the grammar EXCEPT that you separate
  statements and score cases by putting each on its own line (there is no `;`).

**Reserved words** (cannot be assignment/model/universe/signal/score names):
`and annual at backtest by cash costs daily def else equal export exposure
fallback false for from gate hold_band hourly if import in learn median minute
model monthly not on on_missing or pit_lag quarterly rebalance report score
select signal skip strategy tbills to top true universe validate value weekly
weight weighting weights where zero`. They ARE legal as trailing dotted-path
components (`meta.value` is fine; `value = …` is not).

---

## 2. Operators and precedence

Loosest → tightest; parenthesize to override. (`v if c else e` is loosest,
`@`/call/ref/literal tightest.)

| Lvl | Construct | Assoc | Note |
|---|---|---|---|
| 1 | `v if c else e` (ternary) | right | `2 if a else 1 if b else 0` = `2 if a else (1 if b else 0)` |
| 2 | `or` | left | |
| 3 | `and` | left | `a or b and c` = `a or (b and c)` |
| 4 | `not` (prefix) | — | `not a > 1` = `not (a > 1)` |
| 5 | `== != > < >= <=`, `in (...)` | non-chaining | `a < b < c` is a **syntax error** |
| 6 | `??` (coalesce) | left | `a ?? b + c` = `a ?? (b + c)` |
| 7 | `+ -` | left | |
| 8 | `* / %` | left | `a + b * c` = `a + (b * c)` |
| 9 | `^` (power) | right | `2 ^ 3 ^ 2` = `2 ^ (3 ^ 2)` = 512 |
| 10 | unary `-` (prefix) | — | `-x ^ 2` = `-(x ^ 2)`; write `(-x)^2` when you mean that |
| 11 | `@` qualifier (postfix) | — | binds tighter than unary minus and every binary op |
| 12 | call, ref, literal, `( )` | — | |

### Arithmetic `+ - * / % ^`
Cell-wise. `^` takes real exponents (`x ^ (1/3)` is a cube root). Division and
modulo by zero yield **null**, never ±∞.

```
income.revenue ^ 0.5                          # sqrt via power
  A=100 -> 10.0   B=200 -> 14.1421   C=300 -> 17.3205
income.net_income ^ (1/3)                      # real cube root, verified
(income.revenue - income.cogs) / income.revenue   # gross margin
  A(100,60) -> 0.4   B(200,90) -> 0.55   C(300,210) -> 0.3
```

### Comparisons → boolean panels
`meta.sector == "Tech"` (string equality is legal). Ordering on strings is
reserved (`E-TYPE-ORDER`, not currently checked). Comparisons do not chain.

### `in (...)` — membership against a literal list only
```
1 if meta.sector in ("Tech", "Energy") else 0     # list holds literals, not expressions
```

### Boolean `and` / `or` / `not` — three-valued logic
`true or null → true`, `false and null → false`, `true and null → null`,
`not null → null`.

### Ternary `v if c else e` — cell-wise, right-associating (first-match-wins)
```
3 if meta.market_cap > 10000 else 2 if meta.market_cap > 4000 else 1
  A(5000) -> 2   B(20000) -> 3   C(8000) -> 2   D(2000) -> 1
```

### Coalesce `x ?? y` — first non-null; left-associative chains `a ?? b ?? 0`
```
cash.stock_issued ?? 0        # null issuance -> 0.0    (verified: null -> 0.0, 5.0 -> 5.0)
```

---

## 3. Field references and namespaces

Field references are **always dotted**: `namespace.field`. A bare NAME resolves to
an earlier assignment in the same model, never to a field — with two reserved
exceptions: **`time`** (the period-end datetime of each cell) and **`entity`**
(the cell's canonical id) are the panel's own index columns and may be referenced
directly (`year(time)`, `entity == "SPY"`). This is what makes
`roa > lag(roa, 1)` unambiguous.

Standard namespaces (call `schema(namespace)` for the exact field list + kinds):

| Namespace | Content | Core fields (examples) |
|---|---|---|
| `income.*` | income statement (mostly `flow`) | `revenue`, `cogs`, `gross_profit`, `operating_income`, `net_income`, `interest_expense`, `income_tax_expense`, `income_before_tax`, `ebitda`, `depreciation_amortization`, `sga`, `eps_diluted` (per_share), `weighted_average_shares_diluted` (stock) |
| `balance.*` | balance sheet (`stock`) | `total_assets`, `current_assets`, `current_liabilities`, `total_liabilities`, `long_term_debt`, `total_debt`, `total_equity`, `retained_earnings`, `inventory`, `accounts_receivable`, `accounts_payable`, `net_fixed_assets`, `cash_and_equivalents`, `goodwill`, `minority_interest` |
| `cash.*` | cash flow (`flow`) | `cfo`, `capex`, `free_cash_flow`, `stock_issued`, `cfi`, `cff`, `dividends_paid`, `net_change_in_cash` |
| `price.*` | market data | `adj_close` (price), `dividends` (per_share) |
| `meta.*` | classification (`meta`) | `sector`, `exchange`, `market_cap`, `is_active`, `country` |

Other namespaces exist in the spec's data model but are provided by data-source
packages, not the core schema: `index.*`, `macro.*`, `estimates.*`, `insider.*`,
`ownership.*`, `sentiment.*`, and provider vocabularies like `gmd.*` (macro,
country-keyed) and `fmp.*` (snapshots). A source package contributes its
namespace + kinds via the `trail.schema` entry point; contributed fields validate
and resample by kind exactly like built-ins. **`eval`/`run` reject any field not
in the *active* schema** (`E-FIELD-UNKNOWN`), so only reference fields your
configured sources actually contribute.

### Field kinds and the stock/flow lint
Every field carries a **kind** — `flow`, `stock`, `level`, `price`, `rate`,
`ratio`, `return`, `index`, `per_share`, `meta`, … — that drives how it
aggregates across frequencies (§7). Dividing a `flow` by a point-in-time `stock`
without averaging emits the **warning** `W-KIND-STOCK-FLOW` (not an error):
```trail
inventory_turnover = income.cogs / balance.inventory          # W-KIND-STOCK-FLOW
inventory_turnover = income.cogs / avg2(balance.inventory)    # clean: flow over averaged stock
```

---

## 4. Null semantics

Missing data is **null** (not NaN); nothing raises at runtime.

1. **Arithmetic/comparison propagate null.** `null + 5 → null`; `null > 0.12 → null`
   (a boolean flag can be null).
2. **Division/modulo by zero → null.**
3. **Domain violations → null:** `sqrt(-4)`, `log(0)`, `(-8) ^ 0.5`.
4. **Null conditions do not match** in scores/ternaries — they fall through to the
   next case / `else`. **Exception:** if *every* case condition is null the score is
   **null** (not `else`). This is what makes `on_missing skip` meaningful.
5. **Coalesce** `x ?? y`.
6. Boolean connectives follow three-valued logic (§2).
7. **Windowed ops require a full window:** `roll_*(x, n)` is null until `n`
   consecutive periods exist. Cross-sectional ops skip nulls (the group is the
   non-null members).

**Practical consequence — `count` vs `count_true`:**
```
count(income.revenue > 120, cash.stock_issued > 0)
  # any null flag nulls the whole tally:  A -> None,  B -> None
count_true(income.revenue > 120, cash.stock_issued > 0)
  # null folded to false, present evidence still counts:  A -> 0,  B -> 1
(cash.stock_issued ?? 0) == 0            # guard: null issuance counts as "no issuance"
  A -> True   B -> False
```

---

## 5. The `by <group>` clause (cross-sectional grouping)

Cross-sectional operators compute within the **group** `(time × universe)`,
refined to `(time × universe × field-value)` by a trailing `by <dotted-field>`.
The clause attaches to the *call*, grouping that op's reducer — it does not leak
into the operands.

```
zscore(income.net_income / income.revenue) by meta.sector
  # A,B in "Tech"; C,D in "Health" -> each standardized within its sector
  A -> -0.7071   B -> 0.7071   C -> 0.7071   D -> -0.7071
rank(income.net_income) by meta.sector
  A -> 1.0   B -> 2.0   C -> 2.0   D -> 1.0
```

Without `by`, the group is the whole universe/panel for that period:
```
zscore(income.revenue)         # standardize across all entities that period
```

A `by` on a **`def` call** threads into every cross-sectional op inside the
inlined body that has no `by` of its own (`robust_zscore(x) by meta.sector`
groups the internal `xs_median`/`xs_mad` by sector).

---

## 6. The `@` field-reference qualifiers

`@` is a single postfix operator on a **schema field reference only** —
`(a + b) @ fmp` is a syntax error. A reference carries **at most one** `@`
qualifier; the three are mutually exclusive. Each binds tighter than every binary
operator and than unary minus.

| Qualifier | Form | Meaning |
|---|---|---|
| **`@ <source>`** | bare source name | pin to exactly one configured source, skipping cross-source coalescing |
| **`@ entity("<id>")`** | quoted canonical entity id | read one entity's series and broadcast it across the grid |
| **`@ align(<temporal-expr>)`** | temporal expr over the source's date columns | override the field's point-in-time alignment coordinate |

```trail
# source pin — source-disagreement forensics (validate-verified)
signal gap at annual = abs(income.revenue @ fmp - income.revenue @ edgar)
                     / (income.revenue @ edgar ?? income.revenue @ fmp)

# entity pin — excess return vs a benchmark entity (validate-verified;
# execution needs source-backed data, not a plain {"rows"} panel)
signal excess at annual = price.return - price.return @ entity("SPY")

# align override — align on filing-year instead of the provider default
signal s at annual = income.revenue @ align(year(filing_date))
```

The **frequency prefix** (§7) is a *prefix*, not an `@` qualifier, and composes
with any one of them: `annual.income.revenue @ edgar`,
`daily.price.adj_close @ entity("SPY")`.

`@ align(...)` may reference only the source's date-column names and temporal
functions (§`functions.md`); a schema field there is `E-ALIGN-EXPR`.

---

## 7. Frequency, alignment, and point-in-time (usage level)

Every source has a native **frequency** on the ladder
`annual < quarterly < monthly < weekly < daily < hourly < minute` (coarse→fine).
A `model`/`signal` computes on one **target frequency** set by `at <freq>`; when
omitted it defaults to the **finest** frequency among the fields referenced. The
target defines one step for every time-series op: `lag(x, 1)` is one target
period, `roll_mean(x, 3)` three target periods.

**Automatic alignment** by kind when a field's native frequency ≠ target:
- **coarser → target (upsample):** carry last known value forward (as-of). Safe
  for `stock`/`level`/`price`/`rate`/`ratio`/`index`/`meta`; a `flow`/`return`/
  `per_share` upsample warns `W-UPSAMPLE-FLOW`.
- **finer → target (downsample):** aggregate by kind — `flow`→sum, `stock`/`level`/
  `index`→last, `rate`/`ratio`→mean, `return`→compound.

**Frequency-qualified references** pull a field at a named native frequency, and
one block may use several: `quarterly.income.revenue / annual.balance.total_assets`
(validate-verified). Requesting a frequency no source serves is `E-FREQ-UNAVAILABLE`.

**Explicit transforms** override the automatic rule — `resample(x, freq, agg)`,
`to_annual/quarterly/monthly/daily(x)`, `asof(x)`, `ttm(x)`, `trailing(x, "1y")`,
and duration-string rolling windows (`roll_sum(x, "1y")`). See `functions.md §5`.

**Point-in-time.** `time` is the period-end label; a field may also carry a
**known-date** coordinate (filing/trade date). Under `panel.pit: auto` (default)
each value is placed on the grid by *when it became knowable*, so no expression
can observe a future period (invariant I4). This is engine-enforced; you cannot
write look-ahead. `estimates.*` are opinions knowable at time t, not future refs.

---

## 8. Declarations

A program is one or more declarations. Universe/model/signal names share one
namespace; redeclaring is `E-NAME-REBOUND`. Declaration order does not matter
across declarations; assignment order matters *within* a model.

### `universe` — a screened, point-in-time membership set
```trail
universe us_main = stocks where meta.exchange in ("NYSE","NASDAQ") and meta.is_active
universe nonfin  = us_main where meta.sector != "Financial Services"   # universes compose
universe liquid  = nonfin where meta.market_cap > 200e6
```
`root` is `stocks` (merged canonical listing), a pinned listing (`fmp.stocks`), or
another universe's name. The `where` clause is any boolean expression over fields.

### `signal` — a one-export executable (sugar for a single-export model)
```trail
signal value_composite on nonfin at annual =
    ( zscore(-price.adj_close) ) / 1        # runs to an (entity, time, value) series
```
`on UNIVERSE` and `at FREQ` are optional (defaults: full panel, finest freq).

### `model` — assignments + scores + exports
```
model NAME [on UNIVERSE] [at FREQ] {
    [desc STRING]
    [on_missing skip|zero|median]     # default skip; median parses but is treated as skip
    (assignment | score | export)+
}
```
Defaults: `at annual`, `on_missing skip`. `on` may be omitted only when the
program declares ≤1 universe. Verified full model:
```trail
model quality at annual {
    desc "Margin + return quality, weighted rollup"
    on_missing skip
    gm    = income.gross_profit / income.revenue      # assignment: visible to later stmts
    roa   = income.net_income / balance.total_assets
    f_roa = roa > 0
    export checklist = count(f_roa)                    # export = assignment that is output
    score gm_score weight 7 {                          # first-match-wins, mandatory else
        2 if gm > 0.4
        1 if gm > 0.2
        else 0
    }
    score roa_score weight 3 {
        1 if roa > 0.03
        else 0
    }
    export composite = weighted_score()                # weighted rollup of the scores
}
# run over {"rows": A(gm=.45,roa=.05), B(gm=.25,roa=.075)} ->
#   A: checklist=1, composite=1.0     B: checklist=1, composite=0.5882
```

- **Assignments** `name = expr` bind a panel visible to *later* statements; forward
  refs are `E-NAME-UNDEFINED`, rebinding `E-NAME-REBOUND`.
- **`export name = expr`** both binds and materializes into the output panel.
- **`export name`** (bare, no RHS) promotes an already-defined local to an output:
  ```trail
  roa = income.net_income / avg2(balance.total_assets)
  export roa
  ```
- **`score name weight N { <cases> else NUMBER }`** — ordered first-match-wins.
  Case values and `else` MUST be non-negative numeric **literals** (a non-literal
  there is a syntax error) — that lets `weighted_score()` compute each score's max
  statically. `weight` is metadata for `weighted_score()`. **Each case on its own
  line — no `;`.**

### `weighted_score()` — model-context rollup (legal only as a full RHS)
Over scores `(sᵢ, wᵢ)` with `maxᵢ` = the largest literal in scoreᵢ:
`numerator = Σ coalesce(sᵢ·wᵢ, 0)`, `denominator = Σ dᵢ` where under
`on_missing skip` a null score contributes 0 to the denominator (gaps
renormalize) and under `zero` it always contributes `wᵢ·maxᵢ`; result
`numerator/denominator` (null if denominator 0). A score whose every condition is
null is itself null. Used anywhere but a full model RHS → `E-MODEL-CONTEXT`.

### `def` — non-recursive expression macros (the stdlib mechanism)
```trail
def gross_margin(gross_profit, revenue) = gross_profit / revenue
def sharpe(r, n)                        = roll_mean(r, n) / roll_std(r, n)
```
Body is a single expression; inlined at each call site before compilation.
Non-recursive (`E-FUNC-RECURSION`), exact arity (`E-FUNC-ARITY`), no keyword
args/defaults/overloads. A parameter flowing into a window/quantile position must
receive a literal at the call site (invariant I3).

### `import "path.trail"` — source-level inclusion
Pulls another file's `def`s and `universe`s in as if written in place (its
`model`/`signal`/etc. execution units are skipped). Path resolves relative to the
importing file's directory. Diamond imports dedupe; cycles are `E-IMPORT-CYCLE`.
When validating via the MCP tool, pass `base_dir` as a *file path* whose directory
holds the import target.
```trail
import "factors.trail"
signal s on nonfin at annual = value_z() by meta.sector   # value_z() defined in factors.trail
```

### Parse-only (post-1.0): `strategy`, `backtest`, `learn`
These parse and validate but do not execute; running one raises
`E-PHASE-DEFERRED`. `fwd_return(horizon)` is legal only as a `learn.target`.

---

## 9. Diagnostics you will actually hit

| Code | Meaning |
|---|---|
| `E-FIELD-UNKNOWN` | field (or `by` target) not in the active schema |
| `E-FUNC-UNKNOWN` | function name not in the registry (also `normal_pdf`, `sply`, registered fns) |
| `E-FUNC-ARITY` | wrong argument count |
| `E-NAME-UNDEFINED` | bare name used before assignment (and not `time`/`entity`) |
| `E-NAME-REBOUND` | a name reused within a model, or a duplicate top-level decl |
| `E-MODEL-CONTEXT` | `weighted_score()` anywhere but a full model RHS |
| `E-UNIVERSE-UNKNOWN` | `on` names an unknown universe, or omitted with ≠1 universe |
| `E-SYNTAX` | parse error — most commonly a `;` used as a separator, or `a < b < c` |
| `W-KIND-STOCK-FLOW` | flow/stock division without `avg2`/`lag` (warning) |
| `W-UPSAMPLE-FLOW` | a flow/return/per_share field upsampled by as-of (warning) |

`validate` returns `valid:true` when there are no `severity:"error"` issues —
warnings still let a program run.
