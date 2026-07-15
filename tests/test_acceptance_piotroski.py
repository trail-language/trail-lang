"""Golden test: Trail-compiled Piotroski vs an independent numpy/pandas re-implementation.

The oracle mirrors Trail's null semantics exactly: a flag is undefined (NaN) where any
of its inputs is missing, and the F-score is undefined if any flag is undefined
(Trail's `count` propagates null). This validates the whole pipeline AND the null model.
"""
from pathlib import Path

import numpy as np

from trail import ast
from trail.compiler import compile_model
from trail.fixtures import load_panel
from trail.pipeline import prepare

EXAMPLE = Path(__file__).parent.parent / "examples" / "piotroski.trail"


def _oracle(pdf):
    out = {}
    for sec, g in pdf.groupby("entity"):
        g = g.sort_values("time").reset_index(drop=True)
        assets = g["balance.total_assets"]
        avg_assets = (assets + assets.shift(1)) / 2
        roa = g["income.net_income"] / avg_assets
        cfo_a = g["cash.cfo"] / avg_assets
        ltd = g["balance.long_term_debt"] / avg_assets
        cur = g["balance.current_assets"] / g["balance.current_liabilities"]
        gm = g["income.gross_profit"] / g["income.revenue"]
        to = g["income.revenue"] / avg_assets

        def flag(cond, *ins):
            f = cond.astype(float).to_numpy().copy()
            for s in ins:
                f[s.isna().to_numpy()] = np.nan
            return f

        flags = [
            flag(roa > 0, roa),
            flag(g["cash.cfo"] > 0, g["cash.cfo"]),
            flag(roa > roa.shift(1), roa, roa.shift(1)),
            flag(cfo_a > roa, cfo_a, roa),
            flag(ltd < ltd.shift(1), ltd, ltd.shift(1)),
            flag(cur > cur.shift(1), cur, cur.shift(1)),
            flag(g["cash.stock_issued"].fillna(0) == 0),  # coalesced -> never null
            flag(gm > gm.shift(1), gm, gm.shift(1)),
            flag(to > to.shift(1), to, to.shift(1)),
        ]
        total = np.column_stack(flags).sum(axis=1)  # NaN if any flag NaN
        for period, val in zip(g["time"], total):
            out[(sec, period.year)] = None if np.isnan(val) else int(val)
    return out


def test_fscore_matches_pandas_oracle():
    program = prepare(EXAMPLE.read_text())
    universes = {d.name: d for d in program.decls if isinstance(d, ast.UniverseDecl)}
    model = next(d for d in program.decls if isinstance(d, ast.ModelDecl))
    result = compile_model(model, universes).run(load_panel())

    expected = _oracle(load_panel().to_pandas())
    checked = 0
    for row in result.iter_rows(named=True):
        key = (row["entity"], row["time"].year)
        assert row["fscore"] == expected[key], (key, row["fscore"], expected[key])
        checked += 1
    assert checked == 48
    # Sanity: at least some cells are fully computable and land in 0..9.
    non_null = [v for v in expected.values() if v is not None]
    assert non_null and all(0 <= v <= 9 for v in non_null)
