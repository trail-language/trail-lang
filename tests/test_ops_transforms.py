import datetime as dt

import polars as pl
import pytest

from trail import ast
from trail.compiler import compile_model
from trail.ops import ENTITY, TIME, build
from trail.pipeline import prepare
from trail.validate import validate

_Q = [dt.datetime(y, m, 1) for y in (2022, 2023) for m in (3, 6, 9, 12)]


def test_asof_carries_last_known_value_forward_per_entity():
    df = pl.DataFrame({
        ENTITY: ["X", "X", "X", "X", "Y", "Y", "Y", "Y"],
        TIME: _Q[:4] * 2,
        "v": [1.0, None, None, 4.0, None, 2.0, None, None],
    }).with_columns(pl.col(TIME).cast(pl.Datetime("us"))).sort([ENTITY, TIME])
    out = df.with_columns(build("asof", [pl.col("v")], {}, None).alias("o"))["o"].to_list()
    # leading null cannot be filled; each known value carries forward until the next; no cross-entity leak
    assert out[:4] == [1.0, 1.0, 1.0, 4.0]
    assert out[4:] == [None, 2.0, 2.0, 2.0]


def _run(program_src: str, panel: pl.DataFrame) -> pl.DataFrame:
    prog = prepare(program_src)
    assert not [i for i in validate(prog) if i.severity == "error"]
    model = next(d for d in prog.decls if isinstance(d, ast.ModelDecl))
    return compile_model(model, {}).run(panel)


def test_to_annual_defaults_aggregation_by_kind():
    panel = pl.DataFrame({
        ENTITY: ["X"] * 8,
        TIME: _Q,
        "income.revenue": [1.0, 2, 3, 4, 5, 6, 7, 8],       # flow -> sum
        "balance.total_assets": [100.0, 110, 120, 130, 140, 150, 160, 170],  # stock -> last
    }).with_columns(pl.col(TIME).cast(pl.Datetime("us"))).sort([ENTITY, TIME])
    out = _run(
        'model t {\n'
        '  export rev_a = to_annual(income.revenue)\n'
        '  export assets_a = to_annual(balance.total_assets)\n'
        '  export rev_mean = to_annual(income.revenue, "mean")\n'
        '}',
        panel,
    )
    row23 = out.filter(pl.col(TIME) == dt.datetime(2023, 12, 1)).to_dicts()[0]
    assert row23["rev_a"] == pytest.approx(26.0)     # flow summed over the year
    assert row23["assets_a"] == pytest.approx(170.0)  # stock -> last of the year
    assert row23["rev_mean"] == pytest.approx(6.5)    # explicit agg overrides the kind default
    row22 = out.filter(pl.col(TIME) == dt.datetime(2022, 12, 1)).to_dicts()[0]
    assert row22["rev_a"] == pytest.approx(10.0)
    assert row22["assets_a"] == pytest.approx(130.0)


def test_to_quarterly_is_identity_on_quarterly_flow():
    panel = pl.DataFrame({
        ENTITY: ["X"] * 8, TIME: _Q, "income.revenue": [1.0, 2, 3, 4, 5, 6, 7, 8],
    }).with_columns(pl.col(TIME).cast(pl.Datetime("us"))).sort([ENTITY, TIME])
    out = _run('model t { export q = to_quarterly(income.revenue) }', panel).sort(TIME)
    assert out["q"].to_list() == pytest.approx([1, 2, 3, 4, 5, 6, 7, 8])
