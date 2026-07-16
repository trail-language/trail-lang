import datetime as dt

import polars as pl
import pytest

from trail.ops import ENTITY, TIME, build

# 2 entities x 8 quarters (2022-2023), value = 1..8 so aggregations are distinguishable
_Q = [dt.datetime(y, m, 1) for y in (2022, 2023) for m in (3, 6, 9, 12)]
_DF = pl.DataFrame({
    ENTITY: ["X"] * 8 + ["Y"] * 8,
    TIME: _Q * 2,
    "v": [1.0, 2, 3, 4, 5, 6, 7, 8] * 2,
}).with_columns(pl.col(TIME).cast(pl.Datetime("us"))).sort([ENTITY, TIME])


def _col(expr):
    return _DF.with_columns(expr.alias("out"))["out"].to_list()


def test_resample_annual_sum_broadcasts_back_within_year():
    out = _col(build("resample", [pl.col("v"), "annual", "sum"], {}, None))
    # 2022 quarters sum to 1+2+3+4=10, 2023 to 5+6+7+8=26; broadcast to each row, per entity
    assert out == pytest.approx([10, 10, 10, 10, 26, 26, 26, 26] * 2)


def test_resample_annual_last_and_mean():
    last = _col(build("resample", [pl.col("v"), "annual", "last"], {}, None))
    assert last == pytest.approx([4, 4, 4, 4, 8, 8, 8, 8] * 2)
    mean = _col(build("resample", [pl.col("v"), "annual", "mean"], {}, None))
    assert mean == pytest.approx([2.5, 2.5, 2.5, 2.5, 6.5, 6.5, 6.5, 6.5] * 2)


def test_resample_does_not_cross_entities():
    out = _col(build("resample", [pl.col("v"), "annual", "sum"], {}, None))
    assert out[:8] == pytest.approx(out[8:])  # X and Y computed independently


def test_ttm_is_trailing_year_sum():
    # duration-window roll_sum over a 1-year trailing window (the ttm building block)
    out = _col(build("roll_sum", [pl.col("v"), "1y"], {}, None))
    assert out[3] == pytest.approx(10.0)   # end of 2022: 1+2+3+4
    assert out[7] == pytest.approx(26.0)   # end of 2023: 5+6+7+8 (rolling 4 quarters)


def test_duration_window_roll_mean():
    out = _col(build("roll_mean", [pl.col("v"), "1y"], {}, None))
    assert out[3] == pytest.approx(2.5)    # mean of 2022 quarters


def test_resample_and_ttm_through_the_model_pipeline():
    from trail import ast
    from trail.compiler import compile_model
    from trail.pipeline import prepare
    from trail.validate import validate

    prog = prepare(
        'model q {\n'
        '  export rev_ttm = ttm(income.revenue)\n'
        '  export rev_annual = resample(income.revenue, "annual", "sum")\n'
        '}'
    )
    assert not [i for i in validate(prog) if i.severity == "error"]

    panel = pl.DataFrame({
        ENTITY: ["X"] * 8,
        TIME: _Q,
        "income.revenue": [1.0, 2, 3, 4, 5, 6, 7, 8],
    }).with_columns(pl.col(TIME).cast(pl.Datetime("us"))).sort([ENTITY, TIME])
    model = next(d for d in prog.decls if isinstance(d, ast.ModelDecl))
    out = compile_model(model, {}).run(panel)

    end_2023 = out.filter(pl.col(TIME) == dt.datetime(2023, 12, 1)).to_dicts()[0]
    assert end_2023["rev_annual"] == pytest.approx(26.0)  # 2023 sum, broadcast
    assert end_2023["rev_ttm"] == pytest.approx(26.0)     # trailing 4 quarters
