import polars as pl
import pytest

from trail.compiler import compile_expr, compile_model
from trail.fixtures import load_panel
from trail.parser import parse_expr, parse_program


def _eval_expr(src: str, defined=frozenset()) -> pl.DataFrame:
    return load_panel().with_columns(compile_expr(parse_expr(src), set(defined)).alias("out"))


def test_arithmetic_on_fields():
    df = _eval_expr("income.net_income / income.revenue")
    assert df["out"].to_list() == pytest.approx([0.12] * 48)


def test_div_by_zero_is_null():
    df = load_panel().with_columns(pl.lit(0.0).alias("z"))
    out = df.with_columns(compile_expr(parse_expr("income.revenue / z"), {"z"}).alias("out"))
    assert out["out"].null_count() == 48


def test_coalesce_fills_null_field():
    df = _eval_expr("cash.stock_issued ?? 0")
    aaa = df.filter(pl.col("entity") == "AAA")
    assert aaa["out"].to_list() == [0.0] * 8


def test_score_block_first_match_and_weighted_score():
    prog = parse_program('''
model m {
    margin = income.operating_income / income.revenue
    score s1 weight 3 { 2 if margin > 0.12
 1 if margin > 0.05
 else 0 }
    score s2 weight 1 { 1 if income.net_income > 0
 else 0 }
    export composite = weighted_score()
}
''')
    model = prog.decls[0]
    result = compile_model(model, {}).run(load_panel())
    # fixture: margin = 0.20 everywhere -> s1 = 2; net_income > 0 -> s2 = 1
    # composite = (2*3 + 1*1) / (2*3 + 1*1) = 1.0
    assert result["composite"].to_list() == pytest.approx([1.0] * 48)
    assert set(result.columns) == {"entity", "period", "composite"}


def test_on_missing_skip_renormalizes():
    prog = parse_program('''
model m {
    score s1 weight 3 { 2 if income.operating_income / income.revenue > 0.12
 else 0 }
    score s2 weight 1 { 1 if income.interest_expense > 0
 else 0 }
    export composite = weighted_score()
}
''')
    result = compile_model(prog.decls[0], {}).run(load_panel())
    fff = result.filter(pl.col("entity") == "FFF")  # interest_expense null -> s2 null -> skipped
    assert fff["composite"].to_list() == pytest.approx([1.0] * 8)  # 2*3 / 2*3


def test_universe_filter_applies():
    prog = parse_program('''
universe tech = stocks where meta.sector == "Tech"
model m on tech { export rev = income.revenue }
''')
    universes = {d.name: d for d in prog.decls[:1]}
    result = compile_model(prog.decls[1], universes).run(load_panel())
    assert set(result["entity"].unique().to_list()) == {"AAA", "BBB", "CCC"}


def test_zscore_scoped_to_universe():
    prog = parse_program('''
universe tech = stocks where meta.sector == "Tech"
model m on tech { export z = zscore(income.revenue) }
''')
    universes = {d.name: d for d in prog.decls[:1]}
    result = compile_model(prog.decls[1], universes).run(load_panel())
    per_period_mean = result.group_by("period").agg(pl.col("z").mean())
    for v in per_period_mean["z"].to_list():
        assert v == pytest.approx(0.0, abs=1e-9)  # z-scored within tech only
