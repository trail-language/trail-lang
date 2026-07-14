from pathlib import Path

import polars as pl
import pytest

from trail import ast
from trail.compiler import compile_model
from trail.fixtures import load_panel
from trail.pipeline import prepare
from trail.validate import validate

EXAMPLE = Path(__file__).parent.parent / "examples" / "performance.trail"


def _run():
    program = prepare(EXAMPLE.read_text())
    assert not [i for i in validate(program) if i.severity == "error"]
    universes = {d.name: d for d in program.decls if isinstance(d, ast.UniverseDecl)}
    model = next(d for d in program.decls if isinstance(d, ast.ModelDecl))
    return compile_model(model, universes).run(load_panel())


def test_composite_bounds_and_null_policy():
    result = _run()
    vals = [v for v in result["composite"].to_list() if v is not None]
    assert vals and all(0.0 <= v <= 1.0 for v in vals)
    # FFF has null interest_expense -> that score skipped, composite still non-null
    fff_2024 = result.filter((pl.col("entity") == "FFF") & (pl.col("period") == 2024))
    assert fff_2024["composite"][0] is not None


def test_revenue_cagr_known_value():
    # AAA grows 10%/yr exactly -> 4y CAGR = 0.10
    result = _run()
    aaa_2024 = result.filter((pl.col("entity") == "AAA") & (pl.col("period") == 2024))
    assert aaa_2024["revenue_growth"][0] == pytest.approx(0.10)


def test_growth_ordering_matches_fixture_design():
    # EEE shrinks (-3%/yr) so its growth is negative; AAA (10%) positive.
    result = _run()
    last = result.filter(pl.col("period") == 2024)
    by_sec = {r["entity"]: r["revenue_growth"] for r in last.iter_rows(named=True)}
    assert by_sec["EEE"] < 0 < by_sec["AAA"]
