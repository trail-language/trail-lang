"""Regressions for the overnight architecture review's verified P1s."""
import polars as pl

from trail import ast
from trail.compiler import compile_model
from trail.fixtures import load_panel
from trail.parser import parse_program
from trail.pipeline import prepare
from trail.validate import validate


def _codes(src):
    return [i.code for i in validate(parse_program(src))]


def _run_model(src, name):
    prog = prepare(src)
    assert not [i for i in validate(prog) if i.severity == "error"]
    universes = {d.name: d for d in prog.decls if isinstance(d, ast.UniverseDecl)}
    model = next(d for d in prog.decls if isinstance(d, ast.ModelDecl) and d.name == name)
    return compile_model(model, universes).run(load_panel())


def test_universe_composition_applies_ancestor_filters():
    # sub = base where ... must keep base's Tech filter (was: returned ALL sectors)
    out = _run_model(
        'universe base = stocks where meta.sector == "Tech"\n'
        "universe sub = base where meta.is_active\n"
        "model m on sub { export r = income.revenue }\n",
        "m",
    )
    assert set(out["entity"].unique().to_list()) == {"AAA", "BBB", "CCC"}  # Tech only


def test_pow_and_mod_domain_violations_are_null_not_nan():
    out = _run_model(
        "model m {\n"
        "  export bad_pow = (0 - 8) ^ 0.5\n"
        "  export bad_mod = income.revenue % 0\n"
        "  score s weight 1 { 1 if bad_pow > 0 else 0 }\n"
        "  export gate = weighted_score()\n"
        "}",
        "m",
    )
    assert out["bad_pow"].is_null().all()  # NaN would silently MATCH `> 0`
    assert out["bad_mod"].is_null().all()
    assert out["gate"].to_list() == [None] * out.height or set(out["gate"].to_list()) <= {0.0, None}


def test_backtest_of_strategy_is_not_a_rebind():
    src = (
        "strategy s1 { universe u signal income.revenue rebalance quarterly }\n"
        "backtest s1 from 2015-01-01 to 2024-12-31 { benchmark index.spx }\n"
    )
    assert "E-NAME-REBOUND" not in _codes(src)


def test_universe_root_validation():
    assert "E-UNIVERSE-UNKNOWN" in _codes("universe u = nosuch where meta.is_active\n"
                                          "model m on u { export r = income.revenue }")
    assert "E-UNIVERSE-CYCLE" in _codes("universe a = b where meta.is_active\n"
                                        "universe b = a where meta.is_active\n"
                                        "model m on a { export r = income.revenue }")


def test_weighted_score_only_as_whole_rhs():
    assert "E-MODEL-CONTEXT" in _codes(
        "model m { score s weight 1 { 1 if income.revenue > 0 else 0 }\n"
        "  export c = weighted_score() + 1 }"
    )
    assert "E-MODEL-CONTEXT" not in _codes(
        "model m { score s weight 1 { 1 if income.revenue > 0 else 0 }\n"
        "  export c = weighted_score() }"
    )
