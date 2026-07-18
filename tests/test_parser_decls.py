from trail import ast
from trail.parser import parse_program

SRC = '''
universe us_main = stocks where meta.exchange in ("NYSE", "NASDAQ") and meta.market_cap > 200e6

model quality on us_main at annual {
    desc "margin quality"
    on_missing skip
    operating_margin = income.operating_income / income.revenue
    score om_score weight 7 {
        2 if operating_margin > 0.12
        1 if operating_margin > 0.05
        else 0
    }
    export composite = weighted_score()
}

signal breadth on us_main at monthly = xs_frac(price.adj_close > roll_mean(price.adj_close, 10))
'''


def test_program_shape():
    prog = parse_program(SRC)
    kinds = [type(d).__name__ for d in prog.decls]
    assert kinds == ["UniverseDecl", "ModelDecl", "SignalDecl"]


def test_universe():
    u = parse_program(SRC).decls[0]
    assert u.name == "us_main" and u.root == ("stocks",)
    assert isinstance(u.where, ast.BoolOp)


def test_model_statements_and_defaults():
    m = parse_program(SRC).decls[1]
    assert m.universe == "us_main" and m.frequency == "annual"
    assert m.desc == "margin quality" and m.on_missing == "skip"
    assign, score, export = m.statements
    assert isinstance(assign, ast.Assignment) and not assign.export
    assert isinstance(score, ast.ScoreDecl) and score.weight == 7
    assert len(score.cases) == 2 and score.default == ast.Literal(0)
    assert isinstance(export, ast.Assignment) and export.export and export.name == "composite"


def test_model_defaults_when_clauses_absent():
    m = parse_program("model x { a = 1 }").decls[0]
    assert m.universe is None and m.frequency is None and m.on_missing == "skip"


def test_signal_shape():
    sig = parse_program(SRC).decls[2]
    assert sig.name == "breadth" and sig.universe == "us_main" and sig.frequency == "monthly"
    assert isinstance(sig.expr, ast.Call) and sig.expr.name == "xs_frac"


def test_strategy_parses_as_opaque():
    src = ("strategy s { universe us_main\n signal a + b\n rebalance monthly\n "
           "select top 25\n weighting equal\n gate a > 1\n fallback tbills }")
    d = parse_program(src).decls[0]
    assert isinstance(d, ast.OpaqueDecl) and d.kind == "strategy" and d.name == "s"


def test_import_parses_to_import_decl():
    d = parse_program('import "metrics/base.trail"').decls[0]
    assert isinstance(d, ast.ImportDecl) and d.path == "metrics/base.trail"


def test_backtest_and_learn_opaque():
    src = ('backtest bt from 2010-01 to 2025-12 { benchmark index.spx\n pit_lag 45d\n report cagr, sharpe }\n'
           'learn weights for m { segment by meta.country\n target fwd_return(12)\n method shrink\n validate cv(5) }')
    kinds = [(d.kind, d.name) for d in parse_program(src).decls]
    assert kinds == [("backtest", "bt"), ("learn", "m")]
