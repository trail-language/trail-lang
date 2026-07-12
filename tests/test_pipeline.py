from trail import ast
from trail.compiler import compile_model
from trail.fixtures import load_panel
from trail.pipeline import prepare


def test_stdlib_is_implicit():
    # signed_log is a stdlib macro; prepare() makes it available with no explicit import.
    prog = prepare("model m { export s = signed_log(income.net_income) }")
    model = next(d for d in prog.decls if isinstance(d, ast.ModelDecl))
    res = compile_model(model, {}).run(load_panel())
    assert res.height == 48 and res["s"].null_count() == 0


def test_no_stdlib_opt_out():
    # without stdlib, a stdlib-only function is not defined -> stays an unknown call
    prog = prepare("model m { export s = signed_log(income.net_income) }", stdlib=False)
    calls = []
    def walk(e):
        if isinstance(e, ast.Call):
            calls.append(e.name)
            for a in e.args:
                walk(a)
    for d in prog.decls:
        if isinstance(d, ast.ModelDecl):
            for st in d.statements:
                if isinstance(st, ast.Assignment):
                    walk(st.expr)
    assert "signed_log" in calls  # not inlined; would fail validation as E-FUNC-UNKNOWN
