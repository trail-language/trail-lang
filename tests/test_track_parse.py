from trail import ast
from trail.parser import parse_program


def test_track_model_sets_flag():
    prog = parse_program("track model m on u at annual { export a = 1.0 }")
    m = next(d for d in prog.decls if isinstance(d, ast.ModelDecl))
    assert m.track is True and m.name == "m" and m.universe == "u" and m.frequency == "annual"


def test_untracked_model_defaults_false():
    m = next(d for d in parse_program("model m { export a = 1.0 }").decls
             if isinstance(d, ast.ModelDecl))
    assert m.track is False


def test_track_signal_sets_flag():
    s = next(d for d in parse_program("track signal s = 1.0").decls
             if isinstance(d, ast.SignalDecl))
    assert s.track is True and s.name == "s"


def test_track_signal_with_universe_and_freq():
    s = next(d for d in parse_program("track signal s on u at annual = 1.0 + 2.0").decls
             if isinstance(d, ast.SignalDecl))
    assert s.track is True and s.universe == "u" and s.frequency == "annual"


def test_untracked_signal_defaults_false():
    s = next(d for d in parse_program("signal s = 1.0").decls if isinstance(d, ast.SignalDecl))
    assert s.track is False
