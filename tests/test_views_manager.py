import pytest

from trail import ast
from trail.config import load_config
from trail.pipeline import prepare
from trail.store import LocalDiskViewStore
from trail.views import ViewManager, expr_hash

PROGRAM = "track signal s at annual = income.revenue + 1.0"


def _universes(prog):
    return {d.name: d for d in prog.decls if isinstance(d, ast.UniverseDecl)}


def _write_config(path, tok):
    path.write_text(
        "sources:\n  fixture:\n    driver: trail.sources.fixture\n"
        f"    options: {{freshness: {tok}}}\n"
        "precedence:\n  default: [fixture]\n"
        "panel:\n  periods: [2019, 2022]\n"
    )
    return load_config(str(path)), str(path)


@pytest.fixture
def fixture_config(tmp_path):
    return _write_config(tmp_path / "trail.yaml", "t0")


def test_expr_hash_stable_and_name_independent():
    a = prepare("track signal s at annual = income.revenue", stdlib=False)
    b = prepare("track signal other at annual = income.revenue", stdlib=False)
    da = next(d for d in a.decls if isinstance(d, ast.SignalDecl))
    db = next(d for d in b.decls if isinstance(d, ast.SignalDecl))
    assert expr_hash(da, {}) == expr_hash(db, {})


def test_expr_hash_changes_with_body():
    a = prepare("track signal s at annual = income.revenue", stdlib=False)
    b = prepare("track signal s at annual = income.revenue + 1.0", stdlib=False)
    da = next(d for d in a.decls if isinstance(d, ast.SignalDecl))
    db = next(d for d in b.decls if isinstance(d, ast.SignalDecl))
    assert expr_hash(da, {}) != expr_hash(db, {})


def test_panel_key_covers_precedence_and_source_options():
    from trail.config import Config, SourceSpec
    from trail.views import _panel_key
    base = Config(sources={"a": SourceSpec("a", "d", {})}, precedence={"default": ["a"]})
    pk = _panel_key(base)
    # a precedence edit changes the frame's identity
    assert _panel_key(Config(sources={"a": SourceSpec("a", "d", {})},
                             precedence={"default": ["a"], "meta": ["a"]})) != pk
    # a per-source options edit (e.g. options.pit) changes the frame's identity
    assert _panel_key(Config(sources={"a": SourceSpec("a", "d", {"pit": "naive"})},
                             precedence={"default": ["a"]})) != pk
    # periods / pit / strict still count
    assert _panel_key(Config(sources={"a": SourceSpec("a", "d", {})},
                             precedence={"default": ["a"]}, periods=(2019, 2022))) != pk


def test_materialize_writes_then_serves_stored(tmp_path, fixture_config):
    store = LocalDiskViewStore({"dir": str(tmp_path / "views")})
    prog = prepare(PROGRAM, stdlib=False)
    mgr = ViewManager(store, *fixture_config)
    assert mgr.materialize(prog, _universes(prog)) == ["s"]      # first build
    assert mgr.materialize(prog, _universes(prog)) == []         # token unchanged -> served, no recompute
    df = store.read("s")
    assert df is not None and "views.s" in df.columns
    mf = store.manifest("s")
    assert mf.kind == "signal" and mf.columns == ("views.s",) and mf.freshness == {"fixture": "t0"}


def test_materialize_recomputes_when_token_changes(tmp_path):
    store = LocalDiskViewStore({"dir": str(tmp_path / "views")})
    prog = prepare(PROGRAM, stdlib=False)
    ViewManager(store, *_write_config(tmp_path / "c0.yaml", "t0")).materialize(prog, _universes(prog))
    rebuilt = ViewManager(store, *_write_config(tmp_path / "c1.yaml", "t1")).materialize(prog, _universes(prog))
    assert rebuilt == ["s"]                                      # token changed -> recompute


def test_materialize_recomputes_when_expr_changes(tmp_path, fixture_config):
    store = LocalDiskViewStore({"dir": str(tmp_path / "views")})
    ViewManager(store, *fixture_config).materialize(
        prepare(PROGRAM, stdlib=False), {})
    changed = prepare("track signal s at annual = income.revenue + 2.0", stdlib=False)
    assert ViewManager(store, *fixture_config).materialize(changed, {}) == ["s"]
