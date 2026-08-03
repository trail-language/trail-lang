"""view-of-view: a tracked view referencing another tracked view.

Covers dependency detection, coarse cross-view invalidation, materialization ordering, and cycle
rejection. The fixture source (trail.sources.fixture) supplies income.* etc.; a `providers.views`
block roots the store so P3's ViewSource injection resolves `views.*` during a dependent's build.
"""
import datetime as dt

import polars as pl
import pytest

from trail import ast
from trail.config import load_config
from trail.pipeline import prepare
from trail.store import LocalDiskViewStore, Manifest
from trail.views import ViewManager


def _write_config(path, store_dir, tok="t0", changed_cells=None):
    """A fixture-backed config whose `views` provider is rooted at `store_dir` (so ViewManager and the
    injected ViewSource share one store). `changed_cells` arms the fixture changefeed for P2 recompute."""
    opts = f"freshness: {tok}"
    if changed_cells is not None:
        cells = ", ".join(f'[{e}, "{t}"]' for e, t in changed_cells)
        opts += f", changed_cells: [{cells}]"
    path.write_text(
        "sources:\n  fixture:\n    driver: trail.sources.fixture\n"
        f"    options: {{{opts}}}\n"
        "precedence:\n  default: [fixture]\n"
        "panel:\n  periods: [2019, 2022]\n"
        "providers:\n  views:\n    driver: views_local\n"
        f"    options: {{dir: {store_dir}}}\n"
    )
    return load_config(str(path)), str(path)


def _decl(prog, name):
    return next(d for d in prog.decls
               if isinstance(d, (ast.ModelDecl, ast.SignalDecl)) and d.name == name)


def _seed_rating(store, tok="hr", built="2026-08-04T00:00:00", freshness="t0"):
    """Seed a leaf `rating` model view directly (no build), so dependents have something to reference."""
    t = dt.datetime(2019, 12, 31)
    df = pl.DataFrame({"entity": ["AAA", "BBB"], "time": [t, t], "views.rating.score": [1.0, 2.0]})
    store.write("rating", df, Manifest("rating", "model", ("score",), tok, ("fixture",),
                                       {"fixture": freshness}, built, ("views.rating.score",),
                                       frequency="annual"))


def _mgr(tmp_path, **cfg_kw):
    store_dir = tmp_path / "views"
    cfg, cfg_path = _write_config(tmp_path / "trail.yaml", store_dir, **cfg_kw)
    return ViewManager(LocalDiskViewStore({"dir": str(store_dir)}), cfg, cfg_path)


# --- T2: detection, fingerprints, closure ---------------------------------------------------------

def test_view_deps_detects_referenced_view(tmp_path):
    mgr = _mgr(tmp_path)
    comp = _decl(prepare("track model comp {\n  export c = views.rating.score + 1.0\n}", stdlib=False),
                 "comp")
    assert mgr._view_deps(comp, {}) == {"rating"}


def test_view_deps_detects_signal_view_reference(tmp_path):
    mgr = _mgr(tmp_path)
    s = _decl(prepare("track signal comp at annual = views.mom * 2.0", stdlib=False), "comp")
    assert mgr._view_deps(s, {}) == {"mom"}


def test_view_deps_empty_for_leaf(tmp_path):
    mgr = _mgr(tmp_path)
    leaf = _decl(prepare("track signal r at annual = income.revenue + 1.0", stdlib=False), "r")
    assert mgr._view_deps(leaf, {}) == set()


def test_view_dep_fingerprints_present_and_missing(tmp_path):
    mgr = _mgr(tmp_path)
    _seed_rating(mgr.store)
    fp = mgr._view_dep_fingerprints({"rating", "absent"})
    assert fp["rating"]                      # non-empty for a built view
    assert fp["absent"] == ""                # empty sentinel for an unbuilt view


def test_in_dep_closure_self_reference_is_cycle(tmp_path):
    mgr = _mgr(tmp_path)
    _seed_rating(mgr.store)
    # a view reachable from its own deps (here, itself) is a cycle
    assert mgr._in_dep_closure("comp", {"comp"}, {}, None) is True
    # a plain leaf dep does not reach back to the target
    assert mgr._in_dep_closure("comp", {"rating"}, {}, None) is False


# --- T3: record view-dep fingerprints at build; invalidate on dep change --------------------------

def test_build_records_view_dep_fingerprints_and_invalidates(tmp_path):
    mgr = _mgr(tmp_path)
    _seed_rating(mgr.store, tok="hr", built="2026-08-04T00:00:00")
    comp = _decl(prepare("track model comp {\n  export c = views.rating.score + 1.0\n}", stdlib=False),
                 "comp")
    mgr._full_build(comp, {}, None)
    mf = mgr.store.manifest("comp")
    assert set(mf.view_deps) == {"rating"}
    assert mf.view_deps["rating"] == "hr:2026-08-04T00:00:00"    # the dep's fingerprint at build time
    assert mgr.is_stale(comp, {}) is False                       # nothing changed -> served from store
    # the dep is rebuilt (new built_at) -> the dependent is coarsely invalidated
    _seed_rating(mgr.store, tok="hr", built="2026-08-05T09:00:00")
    assert mgr.is_stale(comp, {}) is True


# --- T4: serve materializes deps topologically; rejects cycles ------------------------------------

def _tracked_map(mgr, prog):
    return {d.name: d for d in mgr.tracked(prog)}


def test_serve_materializes_same_program_dep_first(tmp_path):
    mgr = _mgr(tmp_path)
    prog = prepare("track signal a at annual = income.revenue\n"
                   "track signal b at annual = views.a * 2.0", stdlib=False)
    decls = _tracked_map(mgr, prog)
    mgr.serve(decls["b"], {}, decls=decls)                    # serving b must build a first
    assert mgr.store.read("a") is not None                   # the dep was materialized
    b = mgr.store.read("b")
    assert b is not None and "views.b" in b.columns
    merged = mgr.store.read("a").join(b, on=["entity", "time"])
    assert (merged["views.b"] == merged["views.a"] * 2.0).all()   # b computed over the fresh dep


def test_serve_rejects_self_reference(tmp_path):
    mgr = _mgr(tmp_path)
    prog = prepare("track signal s at annual = views.s", stdlib=False)
    decls = _tracked_map(mgr, prog)
    with pytest.raises(ValueError, match="E-VIEW-CYCLE"):
        mgr.serve(decls["s"], {}, decls=decls)


def test_serve_rejects_mutual_cycle(tmp_path):
    mgr = _mgr(tmp_path)
    prog = prepare("track signal a at annual = views.b\n"
                   "track signal b at annual = views.a", stdlib=False)
    decls = _tracked_map(mgr, prog)
    with pytest.raises(ValueError, match="E-VIEW-CYCLE"):
        mgr.serve(decls["b"], {}, decls=decls)


# --- T5: end-to-end acceptance through run_tool ---------------------------------------------------

def _config_only(tmp_path):
    """Write the config (rooting the store) and return the run_tool `data` spec + the store dir."""
    store_dir = tmp_path / "views"
    _write_config(tmp_path / "trail.yaml", store_dir)
    return {"config": str(tmp_path / "trail.yaml")}, store_dir


def test_run_same_program_view_of_view(tmp_path):
    from trail.mcp.tools import run_tool
    data, store_dir = _config_only(tmp_path)
    prog = ("track signal a at annual = income.revenue\n"
            "track signal b at annual = views.a * 2.0")
    r = run_tool("b", data, program=prog, no_stdlib=True)
    assert "error" not in r, r
    store = LocalDiskViewStore({"dir": str(store_dir)})
    assert store.read("a") is not None                       # dependency was materialized on the way
    merged = store.read("a").join(store.read("b"), on=["entity", "time"])
    assert (merged["views.b"] == merged["views.a"] * 2.0).all()


def test_run_coarse_invalidation_propagates(tmp_path):
    from trail.mcp.tools import run_tool
    data, store_dir = _config_only(tmp_path)
    base = run_tool("b", data, program=("track signal a at annual = income.revenue\n"
                                        "track signal b at annual = views.a * 2.0"), no_stdlib=True)
    assert "error" not in base, base
    store = LocalDiskViewStore({"dir": str(store_dir)})
    first = store.read("b").sort("entity", "time")["views.b"].to_list()
    # the dependency's recipe changes -> a rebuilds -> b is coarsely invalidated and picks up new values
    r = run_tool("b", data, program=("track signal a at annual = income.revenue * 10.0\n"
                                     "track signal b at annual = views.a * 2.0"), no_stdlib=True)
    assert "error" not in r, r
    second = LocalDiskViewStore({"dir": str(store_dir)}).read("b").sort("entity", "time")["views.b"].to_list()
    assert second == pytest.approx([v * 10.0 for v in first])


def test_run_cross_run_layered(tmp_path):
    from trail.mcp.tools import run_tool
    data, store_dir = _config_only(tmp_path)
    # build `a` in one run, then reference it from a program that declares only `b`
    assert "error" not in run_tool("a", data, program="track signal a at annual = income.revenue",
                                   no_stdlib=True)
    r = run_tool("b", data, program="track signal b at annual = views.a * 3.0", no_stdlib=True)
    assert "error" not in r, r
    store = LocalDiskViewStore({"dir": str(store_dir)})
    merged = store.read("a").join(store.read("b"), on=["entity", "time"])
    assert (merged["views.b"] == merged["views.a"] * 3.0).all()
    assert set(store.manifest("b").view_deps) == {"a"}       # the cross-run dep is recorded


def test_run_cycle_returns_structured_error(tmp_path):
    from trail.mcp.tools import run_tool
    data, _ = _config_only(tmp_path)
    r = run_tool("b", data, program=("track signal a at annual = views.b\n"
                                     "track signal b at annual = views.a"), no_stdlib=True)
    assert r.get("error", {}).get("code") == "E-VIEW-CYCLE"
