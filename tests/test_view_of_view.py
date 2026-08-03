"""view-of-view: a tracked view referencing another tracked view.

Covers dependency detection, coarse cross-view invalidation, materialization ordering, and cycle
rejection. The fixture source (trail.sources.fixture) supplies income.* etc.; a `providers.views`
block roots the store so P3's ViewSource injection resolves `views.*` during a dependent's build.
"""
import datetime as dt

import polars as pl

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
