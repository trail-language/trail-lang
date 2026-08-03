from trail.mcp.tools import fetch_tool, run_tool

CONFIG = ("sources:\n  fixture:\n    driver: trail.sources.fixture\n"
          "precedence:\n  default: [fixture]\n"
          "panel:\n  periods: [2019, 2022]\n")
TRACK = "track model rating at annual { export score = income.revenue }"


def _cfg(tmp_path):
    p = tmp_path / "trail.yaml"
    p.write_text(CONFIG)
    return str(p)


def test_views_field_is_queryable_after_track(tmp_path):
    cfg = _cfg(tmp_path)
    run_tool("rating", {"config": cfg}, program=TRACK, format="records")      # persist views.rating.score
    # a SEPARATE query references views.rating.score joined with a source field
    r = fetch_tool(["views.rating.score", "income.revenue"], {"config": cfg}, format="records")
    assert "error" not in r, r
    assert r["records"]
    row = r["records"][0]
    assert "views.rating.score" in row and "income.revenue" in row
    assert row["views.rating.score"] == row["income.revenue"]                # score == income.revenue


def test_non_views_load_unaffected(tmp_path):
    r = fetch_tool(["income.revenue"], {"config": _cfg(tmp_path)}, format="records")
    assert "error" not in r and r["records"]


def test_unstored_view_field_stays_unserved(tmp_path):
    # a views.* field that no stored view provides is still an error (nothing to serve)
    cfg = _cfg(tmp_path)
    run_tool("rating", {"config": cfg}, program=TRACK, format="records")
    r = fetch_tool(["views.nope.x"], {"config": cfg}, format="records")
    assert "error" in r


def test_panel_key_unchanged_by_views_reference(tmp_path):
    # the synthetic views source must never reach ViewManager's config: a tracked view's stored
    # panel_key stays identical whether or not another query references views.*
    from trail.store import LocalDiskViewStore
    cfg = _cfg(tmp_path)
    run_tool("rating", {"config": cfg}, program=TRACK, format="records")
    store = LocalDiskViewStore({"dir": str(tmp_path / ".trail" / "views")})
    pk1 = store.manifest("rating").panel_key
    fetch_tool(["views.rating.score", "income.revenue"], {"config": cfg}, format="records")  # injects views src
    run_tool("rating", {"config": cfg}, program=TRACK, format="records")                     # re-serve
    assert store.manifest("rating").panel_key == pk1                                          # no churn


def test_view_declared_without_at_is_queryable(tmp_path):
    # a tracked model with no `at` stores frequency=None; a bare reference must still route (the reviewer's
    # HIGH case: discovery advertised it but routing failed E-FIELD-UNSERVED)
    cfg = _cfg(tmp_path)
    run_tool("r2", {"config": cfg}, program="track model r2 { export s = income.revenue }", format="records")
    r = fetch_tool(["views.r2.s"], {"config": cfg}, format="records")
    assert "error" not in r and r["records"]


def test_screen_over_views_field(tmp_path):
    cfg = _cfg(tmp_path)
    run_tool("rating", {"config": cfg}, program=TRACK, format="records")
    r = fetch_tool(["views.rating.score"], {"config": cfg},
                   where="views.rating.score > 0", format="records")
    assert "error" not in r
