from trail.mcp.tools import drop_tool, refresh_tool, run_tool, views_tool
from trail.store import LocalDiskViewStore

CONFIG = (
    "sources:\n  fixture:\n    driver: trail.sources.fixture\n    options: {{freshness: {tok}}}\n"
    "precedence:\n  default: [fixture]\n"
    "panel:\n  periods: [2019, 2022]\n"
)
MODEL = "track model factor at annual { export v = income.revenue + 1.0 }"


def _config(tmp_path, tok="t0"):
    p = tmp_path / "trail.yaml"
    p.write_text(CONFIG.format(tok=tok))
    return str(p)


def _store(tmp_path):
    return LocalDiskViewStore({"dir": str(tmp_path / ".trail" / "views")})


def test_run_tracked_model_persists_and_serves(tmp_path):
    cfg = _config(tmp_path)
    r1 = run_tool("factor", {"config": cfg}, program=MODEL, format="records")
    assert "error" not in r1 and r1["records"]
    assert "views.factor.v" in r1["records"][0]

    store = _store(tmp_path)
    assert store.read("factor") is not None
    built0 = store.manifest("factor").built_at

    r2 = run_tool("factor", {"config": cfg}, program=MODEL, format="records")
    assert store.manifest("factor").built_at == built0        # served from store, not rebuilt
    assert r2["records"] == r1["records"]


def test_run_tracked_model_recomputes_on_token_change(tmp_path):
    run_tool("factor", {"config": _config(tmp_path, "t0")}, program=MODEL, format="records")
    built0 = _store(tmp_path).manifest("factor").built_at
    r2 = run_tool("factor", {"config": _config(tmp_path, "t1")}, program=MODEL, format="records")
    assert _store(tmp_path).manifest("factor").built_at != built0   # token changed -> recompute
    assert "error" not in r2


def test_run_untracked_model_unaffected(tmp_path):
    cfg = _config(tmp_path)
    r = run_tool("plain", {"config": cfg},
                 program="model plain at annual { export v = income.revenue }", format="records")
    assert "error" not in r and r["records"]
    assert _store(tmp_path).read("plain") is None       # nothing persisted for an untracked model


def test_views_list_and_drop(tmp_path):
    cfg = _config(tmp_path)
    run_tool("factor", {"config": cfg}, program=MODEL, format="records")
    listed = views_tool({"config": cfg})
    entry = next(v for v in listed["views"] if v["name"] == "factor")
    assert entry["kind"] == "model" and entry["columns"] == ["views.factor.v"]
    assert drop_tool("factor", {"config": cfg})["dropped"] is True
    assert drop_tool("factor", {"config": cfg})["dropped"] is False       # already gone
    assert [v["name"] for v in views_tool({"config": cfg})["views"]] == []


def test_drop_forces_full_recompute(tmp_path):
    cfg = _config(tmp_path)
    run_tool("factor", {"config": cfg}, program=MODEL, format="records")
    built0 = _store(tmp_path).manifest("factor").built_at
    drop_tool("factor", {"config": cfg})
    run_tool("factor", {"config": cfg}, program=MODEL, format="records")   # rebuilds from scratch
    assert _store(tmp_path).manifest("factor").built_at != built0


def test_refresh_rebuilds(tmp_path):
    cfg = _config(tmp_path)
    run_tool("factor", {"config": cfg}, program=MODEL, format="records")
    built0 = _store(tmp_path).manifest("factor").built_at
    r = refresh_tool("factor", {"config": cfg}, program=MODEL)
    assert r["refreshed"] == "factor" and r["shape"][0] is not None
    assert _store(tmp_path).manifest("factor").built_at != built0         # eager rebuild


def test_view_tools_require_config():
    assert drop_tool("x", {"rows": []})["error"]["code"] == "E-ARGS"
    assert views_tool({"rows": []})["error"]["code"] == "E-ARGS"


def test_config_panel_edit_triggers_recompute(tmp_path):
    # H1/M2: editing the panel window is part of the view's identity; a stale frame must not be served,
    # and the rebuild must observe the NEW window (not a cached panel).
    p = tmp_path / "trail.yaml"
    store = _store(tmp_path)

    def write(periods):
        p.write_text("sources:\n  fixture:\n    driver: trail.sources.fixture\n"
                     "precedence:\n  default: [fixture]\n"
                     f"panel:\n  periods: {periods}\n")
        return str(p)

    run_tool("factor", {"config": write("[2019, 2022]")}, program=MODEL, format="records")
    m0 = store.manifest("factor")
    r = run_tool("factor", {"config": write("[2020, 2021]")}, program=MODEL, format="records")
    m1 = store.manifest("factor")
    assert m1.panel_key != m0.panel_key and m1.built_at != m0.built_at   # H1: recomputed
    years = {rec["time"][:4] for rec in r["records"]}
    assert years and years <= {"2020", "2021"}                          # M2: fresh narrow panel


def test_source_option_edit_triggers_recompute(tmp_path):
    # H1 residual: a per-source recipe edit (options.pit) is part of the view identity
    p = tmp_path / "trail.yaml"
    store = _store(tmp_path)

    def write(src_opts):
        p.write_text("sources:\n  fixture:\n    driver: trail.sources.fixture\n" + src_opts +
                     "precedence:\n  default: [fixture]\n"
                     "panel:\n  periods: [2019, 2022]\n")
        return str(p)

    run_tool("factor", {"config": write("")}, program=MODEL, format="records")
    pk0 = store.manifest("factor").panel_key
    run_tool("factor", {"config": write("    options: {pit: naive}\n")}, program=MODEL, format="records")
    assert store.manifest("factor").panel_key != pk0


def test_drop_rejects_traversing_name(tmp_path):
    # H2: a raw MCP-supplied name must not escape the store dir
    victim = tmp_path / "victim.parquet"
    victim.write_text("x")
    cfg = _config(tmp_path)
    r = drop_tool("../victim", {"config": cfg})
    assert r["error"]["code"] == "E-VIEW-NAME"
    assert victim.exists()                                              # untouched


def test_provider_without_dir_returns_structured_error(tmp_path):
    # M1: a misconfigured provider must not crash with an uncaught KeyError
    p = tmp_path / "trail.yaml"
    p.write_text("sources:\n  fixture:\n    driver: trail.sources.fixture\n"
                 "precedence:\n  default: [fixture]\n"
                 "providers:\n  views:\n    driver: views_local\n")
    r = views_tool({"config": str(p)})
    assert "error" in r


def test_lifecycle_persist_reuse_drop_recompute(tmp_path):
    # day 1 — build + persist
    cfg = _config(tmp_path, "t0")
    run_tool("factor", {"config": cfg}, program=MODEL, format="records")
    assert [v["name"] for v in views_tool({"config": cfg})["views"]] == ["factor"]
    built0 = _store(tmp_path).manifest("factor").built_at
    # same day, same data — served from the store, no recompute
    run_tool("factor", {"config": cfg}, program=MODEL, format="records")
    assert _store(tmp_path).manifest("factor").built_at == built0
    # day 2 — new data (source freshness flips) — recompute on next reference
    cfg = _config(tmp_path, "t1")  # same path/store dir; only the source token changed
    run_tool("factor", {"config": cfg}, program=MODEL, format="records")
    assert _store(tmp_path).manifest("factor").built_at != built0
    # force a full recompute by deleting the stored frame
    assert drop_tool("factor", {"config": cfg})["dropped"] is True
    assert views_tool({"config": cfg})["views"] == []
