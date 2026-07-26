from trail.mcp.tools import run_tool
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
    r1 = run_tool("factor", {"config": _config(tmp_path, "t0")}, program=MODEL, format="records")
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
