"""`entities` scopes the FETCH, not just the result.

`load_panel_for` always accepted an `entities=` scope, but no MCP caller populated it, so every
query paid to load the whole configured universe. That makes refreshing a handful of names as
expensive as rebuilding everything. These pin the wiring and the cache-key hazard it introduces.
"""
from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from trail.mcp import _config_data, tools
from trail.source import Capabilities, DataSource, FieldInfo


class _RecordingSource(DataSource):
    """Stands in for a configured source; records the entity scope it was asked for."""

    name = "probe"
    requested: list = []

    def __init__(self, options=None):
        super().__init__(options or {})

    def load(self, request):
        _RecordingSource.requested.append(
            sorted(request.entities) if request.entities else None
        )
        wanted = list(request.entities) if request.entities else ["AAA", "BBB", "CCC"]
        return pl.DataFrame({
            "entity": wanted,
            "time": [dt.datetime(2024, 12, 31)] * len(wanted),
            "meta.market_cap": [100.0] * len(wanted),
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))

    def available_fields(self, frequency=None):
        return ["meta.market_cap"]

    def capabilities(self):
        return Capabilities(frequency="annual", frequencies=("annual",), pit=False)

    def describe_field(self, field):
        return FieldInfo(field, True, "direct", "test") if field == "meta.market_cap" else None

    def entities(self, universe=None):
        return ["AAA", "BBB", "CCC"]

    def close(self):
        pass


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    monkeypatch.setattr("trail.registry.resolve_driver", lambda name: _RecordingSource)
    monkeypatch.setattr("trail.sources.resolve_driver", lambda name: _RecordingSource)
    monkeypatch.setattr("trail.mcp._config_data.resolve_driver", lambda name: _RecordingSource)
    cfg = tmp_path / "trail.yaml"
    cfg.write_text(
        "sources:\n  probe:\n    driver: probe\n    options: {}\n"
        "precedence:\n  default: [probe]\n"
        "panel:\n  periods: [2024, 2024]\n",
        encoding="utf-8",
    )
    _RecordingSource.requested = []
    _config_data._CACHE.clear()
    yield str(cfg)
    _RecordingSource.requested = []
    _config_data._CACHE.clear()


def test_entities_scopes_the_fetch(config_path):
    tools.eval_tool("meta.market_cap", {"config": config_path}, entities=["BBB"], format="records")
    assert _RecordingSource.requested == [["BBB"]], (
        "the source must be asked for only the requested entity, not the whole universe"
    )


def test_without_entities_the_whole_universe_is_fetched(config_path):
    tools.eval_tool("meta.market_cap", {"config": config_path}, format="records")
    assert _RecordingSource.requested == [None]


def test_scoped_panel_is_not_served_to_an_unscoped_request(config_path):
    """The panel cache is keyed by (config, fields, freq); without `entities` in that key a
    narrow scoped load would be handed back for a request covering the full universe."""
    scoped = tools.eval_tool("meta.market_cap", {"config": config_path},
                             entities=["BBB"], format="records")
    full = tools.eval_tool("meta.market_cap", {"config": config_path}, format="records")

    assert len(scoped["records"]) == 1
    assert len(full["records"]) == 3, "full request must re-load, not reuse the scoped panel"
    assert _RecordingSource.requested == [["BBB"], None]


def test_same_scope_is_served_from_cache(config_path):
    tools.eval_tool("meta.market_cap", {"config": config_path}, entities=["BBB"], format="records")
    tools.eval_tool("meta.market_cap", {"config": config_path}, entities=["BBB"], format="records")
    assert _RecordingSource.requested == [["BBB"]], "identical scope must hit the cache"


def test_different_scopes_do_not_collide(config_path):
    tools.eval_tool("meta.market_cap", {"config": config_path}, entities=["AAA"], format="records")
    out = tools.eval_tool("meta.market_cap", {"config": config_path}, entities=["CCC"], format="records")
    assert out["records"][0]["entity"] == "CCC"
    assert _RecordingSource.requested == [["AAA"], ["CCC"]]


def test_fetch_tool_also_scopes(config_path):
    tools.fetch_tool(["meta.market_cap"], {"config": config_path},
                     entities=["CCC"], format="records")
    assert _RecordingSource.requested == [["CCC"]]
