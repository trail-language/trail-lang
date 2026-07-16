"""load_panel_for over multiple sources: field assignment by precedence + cross-frequency merge."""
import datetime as dt

import polars as pl

from trail import sources
from trail.config import Config, SourceSpec
from trail.source import Capabilities, ExtendedDataSource


class _PxSource(ExtendedDataSource):
    def load(self, fields, *, periods=None):
        return pl.DataFrame({
            "entity": ["AAA", "AAA"],
            "time": [dt.datetime(2023, 1, 3), dt.datetime(2023, 1, 4)],
            "price.adj_close": [10.0, 11.0],
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))

    def available_fields(self):
        return {"price.adj_close"}

    def describe_field(self, field):
        return None

    def entities(self, universe=None):
        return ["AAA"]

    def capabilities(self):
        return Capabilities(frequency="daily")


class _MacroSource(ExtendedDataSource):
    def load(self, fields, *, periods=None):
        return pl.DataFrame({
            "entity": ["AAA"],
            "time": [dt.datetime(2022, 12, 31)],
            "balance.total_assets": [500.0],
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))

    def available_fields(self):
        return {"balance.total_assets"}

    def describe_field(self, field):
        return None

    def entities(self, universe=None):
        return ["AAA"]

    def capabilities(self):
        return Capabilities(frequency="annual")


def _two_source_config():
    return Config(
        sources={
            "px": SourceSpec("px", "px-driver"),
            "macro": SourceSpec("macro", "macro-driver"),
        },
        precedence={"default": ["px", "macro"]},
    )


def test_load_panel_for_merges_daily_price_and_annual_macro(monkeypatch):
    drivers = {"px-driver": _PxSource, "macro-driver": _MacroSource}
    monkeypatch.setattr(sources, "resolve_driver", lambda ref: drivers[ref])
    panel = sources.load_panel_for(
        _two_source_config(),
        {"price.adj_close", "balance.total_assets"},
        target_freq="daily",
    ).sort("time")
    assert panel.height == 2  # the daily grid, no annual phantom row
    assert set(panel.columns) == {"entity", "time", "price.adj_close", "balance.total_assets"}
    assert panel["price.adj_close"].to_list() == [10.0, 11.0]
    assert panel["balance.total_assets"].to_list() == [500.0, 500.0]  # as-of carried onto each day


def test_load_panel_for_single_source_no_target_is_passthrough(monkeypatch):
    monkeypatch.setattr(sources, "resolve_driver", lambda ref: _PxSource)
    cfg = Config(sources={"px": SourceSpec("px", "px-driver")}, precedence={"default": ["px"]})
    panel = sources.load_panel_for(cfg, {"price.adj_close"}).sort("time")
    assert panel.height == 2
    assert panel["price.adj_close"].to_list() == [10.0, 11.0]
