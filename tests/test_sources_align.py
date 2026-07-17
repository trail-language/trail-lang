"""load_panel_for over multiple sources: field assignment by precedence + cross-frequency merge."""
import datetime as dt

import polars as pl
import pytest

import trail.schema as schema
from trail import sources
from trail.config import Config, SourceSpec
from trail.schema import FieldSpec
from trail.source import BROADCAST_ENTITY, Capabilities, DataSource


@pytest.fixture
def macro_plugin(monkeypatch):
    monkeypatch.setattr(schema, "_plugin_fields",
                        lambda: {"macro.risk_free": FieldSpec("macro.risk_free", "rate")})


@pytest.fixture
def gmd_plugin(monkeypatch):
    monkeypatch.setattr(schema, "_plugin_fields",
                        lambda: {"gmd.gdp": FieldSpec("gmd.gdp", "level")})


class _PxSource(DataSource):
    def load(self, request):
        return pl.DataFrame({
            "entity": ["AAA", "AAA"],
            "time": [dt.datetime(2023, 1, 3), dt.datetime(2023, 1, 4)],
            "price.adj_close": [10.0, 11.0],
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))

    def available_fields(self, frequency=None):
        return {"price.adj_close"}

    def describe_field(self, field):
        return None

    def entities(self, universe=None):
        return ["AAA"]

    def capabilities(self):
        return Capabilities(frequency="daily")


class _MacroSource(DataSource):
    def load(self, request):
        return pl.DataFrame({
            "entity": ["AAA"],
            "time": [dt.datetime(2022, 12, 31)],
            "balance.total_assets": [500.0],
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))

    def available_fields(self, frequency=None):
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


class _TwoStockPxSource(DataSource):
    def load(self, request):
        return pl.DataFrame({
            "entity": ["AAA", "AAA", "BBB", "BBB"],
            "time": [dt.datetime(2023, 1, 3), dt.datetime(2023, 1, 4)] * 2,
            "price.adj_close": [10.0, 11.0, 20.0, 21.0],
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))

    def available_fields(self, frequency=None):
        return {"price.adj_close"}

    def describe_field(self, field):
        return None

    def entities(self, universe=None):
        return ["AAA", "BBB"]

    def capabilities(self):
        return Capabilities(frequency="daily")


class _GlobalMacroSource(DataSource):
    """A single global series keyed by the broadcast sentinel - applies to every entity."""

    def load(self, request):
        return pl.DataFrame({
            "entity": [BROADCAST_ENTITY],
            "time": [dt.datetime(2022, 12, 31)],
            "macro.risk_free": [0.02],
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))

    def available_fields(self, frequency=None):
        return {"macro.risk_free"}

    def describe_field(self, field):
        return None

    def entities(self, universe=None):
        return [BROADCAST_ENTITY]

    def capabilities(self):
        return Capabilities(frequency="annual")


def test_load_panel_for_broadcasts_global_macro_onto_every_stock(macro_plugin, monkeypatch):
    # the spec worked example: price.return - macro.risk_free, macro being a single global series
    drivers = {"px-driver": _TwoStockPxSource, "macro-driver": _GlobalMacroSource}
    monkeypatch.setattr(sources, "resolve_driver", lambda ref: drivers[ref])
    cfg = Config(
        sources={"px": SourceSpec("px", "px-driver"), "macro": SourceSpec("macro", "macro-driver")},
        precedence={"default": ["px", "macro"]},
    )
    panel = sources.load_panel_for(cfg, {"price.adj_close", "macro.risk_free"},
                                   target_freq="daily").sort(["entity", "time"])
    assert panel.height == 4  # 2 stocks x 2 days; no sentinel row leaks through
    assert set(panel["entity"].to_list()) == {"AAA", "BBB"}
    assert panel["macro.risk_free"].to_list() == [0.02, 0.02, 0.02, 0.02]  # broadcast to every stock/day
    assert panel["price.adj_close"].to_list() == [10.0, 11.0, 20.0, 21.0]


class _StockSource(DataSource):
    def load(self, request):
        fields = request.fields
        cols = {"entity": ["AAA", "BBB"], "time": [dt.datetime(2022, 12, 31)] * 2}
        if "income.net_income" in fields:
            cols["income.net_income"] = [10.0, 20.0]
        if "meta.country" in fields:
            cols["meta.country"] = ["USA", "CAN"]
        return pl.DataFrame(cols).with_columns(pl.col("time").cast(pl.Datetime("us")))

    def available_fields(self, frequency=None):
        return {"income.net_income", "meta.country"}

    def describe_field(self, field):
        return None

    def entities(self, universe=None):
        return ["AAA", "BBB"]

    def capabilities(self):
        return Capabilities(frequency="annual")


class _CountryMacroSource(DataSource):
    def load(self, request):
        return pl.DataFrame({
            "entity": ["USA", "CAN"], "time": [dt.datetime(2022, 12, 31)] * 2, "gmd.gdp": [1000.0, 500.0],
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))

    def available_fields(self, frequency=None):
        return {"gmd.gdp"}

    def describe_field(self, field):
        return None

    def entities(self, universe=None):
        return ["USA", "CAN"]

    def capabilities(self):
        return Capabilities(frequency="annual", entity_dim="country", bridge_field="meta.country")


def test_load_panel_for_remaps_country_macro_onto_stocks(gmd_plugin, monkeypatch):
    # the model references income.net_income + gmd.gdp but NOT meta.country; the bridge is
    # auto-injected so the country source can be remapped onto each stock.
    drivers = {"stock-drv": _StockSource, "gmd-drv": _CountryMacroSource}
    monkeypatch.setattr(sources, "resolve_driver", lambda ref: drivers[ref])
    cfg = Config(
        sources={"stock": SourceSpec("stock", "stock-drv"), "gmd": SourceSpec("gmd", "gmd-drv")},
        precedence={"default": ["stock", "gmd"]},
    )
    panel = sources.load_panel_for(cfg, {"income.net_income", "gmd.gdp"}, target_freq="annual")
    d = {r["entity"]: r for r in panel.iter_rows(named=True)}
    assert d["AAA"]["gmd.gdp"] == 1000.0  # AAA -> USA
    assert d["BBB"]["gmd.gdp"] == 500.0   # BBB -> CAN
    assert d["AAA"]["income.net_income"] == 10.0
    assert set(panel["entity"].to_list()) == {"AAA", "BBB"}
