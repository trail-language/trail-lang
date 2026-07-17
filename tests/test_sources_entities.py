"""Gen A: the entities load seam - LoadRequest.entities reaches every source uniformly;
a source is free to ignore it."""
import datetime as dt

import polars as pl

from trail import sources
from trail.config import Config, SourceSpec
from trail.source import Capabilities, DataSource


def _panel():
    return pl.DataFrame({
        "entity": ["AAA"], "time": [dt.datetime(2023, 12, 31)], "income.revenue": [1.0],
    }).with_columns(pl.col("time").cast(pl.Datetime("us")))


class _OptIn(DataSource):
    last_entities: object = "unset"

    def load(self, request):
        type(self).last_entities = request.entities
        return _panel()

    def available_fields(self, frequency=None):
        return {"income.revenue"}

    def capabilities(self):
        return Capabilities(frequency="annual")


class _Legacy(DataSource):
    called = False

    def load(self, request):  # ignores request.entities
        type(self).called = True
        return _panel()

    def available_fields(self, frequency=None):
        return {"income.revenue"}

    def capabilities(self):
        return Capabilities(frequency="annual")


def _cfg():
    return Config(sources={"s": SourceSpec("s", "drv")}, precedence={"default": ["s"]})


def test_opted_in_source_receives_entities(monkeypatch):
    _OptIn.last_entities = "unset"
    monkeypatch.setattr(sources, "resolve_driver", lambda ref: _OptIn)
    sources.load_panel_for(_cfg(), {"income.revenue"}, entities=["AAA", "BBB"])
    assert _OptIn.last_entities == ("AAA", "BBB")  # request.entities is a tuple


def test_entities_omitted_when_caller_passes_none(monkeypatch):
    _OptIn.last_entities = "unset"
    monkeypatch.setattr(sources, "resolve_driver", lambda ref: _OptIn)
    sources.load_panel_for(_cfg(), {"income.revenue"})  # entities defaults to None
    assert _OptIn.last_entities is None


def test_source_ignoring_entities_still_loads(monkeypatch):
    _Legacy.called = False
    monkeypatch.setattr(sources, "resolve_driver", lambda ref: _Legacy)
    # entities supplied at the API; a source whose load() never reads request.entities must
    # still load cleanly - every source gets the same LoadRequest, reading it is optional
    sources.load_panel_for(_cfg(), {"income.revenue"}, entities=["AAA"])
    assert _Legacy.called
