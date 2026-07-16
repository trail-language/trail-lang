"""Gen A: the optional entities= load seam - opted-in sources receive it, legacy sources never do."""
import datetime as dt

import polars as pl

from trail import sources
from trail.config import Config, SourceSpec
from trail.source import DataSource


def _panel():
    return pl.DataFrame({
        "entity": ["AAA"], "time": [dt.datetime(2023, 12, 31)], "income.revenue": [1.0],
    }).with_columns(pl.col("time").cast(pl.Datetime("us")))


class _OptIn(DataSource):
    last_entities: object = "unset"

    def load(self, fields, *, periods=None, entities=None):
        type(self).last_entities = entities
        return _panel()


class _Legacy(DataSource):
    called = False

    def load(self, fields, *, periods=None):
        type(self).called = True
        return _panel()


def _cfg():
    return Config(sources={"s": SourceSpec("s", "drv")}, precedence={"default": ["s"]})


def test_accepts_entities_detects_named_param_and_kwargs_and_rejects_legacy():
    assert sources._accepts_entities(_OptIn.load)
    assert not sources._accepts_entities(_Legacy.load)

    class _Kwargs(DataSource):
        def load(self, fields, **kwargs):
            return _panel()

    assert sources._accepts_entities(_Kwargs.load)


def test_opted_in_source_receives_entities(monkeypatch):
    _OptIn.last_entities = "unset"
    monkeypatch.setattr(sources, "resolve_driver", lambda ref: _OptIn)
    sources.load_panel_for(_cfg(), {"income.revenue"}, entities=["AAA", "BBB"])
    assert _OptIn.last_entities == ["AAA", "BBB"]


def test_entities_omitted_when_caller_passes_none(monkeypatch):
    _OptIn.last_entities = "unset"
    monkeypatch.setattr(sources, "resolve_driver", lambda ref: _OptIn)
    sources.load_panel_for(_cfg(), {"income.revenue"})  # entities defaults to None
    assert _OptIn.last_entities is None


def test_legacy_source_never_receives_entities(monkeypatch):
    _Legacy.called = False
    monkeypatch.setattr(sources, "resolve_driver", lambda ref: _Legacy)
    # entities supplied at the API, but a legacy load(self, fields, *, periods) must not get it
    sources.load_panel_for(_cfg(), {"income.revenue"}, entities=["AAA"])
    assert _Legacy.called  # loaded without a TypeError; entities silently omitted
