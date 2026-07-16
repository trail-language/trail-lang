"""load_panel_for with frequency-qualified fields: a source serving multiple frequencies."""
import datetime as dt

import polars as pl
import pytest

from trail import sources
from trail.config import Config, ConfigError, SourceSpec
from trail.source import Capabilities, DataSource, ExtendedDataSource

_T = dt.datetime(2022, 12, 31)


class _DualFreq(ExtendedDataSource):
    calls: list = []

    def load(self, fields, *, periods=None, frequency=None):
        type(self).calls.append(frequency)
        val = {"annual": 100.0, "quarterly": 25.0}[frequency]
        return pl.DataFrame({"entity": ["AAA"], "time": [_T], "income.revenue": [val]}).with_columns(
            pl.col("time").cast(pl.Datetime("us")))

    def available_fields(self):
        return {"income.revenue"}

    def describe_field(self, field):
        return None

    def entities(self, universe=None):
        return ["AAA"]

    def capabilities(self):
        return Capabilities(frequency="annual", frequencies=("annual", "quarterly"))


class _LegacyAnnual(DataSource):
    def load(self, fields, *, periods=None):  # no frequency param -> single-frequency
        return pl.DataFrame({"entity": ["AAA"], "time": [_T], "income.revenue": [7.0]}).with_columns(
            pl.col("time").cast(pl.Datetime("us")))


def _cfg():
    return Config(sources={"s": SourceSpec("s", "d")}, precedence={"default": ["s"]})


def test_accepts_frequency_detection():
    assert sources._accepts_frequency(_DualFreq.load) is True
    assert sources._accepts_frequency(_LegacyAnnual.load) is False


def test_dual_frequency_source_serves_both_variants(monkeypatch):
    _DualFreq.calls = []
    monkeypatch.setattr(sources, "resolve_driver", lambda ref: _DualFreq)
    panel = sources.load_panel_for(
        _cfg(), {"annual.income.revenue", "quarterly.income.revenue"}, target_freq="annual")
    assert set(_DualFreq.calls) == {"annual", "quarterly"}  # one fetch per frequency
    row = panel.to_dicts()[0]
    assert row["annual.income.revenue"] == 100.0
    assert row["quarterly.income.revenue"] == 25.0  # single quarter summed to the annual bucket


def test_bare_and_annual_qualified_share_one_fetch(monkeypatch):
    _DualFreq.calls = []
    monkeypatch.setattr(sources, "resolve_driver", lambda ref: _DualFreq)
    # bare fetches the source default (annual); annual.* is the same fetch -> a single load call
    panel = sources.load_panel_for(
        _cfg(), {"income.revenue", "annual.income.revenue"}, target_freq="annual")
    assert _DualFreq.calls == ["annual"]
    row = panel.to_dicts()[0]
    assert row["income.revenue"] == 100.0 and row["annual.income.revenue"] == 100.0


def test_unavailable_frequency_raises(monkeypatch):
    monkeypatch.setattr(sources, "resolve_driver", lambda ref: _DualFreq)  # serves annual + quarterly only
    with pytest.raises(ConfigError, match="E-FREQ-UNAVAILABLE"):
        sources.load_panel_for(_cfg(), {"monthly.income.revenue"}, target_freq="monthly")


def test_legacy_single_frequency_source_still_bare_loads(monkeypatch):
    monkeypatch.setattr(sources, "resolve_driver", lambda ref: _LegacyAnnual)
    panel = sources.load_panel_for(_cfg(), {"income.revenue"}, target_freq="annual")
    assert panel.to_dicts()[0]["income.revenue"] == 7.0  # no frequency= handed to a legacy load
