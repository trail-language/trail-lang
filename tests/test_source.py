import datetime as dt

import polars as pl
import pytest

from trail.catalog import describe
from trail.config import DEFAULT_CONFIG
from trail.source import (
    Capabilities,
    DataSource,
    ExtendedDataSource,
    FieldInfo,
    SupportsCapabilities,
    SupportsDiscovery,
    SupportsUniverse,
)
from trail.sources import FixtureSource
from trail.testing import assert_source_conforms


def test_datasource_is_abstract():
    with pytest.raises(TypeError):
        DataSource()  # load is abstract


def test_fixture_is_core_tier_and_conforms():
    src = FixtureSource({})
    assert isinstance(src, DataSource)
    assert not isinstance(src, SupportsDiscovery)  # core-tier only
    assert_source_conforms(src, {"income.revenue", "balance.total_assets"})


def test_close_is_idempotent():
    src = FixtureSource({})
    src.close()
    src.close()


class _FullSource(ExtendedDataSource):
    name = "full"

    def load(self, fields, *, periods=None):
        return pl.DataFrame(
            {
                "entity": ["A", "A", "B"],
                "time": [dt.datetime(2020, 12, 31), dt.datetime(2021, 12, 31), dt.datetime(2020, 12, 31)],
                "income.revenue": [1.0, 2.0, 3.0],
            }
        ).with_columns(pl.col("time").cast(pl.Datetime("us")))

    def available_fields(self):
        return {"income.revenue"}

    def describe_field(self, field):
        return FieldInfo(field, True, "direct") if field == "income.revenue" else None

    def entities(self, universe=None):
        return ["A", "B"]

    def capabilities(self):
        return Capabilities(frequency="annual", period_range=(2020, 2021), provenance="test")


def test_extended_source_satisfies_all_protocols():
    src = _FullSource({})
    assert isinstance(src, SupportsDiscovery)
    assert isinstance(src, SupportsUniverse)
    assert isinstance(src, SupportsCapabilities)
    assert_source_conforms(src, {"income.revenue"})


def test_protocols_are_structural():
    class Duck:
        def available_fields(self):
            return set()

        def describe_field(self, field):
            return None

    assert isinstance(Duck(), SupportsDiscovery)
    assert not isinstance(object(), SupportsDiscovery)


def test_catalog_source_detail_reports_core_tier():
    text = str(describe(("fixture",), DEFAULT_CONFIG))
    assert "fixture" in text and "core-tier" in text
