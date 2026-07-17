import datetime as dt

import polars as pl
import pytest

from trail.catalog import describe
from trail.config import DEFAULT_CONFIG
from trail.source import (
    Capabilities,
    DataSource,
    FieldInfo,
    LoadRequest,
)
from trail.sources import FixtureSource
from trail.testing import assert_source_conforms


def test_datasource_is_abstract():
    with pytest.raises(TypeError):
        DataSource()  # load / available_fields / capabilities are abstract


def test_fixture_conforms():
    src = FixtureSource({})
    assert isinstance(src, DataSource)
    assert "income.revenue" in src.available_fields()  # discovery is core
    assert_source_conforms(src, {"income.revenue", "balance.total_assets"})


def test_close_is_idempotent():
    src = FixtureSource({})
    src.close()
    src.close()


class _FullSource(DataSource):
    name = "full"

    def load(self, request: LoadRequest):
        return pl.DataFrame(
            {
                "entity": ["A", "A", "B"],
                "time": [dt.datetime(2020, 12, 31), dt.datetime(2021, 12, 31), dt.datetime(2020, 12, 31)],
                "income.revenue": [1.0, 2.0, 3.0],
            }
        ).with_columns(pl.col("time").cast(pl.Datetime("us")))

    def available_fields(self, frequency=None):
        return {"income.revenue"}

    def describe_field(self, field):
        return FieldInfo(field, True, "direct") if field == "income.revenue" else None

    def entities(self, universe=None):
        return ["A", "B"]

    def capabilities(self):
        return Capabilities(frequency="annual", period_range=(2020, 2021), provenance="test")


def test_full_source_conforms():
    src = _FullSource({})
    assert src.available_fields() == {"income.revenue"}
    assert src.capabilities().frequency == "annual"
    assert src.entities() == ["A", "B"]
    assert_source_conforms(src, {"income.revenue"})


def test_optional_methods_have_defaults():
    """describe_field / entities / close have safe defaults on the core base."""

    class _Minimal(DataSource):
        def load(self, request):
            return pl.DataFrame(
                {"entity": ["A"], "time": [dt.datetime(2020, 12, 31)], "income.revenue": [1.0]}
            ).with_columns(pl.col("time").cast(pl.Datetime("us")))

        def available_fields(self, frequency=None):
            return {"income.revenue"}

        def capabilities(self):
            return Capabilities(frequency="annual")

    src = _Minimal({})
    assert src.describe_field("income.revenue") is None
    assert src.entities() == []
    assert_source_conforms(src, {"income.revenue"})


def test_catalog_source_detail_reports_discovery():
    text = str(describe(("fixture",), DEFAULT_CONFIG))
    assert "fixture" in text and "provides" in text
