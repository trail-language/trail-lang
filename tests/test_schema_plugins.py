import polars as pl
import pytest

import trail.schema as schema
from trail.catalog import catalog, describe
from trail.config import DEFAULT_CONFIG
from trail.parser import parse_program
from trail.schema import FieldSpec, active_schema, is_field, kind_of
from trail.sources import conform_panel
from trail.validate import validate


@pytest.fixture
def macro_plugin(monkeypatch):
    fake = {
        "macro.gdp": FieldSpec("macro.gdp", "level"),
        "macro.cpi": FieldSpec("macro.cpi", "index"),
    }
    monkeypatch.setattr(schema, "_plugin_fields", lambda: fake)
    return fake


def _codes(src):
    return [i.code for i in validate(parse_program(src))]


def test_active_schema_merges_plugin_fields(macro_plugin):
    a = active_schema()
    assert "income.revenue" in a  # core still present
    assert "macro.gdp" in a and "macro.cpi" in a  # plugin contributed
    assert is_field("macro.gdp")
    assert kind_of("macro.cpi") == "index"


def test_core_wins_on_collision(monkeypatch):
    monkeypatch.setattr(
        schema, "_plugin_fields",
        lambda: {"income.revenue": FieldSpec("income.revenue", "bogus")},
    )
    assert kind_of("income.revenue") == "flow"
    assert active_schema()["income.revenue"].kind == "flow"


def test_validate_accepts_plugin_field(macro_plugin):
    assert "E-FIELD-UNKNOWN" not in _codes("model m { export g = macro.gdp / macro.cpi }")


def test_validate_rejects_field_without_plugin():
    assert "E-FIELD-UNKNOWN" in _codes("model m { export g = macro.gdp }")


def test_conform_panel_keeps_plugin_columns(macro_plugin):
    panel = pl.DataFrame(
        {"security": ["USA"], "period": [2020], "macro.gdp": [1.0]}
    ).with_columns(pl.col("period").cast(pl.Int32))
    # strict: a plugin column must be recognized as in-schema, not an unexpected column
    out = conform_panel(panel, {"macro.gdp"}, strict=True)
    assert "macro.gdp" in out.columns


def test_catalog_surfaces_plugin_namespace(macro_plugin):
    assert "macro" in str(catalog(DEFAULT_CONFIG))
    detail = str(describe(("macro",), DEFAULT_CONFIG))
    assert "macro.gdp" in detail and "macro.cpi" in detail
