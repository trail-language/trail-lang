import polars as pl
import pytest

from trail.config import ConfigError
from trail.sources import PanelConformanceWarning, conform_panel


def test_missing_period_errors_even_when_lenient():
    p = pl.DataFrame({"entity": ["A"], "income.revenue": [1.0]})
    with pytest.raises(ConfigError, match="E-SOURCE-PANEL"):
        conform_panel(p, {"income.revenue"}, strict=False)


def test_conforming_panel_passes_through_unchanged():
    p = pl.DataFrame(
        {"entity": ["A"], "period": [2020], "income.revenue": [1.0]}
    ).with_columns(pl.col("period").cast(pl.Int32))
    out = conform_panel(p, {"income.revenue"}, strict=True)
    assert out.columns == ["entity", "period", "income.revenue"]


def test_strict_rejects_extra_columns_and_missing_fields():
    p = pl.DataFrame({"entity": ["A"], "period": [2020], "junk": [1]}).with_columns(
        pl.col("period").cast(pl.Int32)
    )
    with pytest.raises(ConfigError, match="E-SOURCE-PANEL"):
        conform_panel(p, {"income.revenue"}, strict=True)


def test_lenient_warns_and_coerces():
    p = pl.DataFrame({"entity": ["A"], "period": ["2020"], "junk": [1]})
    with pytest.warns(PanelConformanceWarning):
        out = conform_panel(p, {"income.revenue"}, strict=False)
    assert "junk" not in out.columns  # extra column dropped
    assert "income.revenue" in out.columns  # missing field added as null
    assert out["income.revenue"].null_count() == out.height
    assert out["period"].to_list() == [2020]  # str period coerced to int
