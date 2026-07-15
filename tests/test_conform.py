import datetime as dt

import polars as pl
import pytest

from trail.config import ConfigError
from trail.sources import PanelConformanceWarning, conform_panel


def _t(*years):
    return [dt.datetime(y, 12, 31) for y in years]


def test_missing_time_errors_even_when_lenient():
    p = pl.DataFrame({"entity": ["A"], "income.revenue": [1.0]})
    with pytest.raises(ConfigError, match="E-SOURCE-PANEL"):
        conform_panel(p, {"income.revenue"}, strict=False)


def test_conforming_panel_passes_through_unchanged():
    p = pl.DataFrame({"entity": ["A"], "time": _t(2020), "income.revenue": [1.0]})
    out = conform_panel(p, {"income.revenue"}, strict=True)
    assert out.columns == ["entity", "time", "income.revenue"]


def test_strict_rejects_extra_columns_and_missing_fields():
    p = pl.DataFrame({"entity": ["A"], "time": _t(2020), "junk": [1]})
    with pytest.raises(ConfigError, match="E-SOURCE-PANEL"):
        conform_panel(p, {"income.revenue"}, strict=True)


def test_lenient_warns_and_coerces():
    p = pl.DataFrame({"entity": ["A"], "time": _t(2020), "junk": [1]})
    with pytest.warns(PanelConformanceWarning):
        out = conform_panel(p, {"income.revenue"}, strict=False)
    assert "junk" not in out.columns  # extra column dropped
    assert "income.revenue" in out.columns  # missing field added as null
    assert out["income.revenue"].null_count() == out.height


def test_non_temporal_time_is_a_deviation():
    p = pl.DataFrame({"entity": ["A"], "time": [2020], "income.revenue": [1.0]})
    with pytest.raises(ConfigError, match="E-SOURCE-PANEL"):
        conform_panel(p, {"income.revenue"}, strict=True)


def test_date_time_normalized_to_datetime():
    p = pl.DataFrame({"entity": ["A"], "time": [dt.date(2020, 12, 31)], "income.revenue": [1.0]})
    out = conform_panel(p, {"income.revenue"}, strict=True)
    assert isinstance(out.schema["time"], pl.Datetime)
