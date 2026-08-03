import datetime as dt
import warnings

import polars as pl

from trail.sources import conform_panel


def _panel():
    return pl.DataFrame({"entity": ["A"], "time": [dt.datetime(2024, 1, 1)],
                         "views.rating.score100": [1.5]})


def test_conform_keeps_views_columns_lenient():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = conform_panel(_panel(), set(), strict=False)
    assert "views.rating.score100" in out.columns
    assert not any("unexpected" in str(x.message) for x in w)


def test_conform_views_columns_pass_strict():
    out = conform_panel(_panel(), set(), strict=True)   # must not raise E-SOURCE-PANEL
    assert "views.rating.score100" in out.columns
