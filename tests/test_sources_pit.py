"""End-to-end PIT through the loader: a source that declares a field's alignment coordinate
(describe_field.aligns_on) and emits the __date:* column gets row-shifted, and panel.pit:naive
(global) or options.pit:naive (per-source) turns it off."""
import datetime as dt

import polars as pl

from trail import sources
from trail.config import Config, SourceSpec
from trail.source import Capabilities, DataSource, FieldInfo, LoadRequest, date_col


class _FilingSource(DataSource):
    """Two fiscal years of quarterly revenue, each quarter filed ~45 days after quarter-end."""

    name = "filed"

    def load(self, request: LoadRequest) -> pl.DataFrame:
        time = [dt.datetime(y, m, d) for y in (2022, 2023)
                for m, d in ((3, 31), (6, 30), (9, 30), (12, 31))]
        filing = [dt.datetime(fy, fm, fd) for fy, fm, fd in (
            (2022, 5, 15), (2022, 8, 14), (2022, 11, 14), (2023, 2, 14),
            (2023, 5, 15), (2023, 8, 14), (2023, 11, 14), (2024, 2, 14))]
        return pl.DataFrame({
            "entity": ["X"] * 8, "time": time,
            "income.revenue": [1.0, 2, 3, 4, 5, 6, 7, 8], date_col("filing"): filing,
        }).with_columns([pl.col("time").cast(pl.Datetime("us")),
                         pl.col(date_col("filing")).cast(pl.Datetime("us"))])

    def available_fields(self, frequency=None):
        return {"income.revenue"}

    def describe_field(self, field):
        if field == "income.revenue":
            return FieldInfo(field, True, "direct", aligns_on="filing")
        return None

    def capabilities(self):
        return Capabilities(frequency="quarterly", frequencies=("quarterly",), pit=True)


def _cfg(pit="auto", src_options=None):
    return Config(
        sources={"filed": SourceSpec("filed", "filed-drv", src_options or {})},
        precedence={"default": ["filed"]}, pit=pit)


def test_loader_row_shifts_by_declared_coordinate(monkeypatch):
    monkeypatch.setattr(sources, "resolve_driver", lambda ref: _FilingSource)
    panel = sources.load_panel_for(_cfg(), {"income.revenue"}, target_freq="annual").sort("time")
    by_year = {r["time"].year: r["income.revenue"] for r in panel.iter_rows(named=True)}
    assert by_year[2023] == 10.0   # FY2022 known once Q4 filed 2023-02
    assert by_year[2024] == 26.0   # FY2023 known once Q4 filed 2024-02
    assert "__date:filing" not in panel.columns  # coordinate consumed


def test_panel_pit_naive_disables_the_shift(monkeypatch):
    monkeypatch.setattr(sources, "resolve_driver", lambda ref: _FilingSource)
    panel = sources.load_panel_for(_cfg(pit="naive"), {"income.revenue"}, target_freq="annual").sort("time")
    by_year = {r["time"].year: r["income.revenue"] for r in panel.iter_rows(named=True)}
    assert by_year[2022] == 10.0 and by_year[2023] == 26.0  # period-end placement
    assert 2024 not in by_year


def test_per_source_pit_naive_disables_the_shift(monkeypatch):
    monkeypatch.setattr(sources, "resolve_driver", lambda ref: _FilingSource)
    panel = sources.load_panel_for(
        _cfg(src_options={"pit": "naive"}), {"income.revenue"}, target_freq="annual").sort("time")
    by_year = {r["time"].year: r["income.revenue"] for r in panel.iter_rows(named=True)}
    assert by_year[2022] == 10.0 and by_year[2023] == 26.0
