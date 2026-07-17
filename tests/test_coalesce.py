"""Per-cell coalescing across a namespace precedence chain + live `@ source` pins (v1 phase 5).

The spec §5.1 worked example: precedence {income: [edgar, fmp]}, revenue - EDGAR 2015-2023,
FMP 2015-2024 -> EDGAR's values 2015-2023, FMP's 2024, with per-cell fall-through where EDGAR
has no value. Here source A skips 2020 (an absent cell) so the fall-through onto B is visible
inside the overlap, not only past A's range.
"""
import datetime as dt

import polars as pl
import pytest

from trail import sources
from trail.config import Config, ConfigError, SourceSpec, load_config
from trail.source import Capabilities, DataSource

# A serves 2015-2023 but SKIPS 2020 (a genuinely absent cell -> null after align -> falls to B).
_A_VALUES = {y: 100.0 + (y - 2015) for y in range(2015, 2024) if y != 2020}
# B serves the full 2015-2024, complete - distinct values so the winner is unambiguous.
_B_VALUES = {y: 200.0 + (y - 2015) for y in range(2015, 2025)}


def _annual_source(values_by_year, *, entity="AAA", entity_dim="entity", field="income.revenue"):
    """A fake annual, entity-keyed source serving one field for one entity."""

    class _S(DataSource):
        def load(self, request):
            years = sorted(values_by_year)
            return pl.DataFrame({
                "entity": [entity] * len(years),
                "time": [dt.datetime(y, 12, 31) for y in years],
                field: [values_by_year[y] for y in years],
            }).with_columns(pl.col("time").cast(pl.Datetime("us")))

        def available_fields(self, frequency=None):
            return {field}

        def describe_field(self, f):
            return None

        def entities(self, universe=None):
            return [entity]

        def capabilities(self):
            return Capabilities(frequency="annual", entity_dim=entity_dim)

    return _S


def _drivers(monkeypatch, **by_ref):
    monkeypatch.setattr(sources, "resolve_driver", lambda ref: by_ref[ref])


def _cfg(chain):
    return Config(
        sources={"a": SourceSpec("a", "drv-a"), "b": SourceSpec("b", "drv-b")},
        precedence={"income": list(chain), "default": list(chain)},
    )


def _by_year(panel, col):
    return {t.year: v for t, v in zip(panel["time"], panel[col])}


def test_two_source_per_cell_coalescing(monkeypatch):
    _drivers(monkeypatch, **{"drv-a": _annual_source(_A_VALUES), "drv-b": _annual_source(_B_VALUES)})
    panel = sources.load_panel_for(_cfg(["a", "b"]), {"income.revenue"},
                                   target_freq="annual").sort("time")
    got = _by_year(panel, "income.revenue")
    for y in range(2015, 2024):
        assert got[y] == _A_VALUES.get(y, _B_VALUES[y])  # A wins where it has a value
    assert got[2020] == _B_VALUES[2020]  # per-cell fall-through: A absent at 2020 -> B
    assert got[2024] == _B_VALUES[2024]  # beyond A's range -> B
    assert set(panel.columns) == {"entity", "time", "income.revenue"}  # no #tag leaks out


def test_source_pin_reads_exactly_one_source(monkeypatch):
    _drivers(monkeypatch, **{"drv-a": _annual_source(_A_VALUES), "drv-b": _annual_source(_B_VALUES)})
    # chain is [b, a] so coalescing would prefer B; the pin must still read A alone.
    panel = sources.load_panel_for(_cfg(["b", "a"]), {"income.revenue#a"},
                                   target_freq="annual").sort("time")
    got = _by_year(panel, "income.revenue#a")
    assert set(got) == set(_A_VALUES)  # exactly A's years (its own grid), no B fill
    for y, v in _A_VALUES.items():
        assert got[y] == v
    assert set(panel.columns) == {"entity", "time", "income.revenue#a"}


def test_field_pinned_and_coalesced_in_one_model(monkeypatch):
    _drivers(monkeypatch, **{"drv-a": _annual_source(_A_VALUES), "drv-b": _annual_source(_B_VALUES)})
    panel = sources.load_panel_for(_cfg(["a", "b"]), {"income.revenue", "income.revenue#a"},
                                   target_freq="annual").sort("time")
    assert set(panel.columns) == {"entity", "time", "income.revenue", "income.revenue#a"}
    coalesced = _by_year(panel, "income.revenue")
    pinned = _by_year(panel, "income.revenue#a")
    assert coalesced[2020] == _B_VALUES[2020] and coalesced[2024] == _B_VALUES[2024]
    assert pinned[2015] == _A_VALUES[2015]
    assert pinned[2020] is None  # A absent at 2020 -> the pin stays null (no coalescing)
    assert pinned[2024] is None  # A's range ends 2023


def test_pin_to_source_lacking_field_raises(monkeypatch):
    _drivers(monkeypatch, **{"drv-a": _annual_source(_A_VALUES), "drv-b": _annual_source(_B_VALUES)})
    # neither source serves price.adj_close; pinning to one that lacks it is E-PIN-UNSERVED.
    with pytest.raises(ConfigError, match="E-PIN-UNSERVED"):
        sources.load_panel_for(_cfg(["a", "b"]), {"price.adj_close#a"}, target_freq="annual")


def test_pin_to_unconfigured_source_raises(monkeypatch):
    _drivers(monkeypatch, **{"drv-a": _annual_source(_A_VALUES), "drv-b": _annual_source(_B_VALUES)})
    with pytest.raises(ConfigError, match="E-PIN-SOURCE-UNKNOWN"):
        sources.load_panel_for(_cfg(["a", "b"]), {"income.revenue#zzz"}, target_freq="annual")


def test_coalesce_across_mixed_dimensions_raises(monkeypatch):
    # a chain [entity-source, country-source] both serving the field cannot be coalesced per-cell.
    _drivers(monkeypatch,
             **{"drv-a": _annual_source(_A_VALUES),
                "drv-c": _annual_source({2022: 1000.0}, entity="USA", entity_dim="country")})
    cfg = Config(
        sources={"a": SourceSpec("a", "drv-a"), "c": SourceSpec("c", "drv-c")},
        precedence={"income": ["a", "c"], "default": ["a", "c"]},
    )
    with pytest.raises(ConfigError, match="E-COALESCE-DIM-MIXED"):
        sources.load_panel_for(cfg, {"income.revenue"}, target_freq="annual")


def test_single_source_chain_leaks_no_tag(monkeypatch):
    _drivers(monkeypatch, **{"drv-a": _annual_source(_A_VALUES), "drv-b": _annual_source(_B_VALUES)})
    # income routes only to A -> single-source serving chain: no coalescing, no #tag columns.
    cfg = Config(
        sources={"a": SourceSpec("a", "drv-a"), "b": SourceSpec("b", "drv-b")},
        precedence={"income": ["a"], "default": ["a"]},
    )
    panel = sources.load_panel_for(cfg, {"income.revenue"}, target_freq="annual").sort("time")
    assert set(panel.columns) == {"entity", "time", "income.revenue"}
    assert _by_year(panel, "income.revenue") == dict(_A_VALUES)  # exactly A, unchanged


_INVALID_NAME_YAML = """
sources:
  "bad.name":
    driver: trail.sources.fixture
precedence:
  default: ["bad.name"]
"""


def test_invalid_source_name_rejected(tmp_path):
    f = tmp_path / "trail.yaml"
    f.write_text(_INVALID_NAME_YAML)
    with pytest.raises(ConfigError, match="E-SOURCE-NAME"):
        load_config(str(f))


def _dated_annual_source(values_by_year, *, entity="AAA", field="income.revenue"):
    """An annual source that also emits a `__date:filing` coordinate (filed ~Feb the next year)
    and declares the field aligned on it - so @align can reference `filing`."""
    from trail.source import FieldInfo, date_col

    class _S(DataSource):
        def load(self, request):
            years = sorted(values_by_year)
            return pl.DataFrame({
                "entity": [entity] * len(years),
                "time": [dt.datetime(y, 12, 31) for y in years],
                field: [values_by_year[y] for y in years],
                date_col("filing"): [dt.datetime(y + 1, 2, 15) for y in years],
            }).with_columns([pl.col("time").cast(pl.Datetime("us")),
                             pl.col(date_col("filing")).cast(pl.Datetime("us"))])

        def available_fields(self, frequency=None):
            return {field}

        def describe_field(self, f):
            return FieldInfo(f, True, "direct", aligns_on="filing") if f == field else None

        def entities(self, universe=None):
            return [entity]

        def capabilities(self):
            return Capabilities(frequency="annual", pit=True)

    return _S


def test_align_override_on_a_coalesced_field(monkeypatch):
    # @align(truncate(filing,"1y")) is materialized PER TAG, then the tags coalesce. Both
    # contributors carry `filing`, so no E-ALIGN-UNKNOWN; A wins its years, B fills the rest.
    from trail.parser import parse_expr
    _drivers(monkeypatch, **{"drv-a": _dated_annual_source({2021: 100.0, 2022: 101.0}),
                             "drv-b": _dated_annual_source({2021: 200.0, 2022: 201.0, 2023: 202.0})})
    ov = {"income.revenue": parse_expr('income.revenue @ align(truncate(filing, "1y"))').align}
    panel = sources.load_panel_for(_cfg(["a", "b"]), {"income.revenue"},
                                   target_freq="annual", align_overrides=ov).sort("time")
    assert set(panel.columns) == {"entity", "time", "income.revenue"}  # no #tag / __date leak
    got = _by_year(panel, "income.revenue")
    assert got[2022] == 100.0   # A's FY2021, filed Feb-2022 -> 2022 decision row (A precedence)
    assert got[2023] == 101.0   # A's FY2022 -> 2023
    assert got[2024] == 202.0   # A absent -> B's FY2023 -> 2024
