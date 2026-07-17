"""@ align(expr): a per-field override of the alignment coordinate, expressed over the source's
date columns and materialized in the loader."""
import datetime as dt

import polars as pl
import pytest

import trail.ast as ast
from trail import sources
from trail.config import Config, ConfigError, SourceSpec
from trail.parser import parse_expr
from trail.pipeline import prepare
from trail.source import Capabilities, DataSource, FieldInfo, LoadRequest, date_col
from trail.validate import validate


class _TwoDateSource(DataSource):
    """One FY2022 row carrying two candidate coordinates: `filing` (in 2022) and `accepted`
    (in 2023), so choosing one vs the other lands the value on a different decision year."""

    name = "twodate"

    def load(self, request: LoadRequest) -> pl.DataFrame:
        cols = {"entity": ["X"], "time": [dt.datetime(2022, 12, 31)], "income.revenue": [100.0],
                date_col("filing"): [dt.datetime(2022, 12, 20)],
                date_col("accepted"): [dt.datetime(2023, 1, 10)]}
        return pl.DataFrame(cols).with_columns(
            [pl.col(c).cast(pl.Datetime("us")) for c in cols if c == "time" or c.startswith("__date:")])

    def available_fields(self, frequency=None):
        return {"income.revenue"}

    def describe_field(self, field):
        return FieldInfo(field, True, "direct", aligns_on="filing") if field == "income.revenue" else None

    def capabilities(self):
        return Capabilities(frequency="annual", frequencies=("annual",), pit=True)


@pytest.fixture(autouse=True)
def _patch_driver(monkeypatch):
    monkeypatch.setattr(sources, "resolve_driver", lambda ref: _TwoDateSource)


def _cfg():
    return Config(sources={"twodate": SourceSpec("twodate", "drv")}, precedence={"default": ["twodate"]})


def _load(align_overrides=None):
    return sources.load_panel_for(_cfg(), {"income.revenue"}, target_freq="annual",
                                  align_overrides=align_overrides).sort("time")


# --- parse ---
def test_align_parses_onto_fieldref():
    e = parse_expr('income.revenue @ align(truncate(filing_date, "1y"))')
    assert isinstance(e, ast.FieldRef) and e.path == ("income", "revenue")
    assert isinstance(e.align, ast.Call) and e.align.name == "truncate"
    assert e.entity is None and e.source is None  # align is distinct from the other qualifiers


def test_entity_pin_still_parses_after_grammar_change():
    e = parse_expr('price.adj_close @ entity("SPY")')
    assert e.entity == "SPY" and e.align is None


# --- validate ---
def test_align_validation_rejects_unknown_function(monkeypatch):
    codes = {i.code for i in validate(prepare('model m { export y = income.revenue @ align(bogus(filing)) }'))}
    assert "E-FUNC-UNKNOWN" in codes


def test_align_validation_rejects_schema_field():
    # a two-part dotted name parses as a FieldRef, which is not a source date column
    codes = {i.code for i in validate(prepare('model m { export y = income.revenue @ align(balance.total_assets) }'))}
    assert "E-ALIGN-EXPR" in codes


def test_align_bare_date_name_validates_clean():
    codes = {i.code for i in validate(prepare('model m { export y = income.revenue @ align(filing) }'))}
    assert "E-FIELD-UNKNOWN" not in codes and "E-ALIGN-EXPR" not in codes


# --- end to end through the loader ---
def test_default_coordinate_is_the_declared_aligns_on():
    out = _load()  # filing (2022-12-20) -> annual-2022
    assert out["time"].dt.year().to_list() == [2022]
    assert out["income.revenue"].to_list() == [100.0]


def test_align_override_swaps_the_coordinate_column():
    out = _load({"income.revenue": ast.NameRef("accepted")})  # accepted (2023-01-10) -> annual-2023
    assert out["time"].dt.year().to_list() == [2023]
    assert out["income.revenue"].to_list() == [100.0]
    assert not [c for c in out.columns if c.startswith("__date:")]  # coordinate consumed


def test_align_non_datetime_expr_is_rejected():
    ov = {"income.revenue": parse_expr("income.revenue @ align(year(filing))").align}
    with pytest.raises(ConfigError, match="E-ALIGN-DTYPE"):
        _load(ov)


def test_align_unknown_date_column_is_rejected():
    with pytest.raises(ConfigError, match="E-ALIGN-UNKNOWN"):
        _load({"income.revenue": ast.NameRef("nonexistent")})


# --- review fixes: malformed qualifiers, chaining, and same-field conflicts ---
import lark  # noqa: E402


def test_entity_non_string_arg_is_a_parse_error():
    # arg widened STRING->expr: a non-string entity arg is now a transformer-level rejection
    with pytest.raises((lark.exceptions.VisitError, ValueError)):
        parse_expr("price.adj_close @ entity(123)")


def test_chaining_two_qualifiers_is_rejected():
    # bare chaining is grammar-rejected (a pin isn't an atom); the parenthesized form reaches
    # the transformer, where _reject_requalified refuses a second qualifier
    with pytest.raises((lark.exceptions.VisitError, ValueError, lark.exceptions.UnexpectedInput)):
        parse_expr('price.adj_close @ entity("SPY") @ align(accepted)')
    with pytest.raises((lark.exceptions.VisitError, ValueError)):
        parse_expr('(price.adj_close @ entity("SPY")) @ align(accepted)')


def test_conflicting_align_on_same_field_is_rejected():
    src = "model m { a = income.revenue\n b = income.revenue @ align(accepted) }"
    codes = {i.code for i in validate(prepare(src))}
    assert "E-ALIGN-CONFLICT" in codes


def test_consistent_align_on_same_field_is_ok():
    src = "model m { a = income.revenue @ align(accepted)\n b = income.revenue @ align(accepted) + 1 }"
    codes = {i.code for i in validate(prepare(src))}
    assert "E-ALIGN-CONFLICT" not in codes


class _NullUnusedDateSource(DataSource):
    """Default coordinate is `accepted` (non-null); `filing` is null and unused by the override."""

    name = "nulldate"

    def load(self, request: LoadRequest) -> pl.DataFrame:
        cols = {"entity": ["X"], "time": [dt.datetime(2022, 12, 31)], "income.revenue": [100.0],
                date_col("filing"): [None], date_col("accepted"): [dt.datetime(2023, 1, 10)]}
        return pl.DataFrame(cols).with_columns(
            [pl.col(c).cast(pl.Datetime("us")) for c in cols if c == "time" or c.startswith("__date:")])

    def available_fields(self, frequency=None):
        return {"income.revenue"}

    def describe_field(self, field):
        return FieldInfo(field, True, "direct", aligns_on="accepted") if field == "income.revenue" else None

    def capabilities(self):
        return Capabilities(frequency="annual", frequencies=("annual",), pit=True)


def test_override_does_not_carry_unused_null_date_columns(monkeypatch, recwarn):
    # the override references only `accepted`; the null, unused `filing` must not be carried
    # (carrying it would emit a spurious W-PIT-PARTIAL for its nulls)
    monkeypatch.setattr(sources, "resolve_driver", lambda ref: _NullUnusedDateSource)
    cfg = Config(sources={"nulldate": SourceSpec("nulldate", "drv")}, precedence={"default": ["nulldate"]})
    ov = {"income.revenue": parse_expr('income.revenue @ align(truncate(accepted, "1y"))').align}
    sources.load_panel_for(cfg, {"income.revenue"}, target_freq="annual", align_overrides=ov)
    assert not [w for w in recwarn.list if "W-PIT-PARTIAL" in str(w.message)]
