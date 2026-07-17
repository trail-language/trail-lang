"""Cross-entity references: x @ entity("SPY") - parse, deps, validate, compile, engine."""
import datetime as dt

import polars as pl
import pytest
from lark.exceptions import VisitError

from trail import ast, sources
from trail.compiler import compile_expr
from trail.config import Config, ConfigError, SourceSpec
from trail.deps import extract
from trail.parser import parse_expr, parse_program
from trail.source import Capabilities, DataSource
from trail.validate import validate


# --- language surface ---

def test_entity_pin_parses():
    e = parse_expr('price.adj_close @ entity("SPY")')
    assert isinstance(e, ast.FieldRef)
    assert e.path == ("price", "adj_close") and e.entity == "SPY" and e.source is None
    assert e.column == "price.adj_close"
    assert e.qualified_column == "price.adj_close@SPY"


def test_entity_pin_composes_with_frequency():
    e = parse_expr('daily.price.adj_close @ entity("SPY")')
    assert e.frequency == "daily" and e.entity == "SPY"
    assert e.qualified_column == "daily.price.adj_close@SPY"


def test_source_pin_still_parses():
    e = parse_expr("income.revenue @ fmp")
    assert e.source == "fmp" and e.entity is None
    # a source named `entity` without parens is still a source pin (nothing reserved)
    e2 = parse_expr("income.revenue @ entity")
    assert e2.source == "entity" and e2.entity is None


def test_unknown_selector_rejected():
    with pytest.raises((VisitError, ValueError), match="unknown pin selector"):
        parse_expr('income.revenue @ bogus("SPY")')


def test_pin_on_expression_is_rejected():
    # pins apply to schema field references only (spec: "pin the fields, not the arithmetic")
    with pytest.raises((VisitError, ValueError), match="schema field reference"):
        parse_expr('(income.revenue + income.cogs) @ entity("SPY")')


def test_deps_surface_pinned_column():
    fields = extract(parse_expr('price.adj_close - price.adj_close @ entity("SPY")')).fields
    assert fields == frozenset({"price.adj_close", "price.adj_close@SPY"})


def test_validate_entity_and_source_pins_are_both_legal():
    codes = [i.code for i in validate(parse_program(
        'model m { export x = price.adj_close @ entity("SPY") }'))]
    assert "E-PIN-UNSUPPORTED" not in codes and "E-FIELD-UNKNOWN" not in codes
    # a `@ source` pin is live now: no static rejection (source existence is a loader check)
    codes = [i.code for i in validate(parse_program("model m { export x = price.adj_close @ fmp }"))]
    assert "E-PIN-UNSUPPORTED" not in codes and "E-FIELD-UNKNOWN" not in codes
    codes = [i.code for i in validate(parse_program(
        'model m { export x = bogus.field @ entity("SPY") }'))]
    assert "E-FIELD-UNKNOWN" in codes  # canonical base still validated


def test_compiler_reads_pinned_column():
    expr = compile_expr(parse_expr('price.adj_close @ entity("SPY")'), set())
    assert expr.meta.output_name() == "price.adj_close@SPY"


# --- engine ---

_DAYS = [dt.datetime(2023, 1, d) for d in (3, 4, 5)]


class _MultiEntity(DataSource):
    """Three stocks + SPY, daily prices."""

    def load(self, request):
        rows = []
        base = {"AAA": 10.0, "BBB": 20.0, "SPY": 100.0}
        for ent, b in base.items():
            for i, t in enumerate(_DAYS):
                rows.append({"entity": ent, "time": t, "price.adj_close": b + i})
        return pl.DataFrame(rows).with_columns(pl.col("time").cast(pl.Datetime("us")))

    def available_fields(self, frequency=None):
        return {"price.adj_close"}

    def describe_field(self, field):
        return None

    def entities(self, universe=None):
        return ["AAA", "BBB", "SPY"]

    def capabilities(self):
        return Capabilities(frequency="daily")


def _cfg():
    return Config(sources={"s": SourceSpec("s", "d")}, precedence={"default": ["s"]})


def test_entity_pin_broadcasts_onto_every_row(monkeypatch):
    monkeypatch.setattr(sources, "resolve_driver", lambda ref: _MultiEntity)
    panel = sources.load_panel_for(
        _cfg(), {"price.adj_close", "price.adj_close@SPY"}, target_freq="daily")
    d = {(r["entity"], r["time"].day): r for r in panel.iter_rows(named=True)}
    # SPY's series (100, 101, 102) lands on every entity, day-matched
    for day, spy_px in ((3, 100.0), (4, 101.0), (5, 102.0)):
        assert d[("AAA", day)]["price.adj_close@SPY"] == spy_px
        assert d[("BBB", day)]["price.adj_close@SPY"] == spy_px
        assert d[("SPY", day)]["price.adj_close@SPY"] == spy_px
    assert d[("AAA", 3)]["price.adj_close"] == 10.0  # own series intact


def test_entity_pin_unknown_entity_raises(monkeypatch):
    monkeypatch.setattr(sources, "resolve_driver", lambda ref: _MultiEntity)
    with pytest.raises(ConfigError, match="E-ENTITY-UNKNOWN.*QQQ"):
        sources.load_panel_for(_cfg(), {"price.adj_close", "price.adj_close@QQQ"}, target_freq="daily")


def test_entity_pin_widens_explicit_fetch_scope(monkeypatch):
    seen = {}

    class _Scoped(_MultiEntity):
        def load(self, request):
            seen["entities"] = request.entities
            return _MultiEntity.load(self, request)

    monkeypatch.setattr(sources, "resolve_driver", lambda ref: _Scoped)
    sources.load_panel_for(_cfg(), {"price.adj_close", "price.adj_close@SPY"},
                           target_freq="daily", entities=["AAA", "BBB"])
    assert seen["entities"] == ("AAA", "BBB", "SPY")  # pin entity added to the scope; request.entities is a tuple
