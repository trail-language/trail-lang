"""The field-name codec is the single definition of the column micro-grammar."""
import pytest

from trail import fieldname as fn


def test_qualified_encode():
    assert fn.qualified("income.revenue") == "income.revenue"
    assert fn.qualified("income.revenue", frequency="annual") == "annual.income.revenue"
    assert fn.qualified("price.adj_close", entity="SPY") == "price.adj_close@SPY"
    assert fn.qualified("price.adj_close", frequency="daily", entity="SPY") == "daily.price.adj_close@SPY"


def test_qualified_source_encode():
    assert fn.qualified("income.revenue", source="edgar") == "income.revenue#edgar"
    assert fn.qualified("income.revenue", frequency="annual", source="edgar") == "annual.income.revenue#edgar"
    # entity (@) and source (#) are mutually exclusive
    with pytest.raises(ValueError):
        fn.qualified("income.revenue", entity="SPY", source="edgar")


def test_split_source_and_canonical_strips_it():
    assert fn.split_source("income.revenue#edgar") == ("income.revenue", "edgar")
    assert fn.split_source("income.revenue") == ("income.revenue", None)
    assert fn.canonical("annual.income.revenue#edgar") == "income.revenue"
    assert fn.canonical("income.revenue#fmp") == "income.revenue"


def test_source_round_trip():
    for freq in (None, "annual", "quarterly"):
        for src in (None, "edgar"):
            col = fn.qualified("income.revenue", frequency=freq, source=src)
            base, s = fn.split_source(col)
            f, canon = fn.split_frequency(base)
            assert (f, canon, s) == (freq, "income.revenue", src)
            assert fn.canonical(col) == "income.revenue"


def test_parse_ref():
    assert fn.parse_ref(("income", "revenue")) == (None, ("income", "revenue"))
    assert fn.parse_ref(("annual", "income", "revenue")) == ("annual", ("income", "revenue"))
    assert fn.parse_ref(("daily", "price")) == (None, ("daily", "price"))  # 2-part not split


def test_split_and_canonical():
    assert fn.split_frequency("annual.income.revenue") == ("annual", "income.revenue")
    assert fn.split_frequency("income.revenue") == (None, "income.revenue")
    assert fn.split_pin("price.adj_close@SPY") == ("price.adj_close", "SPY")
    assert fn.split_pin("price.adj_close") == ("price.adj_close", None)
    assert fn.canonical("daily.price.adj_close@SPY") == "price.adj_close"
    assert fn.canonical("annual.income.revenue") == "income.revenue"


def test_round_trip():
    for freq in (None, "annual", "daily"):
        for ent in (None, "SPY"):
            col = fn.qualified("income.revenue", frequency=freq, entity=ent)
            base, e = fn.split_pin(col)
            f, canon = fn.split_frequency(base)
            assert (f, canon, e) == (freq, "income.revenue", ent)
            assert fn.canonical(col) == "income.revenue"
