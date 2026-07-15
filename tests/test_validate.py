import pytest
from lark.exceptions import UnexpectedInput

from trail.parser import parse_program
from trail.validate import validate


def codes(src):
    return [i.code for i in validate(parse_program(src))]


def test_unknown_field_and_function():
    assert "E-FIELD-UNKNOWN" in codes("model m { a = income.bogus }")
    assert "E-FUNC-UNKNOWN" in codes("model m { a = frobnicate(income.revenue) }")


def test_arity():
    assert "E-FUNC-ARITY" in codes("model m { a = lag(income.revenue) }")  # lag needs 2 args


def test_undefined_name():
    assert "E-NAME-UNDEFINED" in codes("model m { a = b + 1 }")
    assert codes("model m { b = 1\n a = b + 1 }") == []


def test_pin_rejected():
    assert "E-PIN-UNSUPPORTED" in codes("model m { a = income.revenue @ fmp }")


def test_score_non_literal_value_is_parse_error():
    # Grammar enforces numeric score values, so a name value fails at parse time.
    with pytest.raises(UnexpectedInput):
        parse_program("model m { x = income.revenue\n score s weight 1 { x if x > 1\n else 0 } }")


def test_unknown_universe():
    assert "E-UNIVERSE-UNKNOWN" in codes("universe u = stocks\nmodel m on nowhere { a = 1 }")


def test_stock_flow_lint():
    assert "W-KIND-STOCK-FLOW" in codes("model m { t = income.revenue / balance.inventory }")
    assert "W-KIND-STOCK-FLOW" not in codes("model m { t = income.revenue / avg2(balance.inventory) }")


def test_name_rebound():
    assert "E-NAME-REBOUND" in codes("model m { a = 1\n a = 2 }")
    assert "E-NAME-REBOUND" in codes("model m { a = 1 }\nmodel m { b = 2 }")


def test_omitted_on_with_multiple_universes():
    src = "universe u1 = stocks\nuniverse u2 = stocks\nmodel m { a = 1 }"
    assert "E-UNIVERSE-UNKNOWN" in codes(src)
    assert "E-UNIVERSE-UNKNOWN" not in codes("universe u1 = stocks\nmodel m { a = 1 }")


def test_fwd_return_only_in_learn():
    assert "E-FWD-CONTEXT" in codes("model m { a = fwd_return(12) }")


def test_median_deferred_warns():
    assert "W-MEDIAN-DEFERRED" in codes("model m { on_missing median\n a = income.revenue }")


def test_nonannual_period_warns():
    assert "W-PERIOD-DEFERRED" in codes("model m at monthly { a = income.revenue }")
    assert "W-PERIOD-DEFERRED" in codes("signal s at quarterly = income.revenue")
    assert "W-PERIOD-DEFERRED" not in codes("model m at annual { a = income.revenue }")
