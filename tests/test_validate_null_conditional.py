"""W-NULL-COND: one condition driving both a positive and a negated conditional.

Conditions are three-valued. A null is neither true nor false, and BOTH forms of
a ternary take their `else` branch on null, so

    score = raw   if g else -1          -> null gives -1   (as if false)
    grade = "BAD" if not g else "OK"    -> null gives "OK"  (as if true)

disagree about exactly the rows where `g` is null. Nothing about that is an
error -- each expression is individually correct, the model validates, the run
succeeds, and the two exports contradict each other on a subset of rows.

Found in production: a gate flag null on 1,170 of 47,485 rows produced companies
simultaneously SCORED as vetoed and GRADED as fine. Anything filtering on the
grade picked up names whose score was the veto sentinel.
"""
from trail.mcp.tools import validate_tool


def _codes(src: str) -> list[str]:
    return [i["code"] for i in validate_tool(src)["issues"]]


MIXED = """model m at annual {
    g = income.revenue > 0
    export score = income.revenue if g else -1
    export grade = "AVOID" if not g else "OK"
}"""

SAME_POLARITY = """model m at annual {
    g = income.revenue > 0
    export score = income.revenue if g else -1
    export other = income.net_income if g else 0
}"""

BOTH_NEGATED = """model m at annual {
    g = income.revenue > 0
    export a = income.revenue if not g else -1
    export b = income.net_income if not g else 0
}"""

SINGLE = """model m at annual {
    export score = income.revenue if income.revenue > 0 else -1
}"""


def test_mixed_polarity_is_flagged():
    assert "W-NULL-COND" in _codes(MIXED)


def test_same_polarity_is_not_flagged():
    """Two conditionals agreeing on polarity agree on null too."""
    assert "W-NULL-COND" not in _codes(SAME_POLARITY)


def test_both_negated_is_not_flagged():
    assert "W-NULL-COND" not in _codes(BOTH_NEGATED)


def test_a_lone_conditional_is_not_flagged():
    """A single ternary has nothing to contradict; warning here would be noise."""
    assert "W-NULL-COND" not in _codes(SINGLE)


def test_it_is_a_warning_not_an_error():
    """Three-valued logic is legal. This is a smell, not a rejection -- a model
    that deliberately wants null-as-false in one place and null-as-true in
    another must still be able to say so."""
    issues = validate_tool(MIXED)["issues"]
    sev = [i["severity"] for i in issues if i["code"] == "W-NULL-COND"]
    assert sev == ["warning"]


def test_polarity_is_tracked_per_model():
    """Two models may use opposite polarities without their results ever meeting."""
    src = """model a at annual {
    g = income.revenue > 0
    export x = income.revenue if g else -1
}
model b at annual {
    g = income.revenue > 0
    export y = income.revenue if not g else -1
}"""
    assert "W-NULL-COND" not in _codes(src)


def test_nested_conditionals_are_reached():
    """The grade ladder that triggered this in production was a nested chain, so
    the walk has to reach conditions inside a ternary's branches, not just the top."""
    src = """model m at annual {
    g = income.revenue > 0
    export score = income.revenue if g else -1
    export grade = "AVOID" if not g else ("A" if income.revenue > 100 else "B")
}"""
    assert "W-NULL-COND" in _codes(src)


def test_only_one_warning_per_model():
    """Many exports sharing one flag is a single design issue, not N of them."""
    src = """model m at annual {
    g = income.revenue > 0
    export a = income.revenue if g else -1
    export b = income.revenue if g else -2
    export c = "X" if not g else "Y"
    export d = "P" if not g else "Q"
}"""
    assert _codes(src).count("W-NULL-COND") == 1
