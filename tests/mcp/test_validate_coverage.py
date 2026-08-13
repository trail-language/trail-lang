"""`validate` checks that the configured sources SERVE the program's fields.

Without a config, validation resolves the field VOCABULARY only -- whether
`income.revenue` is a known field -- not whether anything provides it. Those are
different questions, and the gap is expensive in exactly one way: a source
declared under `sources:` but omitted from `precedence` serves nothing, so the
model validates clean and dies at run time with E-FIELD-UNSERVED *after* the
fetch. On a full-universe panel that is hours of wasted work for a one-line
config omission.
"""
from trail.mcp.tools import validate_tool

MODEL = "model m at annual { export v = income.revenue + 1.0 }"

SERVED = (
    "sources:\n  fixture:\n    driver: trail.sources.fixture\n    options: {}\n"
    "precedence:\n  default: [fixture]\n"
    "panel:\n  periods: [2019, 2022]\n"
)

# The exact shape of the production bug: the source is declared, and left out of
# the precedence chain. A source outside the chain is never consulted.
DECLARED_BUT_UNROUTED = (
    "sources:\n  fixture:\n    driver: trail.sources.fixture\n    options: {}\n"
    "precedence:\n  default: []\n"
    "panel:\n  periods: [2019, 2022]\n"
)


def _cfg(tmp_path, body, name="trail.yaml"):
    p = tmp_path / name
    p.write_text(body)
    return str(p)


def test_validate_without_a_config_is_unchanged(tmp_path):
    """The existing contract: no config, no coverage check, still valid."""
    out = validate_tool(MODEL)
    assert out["valid"] is True
    assert not [i for i in out["issues"] if i["severity"] == "error"]


def test_validate_with_a_config_accepts_a_served_program(tmp_path):
    out = validate_tool(MODEL, data={"config": _cfg(tmp_path, SERVED)})
    assert out["valid"] is True, out["issues"]


def test_validate_catches_a_source_missing_from_precedence(tmp_path):
    """The regression this whole change exists for.

    Before, this returned valid=True and the failure surfaced only from `run`.
    """
    cfg = _cfg(tmp_path, DECLARED_BUT_UNROUTED)

    blind = validate_tool(MODEL)
    assert blind["valid"] is True, "vocabulary-only validation cannot see this"

    out = validate_tool(MODEL, data={"config": cfg})
    assert out["valid"] is False
    codes = {i["code"] for i in out["issues"]}
    assert "E-FIELD-UNSERVED" in codes, out["issues"]
    assert any("income.revenue" in i["message"] for i in out["issues"])


def test_coverage_reports_every_unserved_field_not_just_the_first(tmp_path):
    """`run` raises on the first; validate exists to list them all.

    Fixing config omissions one run at a time is the expensive loop this replaces.
    """
    model = ("model m at annual { export v = income.revenue + income.net_income "
             "+ balance.total_assets }")
    out = validate_tool(model, data={"config": _cfg(tmp_path, DECLARED_BUT_UNROUTED)})
    unserved = [i for i in out["issues"] if i["code"] == "E-FIELD-UNSERVED"]
    assert len(unserved) >= 3, [i["message"] for i in out["issues"]]


def test_an_unreadable_config_warns_rather_than_silently_skipping(tmp_path):
    """A config that cannot be loaded must not quietly restore the blind spot.

    Swallowing this would mean the check reports "no problems" for a config it
    never actually read -- which is how the underlying bug stayed hidden.
    """
    out = validate_tool(MODEL, data={"config": str(tmp_path / "does-not-exist.yaml")})
    codes = {i["code"] for i in out["issues"]}
    assert "W-COVERAGE-UNCHECKED" in codes, out["issues"]
    # A warning, not an error: the program itself is still valid.
    assert out["valid"] is True


def test_parse_errors_still_short_circuit_before_coverage(tmp_path):
    out = validate_tool("model m at annual { export v = }",
                        data={"config": _cfg(tmp_path, SERVED)})
    assert out["valid"] is False
    assert out["issues"], "a parse error must still be reported"
