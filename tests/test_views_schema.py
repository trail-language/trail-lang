from trail import schema
from trail.pipeline import prepare
from trail.validate import validate


def _error_codes(src):
    prog = prepare(src, stdlib=False)
    return [i.code for i in validate(prog) if i.severity == "error"]


def test_is_field_recognizes_views_namespace():
    assert schema.is_field("views.factor.value") is True
    assert schema.is_field("views.momentum") is True


def test_non_views_unknown_field_still_rejected():
    assert schema.is_field("nope.unknown") is False


def test_views_reference_validates_without_field_error():
    codes = _error_codes("universe u = stocks\nsignal top on u at annual = views.factor.value")
    assert "E-FIELD-UNKNOWN" not in codes


def test_views_reference_in_by_clause_ok():
    codes = _error_codes("universe u = stocks\nsignal top on u at annual = zscore(1.0) by views.grp")
    assert "E-FIELD-UNKNOWN" not in codes
