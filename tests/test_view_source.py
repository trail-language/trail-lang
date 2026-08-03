import datetime as dt

import polars as pl

from trail.source import LoadRequest
from trail.store import LocalDiskViewStore, Manifest
from trail.views import ViewSource


def _seed(tmp_path):
    s = LocalDiskViewStore({"dir": str(tmp_path)})
    t = dt.datetime(2024, 12, 31)
    df = pl.DataFrame({"entity": ["A", "B"], "time": [t, t], "views.rating.score": [1.0, 2.0]})
    s.write("rating", df, Manifest("rating", "model", ("score",), "h", ("fixture",), {"fixture": None},
                                   "2026-08-04T00:00:00", ("views.rating.score",), frequency="annual"))
    return s, t


def test_available_fields_and_capabilities(tmp_path):
    _seed(tmp_path)
    vs = ViewSource({"dir": str(tmp_path)})
    assert vs.available_fields() == {"views.rating.score"}
    assert vs.available_fields("annual") == {"views.rating.score"}
    assert vs.available_fields("quarterly") == set()        # a view is offered only at its own frequency
    assert vs.capabilities().frequency == "annual"


def test_load_selects_and_scopes(tmp_path):
    _seed(tmp_path)
    vs = ViewSource({"dir": str(tmp_path)})
    out = vs.load(LoadRequest(fields=frozenset({"views.rating.score"}), entities=("A",)))
    assert out["entity"].to_list() == ["A"] and out["views.rating.score"].to_list() == [1.0]
    assert set(out.columns) >= {"entity", "time", "views.rating.score"}


def test_load_empty_when_field_not_stored(tmp_path):
    _seed(tmp_path)
    vs = ViewSource({"dir": str(tmp_path)})
    out = vs.load(LoadRequest(fields=frozenset({"views.other.x"})))
    assert out.height == 0


def test_freshness_token_changes_with_store(tmp_path):
    s, _ = _seed(tmp_path)
    tok = ViewSource({"dir": str(tmp_path)}).freshness_token()
    assert tok is not None
    s.delete("rating")
    assert ViewSource({"dir": str(tmp_path)}).freshness_token() != tok
