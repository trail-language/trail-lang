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


def test_available_fields_serves_every_view_regardless_of_frequency(tmp_path):
    _seed(tmp_path)
    vs = ViewSource({"dir": str(tmp_path)})
    # a stored column is offered for ANY requested frequency (frequency is not a routing gate)
    assert vs.available_fields() == {"views.rating.score"}
    assert vs.available_fields("annual") == {"views.rating.score"}
    assert vs.available_fields("quarterly") == {"views.rating.score"}
    assert vs.capabilities().frequency == "annual"


def test_mixed_and_none_frequency_views_all_reachable(tmp_path):
    s, t = _seed(tmp_path)                                   # rating @ annual
    df = pl.DataFrame({"entity": ["A"], "time": [t], "views.qmom.sig": [3.0]})
    s.write("qmom", df, Manifest("qmom", "signal", ("qmom",), "h2", ("fixture",), {"fixture": None},
                                 "2026-08-04T00:00:00", ("views.qmom.sig",), frequency="quarterly"))
    df2 = pl.DataFrame({"entity": ["A"], "time": [t], "views.nof.v": [4.0]})
    s.write("nof", df2, Manifest("nof", "model", ("v",), "h3", ("fixture",), {"fixture": None},
                                 "2026-08-04T00:00:00", ("views.nof.v",), frequency=None))  # declared without `at`
    vs = ViewSource({"dir": str(tmp_path)})
    # every column is available at the default frequency a bare reference resolves to
    got = vs.available_fields(vs.capabilities().frequency)
    assert got == {"views.rating.score", "views.qmom.sig", "views.nof.v"}
    assert set(vs.capabilities().frequencies) == {"annual", "quarterly"}   # canonical, None dropped


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
