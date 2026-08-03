import polars as pl

from trail.store import LocalDiskViewStore, Manifest, view_columns


def _mf(cols):
    return Manifest("v", "signal", ("v",), "h1", ("fixture",), {"fixture": "t0"},
                    "2026-07-26T00:00:00", tuple(cols))


def test_write_read_roundtrip(tmp_path):
    s = LocalDiskViewStore({"dir": str(tmp_path)})
    df = pl.DataFrame({"entity": ["A"], "time": ["2024-01-01"], "views.v": [1.5]})
    s.write("v", df, _mf(["views.v"]))
    got = s.read("v")
    assert got is not None and got["views.v"].to_list() == [1.5]
    assert s.manifest("v").expr_hash == "h1"
    assert s.manifest("v").freshness == {"fixture": "t0"}
    assert s.list() == ["v"]


def test_read_absent_is_none(tmp_path):
    assert LocalDiskViewStore({"dir": str(tmp_path)}).read("missing") is None
    assert LocalDiskViewStore({"dir": str(tmp_path)}).manifest("missing") is None


def test_delete(tmp_path):
    s = LocalDiskViewStore({"dir": str(tmp_path)})
    s.write("v", pl.DataFrame({"entity": ["A"], "time": ["2024-01-01"], "views.v": [1.0]}), _mf(["views.v"]))
    assert s.delete("v") is True
    assert s.delete("v") is False
    assert s.read("v") is None


def test_view_columns():
    assert view_columns("signal", "mom", ("mom",)) == ("views.mom",)
    assert view_columns("model", "factor", ("value", "quality")) == (
        "views.factor.value", "views.factor.quality")
    assert view_columns("model", "factor", ("value",), namespace="store") == ("store.factor.value",)


def test_manifest_roundtrip():
    mf = _mf(["views.v"])
    assert Manifest.from_dict(mf.to_dict()) == mf


def test_manifest_frequency_roundtrips_and_defaults():
    mf = Manifest("v", "signal", ("v",), "h", ("fixture",), {"fixture": None},
                  "2026-08-04T00:00:00", ("views.v",), frequency="annual")
    assert Manifest.from_dict(mf.to_dict()).frequency == "annual"
    d = mf.to_dict()
    d.pop("frequency")                            # an old manifest with no frequency key
    assert Manifest.from_dict(d).frequency is None


def test_manifest_view_deps_roundtrips_and_defaults():
    mf = Manifest("comp", "model", ("c",), "h", ("fmp",), {"fmp": None},
                  "2026-08-04T00:00:00", ("views.comp.c",), view_deps={"rating": "hr:t0"})
    back = Manifest.from_dict(mf.to_dict())
    assert back.view_deps == {"rating": "hr:t0"}
    assert back == mf
    d = mf.to_dict()
    d.pop("view_deps")                            # an old manifest predating view-of-view
    assert Manifest.from_dict(d).view_deps == {}


def test_store_rejects_traversing_names(tmp_path):
    import pytest
    s = LocalDiskViewStore({"dir": str(tmp_path)})
    for bad in ("../escape", "a/b", "with.dot", "with space"):
        with pytest.raises(ValueError, match="E-VIEW-NAME"):
            s.delete(bad)


def test_store_requires_dir():
    import pytest
    from trail.config import ConfigError
    with pytest.raises(ConfigError, match="E-PROVIDER-OPTIONS"):
        LocalDiskViewStore({"namespace": "views"})
