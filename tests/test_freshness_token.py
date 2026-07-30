from trail.sources import FixtureSource


def test_default_freshness_token_is_none():
    assert FixtureSource({}).freshness_token() is None


def test_changed_since_default_none():
    from trail.source import Capabilities, DataSource

    class Bare(DataSource):
        name = "b"
        def load(self, r):  # pragma: no cover
            raise NotImplementedError
        def available_fields(self, frequency=None):
            return set()
        def capabilities(self):
            return Capabilities(frequency=None)

    assert Bare({}).changed_since(None) is None


def test_fixture_changed_since_from_option():
    import datetime as dt
    s = FixtureSource({"changed_cells": [["A", "2021-12-31"], ["B", "2021-12-31"]]})
    got = s.changed_since("w0")
    assert ("A", dt.datetime(2021, 12, 31)) in got and len(got) == 2
    assert FixtureSource({}).changed_since("w0") is None


def test_fixture_freshness_token_from_option():
    assert FixtureSource({"freshness": "t1"}).freshness_token() == "t1"


def test_datasource_base_default_is_none():
    # any source not overriding it reports no freshness signal
    from trail.source import DataSource

    class Bare(DataSource):
        name = "bare"
        def load(self, request):  # pragma: no cover - not exercised
            raise NotImplementedError
        def available_fields(self, frequency=None):
            return set()
        def capabilities(self):
            from trail.source import Capabilities
            return Capabilities(frequency=None)

    assert Bare({}).freshness_token() is None
