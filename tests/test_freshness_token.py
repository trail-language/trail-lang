from trail.sources import FixtureSource


def test_default_freshness_token_is_none():
    assert FixtureSource({}).freshness_token() is None


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
