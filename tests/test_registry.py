import pytest

import trail.registry as reg
from trail.config import ConfigError
from trail.registry import registered_drivers, resolve_driver
from trail.sources import FixtureSource, fixture


def test_dotted_path_resolves_function():
    assert resolve_driver("trail.sources.fixture") is fixture


def test_dotted_path_resolves_class():
    assert resolve_driver("trail.sources.FixtureSource") is FixtureSource


def test_builtin_fixture_entry_point_registered():
    assert "fixture" in registered_drivers()
    assert resolve_driver("fixture") is FixtureSource


def test_bareword_without_registration_errors():
    with pytest.raises(ConfigError, match="E-SOURCE-DRIVER"):
        resolve_driver("not_a_registered_name")


def test_bad_dotted_path_errors():
    with pytest.raises(ConfigError, match="cannot resolve driver"):
        resolve_driver("no.such.module.attr")


def test_registered_name_wins_over_dotted(monkeypatch):
    class FakeEP:
        name = "mymodel"

        def load(self):
            return FixtureSource

    monkeypatch.setattr(reg, "_entry_points", lambda: {"mymodel": FakeEP()})
    assert resolve_driver("mymodel") is FixtureSource
