import pytest

from trail.config import ConfigError
from trail.providers import resolve_provider


def test_resolve_provider_unknown_raises():
    with pytest.raises(ConfigError) as e:
        resolve_provider("nope-not-a-dotted-path")
    assert "E-PROVIDER-DRIVER" in str(e.value)


def test_resolve_provider_dotted_import_error_raises():
    with pytest.raises(ConfigError) as e:
        resolve_provider("trail.store.NoSuchClass")
    assert "E-PROVIDER-DRIVER" in str(e.value)
