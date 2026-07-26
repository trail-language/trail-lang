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


def test_resolve_provider_registered_name():
    from trail.store import LocalDiskViewStore
    assert resolve_provider("views_local") is LocalDiskViewStore


def test_resolve_provider_dotted_fallback():
    from trail.store import LocalDiskViewStore
    assert resolve_provider("trail.store.LocalDiskViewStore") is LocalDiskViewStore
