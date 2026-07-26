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


def test_store_for_config_default(tmp_path):
    from trail.config import Config
    from trail.providers import store_for_config
    from trail.store import LocalDiskViewStore
    cfg = Config(sources={}, precedence={"default": []})
    store = store_for_config(cfg, str(tmp_path / "trail.yaml"))
    assert isinstance(store, LocalDiskViewStore)
    assert store.dir == (tmp_path / ".trail" / "views")
    assert store.namespace == "views"


def test_store_for_config_explicit(tmp_path):
    from trail.config import Config, ProviderSpec
    from trail.providers import store_for_config
    cfg = Config(sources={}, precedence={"default": []},
                 providers={"views": ProviderSpec("views", "views_local",
                                                   {"dir": str(tmp_path / "custom")})})
    store = store_for_config(cfg, str(tmp_path / "trail.yaml"))
    assert store.dir == (tmp_path / "custom")
