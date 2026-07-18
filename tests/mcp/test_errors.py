from trail.config import ConfigError
from trail.mcp.errors import to_error
from trail.pipeline import TrailImportError


def test_import_error_code_extracted_from_message():
    e = TrailImportError("E-IMPORT-NOT-FOUND missing /x/y.trail")
    assert to_error(e) == {"error": {"code": "E-IMPORT-NOT-FOUND",
                                     "message": "E-IMPORT-NOT-FOUND missing /x/y.trail"}}


def test_config_error_maps_to_e_config():
    assert to_error(ConfigError("bad yaml"))["error"]["code"] == "E-CONFIG"


def test_unknown_exception_is_e_internal():
    assert to_error(ValueError("boom"))["error"]["code"] == "E-INTERNAL"
