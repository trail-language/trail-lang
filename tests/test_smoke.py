from importlib.metadata import version

import trail


def test_package_imports():
    # __version__ tracks the installed distribution metadata, not a hardcoded literal
    assert trail.__version__ == version("trail-lang")
