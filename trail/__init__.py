"""Trail - a declarative financial expression language over (entity x time) panels."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("trail-lang")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"
