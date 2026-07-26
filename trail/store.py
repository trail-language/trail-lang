"""View store: durable, dependency-aware persistence of computed panels ("tracked views").

A tracked view is a Polars frame (parquet) plus a JSON manifest sidecar under a store directory.
The store is a writeable provider (see trail/providers.py); the read-back adapter that surfaces a
stored view as `<namespace>.*` panel columns lives in trail/views.py.
"""
from __future__ import annotations

import json
import pathlib
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import polars as pl

from trail.config import ConfigError

#: a view name becomes a filename under the store dir, so it must be a bare identifier — no path
#: separators or `..` (a raw MCP-supplied name reaches delete()/refresh(); guard against traversal).
_VIEW_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _check_name(name: str) -> str:
    if not isinstance(name, str) or not _VIEW_NAME_RE.match(name):
        raise ValueError(f"E-VIEW-NAME invalid view name {name!r}; must match [A-Za-z0-9_-]+")
    return name


@dataclass(frozen=True)
class Manifest:
    """Metadata stored alongside a view frame; drives the staleness decision (see trail/views.py)."""
    name: str
    kind: str                          # "model" | "signal"
    exports: tuple[str, ...]
    expr_hash: str                     # hash of the scoped declaration AST; a mismatch forces rebuild
    sources: tuple[str, ...]           # dependency source names whose freshness gates staleness
    freshness: dict[str, str | None]   # source name -> freshness token at build time (None = no signal)
    built_at: str                      # ISO-8601 UTC
    columns: tuple[str, ...]           # physical view columns, e.g. ("views.factor.value", ...)
    panel_key: str = ""                # fingerprint of frame-affecting config knobs (periods/pit/strict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        return cls(
            name=d["name"], kind=d["kind"], exports=tuple(d["exports"]),
            expr_hash=d["expr_hash"], sources=tuple(d["sources"]),
            freshness=dict(d["freshness"]), built_at=d["built_at"],
            columns=tuple(d["columns"]), panel_key=d.get("panel_key", ""),
        )


def view_columns(kind: str, name: str, exports, namespace: str = "views") -> tuple[str, ...]:
    """Physical column names a view materializes under: a signal is one column `<ns>.<name>`;
    a model is one column per export `<ns>.<name>.<export>`."""
    if kind == "signal":
        return (f"{namespace}.{name}",)
    return tuple(f"{namespace}.{name}.{e}" for e in exports)


class ViewStore(ABC):
    """Read/write/delete a set of named view frames. `namespace` is the field prefix references use."""
    namespace: str = "views"

    @abstractmethod
    def read(self, name: str) -> pl.DataFrame | None: ...
    @abstractmethod
    def manifest(self, name: str) -> Manifest | None: ...
    @abstractmethod
    def write(self, name: str, frame: pl.DataFrame, manifest: Manifest) -> None: ...
    @abstractmethod
    def delete(self, name: str) -> bool: ...
    @abstractmethod
    def list(self) -> list[str]: ...


class LocalDiskViewStore(ViewStore):
    """A ViewStore backed by a local directory: `<name>.parquet` frame + `<name>.json` manifest."""

    def __init__(self, options: dict | None = None) -> None:
        options = options or {}
        self.namespace = options.get("namespace", "views")
        if not options.get("dir"):
            raise ConfigError("E-PROVIDER-OPTIONS local-disk view store requires options.dir")
        self.dir = pathlib.Path(options["dir"])
        self.dir.mkdir(parents=True, exist_ok=True)

    def _frame_path(self, name: str) -> pathlib.Path:
        return self.dir / f"{_check_name(name)}.parquet"

    def _manifest_path(self, name: str) -> pathlib.Path:
        return self.dir / f"{_check_name(name)}.json"

    def read(self, name: str) -> pl.DataFrame | None:
        p = self._frame_path(name)
        return pl.read_parquet(p) if p.exists() else None

    def manifest(self, name: str) -> Manifest | None:
        p = self._manifest_path(name)
        return Manifest.from_dict(json.loads(p.read_text())) if p.exists() else None

    def write(self, name: str, frame: pl.DataFrame, manifest: Manifest) -> None:
        frame.write_parquet(self._frame_path(name))
        self._manifest_path(name).write_text(json.dumps(manifest.to_dict()))

    def delete(self, name: str) -> bool:
        # unlink the manifest first, so a crash mid-delete never leaves a manifest without its frame
        # (which is_stale would read as "fresh" while the frame is gone)
        existed = False
        for p in (self._manifest_path(name), self._frame_path(name)):
            if p.exists():
                p.unlink()
                existed = True
        return existed

    def list(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.parquet"))
