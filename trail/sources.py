"""Source drivers and panel loading. Driver contract: factory(options) -> obj.load(fields) -> DataFrame."""
from __future__ import annotations

import importlib

import polars as pl

from trail.config import Config, ConfigError


def resolve_driver(path: str):
    mod_path, _, attr = path.rpartition(".")
    try:
        return getattr(importlib.import_module(mod_path), attr)
    except (ImportError, AttributeError, ValueError) as e:
        raise ConfigError(f"cannot resolve driver '{path}': {e}") from e


class FixtureSource:
    def __init__(self, options: dict):
        self._options = options

    def load(self, fields: set[str]) -> pl.DataFrame:
        from trail.fixtures import load_panel

        return load_panel()


def fixture(options: dict) -> FixtureSource:
    return FixtureSource(options)


def load_panel_for(config: Config, fields: set[str]) -> pl.DataFrame:
    primary = config.precedence["default"][0]  # phase 1: single effective source
    spec = config.sources[primary]
    source = resolve_driver(spec.driver)(spec.options)
    panel = source.load(fields)
    if config.periods is not None:
        lo, hi = config.periods
        panel = panel.filter((pl.col("period") >= lo) & (pl.col("period") <= hi))
    return panel
