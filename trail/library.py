"""Standard library loader — concatenates the bundled `stdlib/*.trail` macro files.

The stdlib is Trail source (derived functions), brought in as `def`s that are inlined
at compile time. Order is irrelevant (functions are collected before expansion), but a
stable order keeps error messages reproducible.
"""
from __future__ import annotations

from importlib.resources import files

STDLIB_MODULES = ["math", "stats", "transform", "calculus", "geometry", "factor", "timeseries", "core"]


def stdlib_source() -> str:
    """Return the concatenated source of every bundled stdlib module."""
    parts = [
        files("trail.stdlib").joinpath(f"{mod}.trail").read_text()
        for mod in STDLIB_MODULES
    ]
    return "\n".join(parts)
