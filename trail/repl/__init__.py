"""Trail REPL — interactive expression language session.

Usage::

    trail repl

Install prompt_toolkit::

    pip install trail-lang[repl]
"""
from __future__ import annotations

from trail.repl.loop import run_repl
from trail.repl.session import ReplSession, Result

__all__ = ["ReplSession", "Result", "run_repl"]
