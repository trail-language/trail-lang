"""Regression: the REPL error path must survive arbitrary < / > in the message.

0.19.0 crashed (xml.parsers.expat.ExpatError -> REPL exit 1) when an error message contained raw
angle brackets (e.g. polars' multi-line errors): show_error wrapped the raw text in
prompt_toolkit's HTML formatter, whose parser rejected it. The message is now HTML-escaped
before wrapping.
"""
from __future__ import annotations

import pytest

pytest.importorskip("prompt_toolkit", reason="show_error renders through prompt_toolkit")

from trail.repl.loop import show_error  # noqa: E402


def test_show_error_survives_angle_brackets(capsys):
    """'<' / '>'-laden and markup-shaped messages render instead of raising ExpatError."""
    show_error("frame <df> where a < b")
    show_error("bad <b>bold</b> tag")
    out = capsys.readouterr().out
    # escaped before wrapping, so the angle brackets parse back to the literal characters
    assert "frame <df> where a < b" in out
    # markup-shaped text renders as data, not as HTML tags
    assert "bad <b>bold</b> tag" in out
