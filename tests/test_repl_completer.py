"""Tests for trail.repl.completer — tab completion."""
from __future__ import annotations

import pytest

from trail.repl.completer import TrailCompleter, FUNCTIONS, KEYWORDS
from trail.repl.session import ReplSession


class TestTrailCompleter:
    @pytest.fixture
    def session(self):
        return ReplSession()

    @pytest.fixture
    def completer(self, session):
        return TrailCompleter(session)

    def test_completer_initializes(self, session):
        """Completer can be created with a session."""
        c = TrailCompleter(session)
        assert c.session is session

    def test_function_keywords_present(self):
        """Key functions exist in the FUNCTIONS list."""
        assert "lag" in FUNCTIONS
        assert "roll_mean" in FUNCTIONS
        assert "ttm" in FUNCTIONS

    def test_keyword_list_nonempty(self):
        """KEYWORDS list contains basic Trail keywords."""
        assert "def" in KEYWORDS
        assert "model" in KEYWORDS
        assert "export" in KEYWORDS

    @pytest.mark.skipif(
        not pytest.importorskip("prompt_toolkit.document", reason="prompt_toolkit not installed"),
        reason="prompt_toolkit not installed"
    )
    def test_completer_returns_completion_items(self, session, completer):
        """Completer generates Completion objects for partial prefixes."""
        from prompt_toolkit.document import Document

        # "ro" should match roll_mean, roll_sum, etc.
        doc = Document(text="ro", cursor_position=2)
        completions = list(completer.complete(doc, None))
        texts = [c.text for c in completions]
        assert any("roll_mean" in t or "roll_sum" in t for t in texts), f"Expected roll_* in: {texts}"

    def test_completer_handles_prefix_match(self, session, completer):
        """Completer suggests functions starting with typed prefix."""
        doc = type('MockDoc', (), {'text_before_cursor': 'a'})()
        completions = list(completer.complete(doc, None))
        texts = [c.text for c in completions]
        assert any(t.startswith("a") for t in texts), f"Expected 'a*' completions: {texts}"

    def test_completer_prefers_defined_names(self, session):
        """User-defined names are suggested in completions."""
        session.definitions["my_margin"] = None
        c = TrailCompleter(session)
        doc = type('MockDoc', (), {'text_before_cursor': 'my_m'})()
        completions = list(c.complete(doc, None))
        assert any(c.text == "my_margin" for c in completions), f"Expected 'my_margin' in: {[c.text for c in completions]}"

    def test_completer_handles_empty_input(self, session, completer):
        """Empty input returns no completions."""
        doc = type('MockDoc', (), {'text_before_cursor': ''})()
        completions = list(completer.complete(doc, None))
        assert len(completions) == 0
