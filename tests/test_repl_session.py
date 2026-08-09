"""Tests for trail.repl — REPL session processing."""
from __future__ import annotations

import pytest

from trail.repl import ReplSession


class TestReplSession:
    """TDD cycle 1: core session + expression evaluation + assignment."""

    @pytest.fixture
    def session(self):
        """Fresh session with default fixture config."""
        return ReplSession()

    def test_process_bare_arithmetic(self, session):
        """Basic arithmetic: 1 + 1 → column with value ≈ 2.0."""
        result = session.process_input("1 + 1")
        assert result.is_result
        df = result.value
        assert len(df.columns) == 3
        val = df["__repl_result"].to_list()[0]
        assert val == pytest.approx(2.0)

    def test_process_assignment(self, session):
        """Assignment: 'x = 1 + 1' defines x and stores computed value."""
        result = session.process_input("x = 1 + 1")
        assert result.is_result
        assert "x" in session.definitions
        df = result.value
        assert "x" in df.columns

    def test_process_bare_field(self, session):
        """Bare field reference resolves: 'income.revenue' → column with values."""
        result = session.process_input("income.revenue")
        assert result.is_result
        df = result.value
        assert len(df.columns) == 3  # entity, time, __repl_result
        val = df["__repl_result"].to_list()[0]
        assert val == pytest.approx(100.0)  # AAA 2017 revenue

    def test_process_assignment_with_fields(self, session):
        """Assignment with fields: 'margin = income.revenue / balance.total_assets'."""
        result = session.process_input("margin = income.revenue / balance.total_assets")
        assert result.is_result
        assert "margin" in session.definitions
        df = result.value
        assert "margin" in df.columns
        non_null = df["margin"].drop_nulls()
        assert len(non_null) > 0

    def test_process_undefined_name_error(self, session):
        """Assignment with undefined name → error."""
        result = session.process_input("x = undefined_name")
        assert result.is_error
        assert "undefined" in result.message.lower() or "E-NAME-UNDEFINED" in result.message

    def test_process_chain_assignment(self, session):
        """Chain: define margin, then use it in next line."""
        session.process_input("margin = income.revenue / balance.total_assets")
        result = session.process_input("margin * 100")
        assert result.is_result
        df = result.value
        val = df["__repl_result"].to_list()[0]
        # margin for AAA 2017 = 100 / 200 = 0.5, * 100 = 50
        assert val == pytest.approx(50.0)

    def test_process_empty_input(self, session):
        """Empty input returns no-op result."""
        result = session.process_input("")
        assert result.is_result
        assert result.value is None

    def test_process_model(self, session):
        """Model declaration compiles and runs."""
        result = session.process_input(
            "model m { export margin = income.revenue / balance.total_assets }"
        )
        assert result.is_result
        df = result.value
        assert "margin" in df.columns
        assert len(df.columns) == 3  # entity, time, margin

    def test_process_def_function(self, session):
        """Def stores function body in definitions."""
        result = session.process_input("def avg2(x) = (x + x) / 2")
        assert result.is_result
        assert "avg2" in session.definitions

    def test_process_meta_catalog(self, session):
        """Meta-catalog returns a catalog result."""
        result = session.process_input("?")
        assert result.is_result
        assert result.value is not None

    def test_process_meta_describe(self, session):
        """Meta-describe returns describe result."""
        result = session.process_input("? income.revenue")
        assert result.is_result
        assert result.value is not None
