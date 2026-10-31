"""Tests for trail.repl — REPL session processing."""
from __future__ import annotations

import glob

import pytest

from trail.config import load_config
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
        """Assignment with undefined name → coded E-NAME-UNDEFINED."""
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
        result = session.process_input("def dbl(x) = x * 2")
        assert result.is_result
        assert "dbl" in session.definitions

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


class TestReplSessionFunctions:
    """Full function surface: stdlib derived functions, user defs, coded diagnostics."""

    @pytest.fixture
    def session(self):
        return ReplSession()

    def test_stdlib_derived_function(self, session):
        """A stdlib derived function (yoy) evaluates on the fixture, not just primitives."""
        result = session.process_input("yoy(income.revenue)")
        assert result.is_result, result.message
        df = result.value
        first = df["__repl_result"].to_list()[0]
        second = df["__repl_result"].to_list()[1]
        assert first is None  # window edge: no prior period for AAA 2017
        # AAA: 2017=100.0, 2018=110.0 → yoy = 0.1
        assert second == pytest.approx(0.1)

    def test_user_def_call(self, session):
        """def + call: 'def dbl(x) = x * 2' then 'dbl(income.revenue)' doubles the field."""
        assert session.process_input("def dbl(x) = x * 2").is_result
        result = session.process_input("dbl(income.revenue)")
        assert result.is_result, result.message
        val = result.value["__repl_result"].to_list()[0]
        assert val == pytest.approx(200.0)  # 2 * 100.0 (AAA 2017)

    def test_def_stdlib_collision(self, session):
        """A user def shadowing a stdlib name is E-FUNC-DUP (as in the file pipeline)."""
        result = session.process_input("def yoy(x) = x")
        assert result.is_error
        assert "E-FUNC-DUP" in result.message

    def test_func_arity_error(self, session):
        """lag with one argument → coded E-FUNC-ARITY (shared validator)."""
        result = session.process_input("lag(income.revenue)")
        assert result.is_error
        assert "E-FUNC-ARITY" in result.message
        assert "lag" in result.message

    def test_unknown_function_error(self, session):
        """Unknown function → coded E-FUNC-UNKNOWN (shared validator)."""
        result = session.process_input("bogus_fn(income.revenue)")
        assert result.is_error
        assert "E-FUNC-UNKNOWN" in result.message

    def test_user_def_arity_error(self, session):
        """A user def called with the wrong arity → E-FUNC-ARITY (macro pass)."""
        assert session.process_input("def dbl(x) = x * 2").is_result
        result = session.process_input("dbl(income.revenue, 1)")
        assert result.is_error
        assert "E-FUNC-ARITY" in result.message

    def test_model_with_def_call(self, session):
        """A model line uses user defs: statements are macro-expanded before compile."""
        assert session.process_input("def dbl(x) = x * 2").is_result
        result = session.process_input("model m { export r = dbl(income.revenue) }")
        assert result.is_result, result.message
        val = result.value["r"].to_list()[0]
        assert val == pytest.approx(200.0)

    def test_model_scope_guard_allows_named_model(self):
        """--model scope: the named model runs."""
        session = ReplSession(model_scope="m")
        result = session.process_input("model m { export r = income.revenue }")
        assert result.is_result, result.message

    def test_model_scope_guard_blocks_other_models(self):
        """--model scope: any other model/signal decl is refused with a clear error."""
        session = ReplSession(model_scope="m")
        result = session.process_input("model x { export r = income.revenue }")
        assert result.is_error
        assert "scoped to model 'm'" in result.message


class TestReplSessionDeclarations:
    """Universe, strategy, and track declarations have defined outcomes (no silent swallow)."""

    @pytest.fixture
    def session(self):
        return ReplSession()

    def test_universe_decl_binds_model(self, session):
        """A universe decl stores, and a model on it filters the panel."""
        assert session.process_input(
            "universe big = stocks where income.revenue > 150").is_result
        result = session.process_input("model m on big { export r = income.revenue }")
        assert result.is_result, result.message
        df = result.value
        assert "r" in df.columns
        assert len(df) < 48  # fixture = 6 entities x 8 years, unfiltered
        assert df["r"].min() > 150

    def test_universe_duplicate(self, session):
        """A second universe with the same name → E-NAME-REBOUND."""
        assert session.process_input("universe u = stocks where income.revenue > 0").is_result
        result = session.process_input("universe u = stocks where income.revenue > 1")
        assert result.is_error
        assert "E-NAME-REBOUND" in result.message

    def test_strategy_phase_deferred(self, session):
        """strategy parses but execution is a post-1.0 phase (mirrors `trail run`)."""
        result = session.process_input("strategy s { universe stocks signal 1 rebalance monthly }")
        assert result.is_error
        assert "E-PHASE-DEFERRED" in result.message

    def test_track_requires_config(self, session):
        """track without --config: clear error (views need a trail.yaml path), not a silent run."""
        result = session.process_input("track model t { export r = income.revenue }")
        assert result.is_error
        assert "--config" in result.message


class TestReplSessionConfig:
    """Config-backed sessions (--config): real panel, active catalog, scoped runs, views."""

    @pytest.fixture
    def cfg_path(self, tmp_path):
        p = tmp_path / "trail.yaml"
        p.write_text(
            "sources:\n  fx:\n    driver: fixture\n"
            "precedence:\n  default: [fx]\n"
            "panel:\n  periods: [2017, 2018]\n"
        )
        return p

    def _session(self, cfg_path):
        return ReplSession(config=load_config(str(cfg_path)), config_path=str(cfg_path))

    def test_config_panel_loaded(self, cfg_path):
        """The configured panel loads (periods-filtered), not the raw fixture."""
        session = self._session(cfg_path)
        result = session.process_input("income.revenue")
        assert result.is_result, result.message
        df = result.value
        assert len(df) == 12  # 6 entities x 2 periods (2017-2018), not 48
        row = df.filter(df["entity"] == "AAA").sort("time").row(0, named=True)
        assert row["__repl_result"] == pytest.approx(100.0)

    def test_config_catalog_reflects_config(self, cfg_path):
        """?sources lists the ACTIVE config's sources (fx), not the default (fixture)."""
        session = self._session(cfg_path)
        result = session.process_input("?sources")
        assert result.is_result
        assert "fx" in result.value["source"].to_list()
        assert "fixture" not in result.value["source"].to_list()

    def test_config_model_scoped(self, cfg_path):
        """A model line under --config loads a scoped panel (only its referenced fields)."""
        session = self._session(cfg_path)
        result = session.process_input("model m { export r = income.revenue }")
        assert result.is_result, result.message
        df = result.value
        assert set(df.columns) == {"entity", "time", "r"}
        assert len(df) == 12

    def test_config_universe_model(self, cfg_path):
        """Universe + model binding on configured data."""
        session = self._session(cfg_path)
        assert session.process_input(
            "universe big = stocks where income.revenue > 150").is_result
        result = session.process_input("model m on big { export r = income.revenue }")
        assert result.is_result, result.message
        assert result.value["r"].min() > 150

    def test_track_with_config(self, cfg_path):
        """track model persists through the view store (parquet + manifest) and re-serves."""
        session = self._session(cfg_path)
        result = session.process_input("track model t { export r = income.revenue }")
        assert result.is_result, result.message
        frames = glob.glob(str(cfg_path.parent / ".trail" / "views" / "t*"))
        assert frames, "tracked view was not persisted"
        # second run: served from the store (freshness unchanged for the fixture)
        again = session.process_input("track model t { export r = income.revenue }")
        assert again.is_result, again.message


class TestReplSessionImports:
    """import "path": merge a file's defs + universes into the session (defined behavior)."""

    @pytest.fixture
    def defs_file(self, tmp_path):
        p = tmp_path / "defs.trail"
        p.write_text(
            "def dbl(x) = x * 2\n"
            "universe small = stocks where income.revenue < 150\n"
        )
        return p

    def test_import_defs_and_universe(self, defs_file):
        """An import merges the file's def and universe; both then work in the session."""
        session = ReplSession()
        result = session.process_input(f'import "{defs_file}"')
        assert result.is_result, result.message
        assert "dbl" in session.definitions
        assert "small" in session.universes
        # the imported def expands in a later line
        r = session.process_input("dbl(income.revenue)")
        assert r.is_result, r.message
        assert r.value["__repl_result"].to_list()[0] == pytest.approx(200.0)
        # the imported universe binds a model
        m = session.process_input("model mm on small { export r = income.revenue }")
        assert m.is_result, m.message
        assert m.value["r"].max() < 150

    def test_import_missing_file(self, tmp_path):
        """An unresolvable import path → coded E-IMPORT-NOT-FOUND (not silently accepted)."""
        session = ReplSession()
        result = session.process_input(f'import "{tmp_path}/nope.trail"')
        assert result.is_error
        assert "E-IMPORT-NOT-FOUND" in result.message

    def test_import_duplicate(self, defs_file):
        """An import colliding with a session name → E-IMPORT-DUP, session left intact."""
        session = ReplSession()
        assert session.process_input("def dbl(x) = x + 1").is_result
        result = session.process_input(f'import "{defs_file}"')
        assert result.is_error
        assert "E-IMPORT-DUP" in result.message
        # the earlier session def is untouched
        r = session.process_input("dbl(income.revenue)")
        assert r.is_result, r.message
        assert r.value["__repl_result"].to_list()[0] == pytest.approx(101.0)

    def test_import_dedup(self, defs_file):
        """Importing the same file twice is a no-op (dedup, as in prepare)."""
        session = ReplSession()
        assert session.process_input(f'import "{defs_file}"').is_result
        assert session.process_input(f'import "{defs_file}"').is_result
