import pytest
from click.testing import CliRunner

from trail.cli import main
from trail.config import ConfigError, load_config
from trail.sources import load_panel_for, resolve_driver

YAML_OK = """
panel:
  periods: [2019, 2022]
sources:
  fixture:
    driver: trail.sources.fixture
precedence:
  default: [fixture]
"""

YAML_BAD_SOURCE = """
sources:
  fixture:
    driver: trail.sources.fixture
precedence:
  default: [nonexistent]
"""


def test_default_config_when_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no trail.yaml here
    cfg = load_config(None)
    assert "fixture" in cfg.sources and cfg.precedence["default"] == ["fixture"]


def test_yaml_parsed_and_period_bounds_applied(tmp_path):
    f = tmp_path / "trail.yaml"
    f.write_text(YAML_OK)
    cfg = load_config(str(f))
    assert cfg.periods == (2019, 2022)
    panel = load_panel_for(cfg, {"income.revenue"})
    years = panel["time"].dt.year().unique().sort().to_list()
    assert years == [2019, 2020, 2021, 2022]


def test_unknown_precedence_source_rejected(tmp_path):
    f = tmp_path / "trail.yaml"
    f.write_text(YAML_BAD_SOURCE)
    with pytest.raises(ConfigError, match="E-SOURCE-UNKNOWN"):
        load_config(str(f))


def test_unknown_driver_rejected():
    with pytest.raises(ConfigError, match="cannot resolve driver"):
        resolve_driver("no.such.module.driver")


def test_cli_run_with_config(tmp_path):
    cfg = tmp_path / "trail.yaml"
    cfg.write_text(YAML_OK)
    model = tmp_path / "m.trail"
    model.write_text("model m { export margin = income.operating_income / income.revenue }\n")
    res = CliRunner().invoke(main, ["run", str(model), "--model", "m", "--config", str(cfg)])
    assert res.exit_code == 0 and "margin" in res.output


YAML_STRICT = """
sources:
  fixture:
    driver: fixture
precedence:
  default: [fixture]
panel:
  strict: true
"""


def test_panel_strict_parsed(tmp_path):
    f = tmp_path / "trail.yaml"
    f.write_text(YAML_STRICT)
    assert load_config(str(f)).strict is True


def test_strict_defaults_false(tmp_path):
    f = tmp_path / "trail.yaml"
    f.write_text(YAML_OK)
    assert load_config(str(f)).strict is False
