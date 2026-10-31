"""`trail --version` and the generic `trail init <driver>` command (no network, no live marker)."""
import importlib.metadata
from click.testing import CliRunner

import trail.cli as cli
from trail.cli import main
from trail.init import InitField, InitSpec

TEMPLATE = """\
sources:
  demo:
    driver: demo
    options:
      identity: "{identity}"
      tickers: [{tickers}]
panel:
  periods: [{years}]
"""

SPEC = InitSpec(
    driver="demo",
    package="trail-demo",
    install_hint="pipx inject trail-lang trail-demo",
    fields=(
        InitField("identity", "Identity?"),
        InitField("tickers", "Tickers?", default="AAA,BBB"),
        InitField("years", "Years (start,end)?", default="2020,2024"),
    ),
    template=TEMPLATE,
)


# ── --version ──────────────────────────────────────────────────────────────
def test_version_flag():
    res = CliRunner().invoke(main, ["--version"])
    assert res.exit_code == 0
    assert res.output.strip() == f"trail {importlib.metadata.version('trail-lang')}"


# ── init: driver not installed ─────────────────────────────────────────────
def test_init_no_template(monkeypatch):
    monkeypatch.setattr(cli, "resolve_init_spec", lambda driver: None)
    res = CliRunner().invoke(main, ["init", "nosuchdriver"])
    assert res.exit_code == 1
    assert "Error: the nosuchdriver driver is not installed." in res.output
    assert "pipx inject trail-lang trail-nosuchdriver" in res.output
    assert "Traceback" not in res.output


def test_init_driver_missing(monkeypatch):
    # the spec resolves but the package's dist metadata is absent
    monkeypatch.setattr(cli, "resolve_init_spec", lambda driver: SPEC)

    def _missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(cli, "_pkg_version", _missing)
    res = CliRunner().invoke(main, ["init", "demo"])
    assert res.exit_code == 1
    assert "Error: the demo driver is not installed." in res.output
    assert SPEC.install_hint in res.output
    assert "Traceback" not in res.output


# ── init: writes the config ────────────────────────────────────────────────
def _monkey_spec(monkeypatch):
    monkeypatch.setattr(cli, "resolve_init_spec", lambda driver: SPEC)
    monkeypatch.setattr(cli, "_pkg_version", lambda name: "0.1.0")


def test_init_non_interactive_flags(monkeypatch, tmp_path):
    _monkey_spec(monkeypatch)
    out = tmp_path / "trail.yaml"
    res = CliRunner().invoke(
        main,
        ["init", "demo", "--identity", "Jane J j@x.com", "--tickers", "AAA,BBB",
         "--years", "2020,2024", "--output", str(out)],
    )
    assert res.exit_code == 0, res.output
    assert "trail-demo 0.1.0 found (driver: demo)" in res.output
    assert f"wrote {out}" in res.output
    assert out.read_text() == TEMPLATE.format(identity="Jane J j@x.com",
                                              tickers="AAA, BBB", years="2020, 2024")


def test_init_defaults_on_empty_prompts(monkeypatch, tmp_path):
    _monkey_spec(monkeypatch)
    out = tmp_path / "trail.yaml"
    # Enter accepts the default on the tickers/years prompts; identity is typed
    res = CliRunner().invoke(
        main,
        ["init", "demo", "--output", str(out)],
        input="Jane J j@x.com\n\n\n",
    )
    assert res.exit_code == 0, res.output
    assert "Tickers? [AAA,BBB]:" in res.output
    assert "Years (start,end)? [2020,2024]:" in res.output
    assert out.read_text() == TEMPLATE.format(identity="Jane J j@x.com",
                                              tickers="AAA, BBB", years="2020, 2024")


# ── init: overwrite rules ──────────────────────────────────────────────────
def test_init_refuses_overwrite(monkeypatch, tmp_path):
    _monkey_spec(monkeypatch)
    out = tmp_path / "trail.yaml"
    out.write_text("existing\n")
    res = CliRunner().invoke(
        main,
        ["init", "demo", "--identity", "J j@x.com", "--tickers", "AAA",
         "--years", "2020,2024", "--output", str(out)],
    )
    assert res.exit_code == 1
    assert f"Error: {out} already exists (use --force to overwrite)." in res.output
    assert out.read_text() == "existing\n"  # untouched


def test_init_force_overwrite(monkeypatch, tmp_path):
    _monkey_spec(monkeypatch)
    out = tmp_path / "trail.yaml"
    out.write_text("existing\n")
    res = CliRunner().invoke(
        main,
        ["init", "demo", "--identity", "J j@x.com", "--tickers", "AAA",
         "--years", "2020,2024", "--force", "--output", str(out)],
    )
    assert res.exit_code == 0, res.output
    assert out.read_text() == TEMPLATE.format(identity="J j@x.com",
                                              tickers="AAA", years="2020, 2024")
