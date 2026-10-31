"""The bundled fixture registers its field vocabulary via the trail.schema entry point,
so no-config `trail validate` / `trail run` / `ReplSession` accept fixture columns
(instead of rejecting them with E-FIELD-UNKNOWN)."""
import importlib.metadata
from click.testing import CliRunner

from trail.cli import main
from trail.fixtures import fixture_fields
from trail.repl.session import ReplSession

MODEL = "model m at annual { export r = income.revenue }\n"


def test_noconfig_validate_accepts_fixture_field(tmp_path):
    f = tmp_path / "m.trail"
    f.write_text(MODEL)
    res = CliRunner().invoke(main, ["validate", str(f)])
    assert res.exit_code == 0, res.output
    assert "E-FIELD-UNKNOWN" not in res.output


def test_noconfig_run_writes_48_rows(tmp_path):
    f = tmp_path / "m.trail"
    f.write_text(MODEL)
    out = tmp_path / "m.csv"
    res = CliRunner().invoke(main, ["run", str(f), "--model", "m", "--out", str(out)])
    assert res.exit_code == 0, res.output
    lines = out.read_text().strip().splitlines()
    assert lines[0] == "entity,time,r" and len(lines) == 49  # header + 6 entities x 8 years


def test_noconfig_repl_fixture_field():
    s = ReplSession()
    r = s.process_input("income.revenue")
    assert not r.is_error, r.message
    assert r.value.shape == (48, 3)


def test_fixture_schema_entry_point_registered():
    # the dist metadata must actually declare the entry point (the conftest _CORE patch
    # masks this for behavior tests, so the registration itself is verified here)
    eps = [ep for ep in importlib.metadata.entry_points(group="trail.schema")
           if ep.name == "fixture"]
    assert eps, "trail.schema entry point 'fixture' is not registered (reinstall the package)"
    mapping = eps[0].load()
    assert set(mapping) == fixture_fields()
    assert mapping["income.revenue"] == "flow"
    assert mapping["balance.total_assets"] == "stock"
    assert mapping["price.adj_close"] == "price"
    assert mapping["meta.sector"] == "meta"
    # the three kinds that drifted in the original prefix-map version - keep this loud
    assert mapping["cash.cfo"] == "flow"
    assert mapping["income.eps_diluted"] == "per_share"
    assert mapping["price.dividends"] == "per_share"


def test_kinds_match_canonical_for_drifted_columns():
    # the same three columns through the active-schema path (kind_of) - in a clean install
    # (no conftest _CORE patch) this reads exactly what the fixture's entry point registers
    from trail.schema import kind_of
    assert kind_of("cash.cfo") == "flow"
    assert kind_of("income.eps_diluted") == "per_share"
    assert kind_of("price.dividends") == "per_share"
