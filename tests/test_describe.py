"""`trail describe`: surface what a provider actually publishes (fields + categorical values).

Purely descriptive discoverability - no normalization/canonicalization of values.
"""
import datetime as dt

import polars as pl
from click.testing import CliRunner

from trail.cli import main
from trail.describe import (
    categorical_fields, fields_by_namespace, render_describe, value_counts,
)


def _panel() -> pl.DataFrame:
    rows = []
    for ent, sector in [("A", "Tech"), ("B", "Tech"), ("C", "Financial Services")]:
        for yr in (2022, 2023):
            rows.append({
                "entity": ent, "time": dt.datetime(yr, 12, 31),
                "income.revenue": 100.0, "meta.sector": sector, "meta.exchange": "NYSE",
            })
    return pl.DataFrame(rows).with_columns(pl.col("time").cast(pl.Datetime("us")))


def test_value_counts_sorted_by_frequency():
    rows, total = value_counts(_panel(), "meta.sector")
    assert total == 2
    assert rows[0] == ("Tech", 4)  # 2 entities x 2 years, the most frequent first
    assert ("Financial Services", 2) in rows


def test_value_counts_cap_truncates():
    p = pl.DataFrame({
        "entity": [str(i) for i in range(100)],
        "time": [dt.datetime(2023, 12, 31)] * 100,
        "meta.sector": [f"S{i}" for i in range(100)],
    }).with_columns(pl.col("time").cast(pl.Datetime("us")))
    rows, total = value_counts(p, "meta.sector", cap=10)
    assert total == 100 and len(rows) == 10


def test_categorical_fields_detects_strings_not_numbers():
    cats = categorical_fields(_panel())
    assert "meta.sector" in cats and "meta.exchange" in cats
    assert "income.revenue" not in cats


def test_fields_by_namespace_groups_and_sorts():
    ns = fields_by_namespace(_panel())
    assert ns["meta"] == ["meta.exchange", "meta.sector"]
    assert ns["income"] == ["income.revenue"]


def test_render_full_includes_sector_values_and_counts():
    out = render_describe(_panel())
    assert "meta.sector" in out
    assert "Financial Services" in out and "Tech" in out


def test_render_single_field_only():
    out = render_describe(_panel(), field="meta.sector")
    assert "Financial Services" in out and "Tech" in out
    assert "meta.exchange" not in out  # only the requested field is shown


# --- CLI (default fixture config; fixture sectors are Tech/Energy/Financials) ---

MODEL = "model m { export margin = income.operating_income / income.revenue }\n"


def test_describe_cli_shows_sector_distinct_values(tmp_path):
    f = tmp_path / "m.trail"
    f.write_text(MODEL)
    res = CliRunner().invoke(main, ["describe", str(f)])
    assert res.exit_code == 0
    assert "meta.sector" in res.output
    assert "Tech" in res.output and "Financials" in res.output


def test_describe_cli_field_only(tmp_path):
    f = tmp_path / "m.trail"
    f.write_text(MODEL)
    res = CliRunner().invoke(main, ["describe", str(f), "--field", "meta.sector"])
    assert res.exit_code == 0 and "meta.sector" in res.output
    assert "Tech" in res.output
    assert "income.revenue" not in res.output  # only the requested field


def test_describe_cli_model_scoped(tmp_path):
    # scope the panel to a model + its bound universe (same machinery as `run`)
    f = tmp_path / "m.trail"
    f.write_text("model m { keep = meta.sector\n export margin = income.operating_income / income.revenue }\n")
    res = CliRunner().invoke(main, ["describe", str(f), "--model", "m"])
    assert res.exit_code == 0 and "meta.sector" in res.output


def test_describe_cli_unknown_model_errors(tmp_path):
    f = tmp_path / "m.trail"
    f.write_text(MODEL)
    res = CliRunner().invoke(main, ["describe", str(f), "--model", "nope"])
    assert res.exit_code == 1 and "no model named" in res.output
