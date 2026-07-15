import polars as pl

from trail.fixtures import load_panel
from trail.schema import SCHEMA, is_field


def test_schema_has_core_fields_with_kinds():
    assert SCHEMA["income.revenue"].kind == "flow"
    assert SCHEMA["balance.total_assets"].kind == "stock"
    assert SCHEMA["price.adj_close"].kind == "price"
    assert SCHEMA["meta.sector"].kind == "meta"
    assert is_field("cash.cfo") and not is_field("no.such_thing")


def test_panel_shape_and_sort():
    df = load_panel()
    assert df.height == 6 * 8
    assert df["entity"].dtype == pl.Utf8 and isinstance(df["time"].dtype, pl.Datetime)
    assert set(SCHEMA) <= set(df.columns)
    assert df.sort(["entity", "time"]).equals(df)


def test_panel_known_values_and_nulls():
    df = load_panel()
    aaa_2020 = df.filter((pl.col("entity") == "AAA") & (pl.col("time").dt.year() == 2020))
    # revenue = 100 * (1 + 0.10) ** (year - 2017) for AAA (base 100, growth 10%)
    assert abs(aaa_2020["income.revenue"][0] - 100 * 1.10**3) < 1e-9
    assert aaa_2020["cash.stock_issued"][0] is None
    fff = df.filter(pl.col("entity") == "FFF")
    assert fff["income.interest_expense"].null_count() == 8
