import datetime as dt

import polars as pl

from trail.footprint import replace_rows


def test_replace_rows_overwrites_only_keys():
    t = dt.datetime(2021, 12, 31)
    stored = pl.DataFrame({"entity": ["A", "B", "C"], "time": [t, t, t], "views.v": [1.0, 2.0, 3.0]})
    fresh = pl.DataFrame({"entity": ["A", "B"], "time": [t, t], "views.v": [10.0, 20.0]})
    out = replace_rows(stored, fresh, {("A", t), ("B", t)}).sort("entity")
    assert out["views.v"].to_list() == [10.0, 20.0, 3.0]   # C untouched


def test_replace_rows_adds_new_and_drops_missing():
    t = dt.datetime(2021, 12, 31)
    stored = pl.DataFrame({"entity": ["A", "B"], "time": [t, t], "views.v": [1.0, 2.0]})
    fresh = pl.DataFrame({"entity": ["A", "D"], "time": [t, t], "views.v": [10.0, 40.0]})
    # F = {A, B}; B missing from fresh (skipped) -> dropped; D is fresh but not in F -> ignored
    out = replace_rows(stored, fresh, {("A", t), ("B", t)}).sort("entity")
    assert out["entity"].to_list() == ["A"] and out["views.v"].to_list() == [10.0]


def test_replace_rows_empty_keys_is_noop():
    t = dt.datetime(2021, 12, 31)
    stored = pl.DataFrame({"entity": ["A"], "time": [t], "views.v": [1.0]})
    assert replace_rows(stored, stored, set()).equals(stored)
