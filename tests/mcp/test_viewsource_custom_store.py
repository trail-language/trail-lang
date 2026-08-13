"""View composition must work with ANY configured ViewStore, not just local disk.

The ViewStore interface is pluggable through `providers.views`, and the write
path honoured it. The read path did not: ViewSource hardcoded LocalDiskViewStore
and the injection built it from `store.dir` -- an attribute no other
implementation has any reason to expose. A project configuring a custom store
therefore wrote its views correctly and then could not compose them at all:
`views.*` resolved to nothing and every reference failed with E-FIELD-UNSERVED,
naming a column that was sitting in the store the whole time. The AttributeError
was swallowed by a bare `except Exception: pass`, so there was no signal at all.
"""

import pytest

from trail.mcp.tools import run_tool
from trail.store import LocalDiskViewStore

MODEL = "track model factor at annual { export v = income.revenue + 1.0 }"
DEPENDENT = "track model onward at annual { export w = views.factor.v * 2.0 }"


class DirlessViewStore(LocalDiskViewStore):
    """A store that works but exposes no `.dir`, like any non-disk backend.

    Subclassing the disk store keeps the test about the ATTRIBUTE, not about
    reimplementing persistence: everything works, `.dir` is simply not part of
    the public ViewStore contract.
    """

    def __init__(self, options=None):
        super().__init__(options)
        self._hidden_dir = self.dir
        del self.dir            # exactly what a Postgres- or S3-backed store looks like


    def _frame_path(self, name):
        return self._hidden_dir / f"{name}.parquet"

    def _manifest_path(self, name):
        return self._hidden_dir / f"{name}.json"

    def list(self):
        return sorted(p.stem for p in self._hidden_dir.glob("*.parquet"))


CONFIG = (
    "sources:\n  fixture:\n    driver: trail.sources.fixture\n    options: {{}}\n"
    "precedence:\n  default: [fixture]\n"
    "panel:\n  periods: [2019, 2022]\n"
    "providers:\n  views:\n"
    "    driver: tests.mcp.test_viewsource_custom_store.DirlessViewStore\n"
    "    options: {{dir: {dir}}}\n"
)


def _config(tmp_path):
    d = tmp_path / "store"
    d.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "trail.yaml"
    p.write_text(CONFIG.format(dir=str(d)))
    return str(p)


def test_view_of_view_resolves_with_a_store_that_has_no_dir(tmp_path):
    """The regression: build a view, then reference it from another model."""
    cfg = _config(tmp_path)

    first = run_tool("factor", {"config": cfg}, program=MODEL, format="records")
    assert "error" not in first, first
    assert first["records"], "the dependency view did not build"

    onward = run_tool("onward", {"config": cfg}, program=MODEL + "\n" + DEPENDENT,
                      format="records")
    assert "error" not in onward, onward          # was E-FIELD-UNSERVED views.factor.v
    assert onward["records"]
    assert "views.onward.w" in onward["records"][0]


def test_injection_failure_warns_instead_of_silently_unserving(tmp_path, monkeypatch):
    """Degrading is fine; degrading silently is not.

    A bare `except Exception: pass` here meant that anything going wrong while
    making stored views queryable produced no signal whatsoever, and the failure
    surfaced far downstream as E-FIELD-UNSERVED -- an error pointing at the MODEL
    rather than at the view source that could not be built. Two multi-hour builds
    were spent on that misdirection.
    """
    from trail.mcp import _config_data

    def boom(config, config_path):
        raise RuntimeError("store went away")

    monkeypatch.setattr(_config_data, "_with_view_source", boom)
    with pytest.warns(RuntimeWarning, match="W-VIEWSOURCE-UNAVAILABLE"):
        run_tool("onward", {"config": _config(tmp_path)},
                 program=MODEL + "\n" + DEPENDENT, format="records")
