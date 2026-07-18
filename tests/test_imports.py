"""`import "file.trail"` = source-level inclusion: an imported file's reusable top-level
`def`s and `universe`s become visible to the importer. Models/signals are not imported."""
import pytest

from trail import ast
from trail.pipeline import TrailImportError, prepare
from trail.validate import validate


def _codes(prog: ast.Program) -> set[str]:
    return {i.code for i in validate(prog)}


def _write(d, name: str, text: str) -> str:
    p = d / name
    p.write_text(text)
    return str(p)


def test_import_makes_def_available(tmp_path):
    _write(tmp_path, "factors.trail", "def value_z(x) = zscore(x) by meta.sector\n")
    model = 'import "factors.trail"\nmodel m { export v = value_z(income.revenue) }\n'
    mp = _write(tmp_path, "model.trail", model)
    prog = prepare(model, path=mp)
    assert "E-FUNC-UNKNOWN" not in _codes(prog)  # value_z resolved + inlined
    # the def is inlined and stripped: no FuncDef survives into the compile-ready program
    assert all(not (isinstance(d, ast.FuncDef) and d.name == "value_z") for d in prog.decls)


def test_missing_import_means_unknown_function(tmp_path):
    model = "model m { export v = value_z(income.revenue) }\n"
    mp = _write(tmp_path, "model.trail", model)
    prog = prepare(model, path=mp)  # no import -> value_z is undefined
    assert "E-FUNC-UNKNOWN" in _codes(prog)


def test_transitive_import(tmp_path):
    _write(tmp_path, "b.trail", "def base(x) = x + 1\n")
    _write(tmp_path, "a.trail", 'import "b.trail"\ndef mid(x) = base(x) * 2\n')
    model = 'import "a.trail"\nmodel m { export y = mid(income.revenue) }\n'
    mp = _write(tmp_path, "model.trail", model)
    prog = prepare(model, path=mp)  # mid (from a) uses base (from b, pulled transitively)
    assert "E-FUNC-UNKNOWN" not in _codes(prog)


def test_import_universe(tmp_path):
    _write(tmp_path, "u.trail", "universe big = stocks where meta.market_cap > 1e9\n")
    model = 'import "u.trail"\nmodel m on big { export y = income.revenue }\n'
    mp = _write(tmp_path, "model.trail", model)
    prog = prepare(model, path=mp)
    assert "E-UNIVERSE-UNKNOWN" not in _codes(prog)
    assert any(isinstance(d, ast.UniverseDecl) and d.name == "big" for d in prog.decls)


def test_imported_model_is_skipped(tmp_path):
    _write(tmp_path, "lib.trail",
           "def libf(x) = x + 1\nmodel other { export z = income.revenue }\n")
    model = 'import "lib.trail"\nmodel m { export y = libf(income.revenue) }\n'
    mp = _write(tmp_path, "model.trail", model)
    prog = prepare(model, path=mp)
    assert {d.name for d in prog.decls if isinstance(d, ast.ModelDecl)} == {"m"}  # 'other' not imported
    assert "E-FUNC-UNKNOWN" not in _codes(prog)  # but the def is


def test_import_cycle(tmp_path):
    _write(tmp_path, "a.trail", 'import "b.trail"\ndef fa(x) = x\n')
    _write(tmp_path, "b.trail", 'import "a.trail"\ndef fb(x) = x\n')
    model = 'import "a.trail"\nmodel m { export y = income.revenue }\n'
    mp = _write(tmp_path, "model.trail", model)
    with pytest.raises(TrailImportError, match="E-IMPORT-CYCLE"):
        prepare(model, path=mp)


def test_self_import_is_a_cycle(tmp_path):
    model = 'import "model.trail"\nmodel m { export y = income.revenue }\n'
    mp = _write(tmp_path, "model.trail", model)
    with pytest.raises(TrailImportError, match="E-IMPORT-CYCLE"):
        prepare(model, path=mp)


def test_import_not_found(tmp_path):
    model = 'import "nope.trail"\nmodel m { export y = income.revenue }\n'
    mp = _write(tmp_path, "model.trail", model)
    with pytest.raises(TrailImportError, match="E-IMPORT-NOT-FOUND"):
        prepare(model, path=mp)


def test_import_duplicate_def_collides_with_importer(tmp_path):
    _write(tmp_path, "dup.trail", "def value_z(x) = x\n")
    model = ('import "dup.trail"\ndef value_z(x) = x + 1\n'
             "model m { export y = value_z(income.revenue) }\n")
    mp = _write(tmp_path, "model.trail", model)
    with pytest.raises(TrailImportError, match="E-IMPORT-DUP"):
        prepare(model, path=mp)


def test_import_duplicate_def_collides_with_stdlib(tmp_path):
    # signed_log is a standard-library `def`; an import may not shadow it
    _write(tmp_path, "s.trail", "def signed_log(x) = x\n")
    model = 'import "s.trail"\nmodel m { export y = income.revenue }\n'
    mp = _write(tmp_path, "model.trail", model)
    with pytest.raises(TrailImportError, match="E-IMPORT-DUP"):
        prepare(model, path=mp)


def test_same_file_imported_twice_dedups(tmp_path):
    _write(tmp_path, "f.trail", "def dd(x) = x\n")
    # two spellings of the same file must load once - no spurious E-IMPORT-DUP
    model = ('import "f.trail"\nimport "./f.trail"\n'
             "model m { export y = dd(income.revenue) }\n")
    mp = _write(tmp_path, "model.trail", model)
    prog = prepare(model, path=mp)  # does not raise
    assert "E-FUNC-UNKNOWN" not in _codes(prog)


def test_resolution_is_relative_to_importing_file_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # CWD is tmp_path, but the model lives in sub/
    sub = tmp_path / "sub"
    sub.mkdir()
    _write(sub, "factors.trail", "def rel_f(x) = x * 3\n")
    model = 'import "factors.trail"\nmodel m { export y = rel_f(income.revenue) }\n'
    mp = _write(sub, "model.trail", model)
    prog = prepare(model, path=mp)  # resolves sub/factors.trail, not tmp_path/factors.trail
    assert "E-FUNC-UNKNOWN" not in _codes(prog)


def test_import_is_independent_of_stdlib(tmp_path):
    _write(tmp_path, "f.trail", "def nf(x) = x + 1\n")
    model = 'import "f.trail"\nmodel m { export y = nf(income.revenue) }\n'
    mp = _write(tmp_path, "model.trail", model)
    prog = prepare(model, stdlib=False, path=mp)  # --no-stdlib still resolves imports
    assert "E-FUNC-UNKNOWN" not in _codes(prog)
