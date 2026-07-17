"""Discovery core: catalog / describe over the schema, function, and source registries.

Shared engine behind every discovery front-end (REPL `?` meta-commands, the `trail
catalog` CLI, and - later - MCP tools and Jupyter magics). Returns CatalogResult (a
titled metadata table), never a panel: discovery is metadata, not computation.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import polars as pl

from trail.ops import OPS

from trail import ast
from trail.config import DEFAULT_CONFIG, Config
from trail.schema import active_schema
from trail.validate import KNOWN_FUNCTIONS


@lru_cache(maxsize=1)
def _stdlib_functions() -> dict[str, int]:
    """Bundled stdlib macro name -> parameter count (the derived layer). Internal
    helpers (names starting with '_') are hidden from discovery."""
    from trail.library import stdlib_source
    from trail.macro import collect_functions
    from trail.parser import parse_program

    funcs = collect_functions(parse_program(stdlib_source()))
    return {name: len(fd.params) for name, fd in funcs.items() if not name.startswith("_")}

# axis/summary derive from the single function registry (trail.ops.OPS)
_FUNC_META: dict[str, tuple[str, str]] = {n: (sp.axis, sp.summary) for n, sp in OPS.items()}


@dataclass(frozen=True)
class CatalogResult:
    """A titled metadata table. Renders in a terminal (str) and in Jupyter (_repr_html_)."""

    title: str
    frame: pl.DataFrame

    def __str__(self) -> str:
        return f"{self.title}\n{self.frame}"

    def _repr_html_(self) -> str:  # Jupyter rich display
        return f"<strong>{self.title}</strong>{self.frame._repr_html_()}"


def namespaces() -> list[str]:
    return sorted({c.split(".", 1)[0] for c in active_schema()})


def fields(namespace: str | None = None) -> CatalogResult:
    items = [
        (c, spec.kind) for c, spec in active_schema().items()
        if namespace is None or c.split(".", 1)[0] == namespace
    ]
    frame = pl.DataFrame({"field": [c for c, _ in items], "kind": [k for _, k in items]}).sort("field")
    title = f"Fields in '{namespace}' ({frame.height})" if namespace else f"All fields ({frame.height})"
    return CatalogResult(title, frame)


def functions() -> CatalogResult:
    rows = {"function": [], "layer": [], "axis": [], "args": [], "summary": []}
    for n in sorted(KNOWN_FUNCTIONS):
        lo, hi = KNOWN_FUNCTIONS[n]
        axis, summary = _FUNC_META.get(n, ("", ""))
        rows["function"].append(n)
        rows["layer"].append("primitive")
        rows["axis"].append(axis)
        rows["args"].append(str(lo) if lo == hi else f"{lo}..{hi}")
        rows["summary"].append(summary)
    std = _stdlib_functions()
    for n in sorted(std):
        rows["function"].append(n)
        rows["layer"].append("derived")
        rows["axis"].append("")
        rows["args"].append(str(std[n]))
        rows["summary"].append("stdlib macro")
    return CatalogResult(f"Functions ({len(KNOWN_FUNCTIONS)} primitive + {len(std)} derived)",
                         pl.DataFrame(rows))


def sources(config: Config = DEFAULT_CONFIG) -> CatalogResult:
    names = sorted(config.sources)
    frame = pl.DataFrame({
        "source": names,
        "driver": [config.sources[n].driver for n in names],
    })
    precedence = ", ".join(f"{ns}=[{', '.join(chain)}]" for ns, chain in config.precedence.items())
    return CatalogResult(f"Sources ({len(names)}) | precedence: {precedence}", frame)


def _kv(title: str, pairs: list[tuple[str, str]]) -> CatalogResult:
    frame = pl.DataFrame({"property": [k for k, _ in pairs], "value": [v for _, v in pairs]})
    return CatalogResult(title, frame)


def _source_detail(name: str, spec) -> CatalogResult:
    """Source metadata plus a best-effort coverage view for a discoverable source.

    Instantiating a source can fail (e.g. a missing credential); that is reported
    inline rather than raised, so discovery stays usable without a live connection.
    """
    from trail.registry import resolve_driver
    from trail.source import SupportsCapabilities, SupportsDiscovery

    rows: list[tuple[str, str]] = [("driver", spec.driver), ("options", str(spec.options))]
    try:
        src = resolve_driver(spec.driver)(spec.options)
    except Exception as e:  # instantiation/credential failure: report, do not raise
        rows.append(("discovery", f"unavailable ({e})"))
        return _kv(f"Source {name}", rows)
    try:
        if isinstance(src, SupportsCapabilities):
            caps = src.capabilities()
            rows.append(("frequency", caps.frequency))
            if caps.period_range:
                rows.append(("period_range", f"{caps.period_range[0]}..{caps.period_range[1]}"))
            if caps.provenance:
                rows.append(("provenance", caps.provenance))
        if isinstance(src, SupportsDiscovery):
            avail = src.available_fields()
            all_fields = active_schema()
            ns = {f.split(".", 1)[0] for f in avail}
            relevant = {c for c in all_fields if c.split(".", 1)[0] in ns}
            rows.append(("provides",
                         f"{len(avail & relevant)}/{len(relevant)} fields in [{', '.join(sorted(ns))}]"))
            missing = sorted(relevant - avail)
            if missing:
                rows.append(("unavailable_fields", ", ".join(missing)))
        else:
            rows.append(("discovery", "not supported (core-tier source)"))
    finally:
        try:
            src.close()
        except Exception:
            pass
    return _kv(f"Source {name}", rows)


def describe(target: tuple[str, ...], config: Config = DEFAULT_CONFIG) -> CatalogResult:
    dotted = ".".join(target)
    # category list-alls
    if target == ("functions",):
        return functions()
    if target == ("sources",):
        return sources(config)
    if target == ("fields",):
        return fields()
    # a specific field (dotted path in the schema)
    _schema = active_schema()
    if dotted in _schema:
        return _kv(f"Field {dotted}", [("column", dotted), ("kind", _schema[dotted].kind)])
    # a namespace
    if len(target) == 1 and target[0] in namespaces():
        return fields(target[0])
    # a primitive function
    if len(target) == 1 and target[0] in KNOWN_FUNCTIONS:
        lo, hi = KNOWN_FUNCTIONS[target[0]]
        axis, summary = _FUNC_META.get(target[0], ("", ""))
        return _kv(f"Function {target[0]}", [
            ("layer", "primitive"), ("axis", axis),
            ("args", str(lo) if lo == hi else f"{lo}..{hi}"), ("summary", summary),
        ])
    # a derived (stdlib macro) function
    if len(target) == 1 and target[0] in _stdlib_functions():
        return _kv(f"Function {target[0]}", [
            ("layer", "derived"), ("args", str(_stdlib_functions()[target[0]])),
            ("kind", "stdlib macro"),
        ])
    # a source
    if len(target) == 1 and target[0] in config.sources:
        return _source_detail(target[0], config.sources[target[0]])
    return CatalogResult(
        f"Unknown catalog target: '{dotted}'",
        pl.DataFrame({"hint": ["try ? , ?fields, ?functions, ?sources, or ?<namespace>"]}),
    )


def catalog(config: Config = DEFAULT_CONFIG) -> CatalogResult:
    ns = namespaces()
    frame = pl.DataFrame({
        "namespace": ns,
        "fields": [len(fields(n).frame) for n in ns],
    })
    title = (f"Trail catalog - {len(active_schema())} fields across {len(ns)} namespaces, "
             f"{len(KNOWN_FUNCTIONS)} primitive + {len(_stdlib_functions())} derived functions, "
             f"{len(config.sources)} source(s). Use ?<namespace>, ?functions, ?sources, ?<name> for detail.")
    return CatalogResult(title, frame)


def evaluate_meta(node, config: Config = DEFAULT_CONFIG) -> CatalogResult:
    """Route a parsed meta-command AST node to the discovery core."""
    match node:
        case ast.MetaCatalog():
            return catalog(config)
        case ast.MetaDescribe():
            return describe(node.target, config)
    raise TypeError(f"not a meta-command: {type(node).__name__}")
