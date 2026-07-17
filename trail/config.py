"""trail.yaml loading and validation. Normative schema: docs/reference.md §10."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

#: a source name is used verbatim as a `#source` column-name tag (see trail.fieldname), so it must
#: not contain the codec's delimiters (`.`/`@`/`#`) or the tag would be ambiguous.
_SOURCE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class SourceSpec:
    name: str
    driver: str
    options: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    sources: dict[str, SourceSpec]
    precedence: dict[str, list[str]]
    periods: tuple[int, int] | None = None
    #: when True, a non-conforming source panel is a hard error (E-SOURCE-PANEL);
    #: when False (default), deviations are warned and coerced (W-SOURCE-PANEL).
    strict: bool = False
    #: point-in-time placement. "auto" (default) = place each value by its known-date coordinate
    #: where a source supplies one (lookahead-safe); "naive" = ignore coordinates and place every
    #: value at its period-end (pure fundamental analysis). A source may also set options.pit.
    pit: str = "auto"

    def __post_init__(self) -> None:
        # a source name is a verbatim `#source` column tag, so guard it on EVERY construction
        # (load_config and programmatic), not just the YAML path.
        for name in self.sources:
            if not _SOURCE_NAME_RE.match(name):
                raise ConfigError(
                    f"E-SOURCE-NAME source name {name!r} is invalid; a source name must match "
                    f"[A-Za-z0-9_-]+ (no '.', '@', or '#') so the #source column tag stays unambiguous"
                )


DEFAULT_CONFIG = Config(
    sources={"fixture": SourceSpec("fixture", "trail.sources.fixture")},
    precedence={"default": ["fixture"]},
)


def load_config(path: str | None = None) -> Config:
    if path is None:
        if Path("trail.yaml").exists():
            path = "trail.yaml"
        else:
            return DEFAULT_CONFIG
    raw = yaml.safe_load(Path(path).read_text()) or {}
    src_raw = raw.get("sources") or {}
    sources = {
        name: SourceSpec(name, spec["driver"], spec.get("options") or {})
        for name, spec in src_raw.items()
    }
    precedence = {k: list(v) for k, v in (raw.get("precedence") or {}).items()}
    if not precedence:
        precedence = {"default": list(sources)}
    if "default" not in precedence:
        raise ConfigError("precedence.default is required")
    for ns, chain in precedence.items():
        for s in chain:
            if s not in sources:
                raise ConfigError(
                    f"E-SOURCE-UNKNOWN precedence.{ns} references undeclared source '{s}'"
                )
    panel = raw.get("panel") or {}
    p = panel.get("periods")
    periods = (int(p[0]), int(p[1])) if p else None
    strict = bool(panel.get("strict", False))
    pit = str(panel.get("pit", "auto")).lower()
    if pit not in ("auto", "naive"):
        raise ConfigError(f"panel.pit must be 'auto' or 'naive', got {pit!r}")
    return Config(sources, precedence, periods, strict, pit)
