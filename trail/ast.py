"""Typed AST for Trail. All nodes are frozen dataclasses compared by value."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Literal:
    value: float | int | str | bool


@dataclass(frozen=True)
class NameRef:
    name: str


@dataclass(frozen=True)
class FieldRef:
    path: tuple[str, ...]
    source: str | None = None

    @property
    def column(self) -> str:
        return ".".join(self.path)


@dataclass(frozen=True)
class BinOp:
    op: str  # add sub mul div mod pow
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Compare:
    op: str  # eq ne gt lt ge le
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class In:
    item: "Expr"
    options: tuple[Literal, ...]


@dataclass(frozen=True)
class BoolOp:
    op: str  # and or
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Not:
    operand: "Expr"


@dataclass(frozen=True)
class Neg:
    operand: "Expr"


@dataclass(frozen=True)
class Coalesce:
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Ternary:
    value: "Expr"
    cond: "Expr"
    orelse: "Expr"


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple["Expr", ...]
    kwargs: tuple[tuple[str, "Expr"], ...] = field(default=())
    by: tuple[str, ...] | None = None


# --- declaration nodes (Task 3) ---


@dataclass(frozen=True)
class Assignment:
    name: str
    expr: "Expr"
    export: bool = False


@dataclass(frozen=True)
class ScoreCase:
    value: "Expr"
    cond: "Expr"


@dataclass(frozen=True)
class ScoreDecl:
    name: str
    weight: float
    cases: tuple[ScoreCase, ...]
    default: "Expr"


@dataclass(frozen=True)
class UniverseDecl:
    name: str
    root: tuple[str, ...]
    where: "Expr | None" = None


@dataclass(frozen=True)
class ModelDecl:
    name: str
    universe: str | None
    frequency: str
    desc: str | None
    on_missing: str
    statements: tuple["Assignment | ScoreDecl", ...]


@dataclass(frozen=True)
class SignalDecl:
    name: str
    universe: str | None
    frequency: str
    expr: "Expr"


@dataclass(frozen=True)
class OpaqueDecl:
    kind: str  # strategy | backtest | learn | import
    name: str
    text: str = ""


@dataclass(frozen=True)
class FuncDef:
    """A user/stdlib function: a non-recursive expression macro over its parameters."""

    name: str
    params: tuple[str, ...]
    body: "Expr"


@dataclass(frozen=True)
class Program:
    decls: tuple[object, ...]


# --- REPL-dialect meta-commands (never valid in a model file; see reference §2.1) ---


@dataclass(frozen=True)
class MetaCatalog:
    """`?` - the full catalog summary."""


@dataclass(frozen=True)
class MetaDescribe:
    """`?<target>` - describe a namespace, field, function, or source
    (targets `functions` / `sources` list those categories)."""

    target: tuple[str, ...]


Expr = (
    Literal | NameRef | FieldRef | BinOp | Compare | In | BoolOp | Not | Neg | Coalesce | Ternary | Call
)
