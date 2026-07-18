"""Map trail/Python exceptions to a structured {error:{code,message}} an agent can act on.
Trail errors already carry a leading E-CODE in their message; we surface it verbatim."""
from __future__ import annotations

import re
import sys

from lark.exceptions import UnexpectedInput, VisitError

from trail.config import ConfigError
from trail.macro import TrailFunctionError
from trail.pipeline import TrailImportError

_CODE_RE = re.compile(r"^(E-[A-Z0-9-]+)")


def to_error(exc: Exception) -> dict:
    msg = str(exc)
    m = _CODE_RE.match(msg)
    if m:                                  # trail error already coded (E-IMPORT-*, E-FIELD-*, ...)
        code = m.group(1)
    elif isinstance(exc, (UnexpectedInput, VisitError)):
        code = "E-SYNTAX"
    elif isinstance(exc, TrailImportError):
        code = "E-IMPORT"
    elif isinstance(exc, TrailFunctionError):
        code = "E-FUNC"
    elif isinstance(exc, ConfigError):
        code = "E-CONFIG"
    else:
        print(f"trail-mcp internal error: {type(exc).__name__}: {msg}", file=sys.stderr)
        code = "E-INTERNAL"
    return {"error": {"code": code, "message": msg}}
