"""Tab completion for the Trail REPL."""
from __future__ import annotations


# Built-in field prefixes (namespace.* patterns)
FIELD_NAMESPACES = [
    "income.",
    "balance.",
    "cash.",
    "price.",
    "meta.",
    "fmp.",
    "gmd.",
    "views.",
]

# Built-in function names
FUNCTIONS = [
    "lag", "roll_mean", "roll_sum", "roll_std", "roll_var",
    "roll_max", "roll_min", "roll_quantile", "roll_median",
    "ewm_mean", "ewm_std", "decay_linear", "resample",
    "asof", "ttm", "trailing",
    "to_annual", "to_quarterly", "to_monthly", "to_daily",
    "cummax", "cumsum", "cumprod", "cummin",
    "cross_mean", "cross_std", "cross_rank", "cross_percentile",
    "avg2", "pct_change", "cagr", "z_score", "rank", "percentile",
    "log", "exp", "sqrt", "abs", "round",
    "weighted_score", "fwd_return",
]

# REPL keywords
KEYWORDS = [
    "def", "model", "signal", "universe", "export", "score",
    "desc", "on_missing", "weight",
    "if", "else", "and", "or", "not", "in",
    "true", "false", "annual", "quarterly", "monthly", "daily",
    "skip", "zero", "median",
]


class TrailCompleter:
    """Completer for Trail REPL using prompt_toolkit API."""

    def __init__(self, session) -> None:  # noqa: ANN001
        self.session = session

    def complete(self, document, event, *args, **kwargs):  # noqa: ANN001
        """Return completions for the current word under cursor."""
        from prompt_toolkit.completion import Completion

        text = document.text_before_cursor
        if not text:
            return

        # Find the current token being typed
        # Handle both simple names and dotted paths
        last_dot = text.rfind(".")
        if last_dot == -1:
            prefix = text
        else:
            # Check if we're inside a dotted path
            prefix = text[last_dot + 1:]

        if not prefix:
            return

        prefix_lower = prefix.lower()
        completions = []

        # Field namespaces (only when we see a dot prefix)
        if last_dot >= 0:
            for ns in FIELD_NAMESPACES:
                if ns.startswith(prefix_lower):
                    completions.append(Completion(ns, start_position=-len(prefix)))
        else:
            # Top-level completions
            for kw in KEYWORDS:
                if kw.startswith(prefix_lower) and kw != prefix:
                    completions.append(Completion(kw, start_position=-len(prefix), display_meta="keyword"))

            for fn in FUNCTIONS:
                if fn.startswith(prefix_lower) and fn != prefix:
                    completions.append(Completion(fn, start_position=-len(prefix), display_meta="function"))

            # User-defined names
            for name in self.session.definitions.keys():
                if name.startswith(prefix_lower) and name != prefix:
                    completions.append(Completion(name, start_position=-len(prefix), display_meta="defined"))

            # Sort and yield
            completions.sort(key=lambda c: c.text)
            for comp in completions:
                yield comp
