"""Interactive loop for Trail REPL using prompt_toolkit."""
from __future__ import annotations

import sys
from pathlib import Path

from trail.repl.format import truncate_table
from trail.repl.session import ReplSession


def run_repl(session: ReplSession) -> None:
    """Run the interactive REPL loop.

    Features:
    - Tab completion (fields, functions, keywords, user-defined names)
    - Syntax highlighting
    - Multi-line input support
    - History
    - Formatted output with truncation
    - Meta-commands: ? (catalog), ?fields, ?functions, ?sources, ?<target>
    """
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.completion import ThreadedCompleter
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.shortcuts import print_formatted_text, HTML
        from prompt_toolkit.styles import Style
    except ImportError:
        print("prompt_toolkit is required for the REPL.", file=sys.stderr)
        print("Install it with: pip install trail-lang[repl]", file=sys.stderr)
        sys.exit(1)

    # ── Completions ─────────────────────────────────────────────────
    from trail.repl.completer import TrailCompleter

    # ── Style ────────────────────────────────────────────────────────
    style = Style.from_dict({
        "prompt": "ansibrightblue bold",
        "continuation": "ansigreen",
        "search": "ansiyellow bold",
    })

    # ── Prompt template ─────────────────────────────────────────────
    def prompt() -> str:
        """Build the prompt string."""
        return "trail> "

    completer = TrailCompleter(session)

    # ── Key bindings ─────────────────────────────────────────────────
    bindings = KeyBindings()

    @bindings.add("tab")
    def event(event):  # noqa: ANN001
        """Handle tab: complete or insert whitespace."""
        buffer = event.current_buffer
        if buffer.complete_state:
            buffer.complete_next()
        else:
            buffer.insert_text("    ")

    # ── Print helpers ───────────────────────────────────────────────
    def show_result(df) -> None:  # noqa: ANN001
        """Display a DataFrame result."""
        if df is None:
            print_formatted_text(HTML("  <gray>(no output)</gray>"))
            return
        print_formatted_text(HTML(f"  <gray>({len(df)} rows × {len(df.columns)} cols)</gray>"))
        table = truncate_table(df)
        print_formatted_text(table)

    def show_error(message: str) -> None:
        """Display an error message."""
        print_formatted_text(HTML(f"  <red>Error: {message}</red>"))

    # ── Main loop ───────────────────────────────────────────────────
    session_obj = PromptSession(
        prompt=prompt,
        style=style,
        completer=ThreadedCompleter(completer),
        auto_suggest=AutoSuggestFromHistory(),
        key_bindings=bindings,
        history=FileHistory(str(Path.home() / ".trail_history")),
        search_prompt="  search> ",
    )

    print()
    print_formatted_text(HTML(
        "  <cyan>Trail REPL v0.17.0</cyan>"
    ))
    print_formatted_text(HTML(
        "  <gray>Expression language for financial indicators.</gray>"
    ))
    print_formatted_text(HTML("  <gray>Try: income.revenue / balance.total_assets</gray>"))
    print_formatted_text(HTML("  <gray>Or:  margin = income.operating_income / income.revenue</gray>"))
    print_formatted_text(HTML("  <gray>Help:  ?  ?fields  ?functions  ?sources  ?income</gray>"))
    print_formatted_text(HTML("  <gray>Quit:  Ctrl-D or Ctrl-C twice</gray>"))
    print()

    while True:
        try:
            text = session_obj.prompt()
        except (KeyboardInterrupt, EOFError):
            print()  # newline before prompt
            break

        if not text.strip():
            continue

        # Handle exit commands
        if text.strip() in ("quit", "exit", "q"):
            print_formatted_text(HTML("  <gray>Bye.</gray>"))
            break

        # Process input through the session
        result = session.process_input(text)

        if result.is_error:
            show_error(result.message)
        elif result.is_result:
            show_result(result.value)
