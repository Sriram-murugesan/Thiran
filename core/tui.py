import asyncio
import sys
import time
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, RichLog, Static
from textual.containers import Vertical, Horizontal
from textual.reactive import reactive

# Singleton TUI app reference for easy access
_TUI_APP = None


def get_tui():
    """Retrieve the global TUI app instance."""
    return _TUI_APP


def set_tui_status(text: str):
    """Update the status pane if the TUI is running, otherwise print to stdout."""
    if _TUI_APP:
        _TUI_APP.update_status(text)
    else:
        print(text)


def set_tui_intent(intent: str | None, success: bool | None = None, message: str | None = None):
    """Update the intent display panel (intent, outcome, response)."""
    if _TUI_APP:
        _TUI_APP.update_intent(intent, success, message)


class UIStream:
    """A stream object that forwards write() calls to the Textual RichLog."""

    def __init__(self, tui_app):
        self.tui_app = tui_app

    def write(self, s):
        # Prevent writing empty newlines from disrupting logs
        clean_s = s.rstrip("\r\n")
        if clean_s:
            try:
                import threading

                if getattr(self.tui_app, "_thread_id", None) == threading.get_ident():
                    self.tui_app.log_message(clean_s)
                else:
                    self.tui_app.call_from_thread(self.tui_app.log_message, clean_s)
            except Exception:
                pass  # App might not be fully mounted yet

    def flush(self):
        pass


class IntentPanel(Static):
    """Panel showing the last resolved intent and its outcome."""

    intent = reactive("—")
    outcome = reactive("")
    response = reactive("")

    def render(self) -> str:
        parts = [f"[b cyan]Intent:[/b cyan]  {self.intent}"]
        if self.outcome:
            colour = "green" if self.outcome == "success" else "red"
            parts.append(f"[b {colour}]Outcome:[/b {colour}]  {self.outcome}")
        if self.response:
            # Truncate long responses for the panel
            resp = self.response if len(self.response) <= 120 else self.response[:117] + "..."
            parts.append(f"[b white]Response:[/b white]  {resp}")
        return "\n".join(parts)


class StatusPanel(Static):
    """Panel displaying the current agent state (e.g., Recording, Processing...)."""

    status_text = reactive("Initializing...")

    def render(self) -> str:
        return f"[b green]Status[/b green]\n\n{self.status_text}"


class ThiranTUI(App):
    """The main TUI terminal application for Thiran."""

    CSS = """
    #left-panel {
        width: 35%;
        height: 100%;
        background: $boost;
        border-right: solid $panel;
    }
    #status-panel {
        height: auto;
        padding: 1 2;
    }
    #intent-panel {
        height: auto;
        padding: 1 2;
        border-top: solid $panel;
    }
    #log-panel {
        width: 65%;
        height: 100%;
        background: $surface;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, start_agent_coro, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_agent_coro = start_agent_coro
        global _TUI_APP
        _TUI_APP = self

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="left-panel"):
                yield StatusPanel(id="status-panel")
                yield IntentPanel(id="intent-panel")
            yield RichLog(id="log-panel", markup=True, highlight=True, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(StatusPanel).status_text = "System Ready.\nHold [b]SPACE[/b] to speak."
        self.query_one(IntentPanel).intent = "—"
        self.run_worker(self.start_agent_coro, thread=False)

    def update_status(self, new_status: str):
        """Thread-safe status update."""
        def _update():
            try:
                self.query_one(StatusPanel).status_text = new_status
            except Exception:
                pass

        import threading

        if getattr(self, "_thread_id", None) == threading.get_ident():
            _update()
        else:
            self.call_from_thread(_update)

    def update_intent(self, intent: str | None, success: bool | None, message: str | None):
        """Thread-safe intent panel update after a pipeline completes."""
        def _update():
            try:
                panel = self.query_one(IntentPanel)
                panel.intent = intent or "—"
                panel.outcome = ""
                panel.response = ""
                if success is not None:
                    panel.outcome = "success" if success else "failure"
                if message:
                    panel.response = message
            except Exception:
                pass

        import threading

        if getattr(self, "_thread_id", None) == threading.get_ident():
            _update()
        else:
            self.call_from_thread(_update)

    def log_message(self, message: str):
        """Write to the rich log."""
        try:
            self.query_one(RichLog).write(message)
        except Exception:
            pass
