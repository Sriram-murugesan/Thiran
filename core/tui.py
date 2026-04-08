import asyncio
import sys
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

class UIStream:
    """A stream object that forwards write() calls to the Textual RichLog."""
    def __init__(self, tui_app):
        self.tui_app = tui_app

    def write(self, s):
        # Prevent writing empty newlines from disrupting logs
        clean_s = s.rstrip('\r\n')
        if clean_s:
            try:
                import threading
                if getattr(self.tui_app, "_thread_id", None) == threading.get_ident():
                    self.tui_app.log_message(clean_s)
                else:
                    self.tui_app.call_from_thread(self.tui_app.log_message, clean_s)
            except Exception:
                pass # App might not be fully mounted yet

    def flush(self):
        pass

class StatusPanel(Static):
    """Panel displaying the current agent state (e.g., Recording, Processing...)."""
    status_text = reactive("Initializing...")

    def render(self) -> str:
        return f"[b green]Active Status:[/b green]\n\n{self.status_text}"

class ThiranTUI(App):
    """The main TUI terminal application for Thiran."""
    
    CSS = """
    Screen {
        layout: horizontal;
    }
    #status-container {
        width: 30%;
        height: 100%;
        background: $boost;
        border-right: vkey $panel;
        padding: 1 2;
        content-align: center middle;
    }
    #log-panel {
        width: 70%;
        height: 100%;
        background: $surface;
    }
    """
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit")
    ]

    def __init__(self, start_agent_coro, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_agent_coro = start_agent_coro
        global _TUI_APP
        _TUI_APP = self

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="status-container"):
                yield StatusPanel(id="status-panel")
            yield RichLog(id="log-panel", markup=True, highlight=True, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(StatusPanel).status_text = "System Ready\nWaiting for Voice Agent..."
        # Run the existing agent `.start()` natively 
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

    def log_message(self, message: str):
        """Write to the rich log."""
        try:
            self.query_one(RichLog).write(message)
        except Exception:
            pass

