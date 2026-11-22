"""
tools/system/open_app.py
========================
Implements system.open_app intent.

Strategy:
  1. Look up app_name in known_apps.yaml to resolve canonical command.
  2. If not found, attempt to launch app_name directly via subprocess.
  3. Report success/failure via ToolResult.

The known_apps.yaml allowlist is loaded once at import time and cached.
Its path is resolved from the config/ directory relative to the project root.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from core.models import ToolResult
from tools.base import BaseTool, SlotField

# Absolute path to known_apps.yaml, resolved from this file's location
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "known_apps.yaml"


@lru_cache(maxsize=1)
def _load_known_apps() -> dict[str, str]:
    """
    Load and cache the alias→command mapping from known_apps.yaml.

    Returns an empty dict if the file is missing or malformed,
    allowing the tool to fall back to raw subprocess launch.
    """
    if not _CONFIG_PATH.exists():
        return {}
    try:
        with _CONFIG_PATH.open() as fh:
            data = yaml.safe_load(fh) or {}
        return {str(k).lower(): str(v) for k, v in data.items()}
    except Exception:  # noqa: BLE001
        return {}


class SystemOpenAppTool(BaseTool):

    @property
    def name(self) -> str:
        return "system.open_app"

    @property
    def description(self) -> str:
        return "Open a desktop application by name."

    @property
    def slot_fields(self) -> list[SlotField]:
        return [
            SlotField(
                name="app_name",
                type_hint="string",
                required=True,
                description=(
                    "The name of the application to open, e.g. 'chrome', "
                    "'vs code', 'terminal', 'files'."
                ),
            )
        ]

    async def _execute_impl(
        self,
        slots: dict[str, Any],
        request_id: str,
    ) -> ToolResult:
        log = self._log(request_id)
        app_name: str = str(slots["app_name"]).lower().strip()

        known_apps = _load_known_apps()
        command = known_apps.get(app_name)

        if command:
            log.info("system.open_app.known", app=app_name, command=command)
        else:
            # Fall back to treating app_name as the command itself
            command = app_name
            log.info("system.open_app.fallback", app=app_name, command=command)

        try:
            # Non-blocking: detach the process from the agent
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,  # detach from agent's process group
            )
            # Brief wait to catch immediate failures (e.g. command not found)
            try:
                _, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=0.5)
                stderr = stderr_b.decode().strip() if stderr_b else ""
                if proc.returncode is not None and proc.returncode != 0:
                    log.warning(
                        "system.open_app.early_exit",
                        rc=proc.returncode,
                        stderr=stderr,
                    )
                    return ToolResult(
                        request_id=request_id,
                        tool_name=self.name,
                        success=False,
                        message=f"I couldn't open {app_name}. Is it installed?",
                        error_code="APP_LAUNCH_FAILED",
                        data={"app_name": app_name, "command": command, "stderr": stderr},
                        duration_ms=0,
                    )
            except asyncio.TimeoutError:
                # App is still running after 500ms → assume success
                pass

        except FileNotFoundError:
            log.error("system.open_app.not_found", app=app_name)
            return ToolResult(
                request_id=request_id,
                tool_name=self.name,
                success=False,
                message=f"I don't know how to open {app_name}.",
                error_code="UNKNOWN_APP",
                data={"app_name": app_name},
                duration_ms=0,
            )

        display_name = app_name.replace("_", " ").title()
        return ToolResult(
            request_id=request_id,
            tool_name=self.name,
            success=True,
            message=f"Opening {display_name}.",
            data={"app_name": app_name, "command": command},
            duration_ms=0,
        )
