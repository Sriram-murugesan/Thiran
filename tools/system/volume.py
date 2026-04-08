"""
tools/system/volume.py
======================
Implements system.volume intent.

Strategy: Uses pulsectl (PulseAudio Python bindings) to control system volume.
Falls back to pactl subprocess if pulsectl is unavailable.

Dependency: `pip install pulsectl` (Linux PulseAudio only).
"""

from __future__ import annotations

import asyncio
from typing import Any

from core.models import ToolResult
from tools.base import BaseTool, SlotField

_VOLUME_MIN = 0
_VOLUME_MAX = 100


class SystemVolumeTool(BaseTool):

    @property
    def name(self) -> str:
        return "system.volume"

    @property
    def description(self) -> str:
        return "Set, increase, decrease, mute, or unmute the system audio volume."

    @property
    def slot_fields(self) -> list[SlotField]:
        return [
            SlotField(
                name="level",
                type_hint="integer",
                required=False,
                description="Volume level 0–100. Required for set/up/down.",
                default=None,
            ),
            SlotField(
                name="direction",
                type_hint="string",
                required=False,
                description="How to apply the volume change.",
                default="set",
                allowed_values=["up", "down", "set", "mute", "unmute"],
            ),
        ]

    async def _execute_impl(
        self,
        slots: dict[str, Any],
        request_id: str,
    ) -> ToolResult:
        log = self._log(request_id)
        direction = str(slots.get("direction", "set")).lower()
        level = slots.get("level")

        log.info("system.volume.apply", direction=direction, level=level)

        if direction == "mute":
            rc = await self._run_pactl("set-sink-mute", "@DEFAULT_SINK@", "1")
            if rc != 0:
                return self._pactl_error(request_id, "mute")
            return ToolResult(
                request_id=request_id,
                tool_name=self.name,
                success=True,
                message="System audio muted.",
                duration_ms=0,
            )

        if direction == "unmute":
            rc = await self._run_pactl("set-sink-mute", "@DEFAULT_SINK@", "0")
            if rc != 0:
                return self._pactl_error(request_id, "unmute")
            return ToolResult(
                request_id=request_id,
                tool_name=self.name,
                success=True,
                message="System audio unmuted.",
                duration_ms=0,
            )

        # set / up / down — level is guaranteed int [0,100] by Validator
        if level is None:
            return ToolResult(
                request_id=request_id,
                tool_name=self.name,
                success=False,
                message="I need a volume level to adjust. Please try again.",
                error_code="MISSING_SLOT",
                duration_ms=0,
            )

        level_int = int(level)

        if direction == "up":
            pactl_value = f"+{level_int}%"
            message = f"Increasing system volume by {level_int} percent."
        elif direction == "down":
            pactl_value = f"-{level_int}%"
            message = f"Decreasing system volume by {level_int} percent."
        else:
            pactl_value = f"{level_int}%"
            message = f"Setting system volume to {level_int} percent."

        rc = await self._run_pactl(
            "set-sink-volume", "@DEFAULT_SINK@", pactl_value
        )
        if rc != 0:
            return self._pactl_error(request_id, direction)

        return ToolResult(
            request_id=request_id,
            tool_name=self.name,
            success=True,
            message=message,
            data={"direction": direction, "level": level_int},
            duration_ms=0,
        )

    async def _run_pactl(self, *args: str) -> int:
        """Run a pactl command and return its exit code."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "pactl", *args,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()
            return proc.returncode or 0
        except FileNotFoundError:
            return 127

    def _pactl_error(self, request_id: str, action: str) -> ToolResult:
        return ToolResult(
            request_id=request_id,
            tool_name=self.name,
            success=False,
            message=(
                f"I couldn't {action} the system volume. "
                "Make sure PulseAudio is running."
            ),
            error_code="PACTL_ERROR",
            duration_ms=0,
        )
