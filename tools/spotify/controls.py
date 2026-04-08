"""
tools/spotify/controls.py
=========================
Implements spotify.pause, spotify.skip, spotify.volume.

Strategy: Uses xdotool (Linux) to send media key events to the Spotify window,
or falls back to D-Bus (playerctl) for robust media control on Linux desktops.
This is fully local — no API key required.

Dependency: playerctl must be installed (`sudo apt install playerctl`).
"""

from __future__ import annotations

import asyncio
from typing import Any

from core.models import ToolResult
from tools.base import BaseTool, SlotField


class _PlayerctlMixin:
    """Shared helper for running playerctl commands."""

    _PLAYER = "spotify"

    async def _run_playerctl(
        self,
        *args: str,
        request_id: str,
    ) -> tuple[int, str, str]:
        """
        Run a playerctl command and return (returncode, stdout, stderr).
        Never raises — exceptions are caught and returned as non-zero returncode.
        """
        cmd = ["playerctl", f"--player={self._PLAYER}", *args]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await proc.communicate()
            return (
                proc.returncode or 0,
                stdout_b.decode().strip(),
                stderr_b.decode().strip(),
            )
        except FileNotFoundError:
            return (
                127,
                "",
                "playerctl not found. Install via: sudo apt install playerctl",
            )
        except Exception as exc:  # noqa: BLE001
            return (1, "", str(exc))


# ---------------------------------------------------------------------------
# spotify.pause
# ---------------------------------------------------------------------------


class SpotifyPauseTool(_PlayerctlMixin, BaseTool):

    @property
    def name(self) -> str:
        return "spotify.pause"

    @property
    def description(self) -> str:
        return "Pause or resume Spotify playback."

    @property
    def slot_fields(self) -> list[SlotField]:
        return []

    async def _execute_impl(
        self,
        slots: dict[str, Any],
        request_id: str,
    ) -> ToolResult:
        log = self._log(request_id)
        rc, stdout, stderr = await self._run_playerctl("play-pause", request_id=request_id)
        log.debug("spotify.pause.playerctl", rc=rc, stdout=stdout, stderr=stderr)

        if rc != 0:
            return ToolResult(
                request_id=request_id,
                tool_name=self.name,
                success=False,
                message="Spotify doesn't appear to be running. Please open it first.",
                error_code="PLAYERCTL_ERROR",
                data={"stderr": stderr},
                duration_ms=0,
            )

        return ToolResult(
            request_id=request_id,
            tool_name=self.name,
            success=True,
            message="Toggling Spotify playback.",
            duration_ms=0,
        )


# ---------------------------------------------------------------------------
# spotify.skip
# ---------------------------------------------------------------------------


class SpotifySkipTool(_PlayerctlMixin, BaseTool):

    @property
    def name(self) -> str:
        return "spotify.skip"

    @property
    def description(self) -> str:
        return "Skip to the next or previous track on Spotify."

    @property
    def slot_fields(self) -> list[SlotField]:
        return [
            SlotField(
                name="direction",
                type_hint="string",
                required=False,
                description="Which direction to skip.",
                default="next",
                allowed_values=["next", "previous"],
            )
        ]

    async def _execute_impl(
        self,
        slots: dict[str, Any],
        request_id: str,
    ) -> ToolResult:
        log = self._log(request_id)
        direction = str(slots.get("direction", "next")).lower()
        playerctl_cmd = "next" if direction == "next" else "previous"
        direction_label = "next" if direction == "next" else "previous"

        rc, stdout, stderr = await self._run_playerctl(playerctl_cmd, request_id=request_id)
        log.debug("spotify.skip.playerctl", direction=direction, rc=rc, stderr=stderr)

        if rc != 0:
            return ToolResult(
                request_id=request_id,
                tool_name=self.name,
                success=False,
                message="I couldn't skip the track. Is Spotify running?",
                error_code="PLAYERCTL_ERROR",
                data={"stderr": stderr},
                duration_ms=0,
            )

        return ToolResult(
            request_id=request_id,
            tool_name=self.name,
            success=True,
            message=f"Skipping to the {direction_label} track.",
            data={"direction": direction},
            duration_ms=0,
        )


# ---------------------------------------------------------------------------
# spotify.volume
# ---------------------------------------------------------------------------


class SpotifyVolumeTool(_PlayerctlMixin, BaseTool):

    @property
    def name(self) -> str:
        return "spotify.volume"

    @property
    def description(self) -> str:
        return "Adjust, mute, or unmute the Spotify playback volume."

    @property
    def slot_fields(self) -> list[SlotField]:
        return [
            SlotField(
                name="level",
                type_hint="integer",
                required=False,
                description="Volume level (0–100). Required for set/up/down, not for mute/unmute.",
                default=None,
            ),
            SlotField(
                name="direction",
                type_hint="string",
                required=False,
                description="How to adjust volume.",
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

        if direction == "mute":
            rc, _, stderr = await self._run_playerctl("volume", "0", request_id=request_id)
            if rc != 0:
                return self._playerctl_error(request_id, stderr)
            return ToolResult(
                request_id=request_id,
                tool_name=self.name,
                success=True,
                message="Spotify muted.",
                duration_ms=0,
            )

        if direction == "unmute":
            rc, _, stderr = await self._run_playerctl("volume", "1.0", request_id=request_id)
            if rc != 0:
                return self._playerctl_error(request_id, stderr)
            return ToolResult(
                request_id=request_id,
                tool_name=self.name,
                success=True,
                message="Spotify unmuted.",
                duration_ms=0,
            )

        # set / up / down
        # playerctl volume takes a value 0.0–1.0
        if level is None:
            return self._playerctl_error(request_id, "Missing volume level")
        pct = float(int(level)) / 100.0  # level is validated int [0,100]

        if direction == "up":
            volume_arg = f"+{pct:.2f}"
            message = f"Increasing Spotify volume by {int(level)} percent."
        elif direction == "down":
            volume_arg = f"-{pct:.2f}"
            message = f"Decreasing Spotify volume by {int(level)} percent."
        else:  # set
            volume_arg = f"{pct:.2f}"
            message = f"Setting Spotify volume to {int(level)} percent."

        log.debug("spotify.volume.playerctl", volume_arg=volume_arg)
        rc, _, stderr = await self._run_playerctl("volume", volume_arg, request_id=request_id)

        if rc != 0:
            return self._playerctl_error(request_id, stderr)

        return ToolResult(
            request_id=request_id,
            tool_name=self.name,
            success=True,
            message=message,
            data={"direction": direction, "level": level},
            duration_ms=0,
        )

    def _playerctl_error(self, request_id: str, stderr: str) -> ToolResult:
        return ToolResult(
            request_id=request_id,
            tool_name=self.name,
            success=False,
            message="I couldn't adjust Spotify volume. Is Spotify running?",
            error_code="PLAYERCTL_ERROR",
            data={"stderr": stderr},
            duration_ms=0,
        )
