"""
tools/system/timer.py
=====================
Implements system.timer intent.

Strategy:
  - Tracks active timers in a module-level registry (thread-safe).
  - Each timer fires on a background asyncio.Task.
  - On expiry, queues a TTS alert via a shared asyncio.Queue injected
    at registry time — the agent loop drains this queue.
  - Enforces a max of MAX_ACTIVE_TIMERS concurrent timers.

The timer_alert_queue is set once at agent startup via set_alert_queue().
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from core.models import ToolResult
from tools.base import BaseTool, SlotField

MAX_ACTIVE_TIMERS: int = 5

# Module-level registry: timer_id → asyncio.Task
_active_timers: dict[str, asyncio.Task[None]] = {}

# Injected by agent.py at startup — carries TTS alert strings
_alert_queue: asyncio.Queue[str] | None = None


def set_alert_queue(queue: asyncio.Queue[str]) -> None:
    """
    Inject the shared alert queue at agent startup.
    Must be called before any timer fires; not thread-safe beyond initial setup.
    """
    global _alert_queue  # noqa: PLW0603
    _alert_queue = queue


def active_timer_count() -> int:
    """Return the count of currently running timers."""
    _prune_completed()
    return len(_active_timers)


def _prune_completed() -> None:
    """Remove finished/cancelled tasks from the registry."""
    done = [tid for tid, task in _active_timers.items() if task.done()]
    for tid in done:
        del _active_timers[tid]


class SystemTimerTool(BaseTool):

    @property
    def name(self) -> str:
        return "system.timer"

    @property
    def description(self) -> str:
        return "Set a countdown timer for a specified duration."

    @property
    def slot_fields(self) -> list[SlotField]:
        return [
            SlotField(
                name="duration",
                type_hint="integer",
                required=True,
                description="How long the timer should run.",
            ),
            SlotField(
                name="unit",
                type_hint="string",
                required=True,
                description="Time unit for the duration.",
                allowed_values=["seconds", "minutes", "hours"],
            ),
            SlotField(
                name="total_seconds",
                type_hint="integer",
                required=False,
                description="Pre-computed total seconds (set by Validator).",
            ),
        ]

    async def _execute_impl(
        self,
        slots: dict[str, Any],
        request_id: str,
    ) -> ToolResult:
        log = self._log(request_id)

        duration = int(slots["duration"])
        unit = str(slots["unit"])
        total_seconds = int(slots.get("total_seconds") or self._to_seconds(duration, unit))

        _prune_completed()

        if len(_active_timers) >= MAX_ACTIVE_TIMERS:
            log.warning("system.timer.max_reached", count=len(_active_timers))
            return ToolResult(
                request_id=request_id,
                tool_name=self.name,
                success=False,
                message=(
                    f"You already have {MAX_ACTIVE_TIMERS} timers running. "
                    "Please wait for one to finish."
                ),
                error_code="MAX_TIMERS_EXCEEDED",
                duration_ms=0,
            )

        timer_id = str(uuid.uuid4())[:8]
        label = self._human_label(duration, unit)

        log.info(
            "system.timer.start",
            timer_id=timer_id,
            duration=duration,
            unit=unit,
            total_seconds=total_seconds,
        )

        task = asyncio.create_task(
            self._countdown(
                timer_id=timer_id,
                total_seconds=total_seconds,
                label=label,
                request_id=request_id,
            )
        )
        _active_timers[timer_id] = task

        return ToolResult(
            request_id=request_id,
            tool_name=self.name,
            success=True,
            message=f"Timer set for {label}. I'll let you know when it's done.",
            data={
                "timer_id": timer_id,
                "duration": duration,
                "unit": unit,
                "total_seconds": total_seconds,
            },
            duration_ms=0,
        )

    async def _countdown(
        self,
        timer_id: str,
        total_seconds: int,
        label: str,
        request_id: str,
    ) -> None:
        """Background coroutine — sleeps then enqueues a TTS alert."""
        log = self._log(request_id).bind(timer_id=timer_id)
        try:
            await asyncio.sleep(total_seconds)
            alert = f"Your {label} timer is done."
            log.info("system.timer.fired", alert=alert)
            if _alert_queue is not None:
                await _alert_queue.put(alert)
        except asyncio.CancelledError:
            log.info("system.timer.cancelled", timer_id=timer_id)
        finally:
            _active_timers.pop(timer_id, None)

    @staticmethod
    def _to_seconds(duration: int, unit: str) -> int:
        multipliers = {"seconds": 1, "minutes": 60, "hours": 3_600}
        return duration * multipliers.get(unit, 1)

    @staticmethod
    def _human_label(duration: int, unit: str) -> str:
        singular = unit.rstrip("s")
        label_unit = unit if duration != 1 else singular
        return f"{duration} {label_unit}"
