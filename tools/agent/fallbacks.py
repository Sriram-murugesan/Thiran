"""
tools/agent/fallbacks.py
========================
Implements agent.clarify and agent.unknown intents.

These are not real "tools" — they are terminal no-op handlers that
produce TTS responses without performing any external action.
They exist in the tool registry so the Dispatcher can route them
uniformly, without special-casing in the orchestration layer.
"""

from __future__ import annotations

from typing import Any

from core.models import ToolResult
from tools.base import BaseTool, SlotField


class ClarifyTool(BaseTool):
    """
    Handles agent.clarify — emitted when LLM confidence is below threshold.
    Asks the user to repeat their request.
    """

    @property
    def name(self) -> str:
        return "agent.clarify"

    @property
    def description(self) -> str:
        return (
            "Use this when the user's intent is unclear or ambiguous. "
            "Do NOT use for unrecognized commands — use agent.unknown for those."
        )

    @property
    def slot_fields(self) -> list[SlotField]:
        return []

    async def _execute_impl(
        self,
        slots: dict[str, Any],
        request_id: str,
    ) -> ToolResult:
        return ToolResult(
            request_id=request_id,
            tool_name=self.name,
            success=True,
            message="I didn't quite catch that. Could you please repeat?",
            duration_ms=0,
        )


class UnknownTool(BaseTool):
    """
    Handles agent.unknown — emitted when the LLM cannot map the input
    to any known intent.
    """

    @property
    def name(self) -> str:
        return "agent.unknown"

    @property
    def description(self) -> str:
        return (
            "Use this when the user's request does not match any available tool. "
            "Only use as a last resort."
        )

    @property
    def slot_fields(self) -> list[SlotField]:
        return []

    async def _execute_impl(
        self,
        slots: dict[str, Any],
        request_id: str,
    ) -> ToolResult:
        return ToolResult(
            request_id=request_id,
            tool_name=self.name,
            success=True,
            message="I'm not sure how to help with that. Try asking me something else.",
            duration_ms=0,
        )
