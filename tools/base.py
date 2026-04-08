"""
tools/base.py
=============
Abstract base class that every tool must implement.

Design rules enforced by this contract:
  - execute() is async and must complete within the dispatcher's timeout window.
  - execute() must NEVER raise. All exceptions must be caught internally and
    returned as ToolResult(success=False, ...).
  - The 'name' property must exactly match an IntentName enum value string.
  - The 'slots_schema' property is used to auto-generate the LLM system prompt.
    Keep descriptions precise — they directly influence LLM intent extraction.
  - Tools have no access to the Validator, Dispatcher, NLU, or Agent layers.
    They receive only typed slot dicts and return ToolResult.

Logging:
  - Each tool binds request_id at the start of execute() for traceability.
  - Use self._log(request_id) to get a pre-bound structlog logger.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import structlog

from core.models import ToolResult


class SlotField:
    """
    Descriptor for a single slot field in a tool's SlotSchema.

    Used to generate the LLM system prompt and to document tool expectations.
    Not used for runtime validation (that is the Validator's responsibility).
    """

    __slots__ = (
        "name",
        "type_hint",
        "required",
        "description",
        "default",
        "allowed_values",
    )

    def __init__(
        self,
        *,
        name: str,
        type_hint: str,
        required: bool,
        description: str,
        default: Any = None,
        allowed_values: list[str] | None = None,
    ) -> None:
        self.name = name
        self.type_hint = type_hint
        self.required = required
        self.description = description
        self.default = default
        self.allowed_values = allowed_values

    def to_prompt_fragment(self) -> str:
        """Render this field as a human-readable prompt fragment for the LLM."""
        req_label = "REQUIRED" if self.required else f"OPTIONAL (default: {self.default!r})"
        fragment = f"    - {self.name} ({self.type_hint}, {req_label}): {self.description}"
        if self.allowed_values:
            fragment += f". Allowed values: {self.allowed_values}"
        return fragment


class BaseTool(ABC):
    """
    Abstract base for all voice agent tools.

    Subclasses must implement:
      - name: str property → must match IntentName enum value exactly
      - description: str property → used in LLM system prompt
      - slots: list[SlotField] property → slot schema for prompt generation
      - _execute_impl: core business logic, may raise freely

    Subclasses must NOT override:
      - execute() → the public entry point (handles timing, error catching, logging)
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Exact IntentName enum value this tool handles (e.g. 'spotify.play').
        Must match IntentName.<VARIANT>.value precisely.
        """
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """
        One-sentence description of what this tool does.
        Injected verbatim into the LLM system prompt.
        Keep it unambiguous and action-oriented.
        """
        ...

    @property
    @abstractmethod
    def slot_fields(self) -> list[SlotField]:
        """
        Ordered list of slots this tool accepts.
        Used to generate the LLM system prompt and tool documentation.
        """
        ...

    @abstractmethod
    async def _execute_impl(
        self,
        slots: dict[str, Any],
        request_id: str,
    ) -> ToolResult:
        """
        Core business logic. Receives validated, coerced slots.

        MAY raise any exception — BaseTool.execute() will catch all exceptions
        and materialize them as ToolResult(success=False, ...).
        """
        ...

    async def _dry_run_impl(
        self,
        slots: dict[str, Any],
        request_id: str,
    ) -> ToolResult:
        """
        Simulation logic. Default behaviour is to return what the tool would do.
        Subclasses may override this to provide more specific preview responses.
        """
        return ToolResult(
            request_id=request_id,
            tool_name=self.name,
            success=True,
            simulated=True,
            message=f"Simulation: I would execute {self.name} with {slots}." if slots else f"Simulation: I would execute {self.name}.",
            duration_ms=0,
        )

    # ------------------------------------------------------------------
    # Public entry point (do not override)
    # ------------------------------------------------------------------

    async def execute(
        self,
        slots: dict[str, Any],
        request_id: str,
        simulate: bool = False,
    ) -> ToolResult:
        """
        Public execution entry point called by the Dispatcher.

        Wraps _execute_impl with:
          - Wall-clock timing
          - Structured logging (start / success / failure)
          - Universal exception handling (tools must never crash the pipeline)

        Parameters
        ----------
        slots:
            Validated, coerced slot dict from ValidatedIntent.slots.
        request_id:
            Propagated request identifier for end-to-end tracing.

        Returns
        -------
        ToolResult
            Always returns a ToolResult — never raises.
        """
        log = self._log(request_id)
        log.info("tool.start", tool=self.name, slots=slots)

        start_ns = time.monotonic_ns()

        try:
            if simulate:
                result = await self._dry_run_impl(slots=slots, request_id=request_id)
            else:
                result = await self._execute_impl(slots=slots, request_id=request_id)
        except Exception as exc:  # noqa: BLE001
            duration_ms = _elapsed_ms(start_ns)
            log.exception(
                "tool.unhandled_exception",
                tool=self.name,
                duration_ms=duration_ms,
                exc_type=type(exc).__name__,
                exc_message=str(exc),
            )
            return ToolResult(
                request_id=request_id,
                tool_name=self.name,
                success=False,
                message=(
                    "Something went wrong while trying to do that. "
                    "Please try again."
                ),
                error_code="UNHANDLED_EXCEPTION",
                duration_ms=duration_ms,
            )

        duration_ms = _elapsed_ms(start_ns)

        if result.success:
            log.info(
                "tool.success",
                tool=self.name,
                duration_ms=duration_ms,
                message=result.message,
            )
        else:
            log.warning(
                "tool.failure",
                tool=self.name,
                duration_ms=duration_ms,
                error_code=result.error_code,
                message=result.message,
            )

        # Return a copy with the correct duration_ms (tools don't time themselves)
        return result.model_copy(update={"duration_ms": duration_ms})

    # ------------------------------------------------------------------
    # Helpers available to subclasses
    # ------------------------------------------------------------------

    def _log(self, request_id: str) -> structlog.stdlib.BoundLogger:
        """Return a structlog logger pre-bound with request_id and tool name."""
        return structlog.get_logger().bind(
            request_id=request_id,
            tool=self.name,
            layer="tool",
        )

    def to_prompt_block(self) -> str:
        """
        Render this tool as a prompt block for inclusion in the LLM system prompt.

        Example output:
            Tool: spotify.play
            Description: Play a song, artist, or album on Spotify.
            Slots:
              - query (string, REQUIRED): The song, artist, or album to play.
              - artist (string, OPTIONAL): Narrow the search by artist name.
        """
        lines = [
            f"Tool: {self.name}",
            f"Description: {self.description}",
        ]
        if self.slot_fields:
            lines.append("Slots:")
            for field in self.slot_fields:
                lines.append(field.to_prompt_fragment())
        else:
            lines.append("Slots: none")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _elapsed_ms(start_ns: int) -> int:
    """Return elapsed wall-clock time in milliseconds since start_ns."""
    return (time.monotonic_ns() - start_ns) // 1_000_000
