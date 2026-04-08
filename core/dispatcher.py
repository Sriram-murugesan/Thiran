"""
core/dispatcher.py
==================
Action Dispatcher — routes ValidatedIntent to the correct BaseTool.

Responsibilities:
  - Maintain the ToolRegistry (intent name → BaseTool instance mapping)
  - Enforce rate limits per tool namespace
  - Run before_execute / after_execute hooks
  - Wrap tool execution in an asyncio timeout
  - Guarantee a ToolResult is always returned (never raises)

Design rules:
  - The Dispatcher NEVER validates. It receives only ValidatedIntent.
  - The Dispatcher NEVER calls the LLM. It is purely a routing layer.
  - Per-tool timeout is enforced here, not inside tools.
  - Rate limit state is in-process only (no external store required).
  - Tools are registered at construction time; registry is immutable at runtime.

Logging:
  - Every dispatch is logged with request_id, intent, duration, and result.
  - Hook violations are logged as warnings, not errors.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Awaitable

import structlog

from core.models import IntentName, ToolResult, ValidatedIntent, ValidationErrorCode
from tools.base import BaseTool

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

BeforeHook = Callable[[str, dict], Awaitable[None] | None]
AfterHook = Callable[[str, ToolResult], Awaitable[None] | None]


# ---------------------------------------------------------------------------
# Rate Limit Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RateLimitPolicy:
    """
    Token-bucket rate limit for a tool namespace (e.g. 'spotify').

    max_calls: maximum number of calls allowed within window_seconds.
    window_seconds: rolling time window in seconds.
    """

    max_calls: int
    window_seconds: float


# Default policies keyed on the domain prefix of the intent name.
# Override by passing custom policies to Dispatcher.__init__.
DEFAULT_RATE_LIMITS: dict[str, RateLimitPolicy] = {
    "spotify": RateLimitPolicy(max_calls=10, window_seconds=60.0),
    "youtube": RateLimitPolicy(max_calls=20, window_seconds=60.0),
    "system":  RateLimitPolicy(max_calls=30, window_seconds=60.0),
    "web":     RateLimitPolicy(max_calls=20, window_seconds=60.0),
    "agent":   RateLimitPolicy(max_calls=100, window_seconds=60.0),
}


class _RateLimiter:
    """
    Per-namespace sliding-window rate limiter.
    Stores timestamps of recent calls in a deque; pops stale entries on check.
    Thread-safe within a single asyncio event loop.
    """

    def __init__(self, policies: dict[str, RateLimitPolicy]) -> None:
        self._policies = policies
        # domain → deque of call timestamps (monotonic seconds)
        self._windows: dict[str, deque[float]] = defaultdict(deque)

    def check(self, intent: str) -> bool:
        """
        Returns True if the call is allowed under the rate limit.
        Advances the window and records the call if allowed.
        """
        domain = intent.split(".")[0]
        policy = self._policies.get(domain)
        if policy is None:
            return True  # no policy → unrestricted

        now = time.monotonic()
        window = self._windows[domain]

        # Evict timestamps outside the rolling window
        cutoff = now - policy.window_seconds
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= policy.max_calls:
            return False

        window.append(now)
        return True


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class Dispatcher:
    """
    Routes ValidatedIntent instances to registered BaseTool implementations.

    Parameters
    ----------
    tools:
        Sequence of BaseTool instances to register. Each tool's `name`
        property must exactly match an IntentName enum value.
    tool_timeout_seconds:
        Per-tool execution timeout. Applies uniformly to all tools.
        Recommended: 5.0s. Timer tool tasks are exempt (they run detached).
    rate_limit_policies:
        Overrides for DEFAULT_RATE_LIMITS. Merged, not replaced.
    before_hooks:
        Coroutine or sync callables invoked before each tool execution.
        Signature: (intent: str, slots: dict) -> None
        Raise DispatchBlocked to prevent tool execution.
    after_hooks:
        Coroutine or sync callables invoked after each tool execution.
        Signature: (intent: str, result: ToolResult) -> None
        Exceptions in after_hooks are logged and suppressed.
    """

    def __init__(
        self,
        tools: list[BaseTool],
        *,
        tool_timeout_seconds: float = 5.0,
        rate_limit_policies: dict[str, RateLimitPolicy] | None = None,
        before_hooks: list[BeforeHook] | None = None,
        after_hooks: list[AfterHook] | None = None,
    ) -> None:
        self._registry: dict[str, BaseTool] = {}
        self._timeout = tool_timeout_seconds
        self._before_hooks: list[BeforeHook] = before_hooks or []
        self._after_hooks: list[AfterHook] = after_hooks or []

        merged_policies = {**DEFAULT_RATE_LIMITS, **(rate_limit_policies or {})}
        self._rate_limiter = _RateLimiter(merged_policies)

        for tool in tools:
            self._register(tool)

    def _register(self, tool: BaseTool) -> None:
        """Register a tool. Raises ValueError on duplicate name."""
        if tool.name in self._registry:
            raise ValueError(
                f"Duplicate tool registration: '{tool.name}'. "
                "Each tool name must be unique."
            )
        # Validate name is a known IntentName
        try:
            IntentName(tool.name)
        except ValueError as exc:
            raise ValueError(
                f"Tool name '{tool.name}' is not a valid IntentName. "
                "Update IntentName enum first."
            ) from exc
        self._registry[tool.name] = tool

    def registered_tools(self) -> list[BaseTool]:
        """Return all registered tools in registration order."""
        return list(self._registry.values())

    def build_system_prompt_tools_block(self) -> str:
        """
        Generate the tools section of the LLM system prompt.
        Called by the NLU layer to keep the prompt in sync with the registry.
        """
        blocks = [tool.to_prompt_block() for tool in self._registry.values()]
        return "\n\n".join(blocks)

    async def dispatch(self, validated_intent: ValidatedIntent) -> ToolResult:
        """
        Route a ValidatedIntent to its tool and return the ToolResult.

        Always returns a ToolResult — never raises.

        Parameters
        ----------
        validated_intent:
            A fully validated intent from the Validator layer.

        Returns
        -------
        ToolResult
            The result from the matched tool, or an error ToolResult if
            the dispatch was blocked, rate-limited, or timed out.
        """
        log = structlog.get_logger().bind(
            request_id=validated_intent.request_id,
            intent=validated_intent.intent.value,
            layer="dispatcher",
        )

        intent_str = validated_intent.intent.value
        slots = validated_intent.slots
        request_id = validated_intent.request_id

        # ------------------------------------------------------------------
        # 1. Rate limit check
        # ------------------------------------------------------------------
        if not self._rate_limiter.check(intent_str):
            log.warning("dispatcher.rate_limited", intent=intent_str)
            return ToolResult(
                request_id=request_id,
                tool_name=intent_str,
                success=False,
                message="I'm being asked to do that too often. Please wait a moment.",
                error_code=ValidationErrorCode.DISPATCH_BLOCKED.value,
                duration_ms=0,
            )

        # ------------------------------------------------------------------
        # 2. Lookup tool (guaranteed by Validator — no KeyError possible)
        # ------------------------------------------------------------------
        tool = self._registry.get(intent_str)
        if tool is None:
            # Defensive guard: should never be reached if enum and registry are in sync.
            log.error("dispatcher.tool_not_found", intent=intent_str)
            return ToolResult(
                request_id=request_id,
                tool_name=intent_str,
                success=False,
                message="I'm not sure how to handle that request.",
                error_code="TOOL_NOT_FOUND",
                duration_ms=0,
            )

        # ------------------------------------------------------------------
        # 3. Before hooks
        # ------------------------------------------------------------------
        try:
            for hook in self._before_hooks:
                result = hook(intent_str, slots)
                if asyncio.iscoroutine(result):
                    await result
        except DispatchBlocked as exc:
            log.warning("dispatcher.blocked_by_hook", reason=str(exc))
            return ToolResult(
                request_id=request_id,
                tool_name=intent_str,
                success=False,
                message=str(exc) or "That action is currently blocked.",
                error_code=ValidationErrorCode.DISPATCH_BLOCKED.value,
                duration_ms=0,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("dispatcher.before_hook_error", exc=str(exc))
            # Non-blocking hook error — proceed with dispatch

        # ------------------------------------------------------------------
        # 4. Tool execution with timeout
        # ------------------------------------------------------------------
        log.info("dispatcher.executing", tool=intent_str)
        start_ns = time.monotonic_ns()

        try:
            tool_result = await asyncio.wait_for(
                tool.execute(slots=slots, request_id=request_id, simulate=validated_intent.simulate),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            duration_ms = (time.monotonic_ns() - start_ns) // 1_000_000
            log.error(
                "dispatcher.timeout",
                tool=intent_str,
                timeout_s=self._timeout,
                duration_ms=duration_ms,
            )
            tool_result = ToolResult(
                request_id=request_id,
                tool_name=intent_str,
                success=False,
                message="That took too long. Please try again.",
                error_code=ValidationErrorCode.TOOL_TIMEOUT.value,
                duration_ms=duration_ms,
            )

        # ------------------------------------------------------------------
        # 5. After hooks (suppressed — never fail the pipeline)
        # ------------------------------------------------------------------
        for after_hook in self._after_hooks:
            try:
                result_after = after_hook(intent_str, tool_result)
                if asyncio.iscoroutine(result_after):
                    await result_after
            except Exception as exc:  # noqa: BLE001
                log.warning("dispatcher.after_hook_error", exc=str(exc))

        log.info(
            "dispatcher.complete",
            tool=intent_str,
            success=tool_result.success,
            duration_ms=tool_result.duration_ms,
        )
        return tool_result


# ---------------------------------------------------------------------------
# Exception used by before_hooks to block dispatch
# ---------------------------------------------------------------------------


class DispatchBlocked(Exception):
    """
    Raise inside a before_hook to prevent the tool from executing.
    The exception message is used as the TTS response.
    """
