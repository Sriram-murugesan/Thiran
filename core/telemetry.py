"""
core/telemetry.py
=================
Observability layer for structured tracing and JSON logging.

Responsibilities:
  - Provide a mutable `AgentTrace` envelope to capture latency, status,
    and intent tracking per interaction.
  - Expose a `TelemetrySink` interface for flushing completed traces.
  - Provide a default `StructlogTelemetrySink` that logs the canonical
    JSON interaction summary schema.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
import structlog


def _utcnow() -> datetime:
    """Return timezone-aware UTC current time."""
    return datetime.now(tz=timezone.utc)


class AgentTrace(BaseModel):
    """
    Mutable envelope storing the full telemetry lifecycle of a single voice interaction.
    Passed contextually (or accessed sequentially) in the orchestration layer to 
    aggregate end-to-end metrics.
    """

    model_config = ConfigDict(validate_assignment=True)

    request_id: str = Field(description="Unique UUID for this interaction.")
    initiated_at: datetime = Field(default_factory=_utcnow)
    
    # Textual data
    transcript: str | None = Field(default=None)
    resolved_intent: str | None = Field(default=None)
    
    # Completion bounds
    success: bool = Field(default=False)
    error_code: str | None = Field(default=None)
    
    # Systems review markers
    cache_hit: bool = Field(default=False)
    recording_duration_ms: int = Field(default=0)
    time_to_response_ms: int = Field(default=0)
    
    # Latency tracking (milliseconds)
    latencies: dict[str, int] = Field(
        default_factory=lambda: {
            "asr": 0,
            "nlu": 0,
            "validator": 0,
            "tool": 0,
            "tts": 0,
            "total": 0,
        }
    )

    def record_latency(self, stage: str, duration_ms: int) -> None:
        """Record the latency for a specific stage."""
        if stage in self.latencies:
            self.latencies[stage] = duration_ms
        else:
            self.latencies[stage] = duration_ms

    def snapshot(self) -> dict[str, Any]:
        """Convert trace into a standard JSON dictionary matching the schema."""
        return {
            "timestamp": self.initiated_at.isoformat(),
            "level": "ERROR" if not self.success and self.error_code else "INFO",
            "event": "agent.interaction_summary",
            "request_id": self.request_id,
            "intent": self.resolved_intent,
            "success": self.success,
            "error_code": self.error_code,
            "cache_hit": self.cache_hit,
            "recording_duration_ms": self.recording_duration_ms,
            "time_to_response_ms": self.time_to_response_ms,
            "latency_ms": self.latencies,
        }


class TelemetrySink:
    """Base interface for routing completed traces to a destination."""
    
    def flush(self, trace: AgentTrace) -> None:
        """Process and route the fully populated trace."""
        raise NotImplementedError


class StructlogTelemetrySink(TelemetrySink):
    """Default sink that writes the json canonical schema to structlog."""

    def __init__(self) -> None:
        self._log = structlog.get_logger().bind(layer="telemetry")

    def flush(self, trace: AgentTrace) -> None:
        """Write the canonical event log."""
        data = trace.snapshot()
        level = data.pop("level")
        event = data.pop("event")
        
        if level == "ERROR":
            self._log.error(event, **data)
        elif level == "WARNING":
            self._log.warning(event, **data)
        else:
            self._log.info(event, **data)
