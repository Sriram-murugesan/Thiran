"""
core/models.py
==============
Canonical data contracts for the entire voice agent pipeline.

Layer ownership:
  AudioSegment      → VAD layer output
  TranscriptResult  → ASR layer output
  IntentPayload     → NLU layer output (raw, untrusted LLM output)
  ValidatedIntent   → Validator layer output (trusted, dispatch-ready)
  ValidationError   → Validator layer failure path
  ToolResult        → Tool execution layer output
  AgentRequest      → Top-level request envelope (carries request_id)
  AgentResponse     → Top-level response envelope

All models use Pydantic v2 with strict mode where appropriate.
request_id is propagated through every layer for end-to-end traceability.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


def _new_request_id() -> str:
    """Generate a new unique request identifier."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class IntentName(str, Enum):
    """
    Exhaustive set of valid intent identifiers.

    Naming convention: <domain>.<action>
    New intents MUST be added here before they can appear in any payload.
    The LLM system prompt is generated from this enum, so this is the
    single source of truth for the agent's capability surface.
    """

    # Spotify
    SPOTIFY_PLAY = "spotify.play"
    SPOTIFY_PAUSE = "spotify.pause"
    SPOTIFY_SKIP = "spotify.skip"
    SPOTIFY_VOLUME = "spotify.volume"

    # YouTube
    YOUTUBE_PLAY = "youtube.play"
    YOUTUBE_SEARCH = "youtube.search"

    # System
    SYSTEM_OPEN_APP = "system.open_app"
    SYSTEM_VOLUME = "system.volume"
    SYSTEM_TIMER = "system.timer"
    SYSTEM_INSTALL_APP = "system.install_app"
    SYSTEM_UPDATE_APP = "system.update_app"

    # Web
    WEB_SEARCH = "web.search"
    WEB_ANSWER = "web.answer"

    # Gmail
    GMAIL_READ = "gmail.read"

    # Agent-level (never executed as tools, handled by the agent loop)
    AGENT_CLARIFY = "agent.clarify"
    AGENT_UNKNOWN = "agent.unknown"


class SkipDirection(str, Enum):
    NEXT = "next"
    PREVIOUS = "previous"


class VolumeDirection(str, Enum):
    UP = "up"
    DOWN = "down"
    SET = "set"
    MUTE = "mute"
    UNMUTE = "unmute"


class TimerUnit(str, Enum):
    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"


class SearchEngine(str, Enum):
    GOOGLE = "google"
    DUCKDUCKGO = "duckduckgo"


class ValidationErrorCode(str, Enum):
    """
    Machine-readable error codes produced by the Validator layer.
    Downstream consumers (error handler, logging) key on these codes.
    """

    INVALID_JSON = "INVALID_JSON"
    SCHEMA_VIOLATION = "SCHEMA_VIOLATION"
    UNKNOWN_INTENT = "UNKNOWN_INTENT"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    MISSING_SLOT = "MISSING_SLOT"
    INVALID_SLOT_VALUE = "INVALID_SLOT_VALUE"
    UNKNOWN_APP = "UNKNOWN_APP"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    DISPATCH_BLOCKED = "DISPATCH_BLOCKED"


class PipelineStage(str, Enum):
    """Identifies which stage of the pipeline produced an artefact."""

    VAD = "vad"
    ASR = "asr"
    NLU = "nlu"
    VALIDATOR = "validator"
    DISPATCHER = "dispatcher"
    TOOL = "tool"
    RESPONSE_BUILDER = "response_builder"
    TTS = "tts"


# ---------------------------------------------------------------------------
# VAD Layer Output
# ---------------------------------------------------------------------------


class AudioSegment(BaseModel):
    """
    Output of the Voice Activity Detection (VAD) layer.
    Carries raw PCM audio bytes for a single detected speech utterance.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(
        default_factory=_new_request_id,
        description="Unique identifier for this voice interaction.",
    )
    audio_bytes: bytes = Field(
        description="Raw PCM audio bytes (16-bit signed, mono, 16kHz).",
    )
    sample_rate: int = Field(
        default=16_000,
        ge=8_000,
        le=48_000,
        description="Sample rate of the audio in Hz.",
    )
    channels: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Number of audio channels (1=mono, 2=stereo).",
    )
    duration_ms: int = Field(
        ge=0,
        description="Duration of the audio segment in milliseconds.",
    )
    captured_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the segment was captured.",
    )

    @field_validator("audio_bytes")
    @classmethod
    def audio_must_not_be_empty(cls, v: bytes) -> bytes:
        if not v:
            raise ValueError("audio_bytes must not be empty.")
        return v


# ---------------------------------------------------------------------------
# ASR Layer Output
# ---------------------------------------------------------------------------


class TranscriptResult(BaseModel):
    """
    Output of the Automatic Speech Recognition (ASR) layer.
    Carries the raw transcript and quality signals for downstream use.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(description="Propagated from AudioSegment.request_id.")
    text: str = Field(description="Normalized transcript text from the ASR engine.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="ASR engine confidence score (0.0–1.0).",
    )
    language: str = Field(
        default="en",
        min_length=2,
        max_length=5,
        description="Detected language code (BCP-47, e.g. 'en', 'en-US').",
    )
    transcribed_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when transcription completed.",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Transcript text must not be empty or whitespace-only.")
        return stripped


# ---------------------------------------------------------------------------
# NLU Layer Output — Raw (Untrusted)
# ---------------------------------------------------------------------------


class IntentPayload(BaseModel):
    """
    Raw output of the NLU (LLM) layer.

    IMPORTANT: This model represents UNTRUSTED LLM output.
    It must pass through the Validator before being used by any
    downstream layer. No business logic should key on this directly.

    The LLM is constrained to produce JSON matching this schema via a
    system prompt generated from the IntentName enum and slot schemas.
    """

    model_config = ConfigDict(
        # Do not silently ignore extra fields — strip them with a warning instead.
        # Handled explicitly in model_validator below for observability.
        extra="ignore",
    )

    intent: str = Field(
        description="Dot-namespaced intent string from the LLM (e.g. 'spotify.play')."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="LLM self-reported confidence in its intent mapping.",
    )
    slots: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict,
        description="Flat key-value map of extracted entities. No nesting allowed.",
    )
    raw_transcript: str | None = Field(
        default=None,
        description="Echo of the ASR transcript for debugging and cache keying.",
    )

    @field_validator("slots")
    @classmethod
    def slots_must_be_flat(
        cls, v: dict[str, Any]
    ) -> dict[str, str | int | float | bool | None]:
        for key, val in v.items():
            if isinstance(val, (dict, list)):
                raise ValueError(
                    f"Slot '{key}' contains a nested value ({type(val).__name__}). "
                    "All slot values must be scalar (str, int, float, bool, or null)."
                )
        return v


# ---------------------------------------------------------------------------
# Validator Layer Output — Trusted
# ---------------------------------------------------------------------------


class ValidatedIntent(BaseModel):
    """
    Output of the Validator layer.

    Represents a fully validated, schema-conformant intent that is safe to
    dispatch. Every field is resolved and guaranteed to be within bounds.
    The Dispatcher and all Tools consume ONLY this type — never IntentPayload.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(description="Propagated request identifier.")
    intent: IntentName = Field(
        description="Resolved and validated IntentName enum value."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Validated confidence score.",
    )
    slots: dict[str, str | int | float | bool | None] = Field(
        description=(
            "Validated and coerced slot values. "
            "Soft-fail coercions (clamping, defaults) are applied before "
            "this model is constructed. The validation log carries annotations."
        )
    )
    raw_transcript: str | None = Field(
        default=None,
        description="Original transcript for logging and cache keying.",
    )
    simulate: bool = Field(
        default=False,
        description="If True, instructs the dispatcher and tools to perform a dry-run without executing side effects.",
    )
    coercion_log: list[str] = Field(
        default_factory=list,
        description=(
            "Human-readable log of soft-fail mutations applied during validation "
            "(e.g. 'volume clamped from 150 to 100'). Empty on clean validation."
        ),
    )
    validated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when validation completed.",
    )

    @property
    def was_coerced(self) -> bool:
        """True if the Validator successfully injected context or clamped boundaries."""
        return len(self.coercion_log) > 0


# ---------------------------------------------------------------------------
# Validator Layer Output — Failure Path
# ---------------------------------------------------------------------------


class ValidationError(BaseModel):
    """
    Produced by the Validator layer when validation cannot be completed.

    Routed directly to the Error Handler / TTS — never reaches the Dispatcher.
    Carries enough context for structured logging and user-facing TTS messages.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(description="Propagated request identifier.")
    code: ValidationErrorCode = Field(description="Machine-readable error code.")
    field: str | None = Field(
        default=None,
        description="The specific field or slot name that failed validation, if applicable.",
    )
    message: str = Field(
        description=(
            "Human-readable error description. "
            "Used directly as the TTS response to the user. "
            "Must be polite, concise, and actionable."
        )
    )
    raw_payload: str | None = Field(
        default=None,
        description=(
            "Original raw LLM output string. "
            "Stored for debug logging only — never displayed to the user."
        ),
    )
    raw_transcript: str | None = Field(
        default=None,
        description="ASR transcript at time of failure, for correlation.",
    )
    failed_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when validation failed.",
    )


# ---------------------------------------------------------------------------
# Tool Layer Output
# ---------------------------------------------------------------------------


class ToolResult(BaseModel):
    """
    Output of every Tool's execute() method.

    Tools MUST return this type — they must never raise exceptions.
    All exceptions must be caught internally and materialized as
    ToolResult(success=False, ...).

    The message field is TTS-ready: it should be a complete, natural-language
    sentence that makes sense spoken aloud with no visual context.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(description="Propagated request identifier.")
    tool_name: str = Field(
        description="The IntentName string that invoked this tool (e.g. 'spotify.play')."
    )
    success: bool = Field(
        description="True if the action completed as intended."
    )
    message: str = Field(
        description=(
            "TTS-ready natural language summary of the result. "
            "Must be a complete sentence, appropriate for spoken output."
        )
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional structured payload for future UI/logging use. "
            "Not used by TTS. Not guaranteed to be stable between versions."
        ),
    )
    simulated: bool = Field(
        default=False,
        description="True if the outcome represents a safe dry-run instead of real execution.",
    )
    error_code: str | None = Field(
        default=None,
        description=(
            "Machine-readable error identifier when success=False. "
            "Use ValidationErrorCode values or tool-specific strings."
        ),
    )
    duration_ms: int = Field(
        ge=0,
        description="Wall-clock execution time of the tool in milliseconds.",
    )
    executed_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when tool execution completed.",
    )

    @model_validator(mode="after")
    def error_code_required_on_failure(self) -> "ToolResult":
        if not self.success and not self.error_code:
            raise ValueError(
                "error_code must be set when success=False. "
                "Use a ValidationErrorCode value or a tool-specific string."
            )
        return self


# ---------------------------------------------------------------------------
# Context Layer Output
# ---------------------------------------------------------------------------


class InteractionContext(BaseModel):
    """
    Short-term contextual memory to support cross-turn slot filling.
    Maintained securely by the pure agent loop and passed to the Validator.
    """

    model_config = ConfigDict(frozen=True)

    intent: IntentName = Field(
        description="The resolved IntentName enum value of the previous turn.",
    )
    slots: dict[str, str | int | float | bool | None] = Field(
        description="The fully validated and coerced slots from the previous turn.",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when this context was yielded.",
    )

    @property
    def age_seconds(self) -> float:
        """Returns the age of this context instance in seconds."""
        # Using tz-aware arithmetic since _utcnow() yields UTC timezone
        return (_utcnow() - self.timestamp).total_seconds()


# ---------------------------------------------------------------------------
# Top-Level Request / Response Envelopes
# ---------------------------------------------------------------------------


class AgentRequest(BaseModel):
    """
    Top-level request envelope created at push-to-talk activation.

    Carries the AudioSegment and the request_id that will be propagated
    through every layer of the pipeline for end-to-end traceability.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(
        default_factory=_new_request_id,
        description="UUID v4 identifier for this full agent interaction.",
    )
    simulate: bool = Field(
        default=False,
        description="If True, bypasses system execution to safely dry-run intent routing and TTS preview.",
    )
    audio_segment: AudioSegment = Field(
        description="The voice input captured by the VAD layer.",
    )
    initiated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when push-to-talk was released and processing began.",
    )


class AgentResponse(BaseModel):
    """
    Top-level response envelope produced at the end of the pipeline.

    Carries the TTS-ready text and the full result chain for observability.
    The agent loop uses this to drive TTS playback and structured logging.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(description="Propagated from AgentRequest.request_id.")
    response_text: str = Field(
        description="TTS-ready final response text to be spoken to the user.",
    )
    intent: IntentName | None = Field(
        default=None,
        description="The resolved intent, if one was successfully validated.",
    )
    tool_result: ToolResult | None = Field(
        default=None,
        description="The ToolResult, if a tool was successfully executed.",
    )
    validation_error: ValidationError | None = Field(
        default=None,
        description="Set if the pipeline terminated at the Validator layer.",
    )
    pipeline_success: bool = Field(
        description=(
            "True if the full pipeline completed without a hard failure. "
            "A successful tool result counts as True even if the tool action itself "
            "returned success=False (e.g. 'Spotify is busy') — the pipeline ran correctly."
        )
    )
    total_duration_ms: int = Field(
        ge=0,
        description="Total wall-clock time from audio capture to response text, in ms.",
    )
    completed_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the pipeline completed.",
    )

    @model_validator(mode="after")
    def exactly_one_terminal_state(self) -> "AgentResponse":
        """
        Enforce that the response carries exactly one terminal state:
        either a tool_result or a validation_error — not both, not neither
        (unless it's an agent.clarify / agent.unknown which produces only response_text).
        """
        has_tool_result = self.tool_result is not None
        has_validation_error = self.validation_error is not None
        if has_tool_result and has_validation_error:
            raise ValueError(
                "AgentResponse cannot carry both tool_result and validation_error."
            )
        return self
