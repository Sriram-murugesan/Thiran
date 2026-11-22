"""
core/validator.py
=================
Validator layer — the hard trust boundary between the NLU (LLM) and Dispatcher.

Responsibilities:
  1. Parse raw JSON string from the LLM (may be malformed)
  2. Validate root IntentPayload schema
  3. Resolve intent to a known IntentName enum value
  4. Gate on confidence threshold
  5. Apply per-intent slot validation (required fields, types, ranges, enums)
  6. Apply soft-fail coercions (clamping, defaulting) with audit log
  7. Emit ValidatedIntent (trusted) or ValidationError (failure path)

Design constraints:
  - Pure functions only — no I/O, no async, no global state
  - No imports from tools/, dispatcher/, or agent layers
  - All coercions are explicit and logged — nothing is silently mutated
  - Every exit path returns a typed result; never raises to the caller

Logging:
  - Uses structlog with bound request_id on every log record
  - Uses structlog.stdlib.BoundLogger type hint for clarity
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from core.models import (
    IntentName,
    IntentPayload,
    SearchEngine,
    SkipDirection,
    TimerUnit,
    ValidatedIntent,
    ValidationError,
    ValidationErrorCode,
    VolumeDirection,
    InteractionContext,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum LLM confidence to dispatch an intent.
#: Below this threshold the intent is re-routed to agent.clarify.
CONFIDENCE_THRESHOLD: float = 0.50

#: Maximum timer duration in seconds (24 hours).
MAX_TIMER_SECONDS: int = 86_400

#: Volume range enforced for both system and Spotify volume tools.
VOLUME_MIN: int = 0
VOLUME_MAX: int = 100

# ---------------------------------------------------------------------------
# Internal type aliases
# ---------------------------------------------------------------------------

_Slots = dict[str, str | int | float | bool | None]
_CoercionLog = list[str]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate(
    raw_json: str,
    request_id: str,
    raw_transcript: str | None = None,
    simulate: bool = False,
    context: InteractionContext | None = None,
) -> ValidatedIntent | ValidationError:
    """
    Validate raw LLM JSON output and return a typed result.

    Parameters
    ----------
    raw_json:
        The unprocessed JSON string produced by the LLM.
    request_id:
        The request identifier to propagate into the result.
    raw_transcript:
        The ASR transcript that produced this intent, for logging and
        cache-keying. Attached to both success and failure results.

    Returns
    -------
    ValidatedIntent
        When the payload fully passes validation (with optional soft coercions).
    ValidationError
        When a hard validation failure occurs at any step.
    """
    log: structlog.stdlib.BoundLogger = structlog.get_logger().bind(
        request_id=request_id,
        layer="validator",
    )

    log.debug("validator.start", raw_json_length=len(raw_json))

    # Step 1 — Parse JSON
    parse_result = _parse_json(raw_json, request_id, raw_transcript, log)
    if isinstance(parse_result, ValidationError):
        return parse_result
    raw_data: dict[str, Any] = parse_result

    # Step 2 — Strip unknown root keys (warn, do not fail)
    raw_data = _strip_unknown_root_keys(raw_data, log)

    # Step 3 — Validate root schema (required fields, types)
    schema_result = _validate_root_schema(raw_data, request_id, raw_transcript, log)
    if isinstance(schema_result, ValidationError):
        return schema_result
    payload: IntentPayload = schema_result

    # Step 4 — Resolve intent to enum
    intent_result = _resolve_intent(payload, request_id, raw_transcript, log)
    if isinstance(intent_result, ValidationError):
        return intent_result
    intent: IntentName = intent_result

    # Step 5 — Confidence gate
    if payload.confidence < CONFIDENCE_THRESHOLD:
        log.warning(
            "validator.low_confidence",
            confidence=payload.confidence,
            threshold=CONFIDENCE_THRESHOLD,
            intent=intent.value,
        )
        # Re-route to clarify — not a hard failure
        return _build_clarify_intent(
            request_id=request_id,
            raw_transcript=raw_transcript,
            confidence=payload.confidence,
            simulate=simulate,
        )

    # Step 6 — Per-intent slot validation (includes soft-fail coercions)
    slots_result = _validate_slots(
        intent=intent,
        raw_slots=payload.slots,
        request_id=request_id,
        raw_transcript=raw_transcript,
        context=context,
        log=log,
    )
    if isinstance(slots_result, ValidationError):
        return slots_result
    validated_slots, coercion_log = slots_result

    log.info(
        "validator.success",
        intent=intent.value,
        confidence=payload.confidence,
        coercions=coercion_log,
    )

    return ValidatedIntent(
        request_id=request_id,
        intent=intent,
        confidence=payload.confidence,
        slots=validated_slots,
        raw_transcript=raw_transcript or payload.raw_transcript,
        coercion_log=coercion_log,
        simulate=simulate,
    )


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------


def _parse_json(
    raw_json: str,
    request_id: str,
    raw_transcript: str | None,
    log: structlog.stdlib.BoundLogger,
) -> dict[str, Any] | ValidationError:
    """Step 1: Attempt to parse the raw string as JSON."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        log.error("validator.invalid_json", error=str(exc))
        return ValidationError(
            request_id=request_id,
            code=ValidationErrorCode.INVALID_JSON,
            field=None,
            message="I had trouble understanding that. Could you please repeat?",
            raw_payload=raw_json,
            raw_transcript=raw_transcript,
        )

    if not isinstance(data, dict):
        log.error("validator.invalid_json_type", got=type(data).__name__)
        return ValidationError(
            request_id=request_id,
            code=ValidationErrorCode.INVALID_JSON,
            field=None,
            message="I had trouble understanding that. Could you please repeat?",
            raw_payload=raw_json,
            raw_transcript=raw_transcript,
        )

    return data


def _strip_unknown_root_keys(
    data: dict[str, Any],
    log: structlog.stdlib.BoundLogger,
) -> dict[str, Any]:
    """Step 2: Remove keys not in the IntentPayload schema. Warn, do not fail."""
    known_keys = {"intent", "confidence", "slots", "raw_transcript"}
    unknown = set(data.keys()) - known_keys
    if unknown:
        log.warning("validator.unknown_root_keys", keys=sorted(unknown))
        return {k: v for k, v in data.items() if k in known_keys}
    return data


def _validate_root_schema(
    data: dict[str, Any],
    request_id: str,
    raw_transcript: str | None,
    log: structlog.stdlib.BoundLogger,
) -> IntentPayload | ValidationError:
    """Step 3: Validate the stripped dict against the IntentPayload Pydantic model."""
    try:
        payload = IntentPayload.model_validate(data)
    except Exception as exc:  # Pydantic ValidationError
        log.error("validator.schema_violation", error=str(exc))
        return ValidationError(
            request_id=request_id,
            code=ValidationErrorCode.SCHEMA_VIOLATION,
            field=None,
            message="I couldn't parse that request properly. Please try again.",
            raw_payload=json.dumps(data),
            raw_transcript=raw_transcript,
        )
    return payload


def _resolve_intent(
    payload: IntentPayload,
    request_id: str,
    raw_transcript: str | None,
    log: structlog.stdlib.BoundLogger,
) -> IntentName | ValidationError:
    """Step 4: Map the raw intent string to an IntentName enum value."""
    try:
        return IntentName(payload.intent)
    except ValueError:
        log.error("validator.unknown_intent", intent=payload.intent)
        return ValidationError(
            request_id=request_id,
            code=ValidationErrorCode.UNKNOWN_INTENT,
            field="intent",
            message="I'm not sure how to help with that.",
            raw_payload=payload.model_dump_json(),
            raw_transcript=raw_transcript,
        )


def _build_clarify_intent(
    request_id: str,
    raw_transcript: str | None,
    confidence: float,
    simulate: bool,
) -> ValidatedIntent:
    """Construct a ValidatedIntent that routes to the clarify handler."""
    return ValidatedIntent(
        request_id=request_id,
        intent=IntentName.AGENT_CLARIFY,
        confidence=confidence,
        slots={},
        raw_transcript=raw_transcript,
        simulate=simulate,
        coercion_log=[
            f"Intent re-routed to agent.clarify: confidence {confidence:.2f} "
            f"< threshold {CONFIDENCE_THRESHOLD}"
        ],
    )


# ---------------------------------------------------------------------------
# Per-intent slot validation (Step 6)
# ---------------------------------------------------------------------------


def _validate_slots(
    intent: IntentName,
    raw_slots: _Slots,
    request_id: str,
    raw_transcript: str | None,
    context: InteractionContext | None,
    log: structlog.stdlib.BoundLogger,
) -> tuple[_Slots, _CoercionLog] | ValidationError:
    """
    Dispatch to the per-intent slot validator.

    Returns either (validated_slots, coercion_log) or a ValidationError.
    Agent-level intents (clarify, unknown) have no slots and always pass.
    """
    context_coercion: _CoercionLog = []
    
    # 1. Backfill context
    if context and context.age_seconds <= 60.0:
        current_domain = intent.value.split(".")[0]
        context_domain = context.intent.value.split(".")[0]
        
        if intent == context.intent or current_domain == context_domain:
            for key, value in context.slots.items():
                if key not in raw_slots or raw_slots[key] is None or str(raw_slots[key]).strip() == "":
                    raw_slots[key] = value
                    context_coercion.append(f"Context injected missing slot '{key}' = {value!r}")
                    log.info("validator.context_fill", slot=key, value=value, age_seconds=round(context.age_seconds, 2))
        else:
            log.debug("validator.context_ignored", reason="intent mismatch", current=intent.value, context=context.intent.value)

    _slot_validators: dict[IntentName, _SlotValidatorFn] = {
        IntentName.SPOTIFY_PLAY: _slots_spotify_play,
        IntentName.SPOTIFY_PAUSE: _slots_no_slots,
        IntentName.SPOTIFY_SKIP: _slots_spotify_skip,
        IntentName.SPOTIFY_VOLUME: _slots_volume,
        IntentName.YOUTUBE_PLAY: _slots_query_required,
        IntentName.YOUTUBE_SEARCH: _slots_query_required,
        IntentName.SYSTEM_OPEN_APP: _slots_system_open_app,
        IntentName.SYSTEM_VOLUME: _slots_volume,
        IntentName.SYSTEM_TIMER: _slots_system_timer,
        IntentName.SYSTEM_INSTALL_APP: _slots_system_install_app,
        IntentName.SYSTEM_UPDATE_APP: _slots_system_update_app,
        IntentName.WEB_SEARCH: _slots_web_search,
        IntentName.WEB_ANSWER: _slots_query_required,
        IntentName.GMAIL_READ: _slots_gmail_read,
        IntentName.AGENT_CLARIFY: _slots_no_slots,
        IntentName.AGENT_UNKNOWN: _slots_no_slots,
    }

    validator_fn = _slot_validators.get(intent)
    if validator_fn is None:
        # Defensive: should be unreachable given enum exhaustiveness,
        # but guards against future enum additions without slot validators.
        log.error("validator.missing_slot_validator", intent=intent.value)
        return ValidationError(
            request_id=request_id,
            code=ValidationErrorCode.SCHEMA_VIOLATION,
            field="intent",
            message="This feature isn't configured yet. Please try something else.",
            raw_payload=None,
            raw_transcript=raw_transcript,
        )

    result = validator_fn(
        raw_slots,
        request_id,
        raw_transcript,
        log,
    )
    
    if isinstance(result, ValidationError):
        return result
    else:
        validated_slots, specific_coercion = result
        return validated_slots, context_coercion + specific_coercion


# ---------------------------------------------------------------------------
# Type alias for slot validator functions
# ---------------------------------------------------------------------------

from typing import Callable  # noqa: E402 (deliberate placement after module body)

_SlotValidatorFn = Callable[
    [_Slots, str, str | None, structlog.stdlib.BoundLogger],
    "tuple[_Slots, _CoercionLog] | ValidationError",
]


# ---------------------------------------------------------------------------
# Individual slot validators
# ---------------------------------------------------------------------------


def _slots_no_slots(
    slots: _Slots,
    request_id: str,
    raw_transcript: str | None,
    log: structlog.stdlib.BoundLogger,
) -> tuple[_Slots, _CoercionLog]:
    """Validator for intents that require no slots."""
    return {}, []


def _slots_query_required(
    slots: _Slots,
    request_id: str,
    raw_transcript: str | None,
    log: structlog.stdlib.BoundLogger,
) -> tuple[_Slots, _CoercionLog] | ValidationError:
    """
    Validator for intents that require only a non-empty 'query' slot.
    Used by: youtube.play, youtube.search.
    """
    query = slots.get("query")

    if query is None or not str(query).strip():
        log.warning("validator.missing_slot", slot="query")
        return ValidationError(
            request_id=request_id,
            code=ValidationErrorCode.MISSING_SLOT,
            field="query",
            message="What would you like me to search for?",
            raw_payload=None,
            raw_transcript=raw_transcript,
        )

    return {"query": str(query).strip()}, []


def _slots_spotify_play(
    slots: _Slots,
    request_id: str,
    raw_transcript: str | None,
    log: structlog.stdlib.BoundLogger,
) -> tuple[_Slots, _CoercionLog] | ValidationError:
    """Validator for spotify.play."""
    query = slots.get("query")

    if query is None or not str(query).strip():
        log.warning("validator.missing_slot", slot="query", intent="spotify.play")
        return ValidationError(
            request_id=request_id,
            code=ValidationErrorCode.MISSING_SLOT,
            field="query",
            message="What would you like to play on Spotify?",
            raw_payload=None,
            raw_transcript=raw_transcript,
        )

    validated: _Slots = {"query": str(query).strip()}

    artist = slots.get("artist")
    if artist is not None:
        validated["artist"] = str(artist).strip() or None

    album = slots.get("album")
    if album is not None:
        validated["album"] = str(album).strip() or None

    return validated, []


def _slots_spotify_skip(
    slots: _Slots,
    request_id: str,
    raw_transcript: str | None,
    log: structlog.stdlib.BoundLogger,
) -> tuple[_Slots, _CoercionLog] | ValidationError:
    """Validator for spotify.skip. Soft-fails invalid direction to 'next'."""
    coercions: _CoercionLog = []
    raw_direction = slots.get("direction", SkipDirection.NEXT.value)

    try:
        direction = SkipDirection(str(raw_direction).lower())
    except ValueError:
        coercions.append(
            f"Invalid skip direction '{raw_direction}' defaulted to 'next'."
        )
        log.warning(
            "validator.coercion",
            slot="direction",
            original=raw_direction,
            coerced="next",
        )
        direction = SkipDirection.NEXT

    return {"direction": direction.value}, coercions


def _slots_volume(
    slots: _Slots,
    request_id: str,
    raw_transcript: str | None,
    log: structlog.stdlib.BoundLogger,
) -> tuple[_Slots, _CoercionLog] | ValidationError:
    """
    Validator for spotify.volume and system.volume.
    Handles mute/unmute (no level required), clamps level to [0, 100].
    """
    coercions: _CoercionLog = []

    # Resolve direction with default
    raw_direction = slots.get("direction", VolumeDirection.SET.value)
    try:
        direction = VolumeDirection(str(raw_direction).lower())
    except ValueError:
        coercions.append(
            f"Invalid volume direction '{raw_direction}' defaulted to 'set'."
        )
        log.warning(
            "validator.coercion",
            slot="direction",
            original=raw_direction,
            coerced="set",
        )
        direction = VolumeDirection.SET

    # Mute/unmute do not require a level
    if direction in (VolumeDirection.MUTE, VolumeDirection.UNMUTE):
        return {"direction": direction.value, "level": None}, coercions

    # Level is required for up/down/set
    raw_level = slots.get("level")
    if raw_level is None:
        log.warning("validator.missing_slot", slot="level")
        return ValidationError(
            request_id=request_id,
            code=ValidationErrorCode.MISSING_SLOT,
            field="level",
            message="To what volume level?",
            raw_payload=None,
            raw_transcript=raw_transcript,
        )

    try:
        level = int(raw_level)
    except (TypeError, ValueError):
        log.error("validator.invalid_slot_value", slot="level", value=raw_level)
        return ValidationError(
            request_id=request_id,
            code=ValidationErrorCode.INVALID_SLOT_VALUE,
            field="level",
            message="I need a number for the volume level. Please try again.",
            raw_payload=None,
            raw_transcript=raw_transcript,
        )

    # Soft-fail: clamp to [0, 100]
    if not (VOLUME_MIN <= level <= VOLUME_MAX):
        clamped = max(VOLUME_MIN, min(VOLUME_MAX, level))
        coercions.append(
            f"Volume level {level} clamped to {clamped} "
            f"(valid range {VOLUME_MIN}–{VOLUME_MAX})."
        )
        log.warning(
            "validator.coercion",
            slot="level",
            original=level,
            coerced=clamped,
        )
        level = clamped

    return {"direction": direction.value, "level": level}, coercions


def _slots_system_open_app(
    slots: _Slots,
    request_id: str,
    raw_transcript: str | None,
    log: structlog.stdlib.BoundLogger,
) -> tuple[_Slots, _CoercionLog] | ValidationError:
    """
    Validator for system.open_app.
    Normalizes app_name to lowercase-stripped form.
    Allowlist check is deliberately NOT performed here — it belongs in the tool,
    which has access to the known_apps config. The validator verifies structure only.
    """
    app_name = slots.get("app_name")

    if app_name is None or not str(app_name).strip():
        log.warning("validator.missing_slot", slot="app_name")
        return ValidationError(
            request_id=request_id,
            code=ValidationErrorCode.MISSING_SLOT,
            field="app_name",
            message="Which application should I open?",
            raw_payload=None,
            raw_transcript=raw_transcript,
        )

    return {"app_name": str(app_name).strip().lower()}, []


def _slots_system_timer(
    slots: _Slots,
    request_id: str,
    raw_transcript: str | None,
    log: structlog.stdlib.BoundLogger,
) -> tuple[_Slots, _CoercionLog] | ValidationError:
    """Validator for system.timer. Enforces duration > 0 and total < 24 hours."""

    # Validate unit
    raw_unit = slots.get("unit")
    if raw_unit is None:
        log.warning("validator.missing_slot", slot="unit")
        return ValidationError(
            request_id=request_id,
            code=ValidationErrorCode.MISSING_SLOT,
            field="unit",
            message="Should that be seconds, minutes, or hours?",
            raw_payload=None,
            raw_transcript=raw_transcript,
        )

    try:
        unit = TimerUnit(str(raw_unit).lower())
    except ValueError:
        log.error("validator.invalid_slot_value", slot="unit", value=raw_unit)
        return ValidationError(
            request_id=request_id,
            code=ValidationErrorCode.INVALID_SLOT_VALUE,
            field="unit",
            message=(
                f"I don't understand the unit '{raw_unit}'. "
                "Please say seconds, minutes, or hours."
            ),
            raw_payload=None,
            raw_transcript=raw_transcript,
        )

    # Validate duration
    raw_duration = slots.get("duration")
    if raw_duration is None:
        log.warning("validator.missing_slot", slot="duration")
        return ValidationError(
            request_id=request_id,
            code=ValidationErrorCode.MISSING_SLOT,
            field="duration",
            message="For how long should I set the timer?",
            raw_payload=None,
            raw_transcript=raw_transcript,
        )

    try:
        duration = int(raw_duration)
    except (TypeError, ValueError):
        log.error(
            "validator.invalid_slot_value", slot="duration", value=raw_duration
        )
        return ValidationError(
            request_id=request_id,
            code=ValidationErrorCode.INVALID_SLOT_VALUE,
            field="duration",
            message="I need a whole number for the timer duration. Please try again.",
            raw_payload=None,
            raw_transcript=raw_transcript,
        )

    if duration <= 0:
        log.error("validator.invalid_slot_value", slot="duration", value=duration)
        return ValidationError(
            request_id=request_id,
            code=ValidationErrorCode.INVALID_SLOT_VALUE,
            field="duration",
            message="The timer duration must be greater than zero.",
            raw_payload=None,
            raw_transcript=raw_transcript,
        )

    # Check total seconds cap
    multipliers = {
        TimerUnit.SECONDS: 1,
        TimerUnit.MINUTES: 60,
        TimerUnit.HOURS: 3_600,
    }
    total_seconds = duration * multipliers[unit]

    if total_seconds > MAX_TIMER_SECONDS:
        log.error(
            "validator.invalid_slot_value",
            slot="duration",
            total_seconds=total_seconds,
            max=MAX_TIMER_SECONDS,
        )
        return ValidationError(
            request_id=request_id,
            code=ValidationErrorCode.INVALID_SLOT_VALUE,
            field="duration",
            message="I can only set timers up to 24 hours.",
            raw_payload=None,
            raw_transcript=raw_transcript,
        )

    return {"duration": duration, "unit": unit.value, "total_seconds": total_seconds}, []


def _slots_web_search(
    slots: _Slots,
    request_id: str,
    raw_transcript: str | None,
    log: structlog.stdlib.BoundLogger,
) -> tuple[_Slots, _CoercionLog] | ValidationError:
    """Validator for web.search. Defaults engine to 'google' on invalid value."""
    coercions: _CoercionLog = []

    query = slots.get("query")
    if query is None or not str(query).strip():
        log.warning("validator.missing_slot", slot="query", intent="web.search")
        return ValidationError(
            request_id=request_id,
            code=ValidationErrorCode.MISSING_SLOT,
            field="query",
            message="What should I search for?",
            raw_payload=None,
            raw_transcript=raw_transcript,
        )

    raw_engine = slots.get("engine", SearchEngine.GOOGLE.value)
    try:
        engine = SearchEngine(str(raw_engine).lower())
    except ValueError:
        coercions.append(
            f"Unknown search engine '{raw_engine}' defaulted to 'google'."
        )
        log.warning(
            "validator.coercion",
            slot="engine",
            original=raw_engine,
            coerced="google",
        )
        engine = SearchEngine.GOOGLE

    return {"query": str(query).strip(), "engine": engine.value}, coercions


def _slots_system_install_app(
    slots: _Slots,
    request_id: str,
    raw_transcript: str | None,
    log: structlog.stdlib.BoundLogger,
) -> tuple[_Slots, _CoercionLog] | ValidationError:
    """
    Validator for system.install_app.
    Normalises package_name to lowercase. Also accepts 'app_name' as an
    alias for compatibility with LLMs that reuse the open_app slot name.
    """
    # Accept both 'package_name' and 'app_name' (LLM alias)
    pkg = slots.get("package_name") or slots.get("app_name")

    if pkg is None or not str(pkg).strip():
        log.warning("validator.missing_slot", slot="package_name", intent="system.install_app")
        return ValidationError(
            request_id=request_id,
            code=ValidationErrorCode.MISSING_SLOT,
            field="package_name",
            message="Which application or package should I install?",
            raw_payload=None,
            raw_transcript=raw_transcript,
        )

    return {"package_name": str(pkg).strip().lower()}, []


def _slots_system_update_app(
    slots: _Slots,
    request_id: str,
    raw_transcript: str | None,
    log: structlog.stdlib.BoundLogger,
) -> tuple[_Slots, _CoercionLog]:
    """
    Validator for system.update_app.
    package_name is OPTIONAL — absence means 'update everything'.
    Also accepts 'app_name' as an LLM alias.
    """
    pkg = slots.get("package_name") or slots.get("app_name") or None
    if pkg:
        pkg = str(pkg).strip().lower() or None
    return {"package_name": pkg}, []


def _slots_gmail_read(
    slots: _Slots,
    request_id: str,
    raw_transcript: str | None,
    log: structlog.stdlib.BoundLogger,
) -> tuple[_Slots, _CoercionLog] | ValidationError:
    """
    Validator for gmail.read.
    Extracts the 'count' slot, ensures it's a valid integer, and clamps it
    to a reasonable range [1, 10] to prevent excessively large LLM payloads
    and slow TTS readouts.
    """
    coercions: _CoercionLog = []
    
    raw_count = slots.get("count", 5)
    try:
        count = int(raw_count)
    except (TypeError, ValueError):
        coercions.append(f"Invalid count '{raw_count}' defaulted to 5.")
        log.warning("validator.coercion", slot="count", original=raw_count, coerced=5)
        count = 5

    # Clamp the count [1, 10] safely
    if not (1 <= count <= 10):
        clamped_count = max(1, min(10, count))
        coercions.append(f"Count {count} clamped to {clamped_count} (valid range 1-10).")
        log.warning("validator.coercion", slot="count", original=count, coerced=clamped_count)
        count = clamped_count

    return {"count": count}, coercions
