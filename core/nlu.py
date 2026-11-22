"""
core/nlu.py
===========
NLU (Natural Language Understanding) layer — Groq API client.

Responsibilities:
  - Generate the LLM system prompt from the live tool registry
  - Send transcript → Groq REST API with structured JSON output mode
  - Return raw JSON string (IntentPayload, untrusted) for the Validator
  - Manage an in-process intent cache (transcript hash → cached JSON)
  - Handle Groq connectivity failures gracefully

Design rules:
  - This layer produces ONLY raw JSON strings. It never constructs models.
  - It never calls the Validator, Dispatcher, or any Tool.
  - The system prompt is built once and cached until registry changes.
  - The intent cache uses an LRU eviction strategy with TTL expiry.

Intent Cache:
  - Key: sha256(transcript.strip().lower())
  - Value: (raw_json: str, expires_at: float)
  - TTL: configurable (default 300s / 5 minutes)
  - Max entries: configurable (default 128)

Logging:
  - Cache hits/misses, Groq request timing, error responses are all logged.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict
from typing import Any

import aiohttp
import structlog
import numpy as np

from core.dispatcher import Dispatcher

# ---------------------------------------------------------------------------
# Health-check helper (module-level, reusable)
# ---------------------------------------------------------------------------

async def check_groq_reachable() -> bool:
    """
    Verifies if GROQ_API_KEY is configured and valid by pinging the Groq API.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return False
        
    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=4.0)) as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 401:
                    print("\n  \u274c  ERROR: Groq API Key is INVALID (HTTP 401 Unauthorized).\n")
                    return False
                return resp.status == 200
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_HEADER = """\
You are an intent extraction engine for a voice assistant. Your ONLY job is to
analyse the user's spoken request and return a single JSON object.

STRICT RULES:
1. Output ONLY valid JSON. No explanation, no markdown, no code fences.
2. The JSON must conform exactly to the schema below.
3. Choose the intent ONLY from the list of available tools.
4. If you are unsure, set confidence below 0.5 and use agent.clarify.
5. If no tool matches, use agent.unknown.
6. Slots must be flat key-value pairs. No nesting.

OUTPUT SCHEMA:
{
  "intent": "<tool name from the list below>",
  "confidence": <float 0.0 to 1.0>,
  "slots": { <key: value pairs specific to the chosen tool> },
  "raw_transcript": "<echo the user's exact words here>"
}

AVAILABLE TOOLS:
"""

_SYSTEM_PROMPT_FOOTER = """
USER INPUT: respond with the JSON object only.
"""

# ---------------------------------------------------------------------------
# LRU + TTL cache
# ---------------------------------------------------------------------------

class _SemanticIntentCache:
    """
    In-process LRU cache with per-entry TTL for NLU results.

    Uses 'sentence-transformers' to embed transcripts and retrieve cached
    intents via cosine similarity, falling back to exact hash matches if
    the embedding engine is missing.
    """

    def __init__(self, max_size: int = 128, ttl_seconds: float = 300.0, threshold: float = 0.92) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._threshold = threshold
        self._model = None
        # OrderedDict: hash_key -> (transcript, embedding, raw_json, expires_at)
        self._store: OrderedDict[str, tuple[str, np.ndarray | None, str, float]] = OrderedDict()

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model if self._model is not False else None
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            structlog.get_logger().warning(
                "nlu.semantic_cache_disabled",
                msg="Install 'sentence-transformers' for semantic caching. Falling back to exact match.",
            )
            self._model = False
            return None

        try:
            # 'all-MiniLM-L6-v2' is a local/offline-capable model — loading it
            # may probe HuggingFace Hub for optional resources:
            #   - additional_chat_templates → 404 (optional, ignore)
            #   - 1_Pooling/config.json   → 307 redirect (handled automatically)
            # Suppress noisy INFO logs from the hub client during these checks.
            import logging
            hf_logger = logging.getLogger("transformers_hub")
            orig_level = hf_logger.level
            hf_logger.setLevel(logging.ERROR)
            try:
                self._model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
            finally:
                hf_logger.setLevel(orig_level)
        except Exception as exc:
            structlog.get_logger().warning(
                "nlu.semantic_cache_disabled",
                msg=f"sentence-transformers model loading failed: {exc}. "
                    "Falling back to exact-match cache only.",
            )
            self._model = False
        return self._model if self._model is not False else None

    def _embed(self, text: str) -> np.ndarray | None:
        model = self._get_model()
        if not model:
            return None
        try:
            return model.encode(text.strip().lower(), convert_to_numpy=True)
        except Exception as exc:
            structlog.get_logger().warning(
                "nlu.embedding_failed",
                msg=f"Embedding encode failed: {exc}. Skipping semantic cache for this turn.",
            )
            return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get(self, transcript: str) -> str | None:
        """Return cached raw JSON if semantic similarity > threshold."""
        query_emb = self._embed(transcript)
        now = time.monotonic()
        
        # Immediate cleanup of expired entries
        expired_keys = [k for k, v in self._store.items() if now > v[3]]
        for k in expired_keys:
            structlog.get_logger().debug("nlu.cache_eviction", transcript=self._store[k][0])
            del self._store[k]

        best_score = -1.0
        best_key = None

        def _has_antonym_conflict(q_str: str, c_str: str) -> bool:
            """Check if strings have contradicting boolean actions despite high topic similarity."""
            query_w = q_str.lower().split()
            cached_w = c_str.lower().split()
            antonyms = [
                ({"up"}, {"down"}),
                ({"next", "forward"}, {"previous", "back"}),
                ({"start", "play", "resume"}, {"stop", "pause", "halt"}),
                ({"on"}, {"off"}),
            ]
            for set_a, set_b in antonyms:
                if any(w in query_w for w in set_a) and any(w in cached_w for w in set_b):
                    return True
                if any(w in query_w for w in set_b) and any(w in cached_w for w in set_a):
                    return True
            return False

        if query_emb is None:
            # Fallback exact lexical match
            for key, (t, emb, raw_json, _) in self._store.items():
                if t.strip().lower() == transcript.strip().lower():
                    best_key = key
                    break
        else:
            # Semantic search
            for key, (t, emb, raw_json, _) in self._store.items():
                if emb is not None:
                    score = self._cosine_similarity(query_emb, emb)
                    if score > best_score:
                        best_score = score
                        best_key = key

        if best_key and (query_emb is None or best_score >= self._threshold):
            t, emb, raw_json, expires_at = self._store[best_key]
            
            # Opposing semantic bypass check
            if query_emb is not None and _has_antonym_conflict(transcript, t):
                structlog.get_logger().warning("nlu.cache_antonym_conflict", query=transcript, cached=t)
                return None
                
            # LRU shuffle
            del self._store[best_key]
            self._store[best_key] = (t, emb, raw_json, expires_at)
            
            if query_emb is not None:
                structlog.get_logger().debug("nlu.semantic_match", score=best_score, matched_transcript=t)
                
            return raw_json

        return None

    def put(self, transcript: str, raw_json: str) -> None:
        """Insert or update a cache entry with its embedding."""
        emb = self._embed(transcript)
        now = time.monotonic()
        expires_at = now + self._ttl
        
        normalised = transcript.strip().lower()
        key = hashlib.sha256(normalised.encode()).hexdigest()
        
        self._store[key] = (transcript, emb, raw_json, expires_at)
        self._store.move_to_end(key)
        
        if len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def invalidate(self, transcript: str) -> None:
        normalised = transcript.strip().lower()
        key = hashlib.sha256(normalised.encode()).hexdigest()
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

# ---------------------------------------------------------------------------
# NLU Engine
# ---------------------------------------------------------------------------

class NLUEngine:
    """
    Sends transcripts to Groq and returns raw IntentPayload JSON strings.

    Parameters
    ----------
    dispatcher:
        The live Dispatcher instance. Used to generate the tools section
        of the system prompt.
    model:
        Groq model name (default: 'llama-3.1-8b-instant').
    request_timeout_seconds:
        HTTP timeout for the Groq request (default 3 to 5 seconds).
    cache_max_size:
        Maximum number of entries in the intent cache.
    cache_ttl_seconds:
        Time-to-live for cache entries in seconds.
    """

    def __init__(
        self,
        dispatcher: Dispatcher,
        *,
        model: str = "llama-3.1-8b-instant",
        request_timeout_seconds: float = 5.0,
        cache_max_size: int = 128,
        cache_ttl_seconds: float = 300.0,
    ) -> None:
        self._dispatcher = dispatcher
        self._model = model
        self._timeout = aiohttp.ClientTimeout(total=request_timeout_seconds)
        self._cache = _SemanticIntentCache(max_size=cache_max_size, ttl_seconds=cache_ttl_seconds)
        self._system_prompt: str | None = None
        self._last_hit = False
        self._groq_url = "https://api.groq.com/openai/v1/chat/completions"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def infer(
        self,
        transcript: str,
        request_id: str,
    ) -> str:
        log = structlog.get_logger().bind(
            request_id=request_id,
            layer="nlu",
            model=self._model,
        )
        self._last_hit = False

        if transcript.strip().lower() == "unclear input":
            log.warning("nlu.skip_unclear_input", reason="ASR returned 'unclear input'; bypassing LLM call")
            return self._clarify_json(transcript)

        # 1. Cache check
        cached = self._cache.get(transcript)
        if cached is not None:
            self._last_hit = True
            log.info("nlu.cache_hit", transcript_preview=transcript[:60])
            return cached

        log.info("nlu.cache_miss", transcript_preview=transcript[:60])

        # 2. Pre-call health check for Groq
        if not os.environ.get("GROQ_API_KEY"):
            log.warning(
                "nlu.groq_failure",
                action="returning agent.clarify without LLM call",
                hint="Set GROQ_API_KEY environment variable",
            )
            return self._clarify_json(transcript, reason="Groq API key not configured")

        # 3. Build system prompt
        system_prompt = self._get_system_prompt()

        # 4. Call Groq
        start_ns = time.monotonic_ns()
        raw_json = await self._call_groq_with_retry(
            system_prompt=system_prompt,
            user_message=transcript,
            request_id=request_id,
            log=log,
        )
        duration_ms = (time.monotonic_ns() - start_ns) // 1_000_000
        log.info("nlu.groq_response", duration_ms=duration_ms)

        # 5. Store in cache
        self._cache.put(transcript, raw_json)

        return raw_json

    def invalidate_cache_for(self, transcript: str) -> None:
        self._cache.invalidate(transcript)

    def reset_system_prompt(self) -> None:
        self._system_prompt = None

    def cache_size(self) -> int:
        return len(self._cache)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_system_prompt(self) -> str:
        if self._system_prompt is None:
            tools_block = self._dispatcher.build_system_prompt_tools_block()
            self._system_prompt = _SYSTEM_PROMPT_HEADER + tools_block + _SYSTEM_PROMPT_FOOTER
        return self._system_prompt

    async def _call_groq_with_retry(
        self,
        system_prompt: str,
        user_message: str,
        request_id: str,
        log: structlog.stdlib.BoundLogger,
        max_retries: int = 1,
    ) -> str:
        for attempt in range(max_retries + 1):
            result = await self._call_groq(
                system_prompt=system_prompt,
                user_message=user_message,
                request_id=request_id,
                log=log,
            )
            try:
                parsed = json.loads(result)
                if parsed.get("intent") not in ("agent.clarify",) or parsed.get("confidence", 1.0) > 0.0:
                    return result
            except json.JSONDecodeError:
                return result

            if attempt < max_retries:
                log.warning(
                    "nlu.groq_retry",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                )
                await __import__("asyncio").sleep(0.5)

        return result

    async def _call_groq(
        self,
        system_prompt: str,
        user_message: str,
        request_id: str,
        log: structlog.stdlib.BoundLogger,
    ) -> str:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return self._clarify_json(user_message, reason="Groq API key missing")

        log.debug("nlu.groq_request", model=self._model)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }

        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                async with session.post(self._groq_url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        body = await response.text()
                        log.error(
                            "nlu.groq_failure",
                            status=response.status,
                            body=body[:200],
                        )
                        return self._clarify_json(user_message)

                    data = await response.json(content_type=None)
                    
                    try:
                        raw_text = data["choices"][0]["message"]["content"]
                    except (KeyError, IndexError):
                        log.error("nlu.groq_failure", reason="unexpected_response_format", data=data)
                        return self._clarify_json(user_message)

                    if not raw_text or not raw_text.strip():
                        log.error("nlu.groq_failure", reason="empty_response")
                        return self._clarify_json(user_message)

                    log.debug("nlu.raw_response", response_preview=raw_text[:120])
                    return raw_text

        except aiohttp.ClientConnectorError:
            log.error("nlu.groq_failure", reason="api_unreachable")
            return self._clarify_json(user_message)
        except TimeoutError:
            log.error("nlu.groq_failure", reason="timeout", timeout_s=self._timeout.total)
            return self._clarify_json(user_message)
        except Exception as exc:  # noqa: BLE001
            log.exception("nlu.unexpected_error", exc=str(exc))
            return self._clarify_json(user_message)

    @staticmethod
    def _clarify_json(transcript: str, reason: str = "Could you say that again?") -> str:
        return json.dumps(
            {
                "intent": "agent.clarify",
                "confidence": 0.0,
                "slots": {"message": reason},
                "raw_transcript": transcript,
            }
        )

