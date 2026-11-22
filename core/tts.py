"""
core/tts.py
===========
TTS (Text-to-Speech) layer — Coqui TTS with pyttsx3 fallback.

Responsibilities:
  - Accept a response_text string
  - Synthesise and play audio through the system speaker
  - Provide a fast fallback (pyttsx3) if Coqui TTS fails or is not installed
  - Provide a console-print degraded mode if both engines fail
  - Drain the timer alert queue between utterances

Design rules:
  - Speech synthesis is synchronous (most TTS libs are blocking).
    We run in an executor so TTS never blocks the asyncio loop.
  - Only one utterance plays at a time (enforced by an asyncio.Lock).
  - The alert_queue (injected from agent.py) is checked after every
    tool result — timer expirations are spoken without any LLM round-trip.

Engine priority:
  1. Coqui TTS (high quality, fully local, ~300ms latency for short sentences)
  2. pyttsx3  (lightweight system TTS, near-instant)
  3. Console print (silent degraded mode — always succeeds)

Logging:
  - Engine used, synthesis latency, and any fallback events are logged.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any

import structlog

# ALSA error patterns that indicate the audio subsystem is unavailable
_ALSA_HOST_DOWN_PATTERNS = (
    "host is down",
    "audio open error",
    "no such device",
    "connection refused",
    "broken pipe",
)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TTSEngine(str, Enum):
    COQUI = "coqui"
    PYTTSX3 = "pyttsx3"
    CONSOLE = "console"   # Degraded mode — no audio


# ---------------------------------------------------------------------------
# TTS Engine
# ---------------------------------------------------------------------------


class TTSService:
    """
    Text-to-speech service with automatic engine fallback.

    Parameters
    ----------
    preferred_engine:
        Which engine to attempt first.
    coqui_model_name:
        Coqui TTS model identifier (e.g. 'tts_models/en/ljspeech/tacotron2-DDC').
    alert_queue:
        Shared asyncio queue for timer alerts. Injected by agent.py.
        TTSService drains this queue through speak_alerts() after each response.
    max_workers:
        Thread pool size for running blocking TTS calls.
    """

    def __init__(
        self,
        *,
        preferred_engine: TTSEngine = TTSEngine.COQUI,
        coqui_model_name: str = "tts_models/en/ljspeech/tacotron2-DDC",
        alert_queue: asyncio.Queue[str] | None = None,
        max_workers: int = 1,
    ) -> None:
        self._preferred = preferred_engine
        self._coqui_model_name = coqui_model_name
        self._alert_queue = alert_queue
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tts")
        self._lock = asyncio.Lock()  # Prevents overlapping utterances
        self._active_engine: TTSEngine | None = None
        self._coqui_tts: Any | None = None
        self._pyttsx3_engine: Any | None = None
        self._log = structlog.get_logger().bind(layer="tts")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_engine(self) -> None:
        """
        Attempt to initialise the preferred TTS engine.
        Called once at agent startup. Falls back automatically on failure.
        """
        if self._preferred == TTSEngine.COQUI:
            if self._try_load_coqui():
                self._active_engine = TTSEngine.COQUI
                return

        if self._try_load_pyttsx3():
            self._active_engine = TTSEngine.PYTTSX3
            return

        self._log.warning("tts.degraded_mode_console")
        self._active_engine = TTSEngine.CONSOLE

    def _try_load_coqui(self) -> bool:
        try:
            from TTS.api import TTS as CoquiTTS  # type: ignore[import]

            self._log.info("tts.loading_coqui", model=self._coqui_model_name)
            t0 = time.monotonic()
            self._coqui_tts = CoquiTTS(self._coqui_model_name, progress_bar=False)
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            self._log.info("tts.coqui_loaded", load_ms=elapsed_ms)
            return True
        except Exception as exc:  # noqa: BLE001
            self._log.warning("tts.coqui_unavailable", exc=str(exc))
            return False

    def _try_load_pyttsx3(self) -> bool:
        try:
            import pyttsx3  # type: ignore[import]
            import sounddevice as sd
            
            if sd.default.device[1] is None:
                self._log.warning("tts.no_output_device_found")
                return False

            self._log.info("tts.loading_pyttsx3")
            engine = pyttsx3.init()

            # ── Female voice selection ──────────────────────────────────
            # espeak-ng supports female variants via the '+f<N>' suffix.
            # We prefer: en-us+f3 (clearest American female),
            #            en-us+f2, en-gb+f3 as fallbacks.
            _FEMALE_CANDIDATES = [
                "gmw/en-us+f3",   # American English, female variant 3 (clearest)
                "gmw/en-us+f2",   # American English, female variant 2
                "gmw/en+f3",      # British English, female variant 3
                "gmw/en+f2",
            ]
            voices = engine.getProperty("voices")
            voice_ids = {v.id for v in voices}

            selected_voice = None
            for candidate in _FEMALE_CANDIDATES:
                # pyttsx3 espeak driver accepts the variant suffix even if it is
                # not listed as a discrete voice — try each directly.
                try:
                    engine.setProperty("voice", candidate)
                    selected_voice = candidate
                    self._log.info("tts.female_voice_selected", voice=candidate)
                    break
                except Exception:
                    continue

            if selected_voice is None:
                # Last resort: pick the first English voice available
                en_voices = [v for v in voices if "en" in str(v.id)]
                if en_voices:
                    engine.setProperty("voice", en_voices[0].id)
                    self._log.warning("tts.female_voice_fallback", voice=en_voices[0].id)

            # ── Speech quality tuning ───────────────────────────────────
            engine.setProperty("rate", 150)    # Slightly slower → clearer
            engine.setProperty("volume", 1.0)  # Full volume
            # ────────────────────────────────────────────────────────────

            self._pyttsx3_engine = engine
            self._log.info("tts.pyttsx3_loaded")
            return True
        except Exception as exc:  # noqa: BLE001
            self._log.warning("tts.pyttsx3_unavailable", exc=str(exc))
            return False

    def is_ready(self) -> bool:
        return self._active_engine is not None

    def active_engine(self) -> TTSEngine | None:
        return self._active_engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def speak(self, text: str, request_id: str = "") -> None:
        """
        Synthesise and play text through the active TTS engine.

        Acquires the utterance lock to prevent overlapping speech.
        Runs the blocking synthesis call in a thread pool.
        On any failure, falls back to the next available engine.

        Parameters
        ----------
        text:
            The string to speak. Should be a complete sentence.
        request_id:
            Propagated identifier for log correlation.
        """
        if not text.strip():
            return
            
        if getattr(self, "_shutdown", False):
            return

        log = self._log.bind(request_id=request_id)

        async with self._lock:
            log.info("tts.speak", engine=self._active_engine, preview=text[:80])
            start_ns = time.monotonic_ns()

            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(
                    self._executor,
                    self._speak_sync,
                    text,
                    log,
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("tts.speak_error", exc=str(exc))
                # Final fallback: print to console
                print(f"[TTS] {text}")

            duration_ms = (time.monotonic_ns() - start_ns) // 1_000_000
            log.info("tts.speak_complete", duration_ms=duration_ms)

    async def speak_alerts(self, request_id: str = "") -> None:
        """
        Drain pending timer alerts from the shared queue and speak each one.
        Called by the agent loop after each tool response.
        Non-blocking: returns immediately if the queue is empty.
        """
        if self._alert_queue is None:
            return
        while not self._alert_queue.empty():
            try:
                alert_text = self._alert_queue.get_nowait()
                await self.speak(alert_text, request_id=request_id)
            except asyncio.QueueEmpty:
                break

    def set_alert_queue(self, queue: asyncio.Queue[str]) -> None:
        """Inject (or replace) the timer alert queue at runtime."""
        self._alert_queue = queue

    def shutdown(self) -> None:
        """Shut down the TTS thread pool gracefully."""
        self._shutdown = True
        self._executor.shutdown(wait=False)
        self._log.info("tts.shutdown")

    # ------------------------------------------------------------------
    # Internal — blocking, runs in thread pool
    # ------------------------------------------------------------------

    def _speak_sync(self, text: str, log: structlog.stdlib.BoundLogger) -> None:
        """Blocking synthesis + playback. Runs in thread pool executor."""
        engine = self._active_engine

        if engine == TTSEngine.COQUI:
            self._speak_coqui(text, log)
        elif engine == TTSEngine.PYTTSX3:
            self._speak_pyttsx3(text, log)
        else:
            print(f"\n[THIRAN] {text}\n")

    def _speak_coqui(self, text: str, log: structlog.stdlib.BoundLogger) -> None:
        """Synthesise with Coqui TTS and play via sounddevice."""
        try:
            import sounddevice as sd  # type: ignore[import]
            import numpy as np        # type: ignore[import]

            if self._coqui_tts is None:
                return

            wav = self._coqui_tts.tts(text=text)
            audio = np.array(wav, dtype=np.float32)
            sd.play(audio, samplerate=22_050)
            sd.wait()
        except Exception as exc:  # noqa: BLE001
            exc_lower = str(exc).lower()
            if any(p in exc_lower for p in _ALSA_HOST_DOWN_PATTERNS):
                log.error(
                    "tts.alsa_error_fallback",
                    exc=str(exc),
                    action="demoting Coqui → pyttsx3 or console",
                )
                # Demote permanently so we don't keep hitting the broken device
                if self._pyttsx3_engine is not None:
                    self._active_engine = TTSEngine.PYTTSX3
                    self._speak_pyttsx3(text, log)
                else:
                    self._active_engine = TTSEngine.CONSOLE
                    print(f"\n[THIRAN] {text}\n")
            else:
                log.warning("tts.coqui_speak_failed", exc=str(exc), fallback="pyttsx3")
                if self._pyttsx3_engine is not None:
                    self._speak_pyttsx3(text, log)
                else:
                    print(f"\n[THIRAN] {text}\n")

    def _speak_pyttsx3(self, text: str, log: structlog.stdlib.BoundLogger) -> None:
        """Synthesise with pyttsx3 (system TTS)."""
        try:
            if self._pyttsx3_engine is None:
                return
            self._pyttsx3_engine.say(text)
            self._pyttsx3_engine.runAndWait()
        except Exception as exc:  # noqa: BLE001
            exc_lower = str(exc).lower()
            if any(p in exc_lower for p in _ALSA_HOST_DOWN_PATTERNS):
                log.error(
                    "tts.alsa_error_fallback",
                    exc=str(exc),
                    action="demoting pyttsx3 → console permanently",
                )
                self._active_engine = TTSEngine.CONSOLE
            else:
                log.warning("tts.pyttsx3_speak_failed", exc=str(exc), fallback="console")
            print(f"\n[THIRAN] {text}\n")
