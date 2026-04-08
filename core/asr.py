"""
core/asr.py
===========
ASR (Automatic Speech Recognition) layer — faster-whisper wrapper.

Responsibilities:
  - Accept AudioSegment from the VAD layer
  - Transcribe audio bytes → TranscriptResult
  - Apply a confidence gate (configurable threshold)
  - Expose a health check for startup validation

Design rules:
  - The model is loaded once at construction and held in memory.
  - All transcription is synchronous (faster-whisper uses ctranslate2
    internally which releases the GIL — we wrap in run_in_executor).
  - No I/O other than the model file load at startup.

Model selection guide (trade-off: speed vs. accuracy):
  tiny.en   ~32MB  fastest,  lower accuracy
  base.en   ~74MB  fast,     acceptable
  small.en  ~244MB balanced  ← default
  medium.en ~769MB accurate, slower (~2–3× small)
  large-v3  ~1.5GB highest,  requires ~4GB RAM

Logging:
  - Model load time, transcription duration, confidence are logged per request.
"""

from __future__ import annotations

import asyncio
import io
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import structlog

from core.models import AudioSegment, TranscriptResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_SIZE = "small.en"
_DEFAULT_DEVICE = "cpu"
_DEFAULT_COMPUTE_TYPE = "int8"   # int8 = fastest on CPU; float16 on GPU
_DEFAULT_CONFIDENCE_THRESHOLD = 0.0  # Whisper doesn't expose token-level confidence
                                      # directly; we use avg log-prob as proxy
_MIN_RMS_ENERGY = 0.002              # Below this threshold audio is treated as silence
_MIN_CONFIDENCE = 0.15               # Transcripts below this confidence are discarded


class ASREngine:
    """
    Wraps faster-whisper for local, offline speech transcription.

    Parameters
    ----------
    model_size:
        faster-whisper model variant to load (e.g. 'small.en').
    device:
        Inference device. 'cpu' or 'cuda'.
    compute_type:
        Quantization type. 'int8' (CPU), 'float16' (GPU).
    beam_size:
        Beam search width. 1 = greedy (fastest). 5 = more accurate.
    language:
        Force a specific language code (e.g. 'en') to skip auto-detection.
        Set to None to enable auto-detection (slower).
    max_workers:
        Thread pool size for running transcription off the asyncio event loop.
    """

    def __init__(
        self,
        *,
        model_size: str = _DEFAULT_MODEL_SIZE,
        device: str = _DEFAULT_DEVICE,
        compute_type: str = _DEFAULT_COMPUTE_TYPE,
        beam_size: int = 1,
        language: str | None = "en",
        max_workers: int = 1,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._language = language
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="asr")
        self._model: Any | None = None  # WhisperModel — loaded on first use
        self._log = structlog.get_logger().bind(layer="asr", model=model_size)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Load the faster-whisper model into memory.
        Call once at agent startup to avoid cold-start latency on first request.
        """
        from faster_whisper import WhisperModel  # type: ignore[import]

        self._log.info("asr.model_loading", model=self._model_size, device=self._device)
        t0 = time.monotonic()
        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )
        elapsed = (time.monotonic() - t0) * 1000
        self._log.info("asr.model_loaded", load_ms=int(elapsed))

    def is_ready(self) -> bool:
        """Return True if the model is loaded and ready to transcribe."""
        return self._model is not None

    async def health_check(self) -> bool:
        """
        Verify Whisper is loaded. Returns True if ready.
        Called by the agent at startup.
        """
        return self.is_ready()

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    async def transcribe(self, segment: AudioSegment) -> TranscriptResult:
        """
        Transcribe an AudioSegment → TranscriptResult.

        Runs faster-whisper in a thread pool to avoid blocking the asyncio loop.

        Parameters
        ----------
        segment:
            VAD-produced audio segment with PCM bytes.

        Returns
        -------
        TranscriptResult
            Always returns a result. On internal error, returns a result with
            empty text and 0.0 confidence so the pipeline can route gracefully.
        """
        log = self._log.bind(request_id=segment.request_id)

        if self._model is None:
            log.error("asr.model_not_loaded")
            return TranscriptResult(
                request_id=segment.request_id,
                text="unclear input",
                confidence=0.0,
                language="en",
            )
            
        if segment.duration_ms < 300:
            log.warning("asr.empty_transcript_detected", reason="duration < 300ms")
            return TranscriptResult(
                request_id=segment.request_id,
                text="unclear input",
                confidence=0.0,
                language="en",
            )

        log.info("asr.transcribe_start", duration_ms=segment.duration_ms)
        start_ns = time.monotonic_ns()

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                self._executor,
                self._transcribe_sync,
                segment.audio_bytes,
                segment.sample_rate,
                log,
            )
        except Exception as exc:  # noqa: BLE001
            duration_ms = (time.monotonic_ns() - start_ns) // 1_000_000
            log.exception("asr.transcribe_error", exc=str(exc), duration_ms=duration_ms)
            return TranscriptResult(
                request_id=segment.request_id,
                text="unclear input",
                confidence=0.0,
                language="en",
            )

        full_text = result["text"].strip()
        if not full_text:
            log.warning("asr.empty_transcript_detected", reason="empty string")
            full_text = "unclear input"

        # Confidence gate: discard very low-confidence transcriptions
        if result["confidence"] < _MIN_CONFIDENCE and full_text != "unclear input":
            log.warning(
                "asr.low_confidence_fallback",
                confidence=result["confidence"],
                threshold=_MIN_CONFIDENCE,
                discarded_text=full_text[:80],
            )
            full_text = "unclear input"

        # Energy gate (checked post-transcription as a secondary guard)
        if result.get("rms_energy", 1.0) < _MIN_RMS_ENERGY and full_text != "unclear input":
            log.warning(
                "asr.low_energy_fallback",
                rms_energy=result.get("rms_energy"),
                threshold=_MIN_RMS_ENERGY,
            )
            full_text = "unclear input"

        duration_ms = (time.monotonic_ns() - start_ns) // 1_000_000
        log.info(
            "asr.transcribe_complete",
            duration_ms=duration_ms,
            text_preview=full_text[:80],
            confidence=result["confidence"],
            language=result["language"],
        )

        return TranscriptResult(
            request_id=segment.request_id,
            text=full_text,
            confidence=result["confidence"],
            language=result["language"],
        )

    def _transcribe_sync(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        log: structlog.stdlib.BoundLogger,
    ) -> dict[str, Any]:
        """
        Blocking transcription call — runs in thread pool.
        Returns a dict with text, confidence, language.
        """
        import numpy as np  # type: ignore[import]

        # Convert raw PCM bytes to float32 numpy array normalised to [-1, 1]
        audio_np = (
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )
        
        # Ensure mono
        if len(audio_np.shape) > 1 and audio_np.shape[1] > 1:
            audio_np = audio_np.mean(axis=1)

        # Calculate RMS energy safely
        rms = float(np.sqrt(np.mean(audio_np**2)))
        log.debug("asr.audio_quality", rms_energy=round(rms, 4))

        # Energy gate: treat sub-threshold audio as silence before hitting Whisper
        if rms < _MIN_RMS_ENERGY:
            log.warning(
                "asr.low_energy_silence",
                rms_energy=round(rms, 4),
                threshold=_MIN_RMS_ENERGY,
                action="returning unclear input without transcription",
            )
            return {"text": "unclear input", "confidence": 0.0, "language": self._language or "en", "rms_energy": rms}

        # Normalize audio (Scale strictly to [-1.0, 1.0])
        max_amp = np.max(np.abs(audio_np))
        if max_amp > 0:
            audio_np = audio_np / max_amp

        segments, info = self._model.transcribe(
            audio_np,
            beam_size=self._beam_size,
            language=self._language,
            vad_filter=False,  # VAD already done upstream
            word_timestamps=False,
        )

        full_text_parts: list[str] = []
        avg_log_prob_sum = 0.0
        seg_count = 0

        for seg in segments:
            full_text_parts.append(seg.text)
            avg_log_prob_sum += seg.avg_logprob
            seg_count += 1

        full_text = " ".join(full_text_parts).strip()

        # Derive a pseudo-confidence from average log-prob.
        # avg_logprob is typically in range [-2, 0]; we map to [0, 1].
        if seg_count > 0:
            avg_log_prob = avg_log_prob_sum / seg_count
            confidence = float(min(1.0, max(0.0, 1.0 + avg_log_prob / 2.0)))
        else:
            confidence = 0.0

        return {
            "text": full_text,
            "confidence": confidence,
            "language": info.language or (self._language or "en"),
            "rms_energy": rms,
        }

    def shutdown(self) -> None:
        """Gracefully shut down the thread pool."""
        self._executor.shutdown(wait=True)
        self._log.info("asr.shutdown")
