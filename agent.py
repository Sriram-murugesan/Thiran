"""
agent.py
========
Top-level agent loop — push-to-talk voice agent entry point.

Responsibilities:
  - Load configuration from config/config.yaml
  - Wire together all pipeline layers at startup
  - Run the push-to-talk input loop
  - Route each utterance through: VAD → ASR → NLU → Validator → Dispatcher → TTS
  - Drain timer alerts between interactions
  - Handle SIGINT / SIGTERM gracefully

Push-to-Talk mechanics:
  - Hold SPACE (or configured key) to record
  - Release to process and respond
  - Uses the 'keyboard' library for cross-platform key capture

Logging:
  - structlog configured at startup with JSON output to file
    and colourised console output for development.
  - Every log record carries 'request_id' for full pipeline traceability.
"""

from __future__ import annotations

import asyncio
import signal
import sys
import time
from pathlib import Path

import numpy as np
import scipy.signal
import sounddevice as sd
import structlog
import yaml
from pynput import keyboard as pynput_keyboard
from pynput.keyboard import Key

from core.asr import ASREngine
from core.dispatcher import Dispatcher
from core.models import (
    AgentRequest,
    AgentResponse,
    AudioSegment,
    IntentName,
    ValidationError,
    InteractionContext,
    ValidatedIntent,
)
from core.nlu import NLUEngine, check_groq_reachable
from core.tts import TTSEngine, TTSService
from core.validator import validate
from tools.agent.fallbacks import ClarifyTool, UnknownTool
from tools.spotify.controls import SpotifyPauseTool, SpotifySkipTool, SpotifyVolumeTool
from tools.spotify.play import SpotifyPlayTool
from tools.system.open_app import SystemOpenAppTool
from tools.system.package_manager import SystemInstallAppTool, SystemUpdateAppTool
from tools.system.timer import SystemTimerTool, set_alert_queue
from tools.system.volume import SystemVolumeTool
from tools.web.search import WebSearchTool
from tools.web.answer import WebAnswerTool
from tools.youtube.play import YouTubePlayTool, YouTubeSearchTool
from tools.gmail.read import GmailReadTool
from core.telemetry import AgentTrace, StructlogTelemetrySink

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _configure_logging(log_level: str = "INFO", ui_stream=None) -> None:
    """Configure structlog for both console and JSON file output."""
    import logging

    logging.basicConfig(
        format="%(message)s",
        stream=ui_stream if ui_stream else sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def _load_config(config_path: Path = Path("config/config.yaml")) -> dict:
    if not config_path.exists():
        return {}
    with config_path.open() as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Audio recording (push-to-talk)
# ---------------------------------------------------------------------------


class PushToTalkRecorder:
    """
    Records audio from the default microphone while a key is held.

    Parameters
    ----------
    hotkey:
        The keyboard key to hold for recording (default: 'space').
    sample_rate:
        Microphone sample rate in Hz.
    channels:
        Number of audio channels (1 = mono).
    """

    def __init__(
        self,
        *,
        hotkey: str = "space",
        sample_rate: int = 16_000,
        channels: int = 1,
    ) -> None:
        self._hotkey = hotkey
        self._sample_rate = sample_rate
        self._channels = channels
        self._log = structlog.get_logger().bind(layer="ptt")
        self._is_pressed = False
        self._boot_logged = False
        
        # Start background listener
        self._listener = pynput_keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self._listener.start()
        
    def _on_press(self, key) -> None:
        if key == Key.space:
            self._is_pressed = True
            
    def _on_release(self, key) -> None:
        if key == Key.space:
            self._is_pressed = False

    async def record(self, request_id: str) -> AudioSegment | None:
        """
        Block until the hotkey is pressed, record while held, return on release.
        Returns None if recording produced no audio (key released immediately).
        """
        log = self._log.bind(request_id=request_id)
        
        # 1. Device check & fallbacks
        device_id = sd.default.device[0]
        if device_id is None:
            log.error("audio.no_input_device")
            return None
            
        device_info = sd.query_devices(device_id, 'input')
        supported_rate = int(device_info['default_samplerate'])
        actual_rate = self._sample_rate
        
        try:
            sd.check_input_settings(device=device_id, channels=self._channels, samplerate=actual_rate)
        except Exception:
            actual_rate = supported_rate
            
        if not self._boot_logged:
            log.info("audio.initialized", device=device_info['name'], rate=actual_rate)
            self._boot_logged = True
            
        from core.tui import set_tui_status
        log.info("ptt.waiting", hotkey=self._hotkey)
        set_tui_status(f"Ready.\n\nHold [b]{self._hotkey.upper()}[/b] to speak")

        # Wait for key press
        while not self._is_pressed:
            await asyncio.sleep(0.05)

        log.info("ptt.recording_start")
        set_tui_status("🎙️ [blink red]Recording...[/blink red]\n\nRelease to complete")
        frames: list[np.ndarray] = []
        start_time = time.monotonic()

        # Stream audio while key is held
        try:
            with sd.InputStream(
                samplerate=actual_rate,
                channels=self._channels,
                dtype="int16",
            ) as stream:
                while self._is_pressed:
                    chunk, _ = stream.read(1024)
                    frames.append(chunk.copy())
                    await asyncio.sleep(0)  # yield to event loop
        except Exception as exc:
            log.error("ptt.audio_error", error=str(exc))
            self._is_pressed = False
            return None

        duration_ms = int((time.monotonic() - start_time) * 1000)
        log.info("ptt.recording_stop", duration_ms=duration_ms)
        set_tui_status("⚙️ Processing audio...\n\nPlease wait")

        if not frames:
            log.warning("ptt.empty_recording")
            return None

        audio_np = np.concatenate(frames, axis=0)
        
        # Resample array if device fell back to a different rate (e.g., 44100 or 48000)
        if actual_rate != self._sample_rate:
            num_samples = int(len(audio_np) * self._sample_rate / actual_rate)
            audio_np = scipy.signal.resample(audio_np, num_samples).astype(np.int16)
            
        audio_bytes = audio_np.tobytes()

        return AudioSegment(
            request_id=request_id,
            audio_bytes=audio_bytes,
            sample_rate=self._sample_rate,
            channels=self._channels,
            duration_ms=duration_ms,
        )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class VoiceAgent:
    """
    Top-level agent that owns the pipeline and the push-to-talk event loop.

    Wires together all layers and runs the interaction loop.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._log = structlog.get_logger().bind(layer="agent")
        self._running = False
        self._alert_queue: asyncio.Queue[str] = asyncio.Queue()
        self._context: InteractionContext | None = None

        # --- Build tool registry ---
        all_tools = [
            SpotifyPlayTool(),
            SpotifyPauseTool(),
            SpotifySkipTool(),
            SpotifyVolumeTool(),
            YouTubePlayTool(),
            YouTubeSearchTool(),
            SystemOpenAppTool(),
            SystemInstallAppTool(),
            SystemUpdateAppTool(),
            SystemVolumeTool(),
            SystemTimerTool(),
            WebSearchTool(),
            WebAnswerTool(),
            GmailReadTool(),
            ClarifyTool(),
            UnknownTool(),
        ]

        # --- Dispatcher ---
        self._dispatcher = Dispatcher(
            tools=all_tools,
            tool_timeout_seconds=config.get("tool_timeout_seconds", 5.0),
        )

        # --- NLU ---
        nlu_cfg = config.get("nlu", {})
        self._nlu = NLUEngine(
            dispatcher=self._dispatcher,
            model=nlu_cfg.get("model", "llama-3.1-8b-instant"),
            request_timeout_seconds=nlu_cfg.get("timeout_seconds", 5.0),
            cache_max_size=nlu_cfg.get("cache_max_size", 128),
            cache_ttl_seconds=nlu_cfg.get("cache_ttl_seconds", 300.0),
        )

        # --- Shadow NLU ---
        self._shadow_nlu: NLUEngine | None = None
        shadow_model = nlu_cfg.get("shadow_model")
        if shadow_model:
            self._shadow_nlu = NLUEngine(
                dispatcher=self._dispatcher,
                model=shadow_model,
                request_timeout_seconds=nlu_cfg.get("timeout_seconds", 5.0),
                cache_max_size=1, # Bypass standard caching for real timings
                cache_ttl_seconds=0,
            )

        # --- ASR ---
        asr_cfg = config.get("asr", {})
        self._asr = ASREngine(
            model_size=asr_cfg.get("model_size", "small.en"),
            device=asr_cfg.get("device", "cpu"),
            compute_type=asr_cfg.get("compute_type", "int8"),
            beam_size=asr_cfg.get("beam_size", 1),
            language=asr_cfg.get("language", "en"),
        )

        # --- TTS ---
        tts_cfg = config.get("tts", {})
        preferred_engine_str = tts_cfg.get("engine", "pyttsx3")
        try:
            preferred_engine = TTSEngine(preferred_engine_str)
        except ValueError:
            preferred_engine = TTSEngine.PYTTSX3

        self._tts = TTSService(
            preferred_engine=preferred_engine,
            coqui_model_name=tts_cfg.get(
                "coqui_model", "tts_models/en/ljspeech/tacotron2-DDC"
            ),
            alert_queue=self._alert_queue,
        )

        # --- PTT Recorder ---
        ptt_cfg = config.get("ptt", {})
        self._recorder = PushToTalkRecorder(
            hotkey=ptt_cfg.get("hotkey", "space"),
            sample_rate=asr_cfg.get("sample_rate", 16_000),
        )

        # --- Telemetry Sink ---
        self._telemetry = StructlogTelemetrySink()

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Initialise all subsystems, run health checks, then enter the
        push-to-talk loop.
        """
        log = self._log
        log.info("agent.starting")

        # Inject alert queue into timer tool module
        set_alert_queue(self._alert_queue)
        self._tts.set_alert_queue(self._alert_queue)

        # Load models
        log.info("agent.loading_asr")
        self._asr.load_model()

        log.info("agent.loading_tts")
        self._tts.load_engine()

        # --- Fix 2: Groq startup health check (non-fatal) ---
        nlu_alive = await check_groq_reachable()
        if not nlu_alive:
            log.warning(
                "agent.nlu_backend_unavailable",
                action="NLU will use agent.clarify fallback until Groq API is reachable",
                hint="\u26a0\ufe0f  Set GROQ_API_KEY",
            )
            print(
                f"\n"
                f"  \u26a0\ufe0f  WARNING: Groq API Key is not set\n"
                f"            NLU is disabled. Thiran will ask for clarification\n"
                f"            on every request until you configure the GROQ_API_KEY.\n"
                f"            Fix: export GROQ_API_KEY='your_key'\n"
            )
        else:
            log.info("agent.nlu_backend_ok", backend="Groq API")
            
        from core.tui import set_tui_status
        set_tui_status("🔊 Initializing Text-to-Speech...")

        # Startup announcement
        await self._tts.speak("Thiran is ready. Hold space to speak.")

        # Register signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_shutdown)

        self._running = True
        await self._run_loop()

    async def _run_loop(self) -> None:
        """Main push-to-talk event loop."""
        log = self._log
        import uuid

        while self._running:
            request_id = str(uuid.uuid4())
            trace = AgentTrace(request_id=request_id)
            total_start_ns = time.monotonic_ns()

            # 1. Record audio
            segment = await self._recorder.record(request_id=request_id)
            if segment is None:
                continue

            # 2. Process utterance
            response = await self._process(
                AgentRequest(request_id=request_id, audio_segment=segment),
                trace=trace,
            )

            # 3. Update intent panel and speak response
            t0_tts = time.monotonic_ns()
            from core.tui import set_tui_status, set_tui_intent
            intent_label = response.intent.value if response.intent else None
            tool_msg = response.tool_result.message if response.tool_result else None
            set_tui_intent(intent_label, response.pipeline_success, tool_msg)
            set_tui_status("🔊 Speaking...")
            await self._tts.speak(response.response_text, request_id=request_id)
            trace.record_latency("tts", (time.monotonic_ns() - t0_tts) // 1_000_000)

            # 4. Drain timer alerts
            await self._tts.speak_alerts(request_id=request_id)

            trace.record_latency("total", (time.monotonic_ns() - total_start_ns) // 1_000_000)
            
            # Explicit telemetry markers required by systems engineer review
            trace.cache_hit = getattr(self._nlu, "_last_hit", False)
            trace.recording_duration_ms = segment.duration_ms
            trace.time_to_response_ms = (t0_tts - total_start_ns) // 1_000_000
            
            self._telemetry.flush(trace)

        log.info("agent.loop_exited")

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    async def _process(self, request: AgentRequest, trace: AgentTrace) -> AgentResponse:
        """
        Run one full pipeline cycle: ASR → NLU → Validator → Dispatcher → response.

        Always returns an AgentResponse — never raises.
        """
        log = self._log.bind(request_id=request.request_id)
        pipeline_start = time.monotonic_ns()

        try:
            # --- ASR ---
            t0_asr = time.monotonic_ns()
            transcript = await self._asr.transcribe(request.audio_segment)
            trace.record_latency("asr", (time.monotonic_ns() - t0_asr) // 1_000_000)
            trace.transcript = transcript.text
            
            log.info(
                "pipeline.transcript",
                text=transcript.text,
                confidence=transcript.confidence,
            )

            if not transcript.text.strip():
                trace.error_code = "EMPTY_TRANSCRIPT"
                return self._make_response(
                    request_id=request.request_id,
                    text="I didn't catch any speech. Please try again.",
                    pipeline_success=False,
                    start_ns=pipeline_start,
                )

            # --- NLU ---
            t0_nlu = time.monotonic_ns()
            raw_json = await self._nlu.infer(
                transcript=transcript.text,
                request_id=request.request_id,
            )
            trace.record_latency("nlu", (time.monotonic_ns() - t0_nlu) // 1_000_000)

            # --- Shadow NLU ---
            if self._shadow_nlu:
                asyncio.create_task(
                    self._run_shadow_comparison(
                        transcript=transcript.text,
                        request_id=request.request_id,
                        primary_raw_json=raw_json,
                    )
                )

            # --- Validator ---
            t0_val = time.monotonic_ns()
            validation_result = validate(
                raw_json=raw_json,
                request_id=request.request_id,
                raw_transcript=transcript.text,
                simulate=request.simulate,
                context=self._context,
            )
            trace.record_latency("validator", (time.monotonic_ns() - t0_val) // 1_000_000)

            if isinstance(validation_result, ValidationError):
                # Invalidate the cache entry so a retry goes back to the LLM
                self._nlu.invalidate_cache_for(transcript.text)
                log.warning(
                    "pipeline.validation_failed",
                    code=validation_result.code,
                    message=validation_result.message,
                )
                trace.success = False
                trace.error_code = validation_result.code.value
                return self._make_response(
                    request_id=request.request_id,
                    text=validation_result.message,
                    pipeline_success=False,
                    validation_error=validation_result,
                    start_ns=pipeline_start,
                )

            trace.resolved_intent = validation_result.intent.value

            log.info(
                "pipeline.validated",
                intent=validation_result.intent.value,
                confidence=validation_result.confidence,
                coercions=validation_result.coercion_log,
            )

            # --- Dispatcher ---
            t0_tool = time.monotonic_ns()
            tool_result = await self._dispatcher.dispatch(validation_result)
            trace.record_latency("tool", (time.monotonic_ns() - t0_tool) // 1_000_000)
            
            trace.success = tool_result.success
            if not tool_result.success:
                trace.error_code = tool_result.error_code

            log.info(
                "pipeline.dispatched",
                intent=validation_result.intent.value,
                success=tool_result.success,
                duration_ms=tool_result.duration_ms,
            )

            if tool_result.success:
                self._context = InteractionContext(
                    intent=validation_result.intent,
                    slots=validation_result.slots,
                )

            return self._make_response(
                request_id=request.request_id,
                text=tool_result.message,
                intent=validation_result.intent,
                tool_result=tool_result,
                pipeline_success=True,
                start_ns=pipeline_start,
            )

        except Exception as exc:  # noqa: BLE001
            log.exception("pipeline.unhandled_error", exc=str(exc))
            trace.success = False
            trace.error_code = "UNHANDLED_PIPELINE_ERROR"
            return self._make_response(
                request_id=request.request_id,
                text="Something went wrong. Please try again.",
                pipeline_success=False,
                start_ns=pipeline_start,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_response(
        self,
        *,
        request_id: str,
        text: str,
        pipeline_success: bool,
        start_ns: int,
        intent: IntentName | None = None,
        tool_result=None,
        validation_error: ValidationError | None = None,
    ) -> AgentResponse:
        duration_ms = (time.monotonic_ns() - start_ns) // 1_000_000
        return AgentResponse(
            request_id=request_id,
            response_text=text,
            intent=intent,
            tool_result=tool_result,
            validation_error=validation_error,
            pipeline_success=pipeline_success,
            total_duration_ms=duration_ms,
        )

    async def _run_shadow_comparison(
        self,
        transcript: str,
        request_id: str,
        primary_raw_json: str,
    ) -> None:
        """
        Execute shadow model inference and compare with primary.
        Runs fully detached to ensure zero latency impact on user path.
        """
        if not self._shadow_nlu:
            return
            
        if not getattr(self, "_shadow_semaphore", None):
            self._shadow_semaphore = asyncio.Semaphore(1)
            
        if self._shadow_semaphore.locked():
            self._log.bind(request_id=request_id, layer="shadow_nlu").warning("shadow_nlu.dropped_due_to_load", reason="Semaphore locked")
            return
            
        async with self._shadow_semaphore:
            log = self._log.bind(request_id=request_id, layer="shadow_nlu")
            try:
                # Retrieve primary intent identity
                primary_val = validate(primary_raw_json, request_id, transcript, simulate=True)
                primary_intent = primary_val.intent.value if isinstance(primary_val, ValidatedIntent) else None

                # Model extraction
                start_ns = time.monotonic_ns()
                shadow_raw = await self._shadow_nlu.infer(transcript, request_id)
                duration_ms = (time.monotonic_ns() - start_ns) // 1_000_000

                # Translate using exact same validator logic
                shadow_val = validate(shadow_raw, request_id, transcript, simulate=True)
                shadow_intent = shadow_val.intent.value if isinstance(shadow_val, ValidatedIntent) else None

                if primary_intent != shadow_intent:
                    log.warning(
                        "shadow_nlu.mismatch",
                        primary=primary_intent,
                        shadow=shadow_intent,
                        duration_ms=duration_ms,
                        transcript=transcript,
                    )
                else:
                    log.info(
                        "shadow_nlu.match",
                        intent=primary_intent,
                        duration_ms=duration_ms,
                    )

            except Exception as exc:  # noqa: BLE001
                log.exception("shadow_nlu.error", exc=str(exc))

    def _handle_shutdown(self) -> None:
        self._log.info("agent.shutdown_signal")
        self._running = False
        self._asr.shutdown()
        self._tts.shutdown()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    
    config_path = Path(__file__).parent / "config" / "config.yaml"
    config = _load_config(config_path)
    log_level = config.get("log_level", "INFO")

    agent = VoiceAgent(config)
    
    from core.tui import ThiranTUI, UIStream
    async def _run_agent():
        await agent.start()
        
    app = ThiranTUI(start_agent_coro=_run_agent)
    _configure_logging(log_level, ui_stream=UIStream(app))

    await app.run_async()


if __name__ == "__main__":
    asyncio.run(main())
