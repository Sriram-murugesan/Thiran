# Thiran — Local AI Voice Agent

A production-grade, modular voice assistant that runs entirely on your machine.
Push-to-talk → speech recognition → local LLM intent parsing → action execution → spoken response.

---

## Architecture at a Glance

```
[Hold SPACE] → VAD → ASR (Whisper) → NLU (Groq API) → Validator → Dispatcher → Tool → TTS → [Hear response]
```

**Key design principles:**
- LLM outputs **only a JSON `IntentPayload`** — it never executes actions
- A **Validator layer** enforces strict schema before anything reaches a tool
- Each tool is fully isolated — adding a new one requires one class + one YAML entry
- The intent cache eliminates repeat LLM calls for common commands

---

## Prerequisites

### System packages

```bash
# PulseAudio volume control (for system.volume + spotify controls)
sudo apt install playerctl pactl

# Text-to-speech engine (pyttsx3 fallback)
sudo apt install espeak espeak-ng

# Required for push-to-talk key capture
# keyboard library needs root OR uinput group membership:
sudo usermod -aG input $USER   # then log out and back in
```

### Groq API (Cloud LLM)

```bash
# Set up your Groq API Key
# Go to https://console.groq.com/keys and generate an API key
# Create a .env file in the project root:
echo 'GROQ_API_KEY="gsk_your_key_here"' > .env
```

---

## Installation

```bash
# Clone / enter project
cd thiran

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (including python-dotenv for env loading)
pip install -r requirements.txt python-dotenv

# Optional: high-quality Coqui TTS (larger install, ~1GB models)
pip install TTS sounddevice
```

---

## Configuration

Edit `config/config.yaml` to tune each layer:

```yaml
asr:
  model_size: small.en    # tiny.en (fast) → large-v3 (accurate)
  device: cpu             # cpu or cuda

nlu:
  model: llama-3.1-8b-instant      # any Groq supported model
  timeout_seconds: 5.0

tts:
  engine: pyttsx3         # pyttsx3 (instant) or coqui (higher quality)

ptt:
  hotkey: space           # key to hold while speaking
```

### Adding a new app shortcut

Edit `config/known_apps.yaml`:
```yaml
figma: figma-linux
postman: postman
```

---

## Running

```bash
# From the project root with venv active:
python agent.py

# Or via the installed entry point (after: pip install -e .):
thiran
```

Hold **SPACE**, speak your command, release. Thiran responds aloud.

**Example commands:**
- *"Play Bohemian Rhapsody by Queen on Spotify"*
- *"Skip to the next track"*
- *"Set the volume to 40"*
- *"Open VS Code"*
- *"Search YouTube for Python tutorials"*
- *"Set a timer for 10 minutes"*
- *"Search Google for asyncio best practices"*

---

## Running Tests

```bash
# Install dev dependencies
pip install ".[dev]"

# Run all tests
pytest

# Run with coverage report
pytest --cov=core --cov=tools --cov-report=term-missing

# Run only validator tests (no external dependencies)
pytest tests/test_validator.py -v
```

The Validator tests are **pure Python** — no Ollama, no microphone, no models needed.

---

## Project Structure

```
thiran/
├── agent.py                    # Entry point — push-to-talk loop
├── config/
│   ├── config.yaml             # All tuneable parameters
│   └── known_apps.yaml         # App name → shell command aliases
├── core/
│   ├── models.py               # All Pydantic data contracts (single source of truth)
│   ├── validator.py            # Trust boundary between LLM and Dispatcher
│   ├── nlu.py                  # Groq HTTP client + LRU intent cache
│   ├── asr.py                  # faster-whisper wrapper
│   ├── tts.py                  # Coqui / pyttsx3 / console TTS
│   └── dispatcher.py           # Tool registry + rate limits + hooks
├── tools/
│   ├── base.py                 # BaseTool abstract class
│   ├── spotify/                # play, pause, skip, volume
│   ├── youtube/                # play, search
│   ├── system/                 # open_app, volume, timer
│   ├── web/                    # search
│   └── agent/                  # clarify, unknown (no-op fallbacks)
├── tests/
│   ├── test_validator.py       # 50+ pure unit tests for the Validator
│   └── test_models.py          # Pydantic constraint tests
├── requirements.txt
└── pyproject.toml
```

---

## Adding a New Tool

1. **Create** `tools/<domain>/<action>.py` implementing `BaseTool`
2. **Add** the intent to `IntentName` enum in `core/models.py`
3. **Register** the tool in `VoiceAgent.__init__` in `agent.py`
4. **Add** a slot validator function in `core/validator.py`
5. The LLM system prompt updates automatically from the registry

---

## Failure Handling

| Scenario | Response |
|---|---|
| Groq API unreachable  | Validation failure fallback JSON ("I'm not sure how to help with that") |
| LLM schema violation | TTS: "I couldn't parse that — please try again" |
| Low LLM confidence | TTS: "I didn't quite catch that, could you repeat?" |
| Tool timeout (>5s) | TTS: "That took too long, please try again" |
| App not found | TTS: "I don't know how to open {app}" |
| Rate limit hit | TTS: "I'm being asked to do that too often" |
| TTS engine crash | Prints response to console (silent degraded mode) |

---

## License

MIT
