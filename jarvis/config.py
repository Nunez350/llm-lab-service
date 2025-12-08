# jarvis/config.py

from pathlib import Path

# ---- GENERAL ----
WAKE_WORD = "jarvis"            # For logic / intent, not hotword engine
LANGUAGE = "en"                 # Whisper language
DEBUG = True

# ---- AUDIO / MIC ----
SAMPLE_RATE = 16000             # Whisper & Piper default fine
CHANNELS = 1
RECORD_DURATION_SEC = 4         # How long Jarvis listens after wake word

# You can override this if you want a specific input device
PREFERRED_INPUT_DEVICE_SUBSTR = "Elgato"   # use None to take default

# ---- STT (Whisper) ----
WHISPER_MODEL_NAME = "tiny.en"  # good for realtime
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# ---- TTS (Piper) ----
# Adjust path to your downloaded Piper model:
PIPER_MODEL_PATH = (
    Path.home()
    / ".local/share/piper/voices/en_US-lessac-medium.onnx"
)

# Alternative: can also use config.json path
PIPER_CONFIG_PATH = None  # If None, will auto-detect from model path

# ---- HOTWORD ----
USE_HOTWORD = True              # Enable hotword detection (Whisper or Porcupine)
USE_PORCUPINE = False           # True = use Porcupine (requires access key); False = use Whisper-based detection

# If you do use Porcupine:
PORCUPINE_ACCESS_KEY = ""       # put your Picovoice AccessKey here (get from https://console.picovoice.ai/)
PORCUPINE_KEYWORD = "jarvis"    # built-in jarvis keyword in pvporcupine

# ---- MEMORY ----
MAX_MEMORY_TURNS = 30

# ---- LLM (API) ----
LLM_ENABLED = True
LLM_API_URL = "http://localhost:9020/v1/chat/completions"  # OpenAI-compatible API endpoint
LLM_MODEL_NAME = "phi3-mini"  # Model name as registered in the API server
LLM_MAX_NEW_TOKENS = 256
LLM_TEMPERATURE = 0.4
# How many past turns to include in the prompt
LLM_CONTEXT_TURNS = 10
# High-level system prompt / personality
LLM_SYSTEM_PROMPT = """You are Jarvis, a helpful, efficient voice assistant running locally on the user's workstation.

- You speak concisely and clearly.

- You can reference earlier parts of this conversation when helpful.

- You only describe actions; the actual system control is done by other services.

If you are not sure about something, ask a brief clarifying question."""

# ---- MISC ----
TTS_WAV_PATH = "jarvis_tts.wav"
TEMP_RECORDING_PATH = "jarvis_temp.wav"
