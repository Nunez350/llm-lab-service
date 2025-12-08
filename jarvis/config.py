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
USE_HOTWORD = False             # Start with False; set True when Porcupine is set
USE_PORCUPINE = False           # True = use Porcupine; False = press Enter to wake

# If you do use Porcupine:
PORCUPINE_ACCESS_KEY = ""       # put your Picovoice AccessKey here
PORCUPINE_KEYWORD = "jarvis"    # built-in jarvis keyword in pvporcupine

# ---- MISC ----
TTS_WAV_PATH = "jarvis_tts.wav"
TEMP_RECORDING_PATH = "jarvis_temp.wav"
