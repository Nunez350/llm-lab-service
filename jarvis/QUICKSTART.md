# JARVIS Quick Start Guide

Get JARVIS up and running in minutes!

## Prerequisites

### 1. Install Python Dependencies

```bash
# Core dependencies (required)
pip install faster-whisper piper-tts sounddevice numpy soundfile scipy transformers torch

# Optional: Hotword detection
pip install pvporcupine pyaudio

# System dependencies (Ubuntu/Debian)
sudo apt-get install ffmpeg alsa-utils pulseaudio-utils
```

### 2. Configure Settings

Edit `jarvis/config.py`:

```python
# Audio settings
PREFERRED_INPUT_DEVICE_SUBSTR = "Elgato"  # Your mic name, or None for default

# LLM settings (if using local LLM)
LLM_ENABLED = True  # Set False to disable LLM fallback
LLM_MODEL_NAME = "/home/rnu/mnt/models/phi_models/"  # Your model path
LLM_DEVICE = "cuda"  # or "cpu"

# Hotword (if using Porcupine)
USE_HOTWORD = False  # Start with False (Enter key mode)
USE_PORCUPINE = False
PORCUPINE_ACCESS_KEY = ""  # Get from https://console.picovoice.ai/
```

## Running JARVIS

### Option 1: Basic Mode (Press Enter to Activate)

This is the simplest way to start:

```bash
cd jarvis
python -m jarvis.main
```

Or from project root:
```bash
python -m jarvis.main
```

**What happens:**
1. JARVIS loads Whisper (STT) and Piper (TTS)
2. Press **Enter** when prompted
3. Speak your command
4. JARVIS processes and responds

### Option 2: Hotword Mode (Porcupine)

Enable hotword detection:

1. Edit `config.py`:
   ```python
   USE_HOTWORD = True
   USE_PORCUPINE = True
   PORCUPINE_ACCESS_KEY = "your-key-here"
   ```

2. Run:
   ```bash
   python -m jarvis.main
   ```

3. Say **"jarvis"** to activate (no Enter needed)

## First Steps

### 1. Test Audio Recording

```bash
# Run diagnostics
python -m jarvis.main --diagnose
```

This shows:
- Available microphones
- Current audio settings
- Device capabilities

### 2. Test Basic Commands

Try these voice commands:
- **"What time is it?"** → Shows current time
- **"My name is [Your Name]"** → JARVIS remembers it
- **"What's my name?"** → JARVIS recalls your name
- **"Tell me a joke"** → JARVIS tells a joke

### 3. Test LLM Integration

If `LLM_ENABLED = True`, try open-ended questions:
- **"Summarize what we've been talking about"**
- **"Explain quantum computing in simple terms"**
- **"What should I do based on my last git status?"**

## Troubleshooting

### Audio Issues

```bash
# Check microphone
python -m jarvis.main --diagnose

# Test specific device
python -c "import sounddevice as sd; print(sd.query_devices())"

# Adjust volume
pactl set-source-volume <source> 150%
```

### LLM Issues

```python
# Disable LLM temporarily
# In config.py:
LLM_ENABLED = False
```

### Port Conflicts

```bash
# Check if port is in use
./scripts/check_port.sh 9020
```

## Quick Test Without Running Full System

### Test STT Only
```python
python -c "
from jarvis.stt_whisper import STTEngine
from jarvis.audio_io import AudioRecorder
import numpy as np

stt = STTEngine()
recorder = AudioRecorder()
audio = recorder.record(4)  # Record 4 seconds
text = stt.transcribe(audio)
print('Heard:', text)
"
```

### Test TTS Only
```python
python -c "
from jarvis.tts_piper import TTSEngine
tts = TTSEngine()
tts.speak('Hello, this is a test.')
"
```

### Test LLM Only
```python
python -c "
from jarvis.llm_agent import LocalLLMAgent
from jarvis.memory import ConversationMemory

llm = LocalLLMAgent()
memory = ConversationMemory()
reply = llm.generate_reply(memory, 'What is 2+2?')
print('LLM:', reply)
"
```

## Common Workflows

### Development Mode
```bash
# Disable LLM for faster startup
# In config.py: LLM_ENABLED = False
python -m jarvis.main
```

### Production Mode
```bash
# Enable LLM, use Porcupine hotword
# In config.py: LLM_ENABLED = True, USE_PORCUPINE = True
python -m jarvis.main
```

### Testing Mode
```bash
# Run diagnostics first
python -m jarvis.main --diagnose

# Then run with debug output
python -m jarvis.main
# (DEBUG = True in config.py shows detailed logs)
```

## File Structure

```
jarvis/
├── main.py              # Entry point - run this!
├── config.py            # All settings here
├── audio_io.py          # Microphone recording
├── stt_whisper.py       # Speech-to-text
├── tts_piper.py         # Text-to-speech
├── llm_agent.py         # Local LLM integration
├── memory.py            # Conversation memory
├── hotword.py           # Hotword detection
└── commands/            # Command handlers
    ├── system.py        # System commands
    ├── dev.py           # Developer commands
    └── fun.py           # Fun commands
```

## Next Steps

1. ✅ Run diagnostics: `python -m jarvis.main --diagnose`
2. ✅ Test basic mode: `python -m jarvis.main` (press Enter)
3. ✅ Try commands: "what time is it", "my name is..."
4. ✅ Enable LLM if needed: set `LLM_ENABLED = True`
5. ✅ Customize commands: edit `commands/*.py`

Enjoy your JARVIS assistant! 🎤🤖

