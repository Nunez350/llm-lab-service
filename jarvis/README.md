# Jarvis Voice Assistant

Production-ready modular voice assistant with hotword detection, speech-to-text, and text-to-speech.

## Features

- 🎤 **Voice Transcription**: Uses Whisper for speech-to-text
- 🔵 **Hotword Detection**: Porcupine or Enter key activation
- 🔊 **Text-to-Speech**: Piper TTS for voice responses
- 🎙️ **Modular Audio I/O**: Support for multiple audio backends
- 📦 **Pluggable Commands**: Easy to extend with new command handlers
- 🔧 **Production Ready**: Clean architecture, error handling, logging

## Installation

### Core Dependencies

```bash
pip install faster-whisper piper-tts sounddevice numpy soundfile scipy
```

### Optional Dependencies

**For Hotword Detection (Porcupine):**
```bash
pip install pvporcupine pyaudio
```

**System Dependencies:**
```bash
# On Ubuntu/Debian
sudo apt-get install ffmpeg alsa-utils pulseaudio-utils
```

## Quick Start

1. **Configure** - Edit `config.py` to adjust settings:
   ```python
   USE_HOTWORD = False  # Start with Enter key activation
   PREFERRED_INPUT_DEVICE_SUBSTR = "Elgato"  # Your mic name
   ```

2. **Download Piper Voice Model** (if not auto-downloaded):
   ```bash
   # Models will be downloaded automatically on first use
   # Or download manually to ~/.local/share/piper/voices/
   ```

3. **Run**:
   ```bash
   cd jarvis
   python -m jarvis.main
   ```

   Or from project root:
   ```bash
   python -m jarvis.main
   ```

## Project Structure

```
jarvis/
├── main.py              # Main entry point and loop
├── config.py            # Configuration settings
├── audio_io.py          # Audio recording (sounddevice + fallbacks)
├── stt_whisper.py       # Speech-to-text engine
├── tts_piper.py         # Text-to-speech engine
├── hotword.py           # Hotword detection (Porcupine/Enter)
├── commands/            # Command handlers
│   ├── __init__.py      # Command dispatcher
│   ├── system.py        # System commands (time, date, etc.)
│   ├── dev.py           # Developer commands (git, tests)
│   └── fun.py           # Fun commands (jokes, greetings)
└── README.md
```

## Configuration

Edit `config.py` to customize:

- **Audio**: Sample rate, device selection
- **STT**: Whisper model size, device (CPU/GPU)
- **TTS**: Piper model path
- **Hotword**: Enable Porcupine, access key, keyword

## Usage

### Basic Mode (Enter Key)

```python
# In config.py
USE_HOTWORD = False
```

Run:
```bash
python -m jarvis.main
# Press Enter to activate, speak your command
```

### Hotword Mode (Porcupine)

```python
# In config.py
USE_HOTWORD = True
USE_PORCUPINE = True
PORCUPINE_ACCESS_KEY = "your-key-here"  # Get from picovoice.ai
```

Run:
```bash
python -m jarvis.main
# Say "jarvis" to activate
```

## Adding Commands

### Create a New Command Module

Create `jarvis/commands/custom.py`:

```python
from typing import Tuple

def try_handle(text: str) -> Tuple[bool, str]:
    if "my command" in text:
        return True, "Response text here"
    return False, ""
```

### Register in Dispatcher

Edit `jarvis/commands/__init__.py`:

```python
from . import system, dev, fun, custom  # Add custom

def handle_command(text: str):
    for module in (system, dev, fun, custom):  # Add custom
        handled, reply = module.try_handle(text)
        if handled:
            return (reply, module.__name__)
```

## Workflow

The main loop follows this pattern:

1. **Wake** → Wait for hotword/Enter
2. **Record** → Capture audio (default 4 seconds)
3. **Transcribe** → Convert speech to text (Whisper)
4. **Handle** → Process command and generate response
5. **Speak** → Output response via TTS
6. **Idle** → Return to listening state

## Troubleshooting

### Audio Issues

- Check microphone: `python -m jarvis.audio_io` (if you add a test script)
- List devices: `python -c "import sounddevice as sd; print(sd.query_devices())"`
- Adjust `PREFERRED_INPUT_DEVICE_SUBSTR` in config.py

### TTS Issues

- Ensure Piper model exists at path in config
- Models auto-download to `~/.local/share/piper/voices/`
- Check audio output: `paplay test.wav` or `aplay test.wav`

### Hotword Issues

- Get Porcupine access key: https://console.picovoice.ai/
- Test without hotword first: `USE_HOTWORD = False`
- Check microphone permissions

## Development

### Running Tests

```bash
# Test audio recording
python -c "from jarvis.audio_io import AudioRecorder; r = AudioRecorder(); print(r.record(1).shape)"

# Test STT
python -c "from jarvis.stt_whisper import STTEngine; s = STTEngine(); print(s.transcribe(...))"

# Test TTS
python -c "from jarvis.tts_piper import TTSEngine; t = TTSEngine(); t.speak('Test')"
```

## Notes

- Temp files (`jarvis_tts.wav`, `jarvis_temp.wav`) are created in current directory
- All modules have graceful fallbacks for missing dependencies
- Error handling with automatic retry and recovery
- Debug mode provides detailed logging (set `DEBUG = True` in config)

## License

Part of llm-lab-service project.
