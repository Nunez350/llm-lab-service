# jarvis/tts_piper.py

import subprocess
import wave
from pathlib import Path

try:
    from piper import PiperVoice
    from piper.download import ensure_voice_exists, find_voice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False

from .config import PIPER_MODEL_PATH, PIPER_CONFIG_PATH, TTS_WAV_PATH, DEBUG


class TTSEngine:
    def __init__(self):
        if not PIPER_AVAILABLE:
            raise ImportError(
                "Piper TTS not available. Install with: pip install piper-tts"
            )
        
        model_path = Path(PIPER_MODEL_PATH)
        
        # Try to find voice if model doesn't exist at exact path
        if not model_path.exists():
            if DEBUG:
                print(f"⚠️ Model not found at {model_path}, attempting to find/download...")
            
            # Try to find or download default voice
            try:
                voice_name = "en_US-lessac-medium"
                try:
                    voice_path, config_path = find_voice(voice_name)
                    model_path = Path(voice_path)
                    if PIPER_CONFIG_PATH is None:
                        # Update config path if we found one
                        pass
                except:
                    if DEBUG:
                        print(f"   Attempting to download voice: {voice_name}")
                    voice_path, config_path = ensure_voice_exists(voice_name, [voice_name])
                    model_path = Path(voice_path)
            
            except Exception as e:
                if DEBUG:
                    print(f"   Could not auto-download voice: {e}")
                raise FileNotFoundError(
                    f"Piper model not found at {PIPER_MODEL_PATH} and could not auto-download. "
                    f"Please download a voice model manually or adjust PIPER_MODEL_PATH in config.py"
                )
        
        # Determine config path
        config_path = PIPER_CONFIG_PATH
        if config_path is None:
            # Auto-detect: try .json file next to model
            json_path = model_path.with_suffix('.json')
            if json_path.exists():
                config_path = str(json_path)
            else:
                config_path = None
        
        print(f"🗣️ Loading Piper model: {model_path}")
        if config_path:
            self.voice = PiperVoice.load(str(model_path), config_path=config_path, use_cuda=False)
        else:
            self.voice = PiperVoice.load(str(model_path), use_cuda=False)

    def speak(self, text: str):
        text = text.strip()
        if not text:
            return

        # synthesize to WAV
        with wave.open(TTS_WAV_PATH, "wb") as wav_file:
            self.voice.synthesize(
                text=text,
                wav_file=wav_file,
                # You can tweak:
                # length_scale=0.9,
                # noise_scale=0.6,
                # noise_w=0.8,
                # sentence_silence=0.2,
            )

        # play
        try:
            subprocess.run(
                ["paplay", TTS_WAV_PATH],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            try:
                subprocess.run(
                    ["aplay", TTS_WAV_PATH],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                try:
                    subprocess.run(
                        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", TTS_WAV_PATH],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except FileNotFoundError:
                    print("⚠️ Could not play audio (paplay/aplay/ffplay not available)")
                    print(f"   Audio saved to: {TTS_WAV_PATH}")
