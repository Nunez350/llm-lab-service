# jarvis/hotword.py

import time
import subprocess
import os
from typing import Optional

from .config import USE_HOTWORD, USE_PORCUPINE, PORCUPINE_ACCESS_KEY, PORCUPINE_KEYWORD, DEBUG, SAMPLE_RATE


class BaseHotwordDetector:
    def wait_for_wake(self):
        raise NotImplementedError
    
    def cleanup(self):
        """Optional cleanup method for resources."""
        pass


class EnterHotwordDetector(BaseHotwordDetector):
    """Simple: press Enter to wake Jarvis."""

    def wait_for_wake(self):
        try:
            input("\n🔵 Press Enter to talk to Jarvis (Ctrl+C to quit)... ")
        except (EOFError, KeyboardInterrupt):
            raise SystemExit


class WhisperHotwordDetector(BaseHotwordDetector):
    """Whisper-based hotword detection (slower but no extra dependencies)."""
    
    def __init__(self):
        from .config import WAKE_WORD
        from .stt_whisper import STTEngine
        
        self.wake_word = WAKE_WORD.lower()
        self.stt = STTEngine()
        self.check_duration = 2  # Check every 2 seconds
        print(f"🎤 Listening for hotword: '{self.wake_word}' (using Whisper)")
        print("   (This continuously records and checks for the wake word)")
    
    def wait_for_wake(self):
        """Continuously record short chunks and check for wake word."""
        import tempfile
        import soundfile as sf
        from scipy import signal
        from .config import SAMPLE_RATE as TARGET_RATE
        
        while True:
            try:
                # Record a short snippet for hotword checking
                short_temp = os.path.join(tempfile.gettempdir(), f"hotword_check_{os.getpid()}.wav")
                
                # Use arecord to avoid sounddevice issues
                try:
                    cmd = [
                        "arecord",
                        "-q",
                        "-D", "pulse",
                        "-d", str(self.check_duration),
                        "-f", "S16_LE",
                        "-c", "1",
                        "-r", str(TARGET_RATE),
                        short_temp,
                    ]
                    subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
                except Exception:
                    pass
                
                # Quick transcription check
                try:
                    if os.path.exists(short_temp) and os.path.getsize(short_temp) > 1000:
                        audio, sr = sf.read(short_temp, dtype="float32")
                        if audio.ndim > 1:
                            audio = audio[:, 0]
                        
                        # Resample if needed
                        if sr != TARGET_RATE:
                            num_samples = int(len(audio) * TARGET_RATE / sr)
                            audio = signal.resample(audio, num_samples)
                        
                        if audio.size > 0:
                            segments, _ = self.stt.model.transcribe(audio, beam_size=1, language="en", vad_filter=True)
                            text = "".join(seg.text for seg in segments).strip().lower()
                            
                            # Check if wake word is in transcription
                            if self.wake_word in text:
                                print(f"🔵 Hotword '{self.wake_word}' detected!")
                                # Clean up and return
                                if os.path.exists(short_temp):
                                    os.remove(short_temp)
                                return
                except Exception:
                    pass  # Silent failure for hotword check
                
                # Clean up temp file
                if os.path.exists(short_temp):
                    try:
                        os.remove(short_temp)
                    except:
                        pass
                
                time.sleep(0.3)  # Brief pause before next check
                        
            except KeyboardInterrupt:
                raise
            except Exception:
                time.sleep(0.5)  # Brief pause on error


class PorcupineHotwordDetector(BaseHotwordDetector):
    """Real hotword detection with Porcupine."""
    
    def __init__(self):
        try:
            import pvporcupine
            import pyaudio
        except ImportError:
            raise ImportError(
                "Porcupine not available. Install with: pip install pvporcupine pyaudio"
            )

        if not PORCUPINE_ACCESS_KEY:
            raise RuntimeError(
                "Porcupine access key not set in config.PORCUPINE_ACCESS_KEY. "
                "Get one at: https://console.picovoice.ai/"
            )

        try:
            self.porcupine = pvporcupine.create(
                access_key=PORCUPINE_ACCESS_KEY,
                keywords=[PORCUPINE_KEYWORD],
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create Porcupine instance: {e}")

        self.pa = pyaudio.PyAudio()
        
        # Try to find default input device
        device_index = None
        try:
            device_info = self.pa.get_default_input_device_info()
            device_index = device_info['index']
        except:
            pass
        
        self.stream = self.pa.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length,
            input_device_index=device_index,
        )
        
        print(f"🎤 Listening for hotword: '{PORCUPINE_KEYWORD}'")

    def wait_for_wake(self):
        import numpy as np

        while True:
            pcm = self.stream.read(self.porcupine.frame_length, exception_on_overflow=False)
            pcm = np.frombuffer(pcm, dtype=np.int16)
            keyword_index = self.porcupine.process(pcm)
            if keyword_index >= 0:
                print("🔵 Hotword detected!")
                # small pause before recording
                time.sleep(0.1)
                return
    
    def cleanup(self):
        """Clean up Porcupine resources."""
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'pa'):
            self.pa.terminate()
        if hasattr(self, 'porcupine'):
            self.porcupine.delete()


def create_hotword_detector() -> BaseHotwordDetector:
    if not USE_HOTWORD:
        if DEBUG:
            print("💤 Hotword disabled, using Enter key to activate.")
        return EnterHotwordDetector()

    if USE_PORCUPINE:
        try:
            return PorcupineHotwordDetector()
        except Exception as e:
            print(f"⚠️ Failed to initialize Porcupine: {e}")
            print("   Falling back to Whisper-based hotword detection.")
            return WhisperHotwordDetector()

    # Use Whisper-based hotword detection (no extra dependencies)
    return WhisperHotwordDetector()
