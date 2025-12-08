# jarvis/hotword.py

import time
from typing import Optional

from .config import USE_HOTWORD, USE_PORCUPINE, PORCUPINE_ACCESS_KEY, PORCUPINE_KEYWORD, DEBUG


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
            print("   Falling back to Enter key activation.")
            return EnterHotwordDetector()

    # Fallback to Enter
    return EnterHotwordDetector()
