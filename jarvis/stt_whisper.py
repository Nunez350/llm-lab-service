# jarvis/stt_whisper.py

from typing import Optional, Tuple

import numpy as np
from faster_whisper import WhisperModel

from .config import WHISPER_MODEL_NAME, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE, LANGUAGE, DEBUG


class STTEngine:
    def __init__(self):
        print(f"🧠 Loading Whisper model: {WHISPER_MODEL_NAME} ({WHISPER_DEVICE}, {WHISPER_COMPUTE_TYPE})")
        try:
            self.model = WhisperModel(
                WHISPER_MODEL_NAME,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
        except Exception as e:
            print(f"❌ Failed to load Whisper model: {e}")
            # Try without proxy if that's the issue
            import os
            proxy_keys = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
            for key in proxy_keys:
                if key in os.environ and '127.0.0.1' in os.environ[key]:
                    del os.environ[key]
            
            try:
                self.model = WhisperModel(
                    WHISPER_MODEL_NAME,
                    device=WHISPER_DEVICE,
                    compute_type=WHISPER_COMPUTE_TYPE,
                )
                print("✅ Model loaded successfully after disabling proxy")
            except Exception as e2:
                raise RuntimeError(f"Failed to load Whisper model: {e2}")

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        
        try:
            segments, info = self.model.transcribe(
                audio,
                beam_size=1,
                language=LANGUAGE,
            )
            text = "".join(seg.text for seg in segments).strip()
            if DEBUG:
                print(f"🧠 Whisper detected language: {info.language}, prob: {info.language_probability:.2f}")
            return text
        except Exception as e:
            print(f"⚠️ Transcription error: {e}")
            return ""
