# jarvis/audio_io.py

import queue
import threading
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    # Fallback to subprocess-based recording
    import subprocess
    import tempfile
    import os
    import soundfile as sf

from .config import SAMPLE_RATE, CHANNELS, RECORD_DURATION_SEC, PREFERRED_INPUT_DEVICE_SUBSTR


def find_input_device(name_substr: Optional[str] = None) -> Optional[int]:
    """Return device index whose name contains name_substr, else default."""
    if not SOUNDDEVICE_AVAILABLE:
        return None
    
    devices = sd.query_devices()
    default_index = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device

    if not name_substr:
        return default_index

    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0 and name_substr.lower() in dev["name"].lower():
            return idx

    return default_index


class AudioRecorder:
    """Blocking recorder: record N seconds of mono audio at SAMPLE_RATE."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS):
        self.sample_rate = sample_rate
        self.channels = channels
        self._q = queue.Queue()

    def _callback(self, indata, frames, time, status):
        if status:
            print("🔊 Audio callback status:", status)
        # Flatten to mono if needed
        if self.channels > 1:
            data = np.mean(indata, axis=1)
        else:
            data = indata[:, 0]
        self._q.put(data.copy())

    def record(self, duration_sec: float) -> np.ndarray:
        """Record duration_sec seconds and return numpy float32 array."""
        if not SOUNDDEVICE_AVAILABLE:
            return self._record_fallback(duration_sec)
        
        self._q = queue.Queue()
        device_index = find_input_device(PREFERRED_INPUT_DEVICE_SUBSTR)

        frames_needed = int(self.sample_rate * duration_sec)
        collected = []

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._callback,
            device=device_index,
            dtype="float32",
        ):
            while sum(len(chunk) for chunk in collected) < frames_needed:
                try:
                    chunk = self._q.get(timeout=duration_sec + 1)
                except queue.Empty:
                    break
                collected.append(chunk)

        if not collected:
            return np.zeros(0, dtype="float32")

        audio = np.concatenate(collected)
        return audio[:frames_needed].astype("float32")
    
    def _record_fallback(self, duration_sec: float) -> np.ndarray:
        """Fallback recording method using arecord/ffmpeg."""
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file.close()
        
        try:
            # Try ffmpeg with pulse first
            cmd = [
                "ffmpeg",
                "-f", "pulse",
                "-i", "default",
                "-ar", str(self.sample_rate),
                "-ac", "1",
                "-t", str(duration_sec),
                "-y",
                temp_file.name,
            ]
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=duration_sec + 3
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to arecord
            try:
                cmd = [
                    "arecord",
                    "-q",
                    "-D", "pulse",
                    "-d", str(int(duration_sec)),
                    "-f", "S16_LE",
                    "-c", "1",
                    "-r", str(self.sample_rate),
                    temp_file.name,
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError:
                os.unlink(temp_file.name)
                return np.zeros(0, dtype="float32")
        
        # Load audio file
        try:
            audio, sr = sf.read(temp_file.name, dtype="float32")
            if audio.ndim > 1:
                audio = audio[:, 0]  # Convert to mono
            
            # Resample if needed
            if sr != self.sample_rate:
                from scipy import signal
                num_samples = int(len(audio) * self.sample_rate / sr)
                audio = signal.resample(audio, num_samples)
            
            os.unlink(temp_file.name)
            return audio.astype("float32")
        except Exception as e:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            print(f"⚠️ Error loading audio: {e}")
            return np.zeros(0, dtype="float32")
