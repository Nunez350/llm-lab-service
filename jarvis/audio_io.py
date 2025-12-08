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

# Always import subprocess fallback utilities
import subprocess
import tempfile
import os
import soundfile as sf

from .config import SAMPLE_RATE, CHANNELS, RECORD_DURATION_SEC, PREFERRED_INPUT_DEVICE_SUBSTR, DEBUG


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


def get_device_supported_rate(device_index: Optional[int] = None, preferred_rate: int = SAMPLE_RATE) -> int:
    """Get a supported sample rate for the device, falling back to preferred rate."""
    if not SOUNDDEVICE_AVAILABLE:
        return preferred_rate
    
    try:
        device_info = sd.query_devices(device_index if device_index is not None else sd.default.device[0])
        default_rate = int(device_info['default_samplerate'])
        
        # Common supported rates to try
        common_rates = [preferred_rate, default_rate, 44100, 48000, 16000, 22050, 32000]
        
        # Remove duplicates and sort by proximity to preferred
        unique_rates = sorted(set(common_rates), key=lambda x: abs(x - preferred_rate))
        
        # Try to find a supported rate
        for rate in unique_rates:
            try:
                # Quick test to see if rate is supported
                test_stream = sd.InputStream(samplerate=rate, channels=1, device=device_index, dtype='float32')
                test_stream.close()
                if DEBUG and rate != preferred_rate:
                    print(f"   Using sample rate {rate} Hz (device supports {default_rate} Hz)")
                return rate
            except:
                continue
        
        # Fallback to device default
        return default_rate
    except Exception as e:
        if DEBUG:
            print(f"   Could not query device sample rate: {e}, using {preferred_rate} Hz")
        return preferred_rate


class AudioRecorder:
    """Blocking recorder: record N seconds of mono audio at SAMPLE_RATE."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS, prefer_subprocess: bool = False):
        self.sample_rate = sample_rate
        self.channels = channels
        self._q = queue.Queue()
        self.prefer_subprocess = prefer_subprocess
        self._sounddevice_failed = False  # Track if sounddevice has failed before

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
        # Skip sounddevice if we prefer subprocess or if it has failed before
        if self.prefer_subprocess or self._sounddevice_failed or not SOUNDDEVICE_AVAILABLE:
            return self._record_fallback(duration_sec)
        
        self._q = queue.Queue()
        device_index = find_input_device(PREFERRED_INPUT_DEVICE_SUBSTR)
        
        # Get a supported sample rate for this device
        actual_rate = get_device_supported_rate(device_index, self.sample_rate)
        
        # Adjust if rate changed
        if actual_rate != self.sample_rate:
            if DEBUG:
                print(f"🔄 Device sample rate: {actual_rate} Hz (requested {self.sample_rate} Hz)")

        frames_needed = int(actual_rate * duration_sec)
        collected = []

        try:
            with sd.InputStream(
                samplerate=actual_rate,
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
        except Exception as e:
            # Mark sounddevice as failed and fall back
            self._sounddevice_failed = True
            if DEBUG:
                print(f"⚠️ sounddevice error (using subprocess fallback): {str(e)[:100]}")
            return self._record_fallback(duration_sec)

        if not collected:
            return np.zeros(0, dtype="float32")

        audio = np.concatenate(collected)
        audio = audio[:frames_needed].astype("float32")
        
        # Resample if we had to use a different rate
        if actual_rate != self.sample_rate:
            from scipy import signal
            num_samples = int(len(audio) * self.sample_rate / actual_rate)
            audio = signal.resample(audio, num_samples)
        
        return audio
    
    def _record_fallback(self, duration_sec: float) -> np.ndarray:
        """Fallback recording method using arecord/ffmpeg."""
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file.close()
        
        recording_success = False
        
        # Try arecord first (more reliable with PulseAudio)
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
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=duration_sec + 3
            )
            recording_success = True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            # Fallback to ffmpeg
            try:
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
                recording_success = True
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                if os.path.exists(temp_file.name):
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
                if DEBUG:
                    print("   Recording command failed")
                return np.zeros(0, dtype="float32")
        
        if not recording_success:
            if os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
            return np.zeros(0, dtype="float32")
        
        # Wait a moment for file to be fully written
        import time
        time.sleep(0.1)
        
        # Load audio file
        try:
            if not os.path.exists(temp_file.name) or os.path.getsize(temp_file.name) < 1000:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                if DEBUG:
                    print("   Recorded file too small or missing")
                return np.zeros(0, dtype="float32")
            
            audio, sr = sf.read(temp_file.name, dtype="float32")
            if audio.ndim > 1:
                audio = audio[:, 0]  # Convert to mono
            
            # Resample if needed
            if sr != self.sample_rate:
                from scipy import signal
                num_samples = int(len(audio) * self.sample_rate / sr)
                audio = signal.resample(audio, num_samples)
            
            os.unlink(temp_file.name)
            
            # Check if audio has actual content (not just silence/noise)
            if len(audio) == 0 or np.abs(audio).max() < 0.001:
                if DEBUG:
                    print("   Audio appears to be silence")
                return np.zeros(0, dtype="float32")
            
            return audio.astype("float32")
        except Exception as e:
            if os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
            if DEBUG:
                print(f"   Error loading audio: {e}")
            return np.zeros(0, dtype="float32")
