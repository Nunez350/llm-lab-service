#!/usr/bin/env python3
"""
Voice transcription script with Elgato microphone support
Includes comprehensive diagnostics for troubleshooting audio issues
"""

import subprocess
import numpy as np
import soundfile as sf
import os
import sys
import tempfile
import argparse
import time
from scipy import signal
from faster_whisper import WhisperModel

# Optional hotword detection imports
try:
    import pvporcupine
    import pyaudio
    PORCUPINE_AVAILABLE = True
except ImportError:
    PORCUPINE_AVAILABLE = False

# Optional TTS imports
try:
    from piper import PiperVoice
    from piper.download import ensure_voice_exists, find_voice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False

# Check for command-line piper (lazy check)
PIPER_CMD_AVAILABLE = None  # Will be checked on first use

# Temporarily disable proxy if it's causing issues (common with 127.0.0.1 proxy timeouts)
_original_proxy = {}
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    _original_proxy[key] = os.environ.get(key)
    # Unset problematic localhost proxies
    if key in os.environ and '127.0.0.1' in os.environ[key]:
        del os.environ[key]

# ---- CONFIG ----
RECORD_RATE = 48000          # Elgato supports 48000 Hz via PipeWire
SAMPLE_RATE = 16000          # what Whisper expects (will resample)
DURATION_SEC = 4             # how long to listen after you press Enter

# Temp file - use tempfile directory, unique per run
TEMP_WAV = os.path.join(tempfile.gettempdir(), f"jarvis_temp_{os.getpid()}.wav")

# Global config (set by argparse)
USE_PULSE = True
PULSE_SOURCE = None  # Auto-detect if None
ALSA_DEVICE = "hw:5,0"

# Lazy-loaded Whisper model
_whisper_model = None

# Lazy-loaded Piper TTS model
_piper_voice = None
_piper_model_path = None
_piper_config_path = None
# -----------------


def get_whisper():
    """Lazy-load Whisper model (singleton)."""
    global _whisper_model
    if _whisper_model is None:
        try:
            _whisper_model = WhisperModel(
                "tiny.en",
                device="cpu",
                compute_type="int8",
                download_root=None,
            )
        except Exception as e:
            print(f"❌ Failed to load Whisper model: {e}")
            if "proxy" in str(e).lower() or "127.0.0.1" in str(e):
                print("   Proxy issue detected. Trying without proxy...")
                for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
                    os.environ.pop(key, None)
                try:
                    _whisper_model = WhisperModel(
                        "tiny.en",
                        device="cpu",
                        compute_type="int8",
                        download_root=None,
                    )
                    print("✅ Model loaded successfully after disabling proxy")
                except Exception as e2:
                    print(f"❌ Still failed: {e2}")
                    print("\n   To fix:")
                    print("   1. Check your proxy settings: env | grep -i proxy")
                    print("   2. Try: unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy")
                    raise
            else:
                raise
    return _whisper_model


def run_diagnostics():
    """Run comprehensive audio diagnostics to help troubleshoot issues."""
    print("\n" + "="*60)
    print("🔍 AUDIO DIAGNOSTICS")
    print("="*60)
    
    # Check PulseAudio/PipeWire
    if USE_PULSE:
        print("\n📊 PulseAudio/PipeWire Sources:")
        try:
            result = subprocess.run(
                ["pactl", "list", "short", "sources"],
                capture_output=True, text=True, check=True
            )
            sources = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        source_id = parts[0]
                        source_name = parts[1]
                        sources.append((source_id, source_name))
                        marker = " ⭐" if 'Elgato' in source_name or 'Wave' in source_name else ""
                        print(f"   {source_id}: {source_name}{marker}")
            
            if not sources:
                print("   ⚠️ No sources found!")
            
            # Check default source
            try:
                result = subprocess.run(
                    ["pactl", "get-default-source"],
                    capture_output=True, text=True, check=True
                )
                default = result.stdout.strip()
                print(f"\n📌 Default source: {default}")
            except:
                print("\n⚠️ Could not get default source")
            
            # Check source volumes and mute status
            print("\n📊 Source Details (Elgato/Wave):")
            try:
                result = subprocess.run(
                    ["pactl", "list", "sources"],
                    capture_output=True, text=True, check=True
                )
                in_elgato = False
                for line in result.stdout.split('\n'):
                    if 'Elgato' in line or 'Wave' in line:
                        in_elgato = True
                        print(f"   {line.strip()}")
                    elif in_elgato:
                        if any(x in line for x in ['Mute:', 'Volume:', 'Name:', 'State:']):
                            print(f"   {line.strip()}")
                        elif line.strip() == '':
                            in_elgato = False
                        elif 'alsa' in line.lower() or 'card' in line.lower():
                            print(f"   {line.strip()}")
            except Exception as e:
                print(f"   ⚠️ Error checking source details: {e}")
        
        except FileNotFoundError:
            print("   ❌ pactl not found. Is PulseAudio/PipeWire installed?")
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️ Error listing sources: {e}")
    
    # Check ALSA devices
    print("\n📊 ALSA Devices:")
    try:
        result = subprocess.run(
            ["arecord", "-l"],
            capture_output=True, text=True, check=True
        )
        found_elgato = False
        for line in result.stdout.split('\n'):
            if 'Elgato' in line or 'Wave' in line:
                print(f"   ⭐ {line.strip()}")
                found_elgato = True
            elif 'card' in line.lower():
                print(f"   {line.strip()}")
        
        if not found_elgato and not any('card' in line.lower() for line in result.stdout.split('\n')):
            print("   ⚠️ No ALSA devices found!")
    except FileNotFoundError:
        print("   ❌ arecord not found. Is ALSA installed?")
    except subprocess.CalledProcessError as e:
        print(f"   ⚠️ Error listing ALSA devices: {e}")
    
    # Test recording capability
    print("\n🧪 Testing Recording Capability:")
    test_file = os.path.join(tempfile.gettempdir(), "test_recording.wav")
    try:
        if USE_PULSE:
            source = find_elgato_pulse_source() or "default"
            cmd = [
                "ffmpeg",
                "-f", "pulse",
                "-i", source,
                "-ar", "44100",
                "-ac", "1",
                "-t", "1",
                "-y",
                test_file,
            ]
            print(f"   Testing with: ffmpeg -f pulse -i {source}")
        else:
            cmd = [
                "arecord",
                "-D", ALSA_DEVICE,
                "-d", "1",
                "-f", "S16_LE",
                "-c", "1",
                "-r", "44100",
                test_file,
            ]
            print(f"   Testing with: arecord -D {ALSA_DEVICE}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=3,
            check=False
        )
        
        if result.returncode == 0 and os.path.exists(test_file):
            size = os.path.getsize(test_file)
            if size > 1000:
                print(f"   ✅ Test recording successful ({size} bytes)")
                os.remove(test_file)
            else:
                print(f"   ⚠️ Test recording too small ({size} bytes) - may indicate no audio input")
                if os.path.exists(test_file):
                    os.remove(test_file)
        else:
            print(f"   ❌ Test recording failed")
            if result.stderr:
                print(f"   Error: {result.stderr.decode()[:200]}")
    except subprocess.TimeoutExpired:
        print("   ⚠️ Test recording timed out")
    except Exception as e:
        print(f"   ⚠️ Test recording error: {e}")
    finally:
        if os.path.exists(test_file):
            try:
                os.remove(test_file)
            except:
                pass
    
    print("\n" + "="*60)
    print("💡 TROUBLESHOOTING TIPS:")
    print("="*60)
    print("1. Check mic is not muted in system settings")
    print("2. Verify mic volume: pactl set-source-volume <source> 150%")
    print("3. Unmute if needed: pactl set-source-mute <source> 0")
    print("4. Check if another app is using the mic")
    if USE_PULSE:
        print("5. Try switching to ALSA: python jarvis_voice.py --alsa")
    else:
        print("5. Try using PulseAudio: python jarvis_voice.py")
    print("6. Restart audio: systemctl --user restart pipewire pipewire-pulse")
    print("="*60 + "\n")


def find_elgato_pulse_source():
    """Find Elgato microphone in PulseAudio/PipeWire sources."""
    try:
        result = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True, text=True, check=True
        )
        for line in result.stdout.split('\n'):
            if 'Elgato' in line or 'Wave' in line:
                # Extract source name (second field)
                parts = line.split('\t')
                if len(parts) >= 2:
                    source_name = parts[1]
                    # Filter for input sources (not monitors)
                    if 'output' not in source_name.lower() and 'monitor' not in source_name.lower():
                        return source_name
    except:
        pass
    return None


def record_once(device_name=None, temp_file=None):
    """Record DURATION_SEC seconds into temp_file (or TEMP_WAV if None)."""
    if temp_file is None:
        temp_file = TEMP_WAV
    
    actual_source = None
    if USE_PULSE:
        source = device_name or PULSE_SOURCE
        if not source:
            source = find_elgato_pulse_source()
        if not source:
            source = "default"
        actual_source = source
        
        # Try ffmpeg first (most reliable)
        try:
            cmd = [
                "ffmpeg",
                "-f", "pulse",
                "-i", source,
                "-ar", str(RECORD_RATE),
                "-ac", "1",
                "-t", str(DURATION_SEC),
                "-y",
                temp_file,
            ]
            print(f"🎧 Recording {DURATION_SEC} seconds from {source} (ffmpeg)...")
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=DURATION_SEC + 3
            )
            return actual_source
        except subprocess.TimeoutExpired:
            print("⚠️ ffmpeg timed out, trying next method...")
        except FileNotFoundError:
            print("⚠️ ffmpeg not found, trying next method...")
        except subprocess.CalledProcessError as e:
            stderr_msg = e.stderr.decode() if e.stderr else str(e)
            if "No such file or directory" in stderr_msg or "does not exist" in stderr_msg:
                print(f"⚠️ Source '{source}' not found, trying default...")
                if source != "default":
                    source = "default"
                    actual_source = source
                    # Retry with default
                    try:
                        cmd = [
                            "ffmpeg",
                            "-f", "pulse",
                            "-i", "default",
                            "-ar", str(RECORD_RATE),
                            "-ac", "1",
                            "-t", str(DURATION_SEC),
                            "-y",
                            temp_file,
                        ]
                        subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, timeout=DURATION_SEC + 3)
                        return actual_source
                    except:
                        pass
            print(f"⚠️ ffmpeg failed: {stderr_msg[:100]}, trying next method...")
        
        # Fallback: try parecord (PulseAudio native)
        if source and source != "default":
            try:
                cmd = [
                    "parecord",
                    f"--device={source}",
                    f"--file-format=wav",
                    f"--rate={RECORD_RATE}",
                    f"--channels=1",
                    temp_file,
                ]
                print(f"🎧 Recording {DURATION_SEC} seconds from {source} (parecord)...")
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                time.sleep(DURATION_SEC)
                process.terminate()
                process.wait(timeout=1)
                if os.path.exists(temp_file) and os.path.getsize(temp_file) > 1000:
                    return actual_source
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
                print(f"⚠️ parecord failed: {e}, trying next method...")
            except Exception as e:
                print(f"⚠️ parecord error: {e}, trying next method...")
        
        # Last resort: arecord with pulse
        try:
            cmd = [
                "arecord",
                "-q",
                "-D", "pulse",
                "-d", str(DURATION_SEC),
                "-f", "S16_LE",
                "-c", "1",
                "-r", str(RECORD_RATE),
                temp_file,
            ]
            print(f"🎧 Recording {DURATION_SEC} seconds via PulseAudio from {source} (arecord)...")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return actual_source
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"All recording methods failed. Last error: {e.stderr.decode() if e.stderr else str(e)}\n"
                f"Run with --diagnose flag for troubleshooting."
            )
    else:
        # Direct ALSA access
        device = device_name or ALSA_DEVICE
        cmd = [
            "arecord",
            "-q",
            "-D", device,
            "-d", str(DURATION_SEC),
            "-f", "S16_LE",
            "-c", "1",
            "-r", str(RECORD_RATE),
            temp_file,
        ]
        print(f"🎧 Recording {DURATION_SEC} seconds from {device} (ALSA)...")
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            stderr_msg = result.stderr.decode() if result.stderr else "Unknown error"
            raise RuntimeError(f"Recording failed: {stderr_msg}")
        return device


def load_audio(temp_file=None):
    """Load temp_file (or TEMP_WAV) as float32 mono and resample to SAMPLE_RATE."""
    if temp_file is None:
        temp_file = TEMP_WAV
    audio, sr = sf.read(temp_file, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    
    # Resample if needed
    if sr != SAMPLE_RATE:
        print(f"🔄 Resampling from {sr} Hz to {SAMPLE_RATE} Hz...")
        num_samples = int(len(audio) * SAMPLE_RATE / sr)
        audio = signal.resample(audio, num_samples)
    
    return audio


def transcribe(audio: np.ndarray) -> str:
    """Transcribe audio using Whisper (lazy-loaded)."""
    model = get_whisper()
    segments, _ = model.transcribe(audio, beam_size=1, language="en")
    text = "".join(seg.text for seg in segments).strip()
    return text


def find_elgato_device():
    """Try to find Elgato Wave device automatically."""
    try:
        result = subprocess.run(["arecord", "-l"], capture_output=True, text=True, check=True)
        for line in result.stdout.split('\n'):
            if 'Elgato' in line or 'Wave' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'card':
                        card_num = parts[i+1].rstrip(':')
                        return f"hw:{card_num},0"
    except:
        pass
    return None


def check_audio_level(audio: np.ndarray) -> float:
    """Check the RMS level of the audio to see if anything was recorded."""
    if audio.size == 0:
        return 0.0
    rms = np.sqrt(np.mean(audio**2))
    return rms


def play_back(temp_file=None):
    """Play back the last recorded audio file."""
    if temp_file is None:
        temp_file = TEMP_WAV
    if not os.path.exists(temp_file):
        print("⚠️ No recording file found.")
        return
    
    file_size = os.path.getsize(temp_file)
    if file_size < 1000:
        print(f"⚠️ Recording file too small ({file_size} bytes) - likely no audio captured.")
        return
    
    print("🔊 Playing back last recording...")
    try:
        subprocess.run(
            ["aplay", temp_file],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            subprocess.run(
                ["paplay", temp_file],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Could not play audio (aplay/paplay not available)")


def _check_piper_cmd():
    """Check if command-line piper is available."""
    global PIPER_CMD_AVAILABLE
    if PIPER_CMD_AVAILABLE is None:
        try:
            result = subprocess.run(["piper", "--version"], capture_output=True, check=False, timeout=2)
            PIPER_CMD_AVAILABLE = (result.returncode == 0)
        except:
            PIPER_CMD_AVAILABLE = False
    return PIPER_CMD_AVAILABLE


def get_piper_voice():
    """Lazy-load Piper TTS voice (singleton)."""
    global _piper_voice, _piper_model_path, _piper_config_path
    
    if not PIPER_AVAILABLE and not _check_piper_cmd():
        return None
    
    if _piper_voice is None and PIPER_AVAILABLE:
        try:
            # Try to find or download a voice model
            # Default to en_US-lessac-medium (good quality, English)
            voice_name = "en_US-lessac-medium"
            try:
                voice_path, config_path = find_voice(voice_name)
            except:
                # Try to download it
                try:
                    voice_path, config_path = ensure_voice_exists(voice_name, [voice_name])
                except Exception as e:
                    print(f"⚠️ Could not load Piper voice '{voice_name}': {e}")
                    print("   Install with: pip install piper-tts")
                    print("   Or use command-line piper if installed")
                    return None
            
            _piper_model_path = voice_path
            _piper_config_path = config_path
            _piper_voice = PiperVoice.load(_piper_model_path, config_path=_piper_config_path)
        except Exception as e:
            print(f"⚠️ Failed to initialize Piper TTS: {e}")
            return None
    
    return _piper_voice


def generate_audio(text: str, output_file: str = None) -> str:
    """Generate audio from text using Piper TTS.
    
    Args:
        text: Text to synthesize
        output_file: Optional output file path (auto-generated if None)
    
    Returns:
        Path to generated audio file, or None if failed
    """
    if not text or not text.strip():
        return None
    
    if output_file is None:
        output_file = os.path.join(tempfile.gettempdir(), f"jarvis_tts_{os.getpid()}.wav")
    
    try:
        if PIPER_AVAILABLE:
            # Use Python API
            voice = get_piper_voice()
            if voice is None:
                return None
            
            # Generate audio
            with open(output_file, 'wb') as f:
                voice.synthesize(text, f)
            
            return output_file if os.path.exists(output_file) and os.path.getsize(output_file) > 0 else None
        
        elif _check_piper_cmd():
            # Use command-line piper
            cmd = ["piper", "--model", "en_US-lessac-medium", "--output_file", output_file]
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate(input=text.encode('utf-8'))
            
            if process.returncode == 0 and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                return output_file
            else:
                # Try with default model location
                cmd = ["piper", "--output_file", output_file]
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout, stderr = process.communicate(input=text.encode('utf-8'))
                if process.returncode == 0 and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    return output_file
                return None
        else:
            return None
            
    except Exception as e:
        print(f"⚠️ TTS generation error: {e}")
        return None


def play_audio(audio_file: str):
    """Play audio file using available audio player.
    
    Args:
        audio_file: Path to audio file to play
    """
    if not audio_file or not os.path.exists(audio_file):
        return False
    
    try:
        # Try aplay first (ALSA)
        subprocess.run(
            ["aplay", audio_file],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # Fallback to paplay (PulseAudio)
            subprocess.run(
                ["paplay", audio_file],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                # Fallback to ffplay (ffmpeg)
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_file],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("⚠️ Could not play audio (aplay/paplay/ffplay not available)")
                return False


def speak(text: str):
    """Generate audio from text and play it.
    
    Args:
        text: Text to speak
    """
    if not text or not text.strip():
        return
    
    print(f"🔊 Speaking: {text[:50]}{'...' if len(text) > 50 else ''}")
    
    # Generate audio
    audio_file = generate_audio(text)
    
    if audio_file:
        # Play audio
        play_audio(audio_file)
        
        # Clean up temp file after a short delay
        def cleanup():
            time.sleep(1)  # Wait a bit before cleanup
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except:
                pass
        
        import threading
        threading.Thread(target=cleanup, daemon=True).start()
    else:
        print("⚠️ TTS not available. Install with: pip install piper-tts")
        print(f"   (Text: {text})")


def handle_command(text: str):
    t = text.lower().strip()
    if not t:
        print("🤷 Heard nothing useful.")
        return
    print("🧠 Handling command:", t)
    # TODO later: map "open firefox", etc. to subprocess calls
    
    # Example: speak a response
    # You can customize this based on the command
    if "time" in t:
        import datetime
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        speak(f"The current time is {current_time}")
    elif "date" in t:
        import datetime
        current_date = datetime.datetime.now().strftime("%B %d, %Y")
        speak(f"Today's date is {current_date}")
    # Add more command handlers as needed


def process_recording_and_transcribe(consecutive_failures_ref):
    """Process a recording: check file, load audio, transcribe, and handle command.
    Returns True if successful, False otherwise. Modifies consecutive_failures_ref[0].
    """
    # Check file size first
    file_size = os.path.getsize(TEMP_WAV) if os.path.exists(TEMP_WAV) else 0
    if file_size < 1000:
        consecutive_failures_ref[0] += 1
        print(f"⚠️ Recording file too small ({file_size} bytes) - no audio captured.")
        
        if consecutive_failures_ref[0] >= 3:
            print("⚠️ Multiple failures detected, running diagnostics...")
            run_diagnostics()
            consecutive_failures_ref[0] = 0
        else:
            print("   Troubleshooting:")
            print("   1. Make sure you spoke into the mic during recording")
            print("   2. Check mic is not muted in system settings")
            print("   3. Verify mic volume/gain is up")
            print("   4. Run with --diagnose for detailed info")
        return False

    audio = load_audio()
    if audio.size == 0:
        consecutive_failures_ref[0] += 1
        print("⚠️ Got empty audio, try again.")
        
        if consecutive_failures_ref[0] >= 3:
            print("⚠️ Multiple failures detected, running diagnostics...")
            run_diagnostics()
            consecutive_failures_ref[0] = 0
        return False

    # Check audio level
    rms_level = check_audio_level(audio)
    if rms_level < 0.001:
        print(f"⚠️ Very low audio level (RMS: {rms_level:.6f}). Check mic volume/gain.")
        print("   You can press 'p' to playback and verify the recording.")
    else:
        print(f"✅ Audio level OK (RMS: {rms_level:.4f})")
        consecutive_failures_ref[0] = 0  # Reset on successful audio capture

    print("🧠 Transcribing...")
    try:
        text = transcribe(audio)
        print("🗣️ You said:", text if text else "(no speech detected)")
        handle_command(text)
        return True
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return False


class HotwordDetector:
    """Hotword detection using Porcupine or Whisper."""
    
    def __init__(self, method="porcupine", keywords=["jarvis"], access_key=None):
        self.method = method
        self.keywords = keywords if isinstance(keywords, list) else [keywords]
        self.access_key = access_key
        self.porcupine = None
        self.stream = None
        self.pa = None
        self._setup()
    
    def _setup(self):
        """Initialize the hotword detector."""
        if self.method == "porcupine":
            if not PORCUPINE_AVAILABLE:
                raise ImportError("Porcupine not available. Install with: pip install pvporcupine pyaudio")
            
            # Create Porcupine instance
            if self.access_key:
                self.porcupine = pvporcupine.create(access_key=self.access_key, keywords=self.keywords)
            else:
                self.porcupine = pvporcupine.create(keywords=self.keywords)
            
            # Initialize PyAudio
            self.pa = pyaudio.PyAudio()
            
            # Try to find the right audio device
            device_index = None
            if USE_PULSE:
                try:
                    device_index = self.pa.get_default_input_device_info()['index']
                except:
                    pass
            else:
                try:
                    for i in range(self.pa.get_device_count()):
                        info = self.pa.get_device_info_by_index(i)
                        if info['maxInputChannels'] > 0:
                            device_index = i
                            break
                except:
                    pass
            
            # Open audio stream
            self.stream = self.pa.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length,
                input_device_index=device_index
            )
    
    def wait_for_hotword(self):
        """Wait until hotword is detected. Blocks until detected."""
        if self.method == "porcupine":
            while True:
                pcm = self.stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                pcm_array = np.frombuffer(pcm, dtype=np.int16)
                keyword_index = self.porcupine.process(pcm_array)
                if keyword_index >= 0:
                    return True
        elif self.method == "whisper":
            # Use Whisper for continuous hotword detection
            # Record short chunks and check for keyword
            hotword = self.keywords[0].lower()
            print(f"🎤 Listening for '{hotword}' using Whisper...")
            
            while True:
                try:
                    # Record a short snippet (2 seconds for hotword check)
                    short_temp = os.path.join(tempfile.gettempdir(), f"hotword_check_{os.getpid()}.wav")
                    
                    # Record with shorter duration for hotword checking
                    if USE_PULSE:
                        source = PULSE_SOURCE or find_elgato_pulse_source() or "default"
                        cmd = [
                            "arecord",
                            "-q",
                            "-D", "pulse",
                            "-d", "2",  # 2 seconds for hotword check
                            "-f", "S16_LE",
                            "-c", "1",
                            "-r", str(RECORD_RATE),
                            short_temp,
                        ]
                    else:
                        cmd = [
                            "arecord",
                            "-q",
                            "-D", ALSA_DEVICE,
                            "-d", "2",  # 2 seconds for hotword check
                            "-f", "S16_LE",
                            "-c", "1",
                            "-r", str(RECORD_RATE),
                            short_temp,
                        ]
                    
                    subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
                    
                    # Quick transcription check
                    try:
                        if os.path.exists(short_temp) and os.path.getsize(short_temp) > 1000:
                            audio = load_audio(short_temp)
                            if audio.size > 0:
                                model = get_whisper()
                                segments, _ = model.transcribe(audio, beam_size=1, language="en", vad_filter=True)
                                text = "".join(seg.text for seg in segments).strip().lower()
                                
                                # Clean up temp file
                                if os.path.exists(short_temp):
                                    os.remove(short_temp)
                                
                                # Check if hotword is in transcription
                                if hotword in text:
                                    return True
                    except Exception as e:
                        pass  # Silent failure for hotword check
                    
                    # Clean up if file still exists
                    if os.path.exists(short_temp):
                        try:
                            os.remove(short_temp)
                        except:
                            pass
                    
                    time.sleep(0.5)  # Brief pause before next check
                        
                except KeyboardInterrupt:
                    raise
                except:
                    time.sleep(0.5)  # Brief pause before next check
        else:
            raise ValueError(f"Unknown hotword method: {self.method}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.method == "porcupine":
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.pa:
                self.pa.terminate()
            if self.porcupine:
                self.porcupine.delete()


def continuous_loop(hotword_method="porcupine", keywords=["jarvis"], access_key=None):
    """Main continuous loop: Wake → Record → Transcribe → Action → Idle
    
    Args:
        hotword_method: "porcupine" or "whisper"
        keywords: List of hotword keywords
        access_key: Porcupine access key (if using Porcupine)
    """
    detector = None
    consecutive_failures = [0]
    
    try:
        # Initialize hotword detector
        print(f"🎤 Initializing {hotword_method} hotword detection...")
        detector = HotwordDetector(method=hotword_method, keywords=keywords, access_key=access_key)
        keyword_display = keywords[0] if keywords else "jarvis"
        print(f"🎤 Listening for hotword: '{keyword_display}'...")
        print("   Press Ctrl+C to stop")
        
        while True:
            try:
                # Wait for hotword (Wake)
                detector.wait_for_hotword()
                print("\n🔵 Hotword detected!")
                
                # Record (Record)
                print("💬 Recording...")
                try:
                    record_once()
                    consecutive_failures[0] = 0  # Reset on success
                except (subprocess.CalledProcessError, RuntimeError, subprocess.TimeoutExpired) as e:
                    consecutive_failures[0] += 1
                    print(f"❌ Recording failed: {e}")
                    if consecutive_failures[0] >= 3:
                        print("⚠️ Multiple failures detected, running diagnostics...")
                        run_diagnostics()
                        consecutive_failures[0] = 0
                    print("\n🎤 Listening for hotword again...")
                    continue
                except Exception as e:
                    consecutive_failures[0] += 1
                    print(f"❌ Unexpected error: {e}")
                    if consecutive_failures[0] >= 3:
                        print("⚠️ Multiple failures detected, running diagnostics...")
                        run_diagnostics()
                        consecutive_failures[0] = 0
                    print("\n🎤 Listening for hotword again...")
                    continue
                
                # Transcribe (Transcribe)
                audio = load_audio()
                if audio.size == 0:
                    print("⚠️ Got empty audio, returning to idle...")
                    continue
                
                rms_level = check_audio_level(audio)
                if rms_level < 0.001:
                    print(f"⚠️ Very low audio level (RMS: {rms_level:.6f}). Returning to idle...")
                    continue
                
                print("🧠 Transcribing...")
                try:
                    text = transcribe(audio)
                    print("🗣️ You said:", text if text else "(no speech detected)")
                except Exception as e:
                    print(f"❌ Transcription error: {e}")
                    continue
                
                # Handle command (Action)
                if text and text.strip():
                    handle_command(text)
                else:
                    print("🤷 No speech detected, returning to idle...")
                
                # Back to idle (Idle)
                print("\n🎤 Listening for hotword again...")
                
            except KeyboardInterrupt:
                print("\n👋 Stopping...")
                break
            except Exception as e:
                print(f"❌ Error in loop: {e}")
                consecutive_failures[0] += 1
                if consecutive_failures[0] >= 3:
                    print("⚠️ Multiple failures detected, running diagnostics...")
                    run_diagnostics()
                    consecutive_failures[0] = 0
                time.sleep(1)  # Brief pause before retrying
                
    finally:
        if detector:
            detector.cleanup()


def cleanup_temp_file():
    """Clean up temp file on exit."""
    if os.path.exists(TEMP_WAV):
        try:
            os.remove(TEMP_WAV)
        except:
            pass


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Voice transcription with Elgato microphone support"
    )
    parser.add_argument(
        "--alsa", 
        action="store_true",
        help="Use ALSA instead of PulseAudio/PipeWire"
    )
    parser.add_argument(
        "--diagnose", "-d",
        action="store_true",
        help="Run audio diagnostics and exit"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Specific PulseAudio source name (overrides auto-detection)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Specific ALSA device (e.g., hw:5,0)"
    )
    parser.add_argument(
        "--hotword",
        action="store_true",
        help="Enable hotword detection mode (requires pvporcupine and pyaudio)"
    )
    parser.add_argument(
        "--keyword",
        type=str,
        default="jarvis",
        help="Hotword keyword to listen for (default: 'jarvis')"
    )
    parser.add_argument(
        "--porcupine-key",
        type=str,
        default=None,
        help="Porcupine access key (optional, uses default if available)"
    )
    parser.add_argument(
        "--hotword-method",
        type=str,
        choices=["porcupine", "whisper"],
        default="porcupine",
        help="Hotword detection method: 'porcupine' (fast, requires library) or 'whisper' (slower, uses Whisper)"
    )
    
    args = parser.parse_args()
    
    # Set global config
    USE_PULSE = not args.alsa
    if args.source:
        PULSE_SOURCE = args.source
    if args.device:
        ALSA_DEVICE = args.device
    
    # Cleanup on exit
    import atexit
    atexit.register(cleanup_temp_file)
    
    # Check for diagnostic mode
    if args.diagnose:
        run_diagnostics()
        sys.exit(0)
    
    # Check for hotword mode
    if args.hotword:
        # Check method requirements
        if args.hotword_method == "porcupine" and not PORCUPINE_AVAILABLE:
            print("❌ Porcupine method requires pvporcupine and pyaudio")
            print("   Install with: pip install pvporcupine pyaudio")
            print("   Or use Whisper method: --hotword --hotword-method whisper")
            sys.exit(1)
        
        # Detect available devices first
        if USE_PULSE:
            elgato_pulse = find_elgato_pulse_source()
            if elgato_pulse and PULSE_SOURCE is None:
                PULSE_SOURCE = elgato_pulse
            print(f"📌 Audio source: {PULSE_SOURCE or elgato_pulse or 'default'}")
        else:
            device = args.device or ALSA_DEVICE
            print(f"📌 Using ALSA device: {device}")
        
        # Start continuous loop
        continuous_loop(
            hotword_method=args.hotword_method,
            keywords=[args.keyword],
            access_key=args.porcupine_key
        )
        sys.exit(0)
    
    # Detect available devices
    actual_source = None
    if USE_PULSE:
        elgato_pulse = find_elgato_pulse_source()
        if elgato_pulse:
            if PULSE_SOURCE is None:
                PULSE_SOURCE = elgato_pulse
            print(f"🔍 Found Elgato via PulseAudio/PipeWire: {elgato_pulse}")
        else:
            print("🔍 Using default PulseAudio/PipeWire source")
            print("   💡 Tip: Run with --diagnose to see all available sources")
        
        actual_source = PULSE_SOURCE or elgato_pulse or "default"
        print("🎙️ Press Enter, then speak into the selected mic...")
        print(f"📌 Using source: {actual_source}")
    else:
        device = args.device or ALSA_DEVICE
        elgato = find_elgato_device()
        if elgato and device != elgato:
            print(f"🔍 Found Elgato device: {elgato}")
            print(f"   (Using configured device: {device})")
            print(f"   (To use Elgato, run with: --device {elgato})")
        print("🎙️ Press Enter, then speak into the selected mic...")
        print(f"📌 Using ALSA device: {device}")

    consecutive_failures_list = [0]  # Use list for mutable reference
    while True:
        try:
            user_input = input(
                "\n🔵 Press Enter to record, 'p' to playback, 'd' for diagnostics, or Ctrl+C to quit... "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Exiting.")
            break

        if user_input == 'p':
            play_back()
            continue
        elif user_input == 'd':
            run_diagnostics()
            continue

        print("💬 Speak now into your microphone...")
        try:
            used_source = record_once()
            consecutive_failures_list[0] = 0  # Reset on success
        except (subprocess.CalledProcessError, RuntimeError, subprocess.TimeoutExpired) as e:
            consecutive_failures_list[0] += 1
            print(f"❌ Recording failed: {e}")
            
            if consecutive_failures_list[0] >= 3:
                print("⚠️ Multiple failures detected, running diagnostics...")
                run_diagnostics()
                consecutive_failures_list[0] = 0
            else:
                print("   💡 Try running with --diagnose flag to troubleshoot")
            continue
        except Exception as e:
            consecutive_failures_list[0] += 1
            print(f"❌ Unexpected error during recording: {e}")
            
            if consecutive_failures_list[0] >= 3:
                print("⚠️ Multiple failures detected, running diagnostics...")
                run_diagnostics()
                consecutive_failures_list[0] = 0
            else:
                print("   💡 Try running with --diagnose flag to troubleshoot")
            continue

        # Process recording and transcribe
        process_recording_and_transcribe(consecutive_failures_list)
