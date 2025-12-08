# jarvis/tts_piper.py

import subprocess
import wave
import shutil
from pathlib import Path

try:
    from piper import PiperVoice
    try:
        from piper.download import ensure_voice_exists, find_voice
        PIPER_DOWNLOAD_AVAILABLE = True
    except ImportError:
        PIPER_DOWNLOAD_AVAILABLE = False
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    PIPER_DOWNLOAD_AVAILABLE = False

from .config import PIPER_MODEL_PATH, PIPER_CONFIG_PATH, TTS_WAV_PATH, DEBUG


def find_system_tts():
    """Find available system TTS command."""
    for cmd in ["espeak-ng", "espeak", "festival", "say"]:
        if shutil.which(cmd):
            return cmd
    
    # Check for piper-tts CLI as fallback
    for cmd in ["piper-tts", "piper_tts"]:
        if shutil.which(cmd):
            return ("piper-cli", cmd)
    
    # Check if piper can be run as module
    try:
        import subprocess
        result = subprocess.run(["python3", "-m", "piper_tts", "--help"], 
                              capture_output=True, timeout=2)
        if result.returncode == 0:
            return ("piper-module", "python3 -m piper_tts")
    except:
        pass
    
    return None


class TTSEngine:
    def __init__(self):
        self.voice = None
        self.system_tts_cmd = find_system_tts()
        
        # Try to initialize Piper if available
        if PIPER_AVAILABLE:
            try:
                model_path = Path(PIPER_MODEL_PATH)
                
                # Try to find voice if model doesn't exist at exact path
                if not model_path.exists() or not model_path.is_file():
                    if DEBUG:
                        print(f"⚠️ Model not found at {model_path}, attempting to find/download...")
                    
                    # Try to find or download default voice (only if download module is available)
                    if PIPER_DOWNLOAD_AVAILABLE:
                        try:
                            voice_name = "en_US-lessac-medium"
                            try:
                                voice_path, config_path = find_voice(voice_name)
                                model_path = Path(voice_path)
                            except:
                                if DEBUG:
                                    print(f"   Attempting to download voice: {voice_name}")
                                voice_path, config_path = ensure_voice_exists(voice_name, [voice_name])
                                model_path = Path(voice_path)
                        except Exception as e:
                            if DEBUG:
                                print(f"   Could not auto-download voice: {e}")
                            raise FileNotFoundError(
                                f"Piper model not found at {PIPER_MODEL_PATH} and could not auto-download."
                            )
                    else:
                        raise FileNotFoundError(
                            f"Piper model not found at {PIPER_MODEL_PATH}."
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
                print("✅ Piper TTS ready")
            except Exception as e:
                if DEBUG:
                    print(f"⚠️ Failed to initialize Piper: {e}")
                self.voice = None
                if self.system_tts_cmd:
                    print(f"🗣️ Falling back to system TTS: {self.system_tts_cmd}")
        
        # If neither Piper nor system TTS available
        if not self.voice and not self.system_tts_cmd:
            print("⚠️ No TTS available. Install Piper or system TTS (espeak-ng)")
            print("   Install: pip install piper-tts")
            print("   Or: sudo apt install espeak-ng")

    def speak(self, text: str):
        text = text.strip()
        if not text:
            return
        
        # Try Piper first if available
        if self.voice is not None:
            self._speak_piper(text)
            return
        
        # Fallback to system TTS
        if self.system_tts_cmd:
            self._speak_system(text)
            return
        
        # No TTS available - just print
        print(f"🔊 (TTS disabled) Would say: {text}")
    
    def _speak_system(self, text: str):
        """Use system TTS command (espeak, festival, etc.)"""
        if isinstance(self.system_tts_cmd, tuple):
            cmd_type, cmd = self.system_tts_cmd
        else:
            cmd_type, cmd = "standard", self.system_tts_cmd
        
        try:
            if cmd_type == "piper-cli":
                # Use piper-tts CLI: piper-tts --model <path> --output_file <wav> --text "text"
                import tempfile
                temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_wav.close()
                try:
                    # Try to find a default model or use auto-download
                    subprocess.run(
                        [cmd, "--model", "en_US-lessac-medium", "--output_file", temp_wav.name, "--text", text],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=10,
                    )
                    # Play the generated WAV
                    self._play_wav(temp_wav.name)
                finally:
                    if Path(temp_wav.name).exists():
                        Path(temp_wav.name).unlink()
                        
            elif cmd_type == "piper-module":
                # Use piper-tts as Python module
                import tempfile
                temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_wav.close()
                try:
                    subprocess.run(
                        cmd.split() + ["--model", "en_US-lessac-medium", "--output_file", temp_wav.name, "--text", text],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=10,
                    )
                    self._play_wav(temp_wav.name)
                finally:
                    if Path(temp_wav.name).exists():
                        Path(temp_wav.name).unlink()
                        
            elif cmd in ["espeak", "espeak-ng"]:
                subprocess.run(
                    [cmd, "-s", "150", "-v", "en", text],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif cmd == "festival":
                subprocess.run(
                    ["festival", "--tts"],
                    input=text.encode(),
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif cmd == "say":
                subprocess.run(
                    ["say", text],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception as e:
            if DEBUG:
                print(f"⚠️ System TTS error: {e}")
            print(f"🔊 (TTS failed) Would say: {text}")
    
    def _play_wav(self, wav_path: str):
        """Play a WAV file using available audio players"""
        for player in ["paplay", "aplay", "ffplay"]:
            try:
                if player == "ffplay":
                    subprocess.run(
                        [player, "-nodisp", "-autoexit", "-loglevel", "quiet", wav_path],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    subprocess.run(
                        [player, wav_path],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                return
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
        if DEBUG:
            print("⚠️ Could not play audio file")
    
    def _speak_piper(self, text: str):
        """Use Piper TTS"""
        # synthesize to WAV
        with wave.open(TTS_WAV_PATH, "wb") as wav_file:
            self.voice.synthesize(
                text=text,
                wav_file=wav_file,
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
