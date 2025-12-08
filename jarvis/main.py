# jarvis/main.py

import time
import numpy as np

from .config import RECORD_DURATION_SEC, DEBUG, MAX_MEMORY_TURNS, USE_HOTWORD
from .audio_io import AudioRecorder
from .stt_whisper import STTEngine
from .tts_piper import TTSEngine
from .hotword import create_hotword_detector
from .memory import ConversationMemory
from . import commands


def main():
    print("🛰️ Starting JARVIS assistant...")

    # Init subsystems
    try:
        # Prefer subprocess recording when using hotword detection to avoid ALSA conflicts
        recorder = AudioRecorder(prefer_subprocess=USE_HOTWORD)
        stt = STTEngine()
        tts = TTSEngine()
        hotword = create_hotword_detector()
        memory = ConversationMemory(max_turns=MAX_MEMORY_TURNS)
    except Exception as e:
        print(f"❌ Failed to initialize JARVIS: {e}")
        return 1

    try:
        tts.speak("Jarvis is online.")
    except Exception as e:
        print(f"⚠️ Could not speak startup message: {e}")
        print("   Continuing without TTS...")

    consecutive_failures = 0

    try:
        while True:
            try:
                # 1) Wait for wake signal (hotword or Enter)
                hotword.wait_for_wake()

                # 2) Record
                print("💬 Listening...")
                audio = recorder.record(RECORD_DURATION_SEC)
                if audio.size == 0:
                    consecutive_failures += 1
                    print("⚠️ No audio captured.")
                    if consecutive_failures >= 3:
                        print("⚠️ Multiple recording failures. Check microphone connection.")
                        consecutive_failures = 0
                    try:
                        tts.speak("I did not hear anything. Please repeat.")
                    except:
                        pass
                    continue

                consecutive_failures = 0  # Reset on success

                # 3) Transcribe
                print("🧠 Transcribing...")
                text = stt.transcribe(audio)
                print("🗣️ You said:", text or "<empty>")

                if not text:
                    try:
                        tts.speak("I did not understand. Please try again.")
                    except:
                        pass
                    continue

                # Add user turn to memory
                memory.add_turn("user", text)

                # 4) Command handling
                reply, handler = commands.handle_command(text, memory)
                if DEBUG and handler:
                    print(f"⚙️ Handled by: {handler}")

                print("🤖 Jarvis:", reply)
                
                # Store assistant response in memory
                memory.add_turn("assistant", reply)
                if DEBUG:
                    print("🧠 Memory:", memory.debug_summary())
                
                try:
                    tts.speak(reply)
                except Exception as e:
                    print(f"⚠️ TTS error: {e}")
                    print(f"   Response: {reply}")

            except KeyboardInterrupt:
                print("\n👋 Jarvis shutting down.")
                break
            except SystemExit:
                print("\n👋 Jarvis shutting down.")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                if DEBUG:
                    import traceback
                    traceback.print_exc()
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    print("⚠️ Too many errors. Shutting down.")
                    break
                time.sleep(1)  # Brief pause before retrying
    finally:
        # Cleanup
        try:
            if hasattr(hotword, 'cleanup'):
                hotword.cleanup()
        except:
            pass

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
