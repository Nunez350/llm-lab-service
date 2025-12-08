#!/usr/bin/env python3
"""Quick test to check if Elgato mic is receiving audio input"""
import subprocess
import os
import time

SOURCE = "alsa_input.usb-Elgato_Systems_Elgato_Wave_3_A011A52411B55R-00.mono-fallback"
TEST_FILE = "/tmp/elgato_input_test.wav"

print("🎤 Testing Elgato Wave 3 microphone input...")
print(f"   Source: {SOURCE}")
print("\n📊 Current settings:")
subprocess.run(["pactl", "get-source-volume", SOURCE])
print("\n🎙️ Recording 3 seconds - please speak into the microphone now...")
print("   (Countdown starting in 2 seconds...)")
time.sleep(2)

for i in range(3, 0, -1):
    print(f"   {i}...")
    time.sleep(1)

print("\n🔴 Recording NOW - speak clearly...")

try:
    process = subprocess.Popen(
        ["parecord", 
         f"--device={SOURCE}",
         "--file-format=wav",
         "--rate=48000",
         "--channels=1",
         TEST_FILE],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(3)
    process.terminate()
    process.wait(timeout=1)
except Exception as e:
    print(f"Error: {e}")

if os.path.exists(TEST_FILE):
    size = os.path.getsize(TEST_FILE)
    print(f"\n📁 Recording file created: {size} bytes")
    
    if size > 1000:
        # Check audio level
        try:
            import soundfile as sf
            import numpy as np
            audio, sr = sf.read(TEST_FILE, dtype="float32")
            if audio.ndim > 1:
                audio = audio[:, 0]
            rms = np.sqrt(np.mean(audio**2))
            max_level = np.max(np.abs(audio))
            print(f"✅ Audio detected!")
            print(f"   RMS level: {rms:.6f}")
            print(f"   Peak level: {max_level:.6f}")
            if rms < 0.001:
                print("   ⚠️ Very quiet - check hardware gain knob on mic")
            elif rms < 0.01:
                print("   ⚠️ Low level - you may need to speak louder or increase gain")
            else:
                print("   ✅ Good audio level detected!")
        except Exception as e:
            print(f"   ⚠️ Could not analyze audio: {e}")
    else:
        print("❌ No audio data captured (only header)")
        print("\n💡 Troubleshooting:")
        print("   1. Check the gain knob on your Elgato Wave 3")
        print("   2. Verify you spoke during the recording")
        print("   3. Check if mic is working in other apps (e.g., pavucontrol)")
        print("   4. Try: alsamixer (press F4 for capture, find card 5)")
else:
    print("❌ Recording file was not created")
