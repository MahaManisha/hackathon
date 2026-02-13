
import sounddevice as sd
import numpy as np
import time

def callback(indata, frames, time_info, status):
    if status:
        print(f"Status: {status}")
    volume_norm = np.linalg.norm(indata) * 10
    print(f"RMS: {volume_norm:.4f} | {'TALKING' if volume_norm > 0.02 else '...'}")

print("Listening... Press Ctrl+C to stop.")
try:
    with sd.InputStream(callback=callback):
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    print("\nStopped.")
except Exception as e:
    print(f"\nError: {e}")
