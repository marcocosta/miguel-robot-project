import json
import subprocess
import time
import numpy as np
from pathlib import Path
from vosk import Model, KaldiRecognizer

MIC_DEVICE = "hw:CARD=Array,DEV=0"
AUDIO_RATE = 16000
AUDIO_CHANNELS = 2
CHUNK_MS = 250
CHUNK_BYTES = int(AUDIO_RATE * AUDIO_CHANNELS * 2 * (CHUNK_MS / 1000.0))

SPEECH_RMS_THRESHOLD = 250

MODEL_PATH = Path.home() / "robot-project/week3/models/vosk-model-small-en-us-0.15"
model = Model(str(MODEL_PATH))

def stereo_raw_to_mono_bytes(raw_bytes: bytes):
    samples = np.frombuffer(raw_bytes, dtype=np.int16)
    if len(samples) < 2:
        return b"", 0.0

    samples = samples[: len(samples) - (len(samples) % 2)]
    stereo = samples.reshape(-1, 2)
    mono = stereo.mean(axis=1).astype(np.int16)

    rms = float(np.sqrt(np.mean(mono.astype(np.float32) ** 2))) if len(mono) else 0.0
    return mono.tobytes(), rms

cmd = [
    "arecord",
    "-q",
    "-D", MIC_DEVICE,
    "-f", "S16_LE",
    "-r", str(AUDIO_RATE),
    "-c", str(AUDIO_CHANNELS),
    "-t", "raw",
]

print("Miguel Hearing Debug")
print("Say: Hey Miguel, what do you see?")
print("Press Ctrl+C to stop.")
print(f"Current speech threshold: {SPEECH_RMS_THRESHOLD}")
print()

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
rec = KaldiRecognizer(model, AUDIO_RATE)

try:
    while True:
        raw = proc.stdout.read(CHUNK_BYTES)
        if not raw:
            continue

        mono_bytes, rms = stereo_raw_to_mono_bytes(raw)
        speech = "SPEECH" if rms > SPEECH_RMS_THRESHOLD else "quiet "

        if rec.AcceptWaveform(mono_bytes):
            result = json.loads(rec.Result())
            text = result.get("text", "").strip()
            if text:
                print(f"RMS={rms:7.1f} {speech} | FINAL: {text}")
            else:
                print(f"RMS={rms:7.1f} {speech}")
        else:
            partial = json.loads(rec.PartialResult()).get("partial", "").strip()
            if partial:
                print(f"RMS={rms:7.1f} {speech} | partial: {partial}")
            else:
                print(f"RMS={rms:7.1f} {speech}")

        time.sleep(0.02)

except KeyboardInterrupt:
    print("\nStopping hearing debug.")

finally:
    proc.terminate()
