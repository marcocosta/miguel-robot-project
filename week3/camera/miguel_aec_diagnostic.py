"""Explicit, privacy-preserving AEC/self-speech field diagnostic.

Run one condition at a time. Audio is reduced to window RMS statistics unless
``--save-wav`` is explicitly supplied.
"""

import argparse
import audioop
import json
import math
import statistics
import subprocess
import time
import wave
from pathlib import Path

from miguel_respeaker_xvf3800 import XVF3800Monitor


def run(condition: str, device: str, seconds: float, save_wav: str = "") -> dict:
    command = ["arecord", "-q", "-D", device, "-f", "S16_LE", "-r", "16000", "-c", "2", "-t", "raw"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    monitor = XVF3800Monitor()
    rms_values, vad_values, doa_values, raw_chunks = [], [], [], []
    started = time.monotonic()
    try:
        while time.monotonic() - started < seconds:
            chunk = process.stdout.read(6400)
            if not chunk:
                break
            rms_values.append(audioop.rms(chunk, 2))
            if save_wav:
                raw_chunks.append(chunk)
            if monitor.available():
                try:
                    doa, vad = monitor.read_doa_vad()
                    doa_values.append(doa)
                    vad_values.append(bool(vad))
                except Exception as exc:
                    print(f"[AEC_DIAGNOSTIC] sensor_error={type(exc).__name__}: {exc}")
    finally:
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
        monitor.close()
    if save_wav and raw_chunks:
        path = Path(save_wav).expanduser()
        with wave.open(str(path), "wb") as output:
            output.setnchannels(2)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b"".join(raw_chunks))
    sorted_rms = sorted(rms_values)
    p95 = sorted_rms[min(len(sorted_rms) - 1, math.floor(len(sorted_rms) * 0.95))] if sorted_rms else 0
    return {
        "condition": condition,
        "duration_seconds": round(time.monotonic() - started, 3),
        "windows": len(rms_values),
        "rms_mean": round(statistics.fmean(rms_values), 2) if rms_values else 0,
        "rms_peak": max(rms_values, default=0),
        "rms_p95": p95,
        "xvf_available": bool(vad_values or doa_values),
        "vad_active_ratio": round(sum(vad_values) / len(vad_values), 3) if vad_values else None,
        "doa_samples": doa_values,
        "raw_audio_saved": bool(save_wav),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual XVF3800 AEC/self-TTS diagnostic")
    parser.add_argument("condition", choices=("A", "B", "C"), help="A=TTS only, B=human only, C=human+TTS")
    parser.add_argument("--device", default="pulse")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--save-wav", default="", help="Explicitly save raw diagnostic audio")
    args = parser.parse_args()
    print("[AEC_DIAGNOSTIC] Start the requested manual condition now.")
    print(json.dumps(run(args.condition, args.device, args.seconds, args.save_wav), sort_keys=True))


if __name__ == "__main__":
    main()
