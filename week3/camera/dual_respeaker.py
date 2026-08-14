"""Clock-safe selection for two independent ReSpeaker USB arrays."""

from __future__ import annotations

import numpy as np


def strongest_stereo_channel(raw_bytes: bytes) -> tuple[bytes, float]:
    """Return the stronger intact channel and its RMS."""
    samples = np.frombuffer(raw_bytes, dtype=np.int16)
    if len(samples) < 2:
        return b"", 0.0
    samples = samples[: len(samples) - (len(samples) % 2)]
    stereo = samples.reshape(-1, 2)
    channel_rms = np.sqrt(np.mean(stereo.astype(np.float32) ** 2, axis=0))
    index = int(np.argmax(channel_rms))
    mono = stereo[:, index].copy()
    return mono.tobytes(), float(channel_rms[index])


def adaptive_rms_threshold(noise_rms_samples, *, minimum=500.0, maximum=1200.0, multiplier=3.0):
    if not noise_rms_samples:
        return float(minimum)
    noise_floor = float(np.median(np.asarray(noise_rms_samples, dtype=np.float32)))
    return min(float(maximum), max(float(minimum), noise_floor * float(multiplier)))


def raw_stereo_quality(raw_bytes: bytes) -> tuple[float, float]:
    """Return RMS and a clipping-aware quality score for one array chunk."""
    samples = np.frombuffer(raw_bytes, dtype=np.int16)
    if len(samples) < 2:
        return 0.0, 0.0
    samples = samples[: len(samples) - (len(samples) % 2)]
    stereo = samples.reshape(-1, 2).astype(np.float32)
    if not len(stereo):
        return 0.0, 0.0
    channel_rms = np.sqrt(np.mean(stereo ** 2, axis=0))
    channel = stereo[:, int(np.argmax(channel_rms))]
    rms = float(np.sqrt(np.mean(channel ** 2)))
    clipped = float(np.mean(np.abs(channel) >= 32000.0))
    return rms, rms * max(0.05, 1.0 - clipped * 20.0)


class DualMicStream:
    """Read two USB arrays and emit the cleaner chunk.

    Independent USB clocks must not be directly averaged: phase and clock
    drift can cancel speech. Selection retains each XVF3800's beamforming and
    extends coverage to both sides of the robot.
    """

    def __init__(self, streams, *, speech_lock_rms=900.0, switch_ratio=1.6):
        self.streams = list(streams)
        self.stdout = self
        self.selected_chunks = [0 for _ in self.streams]
        self.selected_index = None
        self.speech_locked = False
        self.speech_lock_rms = float(speech_lock_rms)
        self.switch_ratio = float(switch_ratio)

    def read(self, size):
        chunks = [stream.stdout.read(size) for stream in self.streams]
        available = [(index, chunk) for index, chunk in enumerate(chunks) if chunk]
        if not available:
            return b""
        qualities = {index: raw_stereo_quality(chunk)[1] for index, chunk in available}
        best_index, best_chunk = max(available, key=lambda item: qualities[item[0]])
        available_indexes = {index for index, _chunk in available}
        if self.selected_index not in available_indexes:
            self.selected_index = best_index
        elif not self.speech_locked:
            current_quality = qualities[self.selected_index]
            if qualities[best_index] > max(1.0, current_quality) * self.switch_ratio:
                self.selected_index = best_index

        # Once either array sees likely speech, retain that acoustic perspective
        # for the utterance instead of splicing independent USB clocks together.
        if not self.speech_locked and qualities[best_index] >= self.speech_lock_rms:
            self.selected_index = best_index
            self.speech_locked = True

        index = self.selected_index
        chunk = next(chunk for candidate, chunk in available if candidate == index)
        self.selected_chunks[index] += 1
        return chunk

    def terminate(self):
        for stream in self.streams:
            try:
                stream.terminate()
            except Exception:
                pass

    def wait(self, timeout=None):
        for stream in self.streams:
            try:
                stream.wait(timeout=timeout)
            except Exception:
                pass

    def kill(self):
        for stream in self.streams:
            try:
                stream.kill()
            except Exception:
                pass
