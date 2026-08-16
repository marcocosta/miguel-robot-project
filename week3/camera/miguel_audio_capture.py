"""Conversation-aware audio capture owned by the V7.5 runtime.

The backend supplies the established device, PCM, WAV, and final-ASR helpers.
This module owns turn lifecycle and endpointing; it deliberately never calls
the historical V6 ``capture_user_turn`` function.
"""

from collections import deque
import json
import threading
import time
from typing import Any, Optional


class MiguelAudioCapture:
    def __init__(self, backend: Any, conversation_manager: Any, stop_event=None, logger=print):
        self.backend = backend
        self.manager = conversation_manager
        self.stop_event = stop_event
        self.log = logger
        self.last_timing: dict = {}
        self.cancelled = False

    def _partial_recognizer(self) -> Optional[Any]:
        model = getattr(self.backend, "vosk_model", None)
        if model is None:
            return None
        try:
            from vosk import KaldiRecognizer
            return KaldiRecognizer(model, int(self.backend.AUDIO_RATE))
        except Exception as exc:
            self.log(f"[ASR_PARTIAL] unavailable={type(exc).__name__}: {exc}")
            return None

    @staticmethod
    def _read_partial(recognizer: Any, mono_bytes: bytes) -> str:
        if recognizer is None:
            return ""
        try:
            if recognizer.AcceptWaveform(mono_bytes):
                payload = json.loads(recognizer.Result())
                return str(payload.get("text", "")).strip()
            payload = json.loads(recognizer.PartialResult())
            return str(payload.get("partial", "")).strip()
        except Exception:
            return ""

    def capture(self) -> str:
        backend = self.backend
        manager = self.manager
        self.last_timing = {}
        self.cancelled = False
        backend.AUDIO_CAPTURE_ACTIVE.set()
        try:
            proc = backend.open_raw_mic_stream()
        except Exception:
            backend.AUDIO_CAPTURE_ACTIVE.clear()
            raise
        capture_done = threading.Event()
        cleanup_started = threading.Event()
        cleanup_finished = threading.Event()
        cleanup_lock = threading.Lock()

        def stop_stream_once() -> None:
            owns_cleanup = False
            with cleanup_lock:
                if not cleanup_started.is_set():
                    cleanup_started.set()
                    owns_cleanup = True
            if owns_cleanup:
                try:
                    backend.stop_stream(proc)
                finally:
                    cleanup_finished.set()
            else:
                cleanup_finished.wait(timeout=1.5)

        def cancel_on_runtime_stop() -> None:
            # Event waits keep this dormant during normal capture. The short
            # bound only controls shutdown response; it is not audio polling.
            while not capture_done.wait(timeout=0.05):
                if self.stop_event is not None and self.stop_event.is_set():
                    stop_stream_once()
                    return

        cancellation_thread = None
        if self.stop_event is not None:
            cancellation_thread = threading.Thread(
                target=cancel_on_runtime_stop,
                daemon=True,
                name="MiguelAudioCaptureCancel",
            )
            cancellation_thread.start()
        started_wall = time.time()
        started_mono = time.monotonic()
        speech_started_wall = None
        speech_started_mono = None
        last_voice_wall = started_wall
        endpoint_silence_observed = None
        speech_started = False
        silence_reported = False
        mono_chunks = []
        chunk_ms = int(backend.CHUNK_MS)
        pre_roll = deque(maxlen=max(1, round(backend.SPEECH_PREROLL_SECONDS * 1000 / chunk_ms)))
        noise = deque(maxlen=max(5, round(2000 / chunk_ms)))
        recognizer = self._partial_recognizer()
        wav_path = backend.AUDIO_DIR / "miguel_user_turn_openai.wav"
        rms_peak = 0.0

        manager.begin_listening(started_mono)
        try:
            while True:
                if self.stop_event is not None and self.stop_event.is_set():
                    self.cancelled = True
                    break
                now_wall = time.time()
                now_mono = time.monotonic()
                # This is a no-input guard only. Active human speech has no
                # total-duration cutoff in the conversation-aware path.
                if not speech_started and now_wall - started_wall > backend.MAX_TURN_SECONDS:
                    self.log("No-input capture timeout reached.")
                    break

                raw = proc.stdout.read(backend.CHUNK_BYTES)
                if not raw:
                    continue
                mono_bytes, rms = backend.stereo_raw_to_mono_bytes(raw)
                if not mono_bytes:
                    continue
                rms_peak = max(rms_peak, rms)
                threshold = backend.adaptive_speech_threshold(noise)
                voiced = rms > threshold

                if speech_started:
                    mono_chunks.append(mono_bytes)
                else:
                    pre_roll.append(mono_bytes)

                if voiced:
                    if not speech_started:
                        speech_started = True
                        speech_started_wall = now_wall
                        speech_started_mono = now_mono
                        mono_chunks.extend(pre_roll)
                        pre_roll.clear()
                        manager.on_voice(True, now_mono)
                    else:
                        manager.note_voice_activity(now_mono)
                        if silence_reported:
                            manager.on_voice(True, now_mono)
                    silence_reported = False
                    last_voice_wall = now_wall
                elif not speech_started and rms < backend.SPEECH_RMS_MIN_THRESHOLD:
                    noise.append(rms)

                if speech_started:
                    partial = self._read_partial(recognizer, mono_bytes)
                    if partial:
                        manager.on_partial(partial, now_mono)
                    silence_age = now_wall - last_voice_wall
                    if not voiced and not silence_reported:
                        manager.on_voice(False, now_mono)
                        silence_reported = True

                    if manager.config.enable_adaptive_endpoint:
                        if manager.evaluate_endpoint(now_mono).commit:
                            endpoint_silence_observed = silence_age
                            break
                    else:
                        spoken = max(0.0, now_wall - (speech_started_wall or started_wall))
                        endpoint_silence = (
                            backend.SHORT_UTTERANCE_SILENCE_SECONDS
                            if spoken < 1.25 else backend.SILENCE_SECONDS
                        )
                        if now_wall - started_wall >= backend.MIN_TURN_SECONDS and silence_age >= endpoint_silence:
                            manager.legacy_commit(now_mono, round(silence_age * 1000))
                            endpoint_silence_observed = silence_age
                            break

            if self.cancelled or (self.stop_event is not None and self.stop_event.is_set()):
                self.cancelled = True
                manager.finish_listening_without_turn("capture_cancelled")
                return ""
            if not mono_chunks:
                manager.finish_listening_without_turn("no_speech")
                return ""

            backend.write_mono_wav(wav_path, mono_chunks)
            transcription_started = time.monotonic()
            text = backend.transcribe_audio_openai(wav_path)
            transcription_finished = time.monotonic()
            manager.final_asr(text, transcription_finished)
            if not text:
                manager.finish_listening_without_turn("empty_asr")
            self.last_timing = {
                "capture_total_ms": (transcription_finished - started_mono) * 1000,
                "speech_wait_ms": (
                    (speech_started_mono - started_mono) * 1000
                    if speech_started_mono is not None else None
                ),
                "speech_duration_ms": (
                    max(0.0, last_voice_wall - speech_started_wall) * 1000
                    if speech_started_wall is not None else None
                ),
                "endpoint_silence_ms": (
                    endpoint_silence_observed * 1000
                    if endpoint_silence_observed is not None else None
                ),
                "transcription_ms": (transcription_finished - transcription_started) * 1000,
            }
            self.log(f"Final heard by OpenAI transcription: {text}")
            return text
        except Exception:
            manager.finish_listening_without_turn("capture_error")
            raise
        finally:
            capture_done.set()
            stop_stream_once()
            if cancellation_thread is not None:
                cancellation_thread.join(timeout=0.2)
            backend.AUDIO_CAPTURE_ACTIVE.clear()
