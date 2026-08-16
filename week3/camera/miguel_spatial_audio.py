"""Stage 2B shadow-only XVF3800 spatial evidence.

This module does not own conversation behavior.  It stabilizes optional XVF
telemetry and publishes immutable snapshots for calibration and logging.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
import os
import threading
import time
from typing import Any, Callable, Optional


UNKNOWN = "UNKNOWN"
NO_SPEECH = "NO_SPEECH"
FRONT_STRONG = "FRONT_STRONG"
FRONT_POSSIBLE = "FRONT_POSSIBLE"
OFF_AXIS = "OFF_AXIS"
UNSTABLE = "UNSTABLE"
ROBOT_SPEAKING_SUPPRESSED = "ROBOT_SPEAKING_SUPPRESSED"


@dataclass(frozen=True)
class SpatialAudioConfig:
    poll_hz: float = 10.0
    window_seconds: float = 0.7
    minimum_samples: int = 3
    stable_resultant_threshold: float = 0.85
    front_azimuth_deg: float = 177.0
    front_strong_deg: float = 15.0
    front_possible_deg: float = 30.0
    post_tts_settle_seconds: float = 0.15
    sensor_log_interval_seconds: float = 0.5
    sensor_heartbeat_seconds: float = 5.0
    direction_log_change_deg: float = 10.0
    retry_interval_seconds: float = 10.0

    @classmethod
    def from_env(cls, env=None) -> "SpatialAudioConfig":
        source = os.environ if env is None else env

        def number(name: str, default: float) -> float:
            try:
                return float(source.get(name, default))
            except (TypeError, ValueError):
                return default

        def integer(name: str, default: int) -> int:
            try:
                return int(source.get(name, default))
            except (TypeError, ValueError):
                return default

        return cls(
            poll_hz=max(1.0, number("MIGUEL_XVF_POLL_HZ", 10.0)),
            window_seconds=max(0.1, number("MIGUEL_XVF_WINDOW_SECONDS", 0.7)),
            minimum_samples=max(1, integer("MIGUEL_XVF_MIN_STABLE_SAMPLES", 3)),
            stable_resultant_threshold=min(1.0, max(0.0, number("MIGUEL_XVF_STABLE_RESULTANT", 0.85))),
            front_azimuth_deg=number("MIGUEL_XVF_FRONT_AZIMUTH_DEG", 177.0) % 360.0,
            front_strong_deg=max(0.0, number("MIGUEL_XVF_FRONT_STRONG_DEG", 15.0)),
            front_possible_deg=max(0.0, number("MIGUEL_XVF_FRONT_POSSIBLE_DEG", 30.0)),
            post_tts_settle_seconds=max(0.0, number("MIGUEL_XVF_POST_TTS_SETTLE_SECONDS", 0.15)),
            sensor_log_interval_seconds=max(0.1, number("MIGUEL_XVF_LOG_INTERVAL_SECONDS", 0.5)),
            sensor_heartbeat_seconds=max(0.5, number("MIGUEL_XVF_HEARTBEAT_SECONDS", 5.0)),
            direction_log_change_deg=max(1.0, number("MIGUEL_XVF_DIRECTION_LOG_CHANGE_DEG", 10.0)),
            retry_interval_seconds=max(1.0, number("MIGUEL_XVF_RETRY_SECONDS", 10.0)),
        )


@dataclass(frozen=True)
class SpatialAudioEvidence:
    timestamp_monotonic: float
    available: bool
    vad_active: bool
    raw_doa_deg: Optional[float]
    relative_doa_deg: Optional[float]
    stable: bool
    circular_resultant: float
    sample_count: int
    classification: str
    suppressed: bool = False
    suppression_reason: str = ""
    device_identity: Optional[dict] = None
    read_error: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def normalize_relative_doa(raw_doa_deg: float, front_azimuth_deg: float) -> float:
    return ((float(raw_doa_deg) - float(front_azimuth_deg) + 180.0) % 360.0) - 180.0


def circular_mean_degrees(angles) -> tuple[Optional[float], float]:
    values = [float(angle) % 360.0 for angle in angles]
    if not values:
        return None, 0.0
    sin_mean = sum(math.sin(math.radians(value)) for value in values) / len(values)
    cos_mean = sum(math.cos(math.radians(value)) for value in values) / len(values)
    resultant = math.hypot(sin_mean, cos_mean)
    if resultant < 1e-12:
        return None, 0.0
    mean = math.degrees(math.atan2(sin_mean, cos_mean)) % 360.0
    return mean, resultant


class SpatialAudioTracker:
    def __init__(self, config: Optional[SpatialAudioConfig] = None):
        self.config = config or SpatialAudioConfig.from_env()
        self._samples = deque()
        self._lock = threading.Lock()
        self._latest = SpatialAudioEvidence(0.0, False, False, None, None, False, 0.0, 0, UNKNOWN)
        self.samples = 0
        self.vad_positive_samples = 0
        self.stable_snapshots = 0
        self.suppressed_robot_speech_samples = 0
        self.read_errors = 0

    def _prune(self, now: float) -> None:
        cutoff = now - self.config.window_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def update(
        self,
        *,
        timestamp_monotonic: float,
        available: bool,
        vad_active: bool,
        raw_doa_deg: Optional[float],
        suppressed: bool = False,
        suppression_reason: str = "",
        device_identity: Optional[dict] = None,
        read_error: str = "",
    ) -> SpatialAudioEvidence:
        now = float(timestamp_monotonic)
        with self._lock:
            self.samples += 1
            if read_error:
                self.read_errors += 1
            self._prune(now)
            if not available or read_error:
                self._samples.clear()
                evidence = SpatialAudioEvidence(
                    now, False, False, None, None, False, 0.0, 0, UNKNOWN,
                    False, "", device_identity, read_error,
                )
            elif suppressed:
                self._samples.clear()
                if vad_active:
                    self.suppressed_robot_speech_samples += 1
                evidence = SpatialAudioEvidence(
                    now, available, bool(vad_active), raw_doa_deg, None, False, 0.0, 0,
                    ROBOT_SPEAKING_SUPPRESSED, True, suppression_reason or "robot_speaking",
                    device_identity, read_error,
                )
            elif not vad_active:
                # Firmware may retain its last DoA while VAD is false.  Keep
                # raw telemetry visible but never retain it as human evidence.
                self._samples.clear()
                evidence = SpatialAudioEvidence(
                    now, True, False, raw_doa_deg, None, False, 0.0, 0, NO_SPEECH,
                    False, "", device_identity, "",
                )
            elif raw_doa_deg is None:
                evidence = SpatialAudioEvidence(
                    now, True, True, None, None, False, 0.0, len(self._samples), UNKNOWN,
                    False, "", device_identity, "",
                )
            else:
                self.vad_positive_samples += 1
                self._samples.append((now, float(raw_doa_deg) % 360.0))
                self._prune(now)
                mean, resultant = circular_mean_degrees(value for _stamp, value in self._samples)
                count = len(self._samples)
                stable = bool(
                    mean is not None
                    and count >= self.config.minimum_samples
                    and resultant >= self.config.stable_resultant_threshold
                )
                relative = normalize_relative_doa(mean, self.config.front_azimuth_deg) if mean is not None else None
                if not stable:
                    classification = UNSTABLE if count >= self.config.minimum_samples else UNKNOWN
                elif abs(relative) <= self.config.front_strong_deg:
                    classification = FRONT_STRONG
                elif abs(relative) <= self.config.front_possible_deg:
                    classification = FRONT_POSSIBLE
                else:
                    classification = OFF_AXIS
                evidence = SpatialAudioEvidence(
                    now, True, True, float(raw_doa_deg) % 360.0, relative,
                    stable, resultant, count, classification,
                    False, "", device_identity, "",
                )
                if stable:
                    self.stable_snapshots += 1
            self._latest = evidence
            return evidence

    def latest(self) -> SpatialAudioEvidence:
        with self._lock:
            return self._latest

    def reset_human_window(self) -> None:
        with self._lock:
            self._samples.clear()

    def diagnostics(self) -> dict:
        with self._lock:
            return {
                "samples": self.samples,
                "vad_positive_samples": self.vad_positive_samples,
                "stable_snapshots": self.stable_snapshots,
                "suppressed_robot_speech_samples": self.suppressed_robot_speech_samples,
                "read_errors": self.read_errors,
                "last_evidence": self._latest.as_dict(),
            }


class XVFSpatialWorker:
    """Poll one persistent monitor without coupling it to PCM capture."""

    def __init__(
        self,
        monitor: Any,
        tracker: SpatialAudioTracker,
        manager: Any,
        stop_event: threading.Event,
        robot_speaking: Callable[[], bool],
        logger: Callable[[str], None] = print,
    ):
        self.monitor = monitor
        self.tracker = tracker
        self.manager = manager
        self.stop_event = stop_event
        self.robot_speaking = robot_speaking
        self.log = logger
        self._last_log_at = 0.0
        self._last_logged_classification: Optional[str] = None
        self._last_logged_suppressed: Optional[bool] = None
        self._last_logged_relative: Optional[float] = None
        self._last_error_log_at = 0.0
        self._last_retry_at = 0.0
        self._retry_not_before = 0.0
        self._suppressed_until = 0.0

    def _should_log(self, evidence: SpatialAudioEvidence, now: float) -> bool:
        elapsed = now - self._last_log_at
        first = self._last_logged_classification is None
        state_changed = (
            evidence.classification != self._last_logged_classification
            or evidence.suppressed != self._last_logged_suppressed
        )
        direction_changed = False
        if evidence.stable and evidence.relative_doa_deg is not None and self._last_logged_relative is not None:
            difference = abs(
                ((evidence.relative_doa_deg - self._last_logged_relative + 180.0) % 360.0) - 180.0
            )
            direction_changed = difference >= self.tracker.config.direction_log_change_deg
        heartbeat = elapsed >= self.tracker.config.sensor_heartbeat_seconds
        should_log = first or heartbeat or (
            elapsed >= self.tracker.config.sensor_log_interval_seconds
            and (state_changed or direction_changed)
        )
        if should_log:
            self._last_log_at = now
            self._last_logged_classification = evidence.classification
            self._last_logged_suppressed = evidence.suppressed
            if evidence.stable and evidence.relative_doa_deg is not None:
                self._last_logged_relative = evidence.relative_doa_deg
        return should_log

    def _identity(self) -> dict:
        health = self.monitor.health()
        return {
            key: health.get(key)
            for key in (
                "device_count", "selected_device_index", "selection_ambiguous",
                "bus", "address", "serial", "vid", "pid", "firmware",
            )
        }

    def run(self) -> None:
        interval = 1.0 / self.tracker.config.poll_hz
        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                speaking = bool(self.robot_speaking())
                if speaking:
                    self._suppressed_until = started + self.tracker.config.post_tts_settle_seconds
                suppressed = speaking or started < self._suppressed_until
                suppression_reason = "robot_speaking" if speaking else ("post_tts_settle" if suppressed else "")
                identity = self._identity()
                try:
                    available = bool(self.monitor.available()) and started >= self._retry_not_before
                    if not available and started - self._last_retry_at >= self.tracker.config.retry_interval_seconds:
                        self._last_retry_at = started
                        available = bool(self.monitor.restart())
                        identity = self._identity()
                    raw_doa, vad = self.monitor.read_doa_vad() if available else (None, None)
                    evidence = self.tracker.update(
                        timestamp_monotonic=started,
                        available=available,
                        vad_active=bool(vad),
                        raw_doa_deg=raw_doa,
                        suppressed=suppressed,
                        suppression_reason=suppression_reason,
                        device_identity=identity,
                    )
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    self._retry_not_before = started + self.tracker.config.retry_interval_seconds
                    self._last_retry_at = started
                    evidence = self.tracker.update(
                        timestamp_monotonic=started,
                        available=False,
                        vad_active=False,
                        raw_doa_deg=None,
                        suppressed=suppressed,
                        suppression_reason=suppression_reason,
                        device_identity=identity,
                        read_error=message,
                    )
                    if started - self._last_error_log_at >= self.tracker.config.retry_interval_seconds:
                        self._last_error_log_at = started
                        self.log(f"[XVF_EVIDENCE] available=false read_error={message}")
                self.manager.note_spatial_audio_evidence(evidence)
                if self._should_log(evidence, started):
                    marker = "XVF_SUPPRESS" if evidence.suppressed else "XVF_EVIDENCE"
                    self.log(
                        f"[{marker}] vad={str(evidence.vad_active).lower()} "
                        f"raw_doa={evidence.raw_doa_deg} relative_doa={evidence.relative_doa_deg} "
                        f"stable={str(evidence.stable).lower()} resultant={evidence.circular_resultant:.3f} "
                        f"samples={evidence.sample_count} spatial={evidence.classification} "
                        f"reason={evidence.suppression_reason or 'none'}"
                    )
                elapsed = time.monotonic() - started
                self.stop_event.wait(max(0.0, interval - elapsed))
        finally:
            self.monitor.close()

    def diagnostics(self) -> dict:
        result = dict(self.monitor.health())
        result.update(self.tracker.diagnostics())
        return result
