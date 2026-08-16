import sys
import threading
import time
import types
from unittest.mock import patch

import pytest

from miguel_conversation_events import FloorOwner, InteractionState
from miguel_conversation_manager import ConversationManager
from miguel_respeaker_xvf3800 import XVF3800Monitor
from miguel_spatial_audio import (
    FRONT_STRONG,
    FRONT_POSSIBLE,
    NO_SPEECH,
    OFF_AXIS,
    ROBOT_SPEAKING_SUPPRESSED,
    UNKNOWN,
    UNSTABLE,
    SpatialAudioConfig,
    SpatialAudioEvidence,
    SpatialAudioTracker,
    XVFSpatialWorker,
    circular_mean_degrees,
)
from miguel_turn_endpoint import ConversationConfig


def config(**overrides):
    values = dict(
        poll_hz=100.0,
        window_seconds=0.7,
        minimum_samples=3,
        stable_resultant_threshold=0.85,
        front_azimuth_deg=177.0,
        front_strong_deg=15.0,
        front_possible_deg=30.0,
        post_tts_settle_seconds=0.0,
        sensor_log_interval_seconds=60.0,
        retry_interval_seconds=1.0,
    )
    values.update(overrides)
    return SpatialAudioConfig(**values)


def feed(tracker, angles, *, start=1.0):
    result = None
    for index, angle in enumerate(angles):
        result = tracker.update(
            timestamp_monotonic=start + index * 0.1,
            available=True,
            vad_active=True,
            raw_doa_deg=angle,
        )
    return result


def evidence(classification, *, stable=False):
    return SpatialAudioEvidence(
        10.0, classification != UNKNOWN, classification != NO_SPEECH,
        177.0 if classification != UNKNOWN else None,
        0.0 if classification != UNKNOWN else None,
        stable, 0.99 if stable else 0.0, 4 if stable else 0, classification,
    )


def test_circular_mean_wraps_near_zero_and_180():
    mean, resultant = circular_mean_degrees([359, 0, 1])
    assert min(abs(mean), abs(mean - 360.0)) < 0.01
    assert resultant > 0.99
    mean, resultant = circular_mean_degrees([179, 180, 181])
    assert mean == pytest.approx(180.0)
    assert resultant > 0.99


def test_vad_false_ignores_retained_doa():
    tracker = SpatialAudioTracker(config())
    result = tracker.update(
        timestamp_monotonic=1.0, available=True, vad_active=False, raw_doa_deg=177
    )
    assert result.classification == NO_SPEECH
    assert result.relative_doa_deg is None
    assert result.sample_count == 0


@pytest.mark.parametrize(
    "angles,classification,relative",
    [
        ([176, 177, 178], FRONT_STRONG, 0),
        ([269, 270, 271], OFF_AXIS, 93),
        ([74, 75, 76], OFF_AXIS, -102),
    ],
)
def test_stable_spatial_classification(angles, classification, relative):
    result = feed(SpatialAudioTracker(config()), angles)
    assert result.stable
    assert result.classification == classification
    assert result.relative_doa_deg == pytest.approx(relative, abs=1.0)


def test_unstable_directions_are_not_front_evidence():
    result = feed(SpatialAudioTracker(config()), [0, 90, 180, 270])
    assert not result.stable
    assert result.classification == UNSTABLE


def test_robot_speaking_suppresses_and_resets_human_window():
    tracker = SpatialAudioTracker(config())
    feed(tracker, [269, 270], start=1.0)
    suppressed = tracker.update(
        timestamp_monotonic=1.3,
        available=True,
        vad_active=True,
        raw_doa_deg=270,
        suppressed=True,
        suppression_reason="robot_speaking",
    )
    assert suppressed.classification == ROBOT_SPEAKING_SUPPRESSED
    assert suppressed.suppressed
    first_human = tracker.update(
        timestamp_monotonic=1.4, available=True, vad_active=True, raw_doa_deg=177
    )
    assert first_human.sample_count == 1
    assert first_human.classification == UNKNOWN
    assert feed(tracker, [177, 178], start=1.5).classification == FRONT_STRONG


@pytest.mark.parametrize("spatial", [UNKNOWN, FRONT_STRONG, FRONT_POSSIBLE, OFF_AXIS, UNSTABLE])
@pytest.mark.parametrize(
    "setup,expected",
    [
        ("explicit_wake", True),
        ("engaged", True),
        ("expected_reply", True),
        ("idle", False),
    ],
)
def test_spatial_evidence_does_not_change_acceptance(spatial, setup, expected):
    manager = ConversationManager(ConversationConfig(), logger=lambda _line: None)
    manager.begin_listening(9.0)
    manager.note_spatial_audio_evidence(
        evidence(spatial, stable=spatial in {FRONT_STRONG, FRONT_POSSIBLE, OFF_AXIS})
    )
    kwargs = {"timestamp": 10.0}
    if setup == "explicit_wake":
        kwargs["explicit_wake"] = True
    elif setup == "engaged":
        manager.engagement_score = 1.0
        manager.last_engagement_update = 10.0
    elif setup == "expected_reply":
        manager.robot_just_asked_question = True
        manager.expected_reply_until = 20.0
    assert manager.accept_human_turn("hello", **kwargs) is expected


def test_note_spatial_evidence_has_no_floor_endpoint_or_state_side_effects():
    manager = ConversationManager(ConversationConfig(), logger=lambda _line: None)
    manager.state = InteractionState.SPEAKING
    manager.floor_owner = FloorOwner.MIGUEL
    endpoint_state = dict(manager.endpoint.__dict__)
    manager.note_spatial_audio_evidence(evidence(FRONT_STRONG, stable=True))
    assert manager.state == InteractionState.SPEAKING
    assert manager.floor_owner == FloorOwner.MIGUEL
    assert manager.endpoint.__dict__ == endpoint_state


def test_turn_latch_survives_later_no_speech_and_drives_shadow_log():
    lines = []
    manager = ConversationManager(ConversationConfig(), logger=lines.append)
    manager.begin_listening(1.0)
    stable = evidence(FRONT_STRONG, stable=True)
    manager.note_spatial_audio_evidence(stable)
    manager.note_spatial_audio_evidence(
        SpatialAudioEvidence(10.1, True, False, 177, None, False, 0.0, 0, NO_SPEECH)
    )
    assert manager.spatial_audio_evidence().classification == NO_SPEECH
    assert manager.turn_spatial_audio_evidence() is stable
    manager.accept_human_turn("room speech", timestamp=10.2)
    shadow = next(line for line in lines if line.startswith("[ADDRESSEE_SHADOW]"))
    assert "evidence_source=turn_latched" in shadow
    assert "vad=true" in shadow
    assert "spatial=FRONT_STRONG" in shadow


def test_off_axis_evidence_remains_latched_for_the_turn():
    manager = ConversationManager(ConversationConfig(), logger=lambda _line: None)
    manager.begin_listening(1.0)
    off_axis = SpatialAudioEvidence(
        1.2, True, True, 289, 112, True, 1.0, 7, OFF_AXIS
    )
    manager.note_spatial_audio_evidence(off_axis)
    manager.note_spatial_audio_evidence(
        SpatialAudioEvidence(1.3, True, False, 289, None, False, 0.0, 0, NO_SPEECH)
    )
    assert manager.turn_spatial_audio_evidence() is off_axis


def test_new_listening_turn_clears_old_latch_and_does_not_reuse_it():
    manager = ConversationManager(ConversationConfig(), logger=lambda _line: None)
    manager.begin_listening(1.0)
    manager.note_spatial_audio_evidence(evidence(FRONT_STRONG, stable=True))
    assert manager.turn_spatial_audio_evidence() is not None
    manager.begin_listening(2.0)
    manager.note_spatial_audio_evidence(
        SpatialAudioEvidence(2.1, True, False, 177, None, False, 0.0, 0, NO_SPEECH)
    )
    assert manager.turn_spatial_audio_evidence() is None


@pytest.mark.parametrize("reason", ["robot_speaking", "post_tts_settle"])
def test_suppressed_and_post_tts_evidence_never_latches(reason):
    manager = ConversationManager(ConversationConfig(), logger=lambda _line: None)
    manager.begin_listening(1.0)
    manager.note_spatial_audio_evidence(
        SpatialAudioEvidence(
            1.1, True, True, 177, None, False, 0.0, 0,
            ROBOT_SPEAKING_SUPPRESSED, True, reason,
        )
    )
    assert manager.turn_spatial_audio_evidence() is None


class FakeManager:
    def __init__(self):
        self.items = []

    def note_spatial_audio_evidence(self, item):
        self.items.append(item)


class FakeMonitor:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.closed = False
        self.reads = 0

    def health(self):
        return {"available": True, "device_count": 1, "selected_device_index": 0}

    def available(self):
        return True

    def restart(self):
        return True

    def read_doa_vad(self):
        self.reads += 1
        if self.fail:
            raise OSError("usb disappeared")
        return 177, True

    def close(self):
        self.closed = True


def test_xvf_worker_stops_closes_monitor_and_does_not_change_behavior():
    stop = threading.Event()
    monitor = FakeMonitor()
    manager = FakeManager()
    worker = XVFSpatialWorker(
        monitor, SpatialAudioTracker(config()), manager, stop, lambda: True, logger=lambda _line: None
    )
    thread = threading.Thread(target=worker.run, name="XVFWorker")
    thread.start()
    deadline = time.monotonic() + 1.0
    while not manager.items and time.monotonic() < deadline:
        time.sleep(0.005)
    stop.set()
    thread.join(0.5)
    assert not thread.is_alive()
    assert monitor.closed
    assert manager.items[-1].suppressed
    assert manager.items[-1].classification == ROBOT_SPEAKING_SUPPRESSED


def test_usb_read_failure_degrades_to_unknown_without_crashing_worker():
    stop = threading.Event()
    monitor = FakeMonitor(fail=True)
    manager = FakeManager()
    worker = XVFSpatialWorker(
        monitor, SpatialAudioTracker(config()), manager, stop, lambda: False, logger=lambda _line: None
    )
    thread = threading.Thread(target=worker.run)
    thread.start()
    deadline = time.monotonic() + 1.0
    while not manager.items and time.monotonic() < deadline:
        time.sleep(0.005)
    stop.set()
    thread.join(0.5)
    assert manager.items[-1].classification == UNKNOWN
    assert "usb disappeared" in manager.items[-1].read_error
    assert monitor.closed


def test_repetitive_sensor_logs_are_transition_and_heartbeat_limited():
    stop = threading.Event()
    worker = XVFSpatialWorker(
        FakeMonitor(), SpatialAudioTracker(config(sensor_log_interval_seconds=0.5)), FakeManager(), stop,
        lambda: False, logger=lambda _line: None,
    )
    no_speech = SpatialAudioEvidence(
        1.0, True, False, 177, None, False, 0.0, 0, NO_SPEECH
    )
    assert worker._should_log(no_speech, 1.0)
    assert not worker._should_log(no_speech, 1.6)
    assert worker._should_log(no_speech, 6.1)
    suppressed = SpatialAudioEvidence(
        6.7, True, True, 287, None, False, 0.0, 0,
        ROBOT_SPEAKING_SUPPRESSED, True, "robot_speaking",
    )
    assert worker._should_log(suppressed, 6.7)
    assert not worker._should_log(suppressed, 7.3)


def test_dual_device_selection_metadata_is_deterministic():
    class Device:
        idVendor = 0x2886
        idProduct = 0x001A

        def __init__(self, bus, address, serial):
            self.bus = bus
            self.address = address
            self.serial_number = serial

    devices = [Device(2, 4, "right"), Device(1, 8, "left")]
    core = types.SimpleNamespace(find=lambda **_kwargs: list(devices))
    util = types.SimpleNamespace(dispose_resources=lambda _device: None, get_string=lambda *_args: None)
    usb = types.ModuleType("usb")
    usb.core = core
    usb.util = util
    with patch.dict(sys.modules, {"usb": usb, "usb.core": core, "usb.util": util}):
        with patch.object(XVF3800Monitor, "_read", return_value=(2, 0, 6)):
            monitor = XVF3800Monitor(device_index=0)
    health = monitor.health()
    assert health["device_count"] == 2
    assert health["selected_device_index"] == 0
    assert health["bus"] == 1
    assert health["address"] == 8
    assert health["serial"] == "left"
    assert health["selection_ambiguous"] is True
