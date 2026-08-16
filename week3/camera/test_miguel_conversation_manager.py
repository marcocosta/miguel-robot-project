import time
import inspect
import ast
import types
import pytest
import threading
from unittest.mock import patch

from miguel_conversation_events import FloorOwner, InteractionState
from miguel_conversation_manager import ConversationManager
from miguel_audio_capture import MiguelAudioCapture
from miguel_respeaker_xvf3800 import XVF3800Monitor
from miguel_turn_endpoint import AdaptiveEndpointDetector, ConversationConfig


def detector_with(text: str, silence_start: float = 10.0, expected_reply: bool = False):
    detector = AdaptiveEndpointDetector(ConversationConfig())
    detector.speech_started(9.0)
    detector.update_partial(text, 9.8)
    detector.silence_detected(silence_start)
    return detector, {"expected_reply": expected_reply}


def test_complete_question_commits_at_550_ms():
    detector, context = detector_with("what time is it")
    decision = detector.evaluate(10.550, context)
    assert decision.commit
    assert decision.reason == "fast_complete"


def test_incomplete_clause_does_not_commit_at_700_ms():
    detector, context = detector_with("can you tell me...")
    decision = detector.evaluate(10.700, context)
    assert not decision.commit
    assert decision.evidence.unfinished


def test_unfinished_prompt_is_detected_with_wake_or_asr_name_prefix():
    for text in (
        "can you tell me",
        "miguel can you tell me",
        "hey miguel can you tell me",
        "mcgill can you tell me",
        "go can you tell me",
    ):
        detector, context = detector_with(text)
        decision = detector.evaluate(10.760, context)
        assert decision.evidence.unfinished, text
        assert not decision.commit, text


def test_prefixed_unfinished_prompt_with_continuation_remains_complete():
    for text in (
        "miguel can you tell me who invented the telephone",
        "can you tell me what time it is",
        "i think that works",
    ):
        detector, context = detector_with(text)
        decision = detector.evaluate(10.760, context)
        assert not decision.evidence.unfinished, text
        assert decision.commit, text


def test_prefixed_unfinished_prompt_uses_existing_repair_timing():
    detector, context = detector_with("miguel can you tell me")
    assert not detector.evaluate(10.510, context).commit
    assert not detector.evaluate(10.760, context).commit
    repair = detector.evaluate(12.210, context)
    assert repair.repair_consider
    assert not repair.commit


def test_speech_to_eot_delay_uses_commit_minus_last_voice():
    manager = ConversationManager(ConversationConfig(), logger=lambda _line: None)
    manager.turn_timing = {
        "last_detected_voice_monotonic": 10.0,
        "turn_commit_monotonic": 10.6,
        "final_asr_monotonic": 10.9,
    }
    latency = manager.latency_fields()
    assert latency["speech_to_eot_delay_ms"] == pytest.approx(600.0)
    assert latency["eot_to_final_asr_ms"] == pytest.approx(300.0)


def test_trailing_subordinate_that_phrases_are_unfinished():
    for text in ("I think that", "I believe that", "I know that", "I was thinking that", "the reason is that"):
        detector, context = detector_with(text)
        decision = detector.evaluate(10.760, context)
        assert decision.evidence.unfinished, text
        assert "trailing_subordinate_that" in decision.evidence.reason_codes
        assert not decision.commit, text


def test_complete_phrase_containing_that_is_not_marked_unfinished():
    detector, context = detector_with("I think that works")
    decision = detector.evaluate(10.760, context)
    assert not decision.evidence.unfinished
    assert decision.commit
    assert decision.reason == "normal_complete"


def test_trailing_subordinate_that_uses_repair_timing():
    detector, context = detector_with("I think that")
    normal_window = detector.evaluate(10.760, context)
    assert not normal_window.commit
    repair = detector.evaluate(12.210, context)
    assert repair.repair_consider
    assert not repair.commit
    fallback = detector.evaluate(13.010, context)
    assert fallback.commit
    assert fallback.reason == "repair_timeout"


def test_missing_vosk_partial_uses_ambiguous_silence_fallback():
    detector, context = detector_with("")
    decision = detector.evaluate(11.410, context)
    assert decision.commit
    assert decision.reason == "no_partial_fallback"


def test_incomplete_clause_marks_repair_before_later_silence_commit():
    detector, context = detector_with("can you tell me and")
    repair = detector.evaluate(12.210, context)
    assert not repair.commit
    assert repair.repair_consider
    assert repair.reason == "repair_consider"
    fallback = detector.evaluate(13.010, context)
    assert fallback.commit
    assert fallback.repair_consider
    assert fallback.reason == "repair_timeout"


def test_speech_resume_cancels_end_candidate():
    manager = ConversationManager(ConversationConfig(), logger=lambda _line: None)
    manager.begin_listening(1.0)
    manager.on_voice(True, 1.1)
    manager.on_partial("can you tell me and", 1.2)
    manager.on_voice(False, 1.3)
    manager.evaluate_endpoint(1.61)
    assert manager.state == InteractionState.END_CANDIDATE
    manager.on_voice(True, 1.62)
    assert manager.state == InteractionState.LISTENING
    assert manager.floor_owner == FloorOwner.HUMAN


def test_empty_capture_releases_human_floor():
    manager = ConversationManager(ConversationConfig(), logger=lambda _line: None)
    manager.begin_listening(1.0)
    manager.finish_listening_without_turn("no_speech")
    assert manager.floor_owner == FloorOwner.NONE
    assert manager.state == InteractionState.IDLE


def test_state_observer_keeps_end_candidate_explicit():
    states = []
    manager = ConversationManager(
        ConversationConfig(),
        logger=lambda _line: None,
        state_observer=lambda state, reason: states.append((state, reason)),
    )
    manager.begin_listening(1.0)
    manager.on_voice(True, 1.1)
    manager.on_partial("what time is it", 1.2)
    manager.on_voice(False, 1.3)
    manager.evaluate_endpoint(1.61)
    assert (InteractionState.END_CANDIDATE, "silence_candidate") in states


def test_expected_yes_reply_commits_around_500_ms():
    manager = ConversationManager(ConversationConfig(), logger=lambda _line: None)
    manager.robot_speech_ended("Do you want another chapter?", timestamp=5.0)
    manager.begin_listening(5.1)
    manager.on_voice(True, 5.2)
    manager.on_partial("yes", 5.3)
    manager.on_voice(False, 5.4)
    decision = manager.evaluate_endpoint(5.91)
    assert decision.commit
    assert "expected_yes_no" in decision.evidence.reason_codes


def test_expected_person_is_cleared_when_reply_window_expires():
    manager = ConversationManager(ConversationConfig(expected_reply_seconds=1.0), logger=lambda _line: None)
    manager.robot_speech_ended("Do you want another chapter?", person="Marco", timestamp=5.0)
    assert manager.expects_reply(5.5)
    assert not manager.expects_reply(6.1)
    assert manager.expected_reply_person is None


def test_known_expected_person_rejects_different_known_speaker():
    manager = ConversationManager(ConversationConfig(), logger=lambda _line: None)
    manager.robot_speech_ended("Do you want another chapter?", person="Marco", timestamp=5.0)
    assert manager.accept_human_turn("yes", timestamp=5.5, person="Marco")
    assert not manager.accept_human_turn("yes", timestamp=5.5, person="Helena")


def test_engaged_followup_without_wake_is_accepted():
    manager = ConversationManager(ConversationConfig(), logger=lambda _line: None)
    manager.on_wake("Miguel", timestamp=10.0)
    assert manager.accept_human_turn("tell me more", timestamp=12.0)


def test_idle_random_room_speech_is_rejected():
    manager = ConversationManager(ConversationConfig(), logger=lambda _line: None)
    assert not manager.accept_human_turn("someone talking across the room", timestamp=10.0)


def test_xvf_unavailable_does_not_raise():
    with patch.object(XVF3800Monitor, "_probe", lambda self: setattr(self._health, "reason", "test_unavailable")):
        monitor = XVF3800Monitor()
    assert not monitor.available()
    assert monitor.health()["reason"] == "test_unavailable"
    assert "fallback=existing_audio" in monitor.startup_log()


def test_xvf_combined_command_exposes_doa_and_vad():
    class FakeDevice:
        idVendor = 0x2886
        idProduct = 0x001A

        def ctrl_transfer(self, _request_type, _request, value, index, _length, timeout):
            assert timeout == 250
            if (value, index) == (0x80, 48):
                return bytes((0, 2, 0, 6))
            if (value, index) == (0x80 | 18, 20):
                return bytes((0, 123, 0, 1, 0))
            raise AssertionError((value, index))

    monitor = XVF3800Monitor(device=FakeDevice())
    assert monitor.available()
    assert monitor.firmware_version() == "2.0.6"
    assert monitor.read_doa_vad() == (123, True)


def test_xvf_probes_documented_fallback_commands_independently():
    import struct

    class FakeLegacyDevice:
        idVendor = 0x2886
        idProduct = 0x001B

        def ctrl_transfer(self, _request_type, _request, value, index, _length, timeout):
            assert timeout == 250
            if (value, index) == (0x80, 48):
                return bytes((0, 1, 2, 3))
            if (value, index) == (0x80 | 75, 33):
                return bytes((0,)) + struct.pack("<ffff", 0.0, 0.0, 0.0, 1.5707963)
            if (value, index) == (0x80 | 80, 33):
                return bytes((0,)) + struct.pack("<ffff", 0.0, 0.0, 0.0, 0.25)
            raise OSError("unsupported")

    monitor = XVF3800Monitor(device=FakeLegacyDevice())
    assert monitor.available()
    assert monitor.health()["pid"] == 0x001B
    assert monitor.read_doa() == 90
    assert monitor.read_vad() is True


def test_shutdown_and_timer_text_survive_endpointing_for_existing_router():
    shutdown, context = detector_with("shutdown")
    shutdown_decision = shutdown.evaluate(10.510, context)
    assert shutdown_decision.commit
    assert shutdown.partial_text == "shutdown"

    timer, context = detector_with("set a timer for five minutes")
    timer_decision = timer.evaluate(10.760, context)
    assert timer_decision.commit
    assert timer.partial_text == "set a timer for five minutes"


def test_active_speech_has_no_total_duration_endpoint():
    detector = AdaptiveEndpointDetector(ConversationConfig())
    detector.speech_started(1.0)
    detector.voice_detected(120.0)
    decision = detector.evaluate(600.0)
    assert not decision.commit
    assert decision.reason == "speech_active"


def test_v7_capture_component_never_calls_historical_capture_user_turn():
    source = inspect.getsource(MiguelAudioCapture)
    assert "capture_user_turn(" not in source
    assert "xvf" not in source.lower()


def test_historical_v6_capture_signature_remains_parameterless():
    v6_path = __import__("pathlib").Path(__file__).with_name("robot_cloud_brain_v6_threaded.py")
    tree = ast.parse(v6_path.read_text())
    capture = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "capture_user_turn"
    )
    assert not capture.args.args
    assert capture.args.vararg is None
    assert capture.args.kwarg is None


def test_conversation_capture_emits_voice_partial_endpoint_and_final_events(tmp_path):
    class Stdout:
        chunks = [b"voice", b"silence"]

        def read(self, _size):
            return self.chunks.pop(0)

    class Manager:
        config = types.SimpleNamespace(enable_adaptive_endpoint=True)

        def __init__(self):
            self.events = []
            self.voice_active = False

        def begin_listening(self, _at): self.events.append("begin")
        def on_voice(self, active, _at):
            self.voice_active = active
            self.events.append(("voice", active))
        def note_voice_activity(self, _at): self.events.append("activity")
        def on_partial(self, text, _at): self.events.append(("partial", text))
        def evaluate_endpoint(self, _at):
            self.events.append("endpoint")
            return types.SimpleNamespace(commit=not self.voice_active)
        def final_asr(self, text, _at): self.events.append(("final", text))
        def finish_listening_without_turn(self, _reason): pass

    backend = types.SimpleNamespace(
        AUDIO_CAPTURE_ACTIVE=threading.Event(),
        AUDIO_RATE=16000, CHUNK_MS=100, CHUNK_BYTES=5,
        SPEECH_PREROLL_SECONDS=0.0, SPEECH_RMS_MIN_THRESHOLD=500,
        MAX_TURN_SECONDS=20.0, MIN_TURN_SECONDS=1.0,
        SILENCE_SECONDS=0.9, SHORT_UTTERANCE_SILENCE_SECONDS=1.05,
        AUDIO_DIR=tmp_path, vosk_model=None,
        open_raw_mic_stream=lambda: types.SimpleNamespace(stdout=Stdout()),
        stereo_raw_to_mono_bytes=lambda raw: (raw, 2000 if raw == b"voice" else 0),
        adaptive_speech_threshold=lambda _noise: 1000,
        write_mono_wav=lambda _path, _chunks: None,
        transcribe_audio_openai=lambda _path: "what time is it",
        stop_stream=lambda _proc: None,
    )
    manager = Manager()
    capture = MiguelAudioCapture(backend, manager, logger=lambda _line: None)
    capture._partial_recognizer = lambda: object()
    capture._read_partial = lambda _recognizer, raw: "what time is it" if raw == b"voice" else ""

    assert capture.capture() == "what time is it"
    assert manager.events == [
        "begin", ("voice", True), ("partial", "what time is it"),
        "endpoint", ("voice", False), "endpoint", ("final", "what time is it"),
    ]


def test_active_prespeech_capture_cancels_and_terminates_stream_promptly(tmp_path):
    stop_event = threading.Event()
    read_started = threading.Event()
    read_released = threading.Event()
    capture_active = threading.Event()
    stopped = []

    class BlockingStdout:
        def read(self, _size):
            read_started.set()
            read_released.wait(timeout=2.0)
            return b""

    proc = types.SimpleNamespace(stdout=BlockingStdout())

    class Manager:
        config = types.SimpleNamespace(enable_adaptive_endpoint=True)
        def __init__(self): self.finished_reason = None
        def begin_listening(self, _at): pass
        def finish_listening_without_turn(self, reason): self.finished_reason = reason

    def stop_stream(stream):
        stopped.append(stream)
        read_released.set()

    backend = types.SimpleNamespace(
        AUDIO_CAPTURE_ACTIVE=capture_active,
        AUDIO_RATE=16000, CHUNK_MS=100, CHUNK_BYTES=6400,
        SPEECH_PREROLL_SECONDS=0.4, SPEECH_RMS_MIN_THRESHOLD=500,
        MAX_TURN_SECONDS=20.0, MIN_TURN_SECONDS=1.0,
        SILENCE_SECONDS=0.9, SHORT_UTTERANCE_SILENCE_SECONDS=1.05,
        AUDIO_DIR=tmp_path, vosk_model=None,
        open_raw_mic_stream=lambda: proc,
        stop_stream=stop_stream,
    )
    manager = Manager()
    capture = MiguelAudioCapture(backend, manager, stop_event=stop_event, logger=lambda _line: None)
    worker = threading.Thread(target=capture.capture, name="CaptureTest")
    worker.start()
    assert read_started.wait(timeout=0.5)
    assert capture_active.is_set()

    stop_event.set()
    worker.join(timeout=0.75)

    assert not worker.is_alive()
    assert capture.cancelled
    assert manager.finished_reason == "capture_cancelled"
    assert stopped == [proc]
    assert not capture_active.is_set()


def test_legacy_endpoint_mode_remains_configurable():
    config = ConversationConfig.from_env({"MIGUEL_ENABLE_ADAPTIVE_ENDPOINT": "false"})
    assert not config.enable_adaptive_endpoint
