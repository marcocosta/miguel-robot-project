import time
from unittest.mock import patch

from miguel_conversation_events import FloorOwner, InteractionState
from miguel_conversation_manager import ConversationManager
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
