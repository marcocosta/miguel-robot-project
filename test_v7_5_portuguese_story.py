"""Regression tests for Miguel V7.5 Portuguese long-story routing."""

from __future__ import annotations

import importlib
import sys
import threading
import types
from pathlib import Path


CAMERA_DIR = Path(__file__).resolve().parent / "week3" / "camera"


def load_v7_5_module():
    sys.path.insert(0, str(CAMERA_DIR))

    fake_v6 = types.SimpleNamespace(
        speak=lambda text: None,
        is_local_robot_control_request=lambda text: False,
    )
    fake_full = types.ModuleType("robot_cloud_brain_v7_full")
    fake_full.v6 = fake_v6
    fake_full.WAKE_PHRASES = ["hey miguel", "miguel", "mission control"]
    fake_full.CameraManager = object
    fake_full.is_local_robot_control_request = lambda text: False

    fake_camera_intents = types.ModuleType("v7.camera_intents")
    fake_camera_intents.classify_camera_intent = lambda text: "none"
    fake_camera_intents.is_identity_camera_request = lambda text: False
    fake_camera_intents.is_scene_camera_request = lambda text: False

    fake_safety_guard = types.ModuleType("v7.safety_guard")
    fake_safety_guard.SafetyGuard = object

    fake_robot_memory = types.ModuleType("robot_memory")
    fake_robot_memory.select_conversation_logs = lambda **kwargs: []

    fake_robot_timer = types.ModuleType("robot_timer")

    fake_depthai = types.ModuleType("depthai")
    fake_cv2 = types.ModuleType("cv2")
    fake_numpy = types.ModuleType("numpy")

    sys.modules.update(
        {
            "robot_cloud_brain_v7_full": fake_full,
            "robot_memory": fake_robot_memory,
            "robot_timer": fake_robot_timer,
            "depthai": fake_depthai,
            "cv2": fake_cv2,
            "numpy": fake_numpy,
            "v7.camera_intents": fake_camera_intents,
            "v7.safety_guard": fake_safety_guard,
        }
    )
    sys.modules.pop("robot_cloud_brain_v7_5_queue", None)
    return importlib.import_module("robot_cloud_brain_v7_5_queue")


def test_portuguese_long_story_requests_infer_long_story() -> None:
    q = load_v7_5_module()
    texts = [
        "Miguel, conte uma história longa sobre Marquinho e Helena.",
        "Miguel, quero uma história de cinco minutos.",
        "Miguel, entra em modo história.",
        "miguel, eu quero que você conte uma história longa, de cinco minutos, sobre um primo chamado marquinho e uma prima chamada helena.",
    ]

    for text in texts:
        detection = q._story_mode_detection(text)
        assert detection["detected"] is True
        assert detection["language"] == "pt"
        assert detection["mode"] == "long_story"
        assert q.infer_response_length_mode(text, "general", "none") == "long_story"


def test_portuguese_five_minutes_duration_is_parsed() -> None:
    q = load_v7_5_module()

    assert q._extract_long_story_duration_minutes("uma história de cinco minutos") == 5
    assert q._extract_long_story_duration_minutes("historia de 5 minutos") == 5


def test_portuguese_five_minute_story_is_continuous_three_chapters() -> None:
    q = load_v7_5_module()
    text = "Miguel, conta uma história longa de cinco minutos sobre Marquinho e Helena em Brasília."

    detection = q._story_mode_detection(text)

    assert detection["detected"] is True
    assert detection["action"] == "generate_story"
    assert detection["story_mode"] == "story_continuous"
    assert detection["requested_minutes"] == 5
    assert detection["chapter_count"] == 3
    assert detection["auto_continue"] is True


def test_portuguese_ten_minute_story_is_continuous_five_chapters() -> None:
    q = load_v7_5_module()
    text = "Miguel, conte uma história de dez minutos sobre Marquinho e Helena."

    detection = q._story_mode_detection(text)

    assert detection["detected"] is True
    assert detection["action"] == "generate_story"
    assert detection["story_mode"] == "story_continuous"
    assert detection["requested_minutes"] == 10
    assert detection["chapter_count"] == 5
    assert detection["auto_continue"] is True


def test_portuguese_bedtime_story_is_continuous_calm() -> None:
    q = load_v7_5_module()
    text = "Miguel, conte uma história longa para o Marquinho dormir."

    detection = q._story_mode_detection(text)

    assert detection["story_mode"] == "story_continuous"
    assert detection["subtype"] == "bedtime"
    assert detection["chapter_count"] == 5
    assert detection["auto_continue"] is True


def test_portuguese_story_mode_command_sets_story_depth() -> None:
    q = load_v7_5_module()
    spoken = []
    q.v6.speak = spoken.append
    state = q.RobotRuntimeState(stop_event=threading.Event())

    handled = q._route_response_depth_mode("Miguel, entra em modo história.", state)

    assert handled is True
    assert state.response_depth_mode == "long_story"
    assert state.response_length_mode == "long_story"
    assert state.conversation_mode == "story"
    assert spoken


def test_portuguese_story_request_generates_instead_of_mode_confirmation() -> None:
    q = load_v7_5_module()
    spoken = []
    q.v6.speak = spoken.append
    state = q.RobotRuntimeState(stop_event=threading.Event())
    text = "Miguel, conte uma história longa sobre Marquinho e Helena."

    detection = q._story_mode_detection(text)
    assert detection["has_story_request"] is True
    assert q._is_new_story_request(text) is True

    with state.lock:
        state.response_depth_mode = "long_story"
        state.response_length_mode = "long_story"
        state.conversation_mode = "story"

    assert q._route_response_depth_mode(text, state) is False
    assert q._route_long_story_mode(text, state) is False
    assert not any("Long explanation mode on" in reply for reply in spoken)
    assert state.conversation_mode == "story"
    assert state.response_depth_mode == "long_story"


def test_portuguese_continuous_story_enqueues_all_chapters() -> None:
    q = load_v7_5_module()
    spoken = []
    q.v6.speak = spoken.append
    state = q.RobotRuntimeState(stop_event=threading.Event())
    text = "Miguel, conta uma história longa de cinco minutos sobre Marquinho e Helena em Brasília."
    detection = q._story_mode_detection(text)

    handled = q._route_story_continuous_generation(text, state, detection)

    assert handled is True
    assert len(spoken) == 3
    assert not any("Do you want" in reply for reply in spoken)
    assert state.story_session.active is False
    assert state.conversation_mode == "general"


def test_portuguese_conta_story_request_generates_instead_of_confirmation() -> None:
    q = load_v7_5_module()
    spoken = []
    q.v6.speak = spoken.append
    state = q.RobotRuntimeState(stop_event=threading.Event())
    text = "Miguel, conta uma história longa de cinco minutos sobre Marquinho e Helena."

    assert q._story_mode_detection(text)["has_story_request"] is True
    assert q._is_new_story_request(text) is True
    assert q.infer_response_length_mode(text, "general", "none") == "long_story"

    with state.lock:
        state.response_depth_mode = "long_story"
        state.response_length_mode = "long_story"
        state.conversation_mode = "story"

    assert q._route_response_depth_mode(text, state) is False
    assert q._route_long_story_mode(text, state) is False
    assert spoken == []


def test_english_long_story_generates_instead_of_confirmation() -> None:
    q = load_v7_5_module()
    spoken = []
    q.v6.speak = spoken.append
    state = q.RobotRuntimeState(stop_event=threading.Event())
    text = "Miguel, tell me a long story about Marquinho and Helena."

    assert q._story_mode_detection(text)["has_story_request"] is True
    assert q._is_new_story_request(text) is True
    assert q.infer_response_length_mode(text, "general", "none") == "long_story"

    with state.lock:
        state.response_depth_mode = "long_story"
        state.response_length_mode = "long_story"
        state.conversation_mode = "story"

    assert q._route_response_depth_mode(text, state) is False
    assert q._route_long_story_mode(text, state) is False
    assert spoken == []


def test_english_ten_minute_story_is_continuous() -> None:
    q = load_v7_5_module()
    text = "Miguel, tell me a 10 minute story."

    detection = q._story_mode_detection(text)

    assert detection["action"] == "generate_story"
    assert detection["story_mode"] == "story_continuous"
    assert detection["requested_minutes"] == 10
    assert detection["chapter_count"] == 5
    assert detection["auto_continue"] is True


def test_french_multilingual_story_classifier_routes_continuous_story() -> None:
    q = load_v7_5_module()
    q.STORY_INTENT_CACHE.clear()

    def fake_classifier(text):
        return {
            "intent": "story",
            "action": "generate_story",
            "language": "fr",
            "response_language": "fr",
            "mode": "story_continuous",
            "subtype": "general",
            "requested_minutes": 10,
            "characters": ["Marquinho", "Helena"],
            "setting": None,
            "confidence": 0.92,
        }

    q._classify_story_intent_multilingual = fake_classifier
    text = "Miguel, raconte-moi une longue histoire de dix minutes sur Marquinho et Helena."

    detection = q._story_mode_detection(text)

    assert detection["detected"] is True
    assert detection["classifier_used"] is True
    assert detection["language"] == "fr"
    assert detection["response_language"] == "fr"
    assert detection["story_mode"] == "story_continuous"
    assert detection["requested_minutes"] == 10
    assert detection["chapter_count"] == 5
    assert q._is_new_story_request(text) is True

    spoken = []
    q.v6.speak = spoken.append
    state = q.RobotRuntimeState(stop_event=threading.Event())
    assert q._route_story_continuous_generation(text, state, detection) is True
    assert len(spoken) == 5
    assert spoken[0].startswith("Chapitre 1")


def test_spanish_multilingual_bedtime_story_classifier_routes_bedtime() -> None:
    q = load_v7_5_module()
    q.STORY_INTENT_CACHE.clear()

    def fake_classifier(text):
        return {
            "intent": "story",
            "action": "generate_story",
            "language": "es",
            "response_language": "es",
            "mode": "story_continuous",
            "subtype": "bedtime",
            "requested_minutes": None,
            "characters": [],
            "setting": None,
            "confidence": 0.9,
        }

    q._classify_story_intent_multilingual = fake_classifier
    text = "Miguel, cuéntame una historia larga para dormir."

    detection = q._story_mode_detection(text)

    assert detection["detected"] is True
    assert detection["classifier_used"] is True
    assert detection["language"] == "es"
    assert detection["response_language"] == "es"
    assert detection["story_mode"] == "story_continuous"
    assert detection["subtype"] == "bedtime"
    assert detection["chapter_count"] == 5

    spoken = []
    q.v6.speak = spoken.append
    state = q.RobotRuntimeState(stop_event=threading.Event())
    assert q._route_story_continuous_generation(text, state, detection) is True
    assert len(spoken) == 5
    assert spoken[0].startswith("Capítulo 1")


def test_multilingual_story_classifier_low_confidence_falls_back_normal() -> None:
    q = load_v7_5_module()
    q.STORY_INTENT_CACHE.clear()
    spoken = []
    q.v6.speak = spoken.append

    def fake_classifier(text):
        return {
            "intent": "story",
            "action": "generate_story",
            "language": "fr",
            "response_language": "fr",
            "mode": "story_continuous",
            "subtype": "general",
            "requested_minutes": 10,
            "characters": [],
            "setting": None,
            "confidence": 0.4,
        }

    q._classify_story_intent_multilingual = fake_classifier
    detection = q._story_mode_detection("Miguel, raconte-moi quelque chose.")

    assert detection["detected"] is False
    assert detection["action"] == "none"
    assert detection["story_mode"] == ""
    assert q._is_new_story_request("Miguel, raconte-moi quelque chose.") is False

    state = q.RobotRuntimeState(stop_event=threading.Event())
    assert q._route_low_confidence_story_intent(detection, state) is True
    assert spoken == [
        "Je vous ai entendu, mais je ne suis pas sûr que vous vouliez une histoire. Pouvez-vous répéter brièvement?"
    ]


def test_english_long_explanation_mode_still_confirms() -> None:
    q = load_v7_5_module()
    spoken = []
    q.v6.speak = spoken.append
    state = q.RobotRuntimeState(stop_event=threading.Event())

    handled = q._route_response_depth_mode("Miguel, turn on long explanation mode.", state)

    assert handled is True
    assert state.response_depth_mode == "long_explanation"
    assert state.response_length_mode == "detailed"
    assert state.conversation_mode == "general"
    assert spoken == ["Long explanation mode on. I'll explain with more detail."]


def test_long_story_trimming_uses_story_cap_not_normal_limit() -> None:
    q = load_v7_5_module()
    long_reply = " ".join(f"palavra{i}" for i in range(300)) + "."

    trimmed = q.make_robot_reply_concise(
        long_reply,
        context="normal",
        response_length_mode="normal",
        response_depth_mode="long_story",
    )

    assert q._word_len(trimmed) > q._response_word_limit("normal")
    assert 180 <= q._word_len(trimmed) <= 250
