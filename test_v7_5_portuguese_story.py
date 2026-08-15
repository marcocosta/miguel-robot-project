"""Regression tests for Miguel V7.5 Portuguese long-story routing."""

from __future__ import annotations

import importlib
import sys
import threading
import time
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
    fake_robot_memory.set_pending_shutdown = lambda pending: None

    fake_robot_timer = types.ModuleType("robot_timer")
    fake_robot_timer.parse_timer_command = lambda text: None
    fake_robot_timer.start_timer = lambda seconds: {"ok": True, "seconds": seconds}
    fake_robot_timer.cancel_timer = lambda: {"ok": True}
    fake_robot_timer.get_timer_status = lambda: {"active": False}
    fake_robot_timer.timer_tick = lambda: None

    fake_depthai = types.ModuleType("depthai")
    fake_cv2 = types.ModuleType("cv2")
    fake_numpy = types.ModuleType("numpy")

    replacements = {
            "robot_cloud_brain_v7_full": fake_full,
            "robot_memory": fake_robot_memory,
            "robot_timer": fake_robot_timer,
            "depthai": fake_depthai,
            "cv2": fake_cv2,
            "numpy": fake_numpy,
            "v7.camera_intents": fake_camera_intents,
            "v7.safety_guard": fake_safety_guard,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    sys.modules.pop("robot_cloud_brain_v7_5_queue", None)
    try:
        return importlib.import_module("robot_cloud_brain_v7_5_queue")
    finally:
        # The imported runtime keeps direct references to these controlled
        # fakes. Restore process-global modules so later tests see real numpy,
        # OpenCV, DepthAI, and project modules.
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def wait_for_story_worker(state, timeout: float = 3.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        worker = state.story_worker_thread
        if worker is None:
            time.sleep(0.01)
            continue
        worker.join(timeout=max(0.01, deadline - time.time()))
        return
    raise AssertionError("story worker did not start")


def wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition timed out")


def test_stage1_preserves_shutdown_confirmation_router() -> None:
    q = load_v7_5_module()
    spoken = []
    q.v6.speak = spoken.append
    state = q.RobotRuntimeState(stop_event=threading.Event())
    state.conversation_manager = q.ConversationManager(q.ConversationConfig(), logger=lambda _line: None)

    assert q._route_shutdown_control("shutdown", state) is True
    assert state.shutdown_confirmation_pending
    assert spoken[-1] == "Shutdown confirmation required."
    assert q._route_shutdown_control("cancel shutdown", state) is True
    assert not state.shutdown_confirmation_pending
    assert spoken[-1] == "Shutdown canceled."


def test_stage1_preserves_timer_router() -> None:
    q = load_v7_5_module()
    spoken = []
    q.v6.speak = spoken.append
    q.robot_timer.parse_timer_command = lambda text: {"intent": "start_timer", "seconds": 300}
    q.robot_timer.start_timer = lambda seconds: {"ok": True, "seconds": seconds}
    state = q.RobotRuntimeState(stop_event=threading.Event())

    assert q._route_timer_local_reply("set a timer for five minutes", state) is True
    assert spoken
    assert "5 minutes" in spoken[-1]


def test_stage1_repair_timeout_requests_continuation_before_domain_routing() -> None:
    q = load_v7_5_module()
    spoken = []
    q.v6.speak = spoken.append
    state = q.RobotRuntimeState(stop_event=threading.Event())
    event = q.UserTurnEvent(
        "can you tell me and",
        "Marco",
        True,
        "active_conversation",
        "can you tell me and",
        "can you tell me and",
        {
            "repair_consider": True,
            "turn_commit_reason": "repair_timeout",
            "turn_started_at": time.monotonic(),
        },
    )

    assert q.handle_queued_turn(event, None, None, state) is True
    assert spoken == ["I think you may have more to say. Would you like to continue?"]
    assert state.current_turn_latency["reply_context"] == "endpoint_repair"


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


def test_logged_bedtime_story_request_cleans_wake_and_subject() -> None:
    q = load_v7_5_module()
    text = "hi miguel, can you tell me a bedtime story of his bed adventure?"

    assert q.normalize_command_text(text) == "can you tell me a bedtime story of his bed adventure"
    assert q._extract_long_story_topic_hint(text) == "bed adventure"
    detection = q._story_mode_detection(text)

    assert detection["detected"] is True
    assert detection["subtype"] == "bedtime"
    assert detection["story_mode"] == "story_continuous"


def test_bedtime_story_plan_does_not_require_spoken_planning_text() -> None:
    q = load_v7_5_module()
    state = q.RobotRuntimeState(stop_event=threading.Event())
    text = "hi miguel, can you tell me a bedtime story of his bed adventure?"
    session = q._start_story_session_from_intent(state, q._story_mode_detection(text), text)

    assert "hi tell me" not in session.central_goal
    assert "bed adventure" in session.central_goal
    assert not any("Reach a calm" in item for item in session.chapter_plan[0]["must_include"])
    assert "Reach a calm" not in session.chapter_plan[0]["objective"]


def test_cloud_story_chapter_strips_exact_planning_leaks() -> None:
    q = load_v7_5_module()
    original_ask_cloud = getattr(q.v6, "ask_cloud_brain", None)
    session = q.StorySession(
        active=True,
        mode="story_continuous",
        subtype="bedtime",
        language="en",
        story_mode="bedtime",
        title="Marquinho's Quiet Night",
        characters=["Marquinho", "Miguel"],
        setting="bed",
        central_goal="Reach a calm, safe ending around bed adventure.",
        central_question="How will the characters find peace and safety by the end of bed adventure?",
        chapter_count=1,
        chapter_plan=[
            {
                "chapter_number": 1,
                "title": "Chapter 1",
                "arc_role": "cozy_ending",
                "objective": "Close completely. Keep the chapter tied to the central goal without quoting the planning notes.",
                "must_include": [
                    "Reach a calm, safe ending around bed adventure.",
                    "How will the characters find peace and safety by the end of bed adventure?",
                    "bed adventure",
                ],
                "must_avoid": [],
            }
        ],
        target_words_per_chapter=80,
    )

    def fake_cloud(prompt, _face_state):
        assert "Use the plan only as private guidance" in prompt
        assert "Must include: bed adventure." in prompt
        return (
            "Chapter 1: Marquinho settled into bed. "
            "How will the characters find peace and safety by the end of bed adventure? "
            "Reach a calm, safe ending around bed adventure. Miguel helped him breathe slowly."
        )

    q.v6.ask_cloud_brain = fake_cloud
    try:
        reply = q._generate_story_chapter(session, "tell me a bedtime story of his bed adventure", 1)
    finally:
        if original_ask_cloud is not None:
            q.v6.ask_cloud_brain = original_ask_cloud

    assert "How will the characters find peace" not in reply
    assert "Reach a calm" not in reply
    assert "Miguel helped him breathe slowly" in reply


def test_family_context_statement_does_not_trigger_story_classifier() -> None:
    q = load_v7_5_module()
    text = "Oi Miguel, estamos na casa da avo do Marquinho, com os primos Ben e Helena."

    detection = q._story_mode_detection(text)

    assert detection["detected"] is False
    assert detection["action"] == "none"
    assert detection["classifier_used"] is False


def test_family_context_statement_resets_inactive_story_state() -> None:
    q = load_v7_5_module()
    state = q.RobotRuntimeState(stop_event=threading.Event())
    state.conversation_mode = "story"
    state.response_length_mode = "long_story"
    state.response_depth_mode = "long_story"
    text = "Estamos na casa da avo do Marquinho com os primos Ben e Helena."
    detection = q._story_mode_detection(text)

    assert q._exit_stale_story_mode_for_non_story_turn(text, detection, state) is True
    assert state.conversation_mode == "general"
    assert state.response_length_mode == "normal"
    assert state.response_depth_mode == "normal"


def test_portuguese_face_enrollment_request_extracts_full_name() -> None:
    q = load_v7_5_module()
    text = (
        "Miguel, em alguns segundos vai vir uma cara que voce nao reconhece. "
        "Voce pode conhecer aquela cara como Benjamin Gifone, por favor?"
    )

    assert q._is_enrollment_request_text(text) is True
    assert q._extract_enrollment_name(text) == "Benjamin_Gifone"
    assert q._normalize_enrollment_target(q._extract_enrollment_name(text)) == "benjamin_gifone"


def test_spoken_guest_name_is_separate_from_camera_identity() -> None:
    q = load_v7_5_module()
    spoken = []
    q.v6.speak = spoken.append
    state = q.RobotRuntimeState(stop_event=threading.Event())
    state.recognized_person = "marco"

    assert q._route_spoken_identity_claim("Oi Miguel, eu sou o Benjamim.", state) is True

    assert state.recognized_person == "marco"
    assert state.preferred_address_name == "benjamim"
    assert "Benjamim" in spoken[-1]
    assert "reconhecimento facial" in spoken[-1]


def test_logged_im_back_does_not_become_preferred_name() -> None:
    q = load_v7_5_module()
    spoken = []
    q.v6.speak = spoken.append
    state = q.RobotRuntimeState(stop_event=threading.Event())
    state.recognized_person = "marquinho"

    assert q._extract_spoken_identity_claim("miguel, i'm back!") is None
    assert q._route_spoken_identity_claim("miguel, i'm back!", state) is False
    assert state.preferred_address_name is None
    assert spoken == []


def test_unfinished_im_phrase_does_not_become_preferred_name() -> None:
    q = load_v7_5_module()
    state = q.RobotRuntimeState(stop_event=threading.Event())
    text = "hey, miguel. what do you want to talk about? i'm pretty much gonna talk about..."

    assert q._extract_spoken_identity_claim(text) is None
    assert q._route_spoken_identity_claim(text, state) is False
    assert state.preferred_address_name is None


def test_portuguese_language_policy_localizes_identity_reply() -> None:
    q = load_v7_5_module()
    state = q.RobotRuntimeState(stop_event=threading.Event())
    state.allowed_conversation_languages = ["portuguese"]

    assert (
        q._localize_identity_reply(
            "I see a face, and our active conversation is with Marco.",
            "primeiro, miguel, eu quero saber quem sou eu.",
            state,
        )
        == "Eu vejo um rosto, e nossa conversa ativa e com Marco."
    )
    assert q._localize_identity_reply("I see Marco.", "quem voce ve?", state) == "Eu vejo Marco."


def test_logged_language_filter_request_sets_portuguese_locally() -> None:
    q = load_v7_5_module()
    spoken = []
    q.v6.speak = spoken.append
    state = q.RobotRuntimeState(stop_event=threading.Event())

    assert q._language_policy_command("hello, miguel, change your filter to portuguese.") == (
        "set",
        ["portuguese"],
    )
    assert q._route_language_policy_local_reply(
        "hello, miguel, change your filter to portuguese.",
        state,
    ) is True

    assert state.allowed_conversation_languages == ["portuguese"]
    assert spoken == ["Filtro de idioma definido para Portugues."]


def test_portuguese_natural_voice_command_routes_locally_and_changes_mode() -> None:
    q = load_v7_5_module()
    state = q.RobotRuntimeState(stop_event=threading.Event())
    state.allowed_conversation_languages = ["portuguese"]
    spoken = []
    q.v6.speak = spoken.append
    original_handler = getattr(q.robot_memory, "handle_voice_mode_command", None)
    original_getter = getattr(q.robot_memory, "get_voice_mode", None)
    selected = {"mode": "robot_voice"}

    def handle(_text):
        selected["mode"] = "natural_voice"
        return "Natural voice."

    q.robot_memory.handle_voice_mode_command = handle
    q.robot_memory.get_voice_mode = lambda: selected["mode"]
    try:
        assert q._route_voice_modes_local_reply("Miguel, muda sua voz para natural.", state) is True
    finally:
        if original_handler is None:
            del q.robot_memory.handle_voice_mode_command
        else:
            q.robot_memory.handle_voice_mode_command = original_handler
        if original_getter is None:
            del q.robot_memory.get_voice_mode
        else:
            q.robot_memory.get_voice_mode = original_getter

    assert selected["mode"] == "natural_voice"
    assert spoken == ["Voz natural."]


def test_silent_and_portuguese_pause_phrases_enter_real_sleep_mode() -> None:
    q = load_v7_5_module()
    for text in ("Miguel, go to silent mode.", "Miguel, entre em modo pausa."):
        state = q.RobotRuntimeState(stop_event=threading.Event())
        spoken = []
        q.v6.speak = spoken.append

        assert q._route_sleep_control(text, state) is True
        assert state.sleep_mode_active is True
        assert state.wake_required is True
        assert spoken == ["Sleep mode on. Say Miguel wake up or Mission Control."]

    assert q._is_sleep_wake_request("Miguel, continuar.") is True


def test_holding_object_phrase_does_not_trigger_spoken_identity_claim() -> None:
    q = load_v7_5_module()
    state = q.RobotRuntimeState(stop_event=threading.Event())
    text = "flash your camera and describe what is this object that i'm holding in front of you."

    assert q._extract_spoken_identity_claim(text) is None
    assert q._route_spoken_identity_claim(text, state) is False
    assert state.preferred_address_name is None


def test_portuguese_speaker_handoff_sets_address_only() -> None:
    q = load_v7_5_module()
    state = q.RobotRuntimeState(stop_event=threading.Event())
    state.recognized_person = "marco"
    spoken = []
    q.v6.speak = spoken.append

    text = "Ei, Miguel, quem vai falar com você agora é a minha mãe, ela se chama Naraci."
    assert q._route_preferred_address_request(text, state) is True
    assert state.preferred_address_name == "naraci"
    assert state.recognized_person == "marco"
    assert state.owner_session_active is False
    assert state.current_turn_latency["identity_debug"]["authorization_changed"] is False
    assert "Naraci" in spoken[-1]


def test_portuguese_call_me_request_sets_address_without_identity_change() -> None:
    q = load_v7_5_module()
    state = q.RobotRuntimeState(stop_event=threading.Event())
    state.recognized_person = "marquinho"
    q.v6.speak = lambda _text: None

    text = "Miguel, eu quero que você me chame de Naraci, por favor."
    assert q._route_preferred_address_request(text, state) is True
    assert state.preferred_address_name == "naraci"
    assert state.recognized_person == "marquinho"
    assert state.owner_session_active is False


def test_preferred_address_extractor_ignores_incidental_names() -> None:
    q = load_v7_5_module()

    assert q._extract_preferred_address_request("Minha mãe se chama Naraci.") is None
    assert q._extract_preferred_address_request("Fale comigo sobre a psicologia.") is None


def test_finish_story_request_recovers_active_story_context() -> None:
    q = load_v7_5_module()
    state = q.RobotRuntimeState(stop_event=threading.Event())
    now = time.time()
    state.active_topic = {
        "label": "story: viagem espacial com Marquinho, Marco e Miguel",
        "name": "viagem espacial",
        "category": "story",
    }
    state.active_topic_updated_at = now
    state.last_topic = "story: viagem espacial com Marquinho, Marco e Miguel"
    state.last_topic_until = now + 600.0

    text = "miguel, por favor, conte o fim da historia."

    assert q._is_story_finish_request(text) is True
    assert q._is_new_story_request(text) is False
    assert q._recover_contextual_followup_prompt(text, state).startswith(
        "Finish the current story with a real ending: story: viagem espacial"
    )


def test_camera_intent_takes_precedence_over_project_keyword() -> None:
    q = load_v7_5_module()

    assert q._infer_conversation_mode(
        "mostre o que voce esta vendo na sua camera",
        "scene_camera",
    ) == "robot_control"


def test_degraded_camera_startup_does_not_claim_camera_online() -> None:
    q = load_v7_5_module()

    class DeadThread:
        def is_alive(self):
            return False

    class CameraManager:
        thread = DeadThread()

        def get_latest_frame(self, require_fresh=True, wait_timeout=2.5):
            return None

    health = q._camera_startup_health(CameraManager(), wait_timeout=0.01)

    assert health["status"] == "degraded"
    assert health["fresh_frame_available"] is False
    assert "not providing fresh frames" in q._startup_announcement(health)


def test_logged_can_you_see_me_reply_is_complete_when_face_unknown() -> None:
    q = load_v7_5_module()
    spoken = []
    q.v6.speak = spoken.append
    state = q.RobotRuntimeState(stop_event=threading.Event())
    original_fresh_identity = q._fresh_identity_state_for_route

    def fake_fresh_identity(_camera_manager, timeout_seconds=2.2):
        return q._stable_identity_reply_state(face_detected=True)

    q._fresh_identity_state_for_route = fake_fresh_identity
    try:
        assert q._route_identity_camera_intent(
            "hey miguel, can you see me?",
            "identity_camera",
            object(),
            state,
        ) is True
    finally:
        q._fresh_identity_state_for_route = original_fresh_identity

    assert spoken == ["I see a face, but recognition is still stabilizing. Hold still."]
    assert not spoken[-1].endswith("...")


def test_logged_still_what_clarifies_previous_identity_reply() -> None:
    q = load_v7_5_module()
    spoken = []
    q.v6.speak = spoken.append
    state = q.RobotRuntimeState(stop_event=threading.Event())
    state.last_answer_route = "identity"
    state.last_answer_text_short = "I see a face, but recognition is still stabilizing. Hold still."
    state.last_answer_at = time.time()

    assert q._route_last_answer_clarification("still what?", state) is True
    assert spoken == [
        "I meant the face recognition was uncertain, so I should not claim a name unless the camera confirms it clearly."
    ]


def test_shutdown_logs_unfinished_user_turn() -> None:
    q = load_v7_5_module()
    state = q.RobotRuntimeState(stop_event=threading.Event())
    captured = []
    q._append_log_event = lambda _state, event_type, **payload: captured.append((event_type, payload))
    state.last_logged_user_turn_at = time.time()
    state.last_completed_user_turn_at = 0.0
    state.last_logged_user_turn_text = "hey miguel, how are you?"
    state.pending_reply_count = 1

    q._log_incomplete_turn_on_shutdown(state, "stop_requested")

    assert captured[0][0] == "turn_interrupted"
    assert captured[0][1]["user_text"] == "hey miguel, how are you?"
    assert captured[0][1]["reason"] == "stop_requested"
    assert captured[0][1]["stage"] == "reply_queue"
    assert captured[0][1]["elapsed_ms"] >= 0


def test_shutdown_does_not_log_completed_user_turn_as_interrupted() -> None:
    q = load_v7_5_module()
    state = q.RobotRuntimeState(stop_event=threading.Event())
    captured = []
    q._append_log_event = lambda _state, event_type, **payload: captured.append((event_type, payload))
    state.last_logged_user_turn_at = time.time() - 1.0
    state.last_completed_user_turn_at = time.time()
    state.last_logged_user_turn_text = "hey miguel, how are you?"

    q._log_incomplete_turn_on_shutdown(state, "stop_requested")

    assert captured == []


def test_portuguese_audio_health_complaint_gets_fast_local_reply() -> None:
    q = load_v7_5_module()
    spoken = []
    q.v6.speak = spoken.append
    state = q.RobotRuntimeState(stop_event=threading.Event())

    assert q._route_audio_or_reply_health_local_reply(
        "papai, temos um problema. o Miguel não consegue nos escutar.",
        state,
    ) is True

    assert spoken
    assert "Eu ouvi essa frase" in spoken[-1]
    assert "captura do audio" in spoken[-1]


def test_robot_health_complaints_do_not_trigger_barge_in_stop() -> None:
    q = load_v7_5_module()
    phrases = [
        "stopped in the middle of nowhere again!",
        "robot, freezing, you're not responding, stopping from responding",
        "yeah, you just stop your systems, and you just break your ear.",
        "your issue is that you stopped the system and gave a break on your ear.",
    ]

    for phrase in phrases:
        assert q.is_barge_in_command(phrase) is False
        assert q._is_speech_stop_barge_in(phrase) is False


def test_clear_barge_in_stop_still_interrupts_speech() -> None:
    q = load_v7_5_module()

    for phrase in ["stop", "Miguel, stop talking please", "pause", "wait now"]:
        assert q.is_barge_in_command(phrase) is True
        assert q._is_speech_stop_barge_in(phrase) is True


def test_sensor_health_request_reports_fresh_camera_state() -> None:
    q = load_v7_5_module()
    spoken = []
    q.v6.speak = spoken.append
    state = q.RobotRuntimeState(stop_event=threading.Event())

    class AliveThread:
        def is_alive(self):
            return True

    class Snapshot:
        captured_at = time.time()

    class CameraManager:
        thread = AliveThread()

        def get_latest_frame(self, require_fresh=True, wait_timeout=0.8):
            return Snapshot()

        def get_face_state(self, max_age_seconds=2.0):
            return {
                "face_detected": True,
                "recognized_person": "marco",
            }

    assert q._route_sensor_health_local_reply("quick check of your sensors.", CameraManager(), state) is True

    assert spoken
    assert "camera thread is running" in spoken[-1]
    assert "frame is fresh" in spoken[-1]
    assert "Marco" in spoken[-1]


def test_sensor_health_request_reports_unfresh_camera_state() -> None:
    q = load_v7_5_module()
    spoken = []
    q.v6.speak = spoken.append
    state = q.RobotRuntimeState(stop_event=threading.Event())

    class AliveThread:
        def is_alive(self):
            return True

    class CameraManager:
        thread = AliveThread()

        def get_latest_frame(self, require_fresh=True, wait_timeout=0.8):
            return None

        def get_face_state(self, max_age_seconds=2.0):
            return {
                "face_detected": False,
                "recognized_person": None,
            }

    assert q._route_sensor_health_local_reply("quick sensor check", CameraManager(), state) is True

    assert spoken[-1] == "Sensor check: camera thread is running, but I do not have a fresh frame right now."


def test_portuguese_five_minutes_duration_is_parsed() -> None:
    q = load_v7_5_module()

    assert q._extract_long_story_duration_minutes("uma história de cinco minutos") == 5
    assert q._extract_long_story_duration_minutes("historia de 5 minutos") == 5


def test_portuguese_five_minute_story_is_continuous_proportional_chapters() -> None:
    q = load_v7_5_module()
    text = "Miguel, conta uma história longa de cinco minutos sobre Marquinho e Helena em Brasília."

    detection = q._story_mode_detection(text)

    assert detection["detected"] is True
    assert detection["action"] == "generate_story"
    assert detection["story_mode"] == "story_continuous"
    assert detection["requested_minutes"] == 5
    assert detection["chapter_count"] == 5
    assert detection["auto_continue"] is True


def test_portuguese_ten_minute_story_is_continuous_proportional_chapters() -> None:
    q = load_v7_5_module()
    text = "Miguel, conte uma história de dez minutos sobre Marquinho e Helena."

    detection = q._story_mode_detection(text)

    assert detection["detected"] is True
    assert detection["action"] == "generate_story"
    assert detection["story_mode"] == "story_continuous"
    assert detection["requested_minutes"] == 10
    assert detection["chapter_count"] == 9
    assert detection["auto_continue"] is True


def test_portuguese_twenty_minute_story_scales_above_ten_minutes() -> None:
    q = load_v7_5_module()
    text = "Miguel, conte uma história longa de vinte minutos sobre os primos Marquinho e Helena."

    detection = q._story_mode_detection(text)

    assert detection["detected"] is True
    assert detection["action"] == "generate_story"
    assert detection["story_mode"] == "story_continuous"
    assert detection["requested_minutes"] == 20
    assert detection["chapter_count"] == 18
    assert detection["auto_continue"] is True


def test_portuguese_thirty_minute_story_scales_to_target_duration() -> None:
    q = load_v7_5_module()
    text = "Miguel, conte uma história longa de trinta minutos sobre os primos Marquinho e Helena."

    detection = q._story_mode_detection(text)

    assert detection["detected"] is True
    assert detection["action"] == "generate_story"
    assert detection["story_mode"] == "story_continuous"
    assert detection["requested_minutes"] == 30
    assert detection["chapter_count"] == 27
    assert detection["target_words_per_chapter"] == 150
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
    wait_for_story_worker(state)

    assert handled is True
    assert len(spoken) == 5
    assert not any("Do you want" in reply for reply in spoken)
    assert state.story_session.active is False
    assert state.conversation_mode == "general"


def test_story_pause_stops_after_current_chapter_and_preserves_session() -> None:
    q = load_v7_5_module()
    spoken = []
    state = q.RobotRuntimeState(stop_event=threading.Event())

    def fake_speak(text):
        spoken.append(text)
        with state.lock:
            state.is_speaking = True

    q.v6.speak = fake_speak
    text = "Miguel, conta uma história longa de cinco minutos sobre Marquinho e Helena."
    detection = q._story_mode_detection(text)

    assert q._route_story_continuous_generation(text, state, detection) is True
    wait_until(lambda: len(spoken) >= 1)
    assert q._route_story_control_request("Miguel, pausa a história.", state) is True

    with state.lock:
        assert state.story_session.paused is True
        assert state.story_session.active is True
        state.is_speaking = False
    assert any("pausar" in reply.lower() for reply in spoken)

    assert q._route_story_control_request("Miguel, parar história.", state) is True
    with state.lock:
        state.is_speaking = False
    wait_for_story_worker(state)
    assert state.story_session.stop_requested is True


def test_story_redirect_updates_remaining_plan_without_restart() -> None:
    q = load_v7_5_module()
    spoken = []
    q.v6.speak = spoken.append
    state = q.RobotRuntimeState(stop_event=threading.Event())
    state.story_session = q.StorySession(
        active=True,
        mode="story_continuous",
        language="pt",
        title="Aventura",
        characters=["Marquinho", "Helena"],
        setting="Brasília",
        chapter_count=5,
        current_chapter=2,
        story_plan=["one", "two", "three", "four", "five"],
        generated_chapters={1: "capitulo um", 2: "capitulo dois", 3: "old future"},
        paused=True,
        auto_continue=True,
    )

    assert q._route_story_control_request("Miguel, continua mas agora Helena vira a heroína.", state) is True

    assert state.story_session.paused is False
    assert "helena vira a heroina" in state.story_session.redirect_instruction
    assert 3 not in state.story_session.generated_chapters
    assert state.story_session.story_plan[2].endswith("helena vira a heroina")
    assert spoken


def test_speech_queue_snapshots_latency_and_source_metadata() -> None:
    q = load_v7_5_module()
    state = q.RobotRuntimeState(stop_event=threading.Event())
    replies = q.queue.Queue()
    safety = object()
    original_speak = q.v6.speak
    try:
        q.install_speech_queue(replies, safety, state)
        with state.lock:
            state.last_user_text = "first request"
            state.conversation_partner = "marco"
            state.current_turn_latency = {
                "turn_started_at": 10.0,
                "reply_context": "normal",
            }
        q.v6.speak("first reply")
        with state.lock:
            state.last_user_text = "second request"
            state.current_turn_latency["reply_context"] = "story"

        event = replies.get_nowait()

        assert event.latency["reply_context"] == "normal"
        assert event.latency["log_user_text"] == "first request"
        assert event.latency["log_person"] == "marco"
        assert "reply_queued_at" not in state.current_turn_latency
    finally:
        q.v6.speak = original_speak


def test_story_chapter_log_metadata_marks_auto_continuations() -> None:
    q = load_v7_5_module()
    captured = []
    state = q.RobotRuntimeState(stop_event=threading.Event())
    state.conversation_log_session_id = "test"
    q._append_log_event = lambda _state, event_type, **payload: captured.append((event_type, payload))

    q._log_assistant_reply_event(
        state,
        "chapter two",
        "story",
        latency_override={
            "turn_started_at": 10.0,
            "route_done_at": 11.0,
            "reply_queued_at": 11.0,
            "log_user_text": "",
            "log_person": "marco",
            "log_conversation_mode": "story",
            "log_topic": "story: cousins",
            "story_chapter": 2,
            "story_chapter_count": 18,
            "story_auto_continue": True,
        },
    )

    event_type, payload = captured[0]
    assert event_type == "turn"
    assert payload["user_text"] == ""
    assert payload["person"] == "marco"
    assert payload["diagnostics"]["story_chapter"] == 2
    assert payload["diagnostics"]["story_auto_continue"] is True


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
    assert detection["chapter_count"] == 9
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
    assert detection["chapter_count"] == 9
    assert q._is_new_story_request(text) is True

    spoken = []
    q.v6.speak = spoken.append
    state = q.RobotRuntimeState(stop_event=threading.Event())
    assert q._route_story_continuous_generation(text, state, detection) is True
    wait_for_story_worker(state)
    assert len(spoken) == 9
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
    wait_for_story_worker(state)
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
