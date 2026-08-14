"""
Miguel Robot Cloud Brain V7.5 Queue Core

Experimental queue-based runner. V7 Full remains the fallback.

Architecture:
- CameraManager owns the OAK queue.
- FaceWorker remains the V7 Full worker.
- AudioWorker captures transcript events.
- BrainWorker routes transcript events through local/camera/safety/cloud paths.
- SpeechWorker owns actual TTS and speaks replies sequentially.
"""

import queue
import os
import difflib
import json
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import robot_cloud_brain_v7_full as full
from miguel_teacher import OpenAICloudTeacherClient, TeacherModeController
import robot_timer

import robot_memory
from v7.camera_intents import classify_camera_intent, is_identity_camera_request, is_scene_camera_request
from v7.safety_guard import SafetyGuard


v6 = full.v6


def _make_teacher_controller() -> TeacherModeController:
    responses_client = getattr(getattr(v6, "client", None), "responses", None)
    if responses_client:
        return TeacherModeController(
            cloud_client=OpenAICloudTeacherClient(
                responses_client,
                model=getattr(v6, "OPENAI_MODEL", "gpt-4o-mini"),
            )
        )
    return TeacherModeController()


class IdentityTracker:
    def __init__(self, max_observations: int = 20):
        self.observations = deque(maxlen=max_observations)
        self.lock = threading.Lock()

    def update(self, face_state: dict) -> None:
        state = dict(face_state or {})
        observation = {
            "timestamp": time.time(),
            "face_detected": bool(state.get("face_detected")),
            "face_count": state.get("face_count"),
            "recognized_person": state.get("recognized_person"),
            "recognition_score": state.get("recognition_score"),
            "recognition_margin": state.get("recognition_margin"),
            "recognition_votes": state.get("recognition_votes"),
            "recognition_scores": state.get("recognition_scores"),
            "face_position": state.get("face_position"),
            "source": state.get("source") or state.get("recognizer"),
        }
        with self.lock:
            self.observations.append(observation)

    def _recent_observations(self, max_age_seconds: float) -> list[dict]:
        now = time.time()
        with self.lock:
            return [
                dict(obs)
                for obs in self.observations
                if now - float(obs.get("timestamp", 0.0) or 0.0) <= max_age_seconds
            ]

    def _best_face_detected_state(self, observations: list[dict]) -> dict | None:
        detected = [obs for obs in observations if obs.get("face_detected")]
        if not detected:
            return None

        best = max(detected, key=lambda obs: float(obs.get("recognition_score") or -1.0))
        return self._state_from_observation(best, recognized_person=None)

    def _state_from_observation(self, observation: dict, recognized_person=None, score=None, margin=None) -> dict:
        state = dict(observation)
        state["updated_at"] = state.pop("timestamp", time.time())
        state["age"] = time.time() - float(state["updated_at"])
        state["recognized_person"] = recognized_person
        if score is not None:
            state["recognition_score"] = score
        if margin is not None:
            state["recognition_margin"] = margin
        state["source"] = "v7_5_identity_tracker"
        return state

    def _stable_candidate(
        self,
        observations: list[dict],
        allowed_names: set[str] | None,
        min_votes: int,
        min_avg_score: float,
        min_avg_margin: float,
    ) -> tuple[str | None, int, float, float, dict | None]:
        by_person: dict[str, list[dict]] = {}
        for obs in observations:
            person = _normalize_person_name(obs.get("recognized_person"))
            if not person:
                continue
            if allowed_names is not None and person not in allowed_names:
                continue
            by_person.setdefault(person, []).append(obs)

        best_name = None
        best_votes = 0
        best_score = 0.0
        best_margin = 0.0
        best_obs = None

        for person, person_observations in by_person.items():
            votes = len(person_observations)
            avg_score = sum(float(obs.get("recognition_score") or 0.0) for obs in person_observations) / votes
            avg_margin = sum(float(obs.get("recognition_margin") or 0.0) for obs in person_observations) / votes
            candidate_obs = max(person_observations, key=lambda obs: float(obs.get("recognition_score") or 0.0))
            candidate_rank = (votes, avg_score, avg_margin)
            best_rank = (best_votes, best_score, best_margin)
            if candidate_rank > best_rank:
                best_name = person
                best_votes = votes
                best_score = avg_score
                best_margin = avg_margin
                best_obs = candidate_obs

        if best_name:
            print(
                "[V7.5 IDENTITY] "
                f"stable candidate={best_name} votes={best_votes} "
                f"avg_score={best_score:.2f} avg_margin={best_margin:.2f}"
            )

        if (
            best_name
            and best_votes >= min_votes
            and best_score >= min_avg_score
            and best_margin >= min_avg_margin
        ):
            return best_name, best_votes, best_score, best_margin, best_obs

        return None, best_votes, best_score, best_margin, best_obs

    def get_stable_identity(self, max_age_seconds: float = 3.0) -> dict | None:
        observations = self._recent_observations(max_age_seconds)
        if not observations:
            return None

        name, votes, avg_score, avg_margin, obs = self._stable_candidate(
            observations,
            allowed_names=None,
            min_votes=1,
            min_avg_score=0.0,
            min_avg_margin=0.0,
        )
        if name and obs and _identity_candidate_accepted(name, votes, avg_score, avg_margin):
            return self._state_from_observation(obs, recognized_person=name, score=avg_score, margin=avg_margin)

        return self._best_face_detected_state(observations)

    def get_owner_authorization_identity(self, max_age_seconds: float = 3.0) -> dict | None:
        observations = self._recent_observations(max_age_seconds)
        if not observations:
            print("[V7.5 IDENTITY] owner authorization=None avg_score=0.00")
            return None

        name, votes, avg_score, avg_margin, obs = self._stable_candidate(
            observations,
            allowed_names={"marco", "marquinho"},
            min_votes=1,
            min_avg_score=0.0,
            min_avg_margin=0.0,
        )
        accepted_name = name if _identity_candidate_accepted(name, votes, avg_score, avg_margin, owner_context=True) else None
        print(f"[V7.5 IDENTITY] owner authorization={accepted_name} avg_score={avg_score:.2f}")
        if accepted_name and obs:
            return self._state_from_observation(obs, recognized_person=name, score=avg_score, margin=avg_margin)

        return None

    def get_reply_candidate(self, max_age_seconds: float = 3.0) -> tuple[str | None, int, float, float, dict | None]:
        observations = self._recent_observations(max_age_seconds)
        if not observations:
            return None, 0, 0.0, 0.0, None

        return self._stable_candidate(
            observations,
            allowed_names=None,
            min_votes=1,
            min_avg_score=0.0,
            min_avg_margin=0.0,
        )

    def has_recent_face_detected(self, max_age_seconds: float = 3.0) -> bool:
        return any(obs.get("face_detected") for obs in self._recent_observations(max_age_seconds))


@dataclass
class StorySession:
    active: bool = False
    mode: str = ""
    subtype: str = "general"
    language: str = "en"
    story_mode: str = "adventure"
    planner: str = "local"
    fictionality: str = "fictional"
    title: str = ""
    characters: list[str] = field(default_factory=list)
    setting: str = ""
    central_goal: str = ""
    central_question: str = ""
    emotional_tone: str = ""
    message: str = ""
    chapter_count: int = 0
    current_chapter: int = 0
    story_plan: list[str] = field(default_factory=list)
    chapter_plan: list[dict] = field(default_factory=list)
    arc_template: str = ""
    generated_chapters: dict[int, str] = field(default_factory=dict)
    stop_requested: bool = False
    paused: bool = False
    redirect_instruction: str = ""
    target_words_per_chapter: int = 150
    auto_continue: bool = False


@dataclass
class RobotRuntimeState:
    stop_event: threading.Event
    stop_speech_event: threading.Event = field(default_factory=threading.Event)
    shutdown_acknowledged_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    identity_tracker: IdentityTracker = field(default_factory=IdentityTracker)
    interaction_state: str = "starting"
    current_status_text: str = ""
    last_state_change_at: float = field(default_factory=time.time)
    last_state_emit_at: float = field(default_factory=time.time)
    last_state_emit_state: str = "starting"
    last_state_emit_text: str = ""
    last_state_emit_recognition_key: str = "none"
    last_state_emit_key: tuple[str, str] = field(default_factory=lambda: ("starting", ""))
    last_face_status: str = ""
    last_face_status_text: str = ""
    last_face_status_at: float = 0.0
    last_face_status_key: tuple[str, str] = field(default_factory=lambda: ("", ""))
    last_face_block_log_at: float = 0.0
    last_unknown_face_visual_at: float = 0.0
    last_face_recognition_key: str = "none"
    face_expression: str = "normal"
    face_expression_source: str = "default"
    last_user_text: str = ""
    previous_user_text: str = ""
    last_non_self_heard_user_text: str = ""
    last_robot_text: str = ""
    last_listen_started_at: float = 0.0
    last_heard_at: float = 0.0
    last_thinking_started_at: float = 0.0
    last_speaking_started_at: float = 0.0
    current_turn_started_at: float = 0.0
    current_turn_latency: dict = field(default_factory=dict)
    pending_reply_count: int = 0
    pending_user_turn_count: int = 0
    turn_processing_active: bool = False
    turn_processing_started_at: float = 0.0
    audio_capture_active: bool = False
    audio_capture_started_at: float = 0.0
    audio_capture_last_heartbeat_at: float = 0.0
    audio_capture_blocked_reason: str | None = None
    last_audio_capture_finished_at: float = 0.0
    last_audio_capture_timing: dict = field(default_factory=dict)
    reply_queue: queue.Queue | None = None
    user_turn_queue: queue.Queue | None = None
    brain_is_processing: bool = False
    last_reply_time: float = 0.0
    conversation_grace_seconds: float = 10.0
    conversation_active: bool = False
    conversation_mode: str = "wake_required"
    conversation_partner: str | None = None
    preferred_address_name: str | None = None
    preferred_address_until: float = 0.0
    conversation_until: float = 0.0
    conversation_started_at: float = 0.0
    last_conversation_activity_at: float = 0.0
    last_robot_question_at: float = 0.0
    last_robot_question_type: str | None = None
    last_user_directed_to_robot: bool = False
    wake_required: bool = True
    wake_required_reason: str = "startup"
    owner_session_active: bool = False
    owner_session_person: str | None = None
    owner_session_until: float = 0.0
    last_owner_session_log_at: float = 0.0
    last_owner_session_logged_person: str | None = None
    password_session_active: bool = False
    password_session_until: float = 0.0
    pending_owner_unlock_until: float = 0.0
    password_env_logged: bool = False
    session_topic: str | None = None
    session_focus: str | None = None
    last_robot_question_text: str = ""
    last_robot_question_expected_slot: str | None = None
    pending_times_table: tuple[int, int, int] | None = None
    last_topic: str | None = None
    last_topic_until: float = 0.0
    recent_conversation_turns: list[str] = field(default_factory=list)
    active_topic: dict = field(default_factory=dict)
    active_topic_updated_at: float = 0.0
    last_user_creative_subject: str | None = None
    last_interrupted_user_topic: dict | None = None
    project_role_discussed_at: float = 0.0
    last_answer_topic: str | None = None
    last_answer_route: str | None = None
    last_answer_text_short: str = ""
    last_answer_at: float = 0.0
    last_logged_user_turn_at: float = 0.0
    last_logged_user_turn_text: str = ""
    last_completed_user_turn_at: float = 0.0
    last_conversation_extend_log_at: float = 0.0
    response_length_mode: str = field(
        default_factory=lambda: os.getenv("MIGUEL_DEFAULT_RESPONSE_LENGTH_MODE", "normal").strip().lower()
        if os.getenv("MIGUEL_DEFAULT_RESPONSE_LENGTH_MODE", "normal").strip().lower() in {"terse", "normal", "detailed", "story", "long_story"}
        else "normal"
    )
    response_depth_mode: str = "normal"
    allowed_conversation_languages: list[str] = field(default_factory=lambda: _default_allowed_conversation_languages())
    long_story_active: bool = False
    long_story_topic: str | None = None
    long_story_segment_index: int = 0
    long_story_max_segments: int = 0
    long_story_target_minutes: int = 0
    long_story_style: str = ""
    recovered_story_context: str = ""
    story_session: StorySession = field(default_factory=StorySession)
    story_worker_thread: threading.Thread | None = None
    recent_story_modes_used: list[str] = field(default_factory=list)
    teacher_controller: TeacherModeController = field(default_factory=_make_teacher_controller)
    last_prompt_type: str | None = None
    last_prompt_text: str | None = None
    last_joke_punchline: str | None = None
    current_mode: str = "normal"
    sleep_mode_active: bool = False
    sleep_mode_until: float = 0.0
    shutdown_pending: bool = False
    shutdown_confirmation_pending: bool = False
    shutdown_confirmation_until: float = 0.0
    debug_handoff_requested: bool = False
    face_detected: bool = False
    face_count: int = 0
    known_person_present: bool = False
    recognized_person: str | None = None
    recognized_person_updated_at: float = 0.0
    is_speaking: bool = False
    last_speech_started_at: float = 0.0
    last_speech_finished_at: float = 0.0
    last_spoken_text: str = ""
    enrollment_state: str = "idle"
    enrollment_target_name: str | None = None
    enrollment_approved_by: str | None = None
    enrollment_approved_at: float = 0.0
    last_weather_temp_f: float | None = None
    last_ready_cue_at: float = 0.0
    ready_cue_enabled: bool = True
    ready_cue_mode: str = field(
        default_factory=lambda: os.getenv("MIGUEL_READY_CUE_MODE", "visual").strip().lower()
        if os.getenv("MIGUEL_READY_CUE_MODE", "visual").strip().lower() in {"visual", "beep", "spoken", "off"}
        else "visual"
    )
    suppress_next_ready_cue: bool = False
    conversation_log_session_id: str | None = None


@dataclass
class UserTurnEvent:
    text: str
    recognized_person: str | None = None
    authorized: bool = False
    authorization_source: str = ""
    normalized_text: str = ""
    stripped_text: str = ""
    latency: dict = field(default_factory=dict)


@dataclass
class ReplyEvent:
    text: str
    latency: dict = field(default_factory=dict)
    context: str = "normal"


_speech_enqueue_context = threading.local()


TTS_CACHE_CANDIDATES = {
    "Here.",
    "Good.",
    "I hear you.",
    "Looking.",
    "I see Marco.",
    "I see Marquinho.",
    "Robot voice.",
    "Natural voice.",
    "Confirmed.",
}


def _state_repeat_log_interval_seconds() -> float:
    try:
        return max(0.0, float(os.getenv("MIGUEL_STATE_REPEAT_LOG_INTERVAL_SECONDS", "5.0")))
    except ValueError:
        return 5.0


def _face_status_repeat_interval_seconds() -> float:
    try:
        return max(0.0, float(os.getenv("MIGUEL_FACE_STATUS_REPEAT_INTERVAL_SECONDS", "2.0")))
    except ValueError:
        return 2.0


def _is_idle_neutral_status(status_text: str) -> bool:
    return _normalize_for_echo(status_text) in {"", "unknown face"}


def _interaction_status_key(interaction_state: str, status_text: str = "") -> str:
    if interaction_state == "idle" and _is_idle_neutral_status(status_text):
        return ""
    return _normalize_for_echo(status_text)


def _face_recognition_key(face_detected: bool, recognized_person: str | None) -> str:
    recognized = _normalize_person_name(recognized_person)
    if recognized:
        return f"known:{recognized}"
    if face_detected:
        return "unknown"
    return "none"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _response_word_limit(mode: str) -> int:
    mode = str(mode or "normal").strip().lower()
    if mode == "terse":
        return max(4, _env_int("MIGUEL_TERSE_MAX_WORDS", 18))
    if mode == "detailed":
        return max(20, _env_int("MIGUEL_DETAILED_MAX_WORDS", 110))
    if mode == "story":
        return max(30, _env_int("MIGUEL_STORY_MAX_WORDS", 160))
    if mode == "long_story":
        return _long_story_spoken_cap_words(True)
    return max(10, _env_int("MIGUEL_NORMAL_MAX_WORDS", 24))


def _long_story_segment_words() -> int:
    return max(40, _env_int("MIGUEL_LONG_STORY_SEGMENT_WORDS", 120))


def _long_story_words_per_chapter() -> int:
    return max(120, _env_int("MIGUEL_LONG_STORY_WORDS_PER_CHAPTER", 150))


def _long_story_max_segments() -> int:
    default = (_long_story_max_target_minutes() * _long_story_words_per_minute() + _long_story_words_per_chapter() - 1) // _long_story_words_per_chapter()
    return max(1, _env_int("MIGUEL_LONG_STORY_MAX_SEGMENTS", default))


def _long_story_duration_chapters(requested_minutes: int) -> int:
    target_words = _long_story_target_words(requested_minutes)
    if not target_words:
        return 0
    words_per_chapter = _long_story_words_per_chapter()
    return max(1, (target_words + words_per_chapter - 1) // words_per_chapter)


def _long_story_words_per_minute() -> int:
    return max(80, _env_int("MIGUEL_LONG_STORY_WORDS_PER_MINUTE", 135))


def _long_story_max_target_minutes() -> int:
    return max(1, _env_int("MIGUEL_LONG_STORY_MAX_TARGET_MINUTES", 30))


NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "um": 1,
    "uma": 1,
    "dois": 2,
    "duas": 2,
    "tres": 3,
    "três": 3,
    "quatro": 4,
    "cinco": 5,
    "seis": 6,
    "sete": 7,
    "oito": 8,
    "nove": 9,
    "dez": 10,
    "onze": 11,
    "doze": 12,
    "treze": 13,
    "catorze": 14,
    "quatorze": 14,
    "quinze": 15,
    "dezesseis": 16,
    "dezasseis": 16,
    "dezessete": 17,
    "dezassete": 17,
    "dezoito": 18,
    "dezenove": 19,
    "dezanove": 19,
    "vinte": 20,
    "trinta": 30,
}


def _extract_long_story_duration_minutes(text: str) -> int:
    normalized = normalize_command_text(text)
    if not normalized or not re.search(r"\b(?:minutes?|minutos?)\b", normalized):
        return 0
    match = re.search(r"\b(\d{1,2})\s*(?:-| )?\s*(?:minutes?|minutos?)\b", normalized)
    if match:
        minutes = int(match.group(1))
        return max(1, min(_long_story_max_target_minutes(), minutes))
    match = re.search(r"\b([a-z]+)\s*(?:-| )?\s*(?:minutes?|minutos?)\b", normalized)
    if not match:
        return 0
    minutes = NUMBER_WORDS.get(match.group(1), 0)
    if not minutes:
        return 0
    return max(1, min(_long_story_max_target_minutes(), minutes))


def _long_story_target_words(minutes: int) -> int:
    minutes = max(0, int(minutes or 0))
    if not minutes:
        return 0
    return max(120, minutes * _long_story_words_per_minute())


def _long_story_spoken_cap_words(explicit_request: bool = True) -> int:
    configured = _env_int("MIGUEL_LONG_STORY_MAX_WORDS", 250)
    cap = min(250, max(180, configured))
    return cap if explicit_request else min(cap, 180)


def _format_long_story_duration(minutes: int) -> str:
    minutes = int(minutes or 0)
    if minutes <= 0:
        return ""
    return f"{minutes} minute" + ("" if minutes == 1 else "s")


def _conversation_timeout_seconds(mode: str = "general") -> float:
    if mode == "owner_password":
        return _env_float("MIGUEL_PASSWORD_SESSION_TIMEOUT_SECONDS", 600.0)
    if mode in {"creative", "story", "project"}:
        return _env_float("MIGUEL_CREATIVE_CONVERSATION_TIMEOUT_SECONDS", 45.0)
    return _env_float("MIGUEL_CONVERSATION_TIMEOUT_SECONDS", 60.0)


def _owner_session_timeout_seconds() -> float:
    return _env_float("MIGUEL_OWNER_SESSION_TIMEOUT_SECONDS", 120.0)


def _wake_required_face_text() -> str:
    return os.getenv("MIGUEL_WAKE_REQUIRED_FACE_TEXT", "SAY HEY MIGUEL").strip() or "SAY HEY MIGUEL"


def _wake_required_display_text() -> str:
    return 'Say "Hey Miguel"'


def _active_conversation_face_text(state: RobotRuntimeState) -> str:
    base = os.getenv("MIGUEL_ACTIVE_CONVERSATION_FACE_TEXT", "YOUR TURN").strip() or "YOUR TURN"
    mode = getattr(state, "conversation_mode", "general") or "general"
    if mode in {"creative", "story", "project", "owner_password"}:
        label = "OWNER MODE" if mode == "owner_password" else mode.upper()
        return f"{label} {base}"
    return base


def _face_supports_status(status: str) -> bool:
    face = getattr(full, "face", None)
    supported = getattr(face, "supported_statuses", None) or getattr(face, "SUPPORTED_STATUSES", None)
    return bool(supported and status in supported)


def _ready_face_state() -> str:
    return "ready" if _face_supports_status("ready") else "idle"


def _ready_face_text(state: RobotRuntimeState) -> str:
    mode = getattr(state, "conversation_mode", "general") or "general"
    if getattr(state, "conversation_active", False) and mode in {"creative", "story", "project", "owner_password"}:
        label = "OWNER MODE" if mode == "owner_password" else mode.upper()
        return f"{label} Ready"
    return "Ready"


FACE_EXPRESSIONS = {"normal", "happy", "angry", "sad", "scared", "concerned", "motivated"}
FACE_EXPRESSION_ORDER = ("normal", "happy", "angry", "sad", "scared", "concerned", "motivated")


def _resting_face_expression(state: RobotRuntimeState) -> str:
    expression = str(getattr(state, "face_expression", "normal") or "normal").lower()
    return expression if expression in FACE_EXPRESSIONS else "normal"


def _persistent_expression_payload(
    state: RobotRuntimeState,
    status_text: str,
) -> tuple[str, str] | None:
    """Keep a selected expression through ordinary listen/think transitions."""
    expression = _resting_face_expression(state)
    if expression == "normal":
        return None
    return expression, status_text


def _face_priority(status: str, text: str = "") -> int:
    normalized_status = str(status or "").strip().lower()
    normalized_text = _normalize_for_echo(text)
    if normalized_status in {"shutdown_pending", "shutdown", "confirm"}:
        return 70
    if normalized_status == "sleeping" or normalized_text == "sleep":
        return 60
    if normalized_status == "speaking":
        return 50
    if normalized_status in {"thinking", "heard", "looking", "enrolling"}:
        return 40
    if normalized_status == "listening" or normalized_text == "your turn":
        return 30
    if normalized_status in {"ready", "idle"} and normalized_text in {"ready", "creative ready", "story ready", "project ready", "owner mode ready", ""}:
        return 20
    if normalized_status == "wake_required" or normalized_text in {"say hey miguel", "say miguel", "say hey miguel"}:
        return 10
    return 20


def _current_face_priority_locked(state: RobotRuntimeState) -> tuple[int, str]:
    if state.shutdown_pending or state.shutdown_confirmation_pending:
        return 70, "shutdown_pending"
    if state.sleep_mode_active or state.current_mode == "sleep":
        return 60, "sleeping"
    if state.is_speaking:
        return 50, "speaking"
    if state.turn_processing_active and state.interaction_state in {"heard", "thinking", "looking", "enrolling"}:
        return 40, state.interaction_state
    return 20, state.interaction_state


TERSE_ALLOWED_ROUTES = {
    "barge_in",
    "clarification",
    "enrollment",
    "greeting",
    "identity",
    "local_ack",
    "language_policy",
    "owner_password_ack",
    "repeat",
    "robot_control",
    "safety_refusal",
    "scene_prelude",
    "shutdown",
    "shutdown_cancel",
    "shutdown_confirm",
    "status",
    "time_status",
    "utility",
    "voice_command",
}

TERSE_ALLOWED_EXACT_REPLIES = {
    "confirmed.",
    "deep voice.",
    "face mode on.",
    "friendly voice.",
    "here.",
    "i hear you.",
    "looking.",
    "natural voice.",
    "owner mode on.",
    "robot voice.",
    "short mode.",
    "shutdown canceled.",
    "stopped.",
    "story voice.",
}


def _short_log_text(text: str, limit: int = 80) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    return compact[:limit]


def _current_owner_partner(state: RobotRuntimeState) -> str | None:
    with state.lock:
        if state.owner_session_active and time.time() <= float(state.owner_session_until or 0.0):
            return state.owner_session_person
        recognized = _normalize_person_name(state.recognized_person)
    return recognized if _is_owner(recognized) else None


def start_conversation_session(
    state: RobotRuntimeState,
    mode: str = "general",
    partner: str | None = None,
    timeout_seconds: float | None = None,
    reason: str = "",
) -> None:
    now = time.time()
    allowed = {
        "wake_required",
        "general",
        "creative",
        "story",
        "project",
        "robot_control",
        "enrollment",
        "owner_password",
    }
    if mode not in allowed:
        mode = "general"
    timeout = _conversation_timeout_seconds(mode) if timeout_seconds is None else float(timeout_seconds)
    resolved_partner = partner or _current_owner_partner(state) or "unknown_wake_user"
    with state.lock:
        was_active = bool(state.conversation_active)
        state.conversation_active = True
        state.conversation_mode = mode
        state.conversation_partner = partner or state.conversation_partner or resolved_partner
        if not was_active:
            state.conversation_started_at = now
        state.last_conversation_activity_at = now
        state.conversation_until = now + timeout
        state.wake_required = False
        state.wake_required_reason = ""
        capture_active = bool(state.audio_capture_active)
    print(f"[V7.14 CONVERSATION] started mode={mode} partner={partner} reason={reason}")
    if capture_active:
        notify_face_status(state, "listening", "YOUR TURN")
    else:
        notify_face_status(state, _ready_face_state(), _ready_face_text(state))


def extend_conversation_session(state: RobotRuntimeState, seconds: float | None = None, reason: str = "") -> None:
    now = time.time()
    timeout = _conversation_timeout_seconds(getattr(state, "conversation_mode", "general")) if seconds is None else float(seconds)
    should_log = False
    with state.lock:
        if not state.conversation_active:
            return
        state.last_conversation_activity_at = now
        state.conversation_until = max(float(state.conversation_until or 0.0), now + timeout)
        remaining = max(0.0, float(state.conversation_until or now) - now)
        if now - float(state.last_conversation_extend_log_at or 0.0) >= 10.0:
            state.last_conversation_extend_log_at = now
            should_log = True
        mode = state.conversation_mode
    if should_log:
        print(f"[V7.14 CONVERSATION] extended mode={mode} remaining={remaining:.1f} reason={reason}")


def _session_partner_for_mode(state: RobotRuntimeState, partner: str | None = None) -> str:
    resolved = _normalize_person_name(partner) or _current_owner_partner(state)
    with state.lock:
        resolved = resolved or _normalize_person_name(state.recognized_person) or state.conversation_partner
    return resolved or "unknown_wake_user"


def _force_active_after_mode(
    state: RobotRuntimeState,
    mode: str,
    partner: str | None = None,
    timeout_seconds: float = 120.0,
    reason: str = "mode_activation",
) -> None:
    mode = str(mode or "general").strip().lower()
    if mode not in {"general", "creative", "story", "project", "owner_password"}:
        mode = "general"
    resolved_partner = _session_partner_for_mode(state, partner)
    now = time.time()
    with state.lock:
        was_active = bool(state.conversation_active)
        state.conversation_active = True
        state.conversation_mode = mode
        state.conversation_partner = resolved_partner
        if not was_active:
            state.conversation_started_at = now
        state.last_conversation_activity_at = now
        state.conversation_until = now + float(timeout_seconds)
        state.wake_required = False
        state.wake_required_reason = ""
        remaining = max(0.0, float(state.conversation_until or now) - now)
    print(f"[V7.15 SESSION] forced_active_after_mode mode={mode} partner={resolved_partner} remaining={remaining:.1f}")


def expire_conversation_session_if_needed(state: RobotRuntimeState) -> bool:
    now = time.time()
    expired = False
    with state.lock:
        if state.conversation_active and now > float(state.conversation_until or 0.0):
            expired_mode = state.conversation_mode
            state.conversation_active = False
            state.conversation_mode = "wake_required"
            state.conversation_partner = None
            state.wake_required = True
            state.wake_required_reason = "timeout"
            if expired_mode == "owner_password":
                state.password_session_active = False
                state.password_session_until = 0.0
            expired = True
    if expired:
        print("[V7.14 CONVERSATION] expired reason=timeout")
        notify_face_status(state, "wake_required", _wake_required_face_text())
    return expired


def is_conversation_active(state: RobotRuntimeState) -> bool:
    expire_conversation_session_if_needed(state)
    with state.lock:
        return bool(state.conversation_active)


def is_wake_phrase(text: str) -> bool:
    return _has_v7_5_wake_phrase(text)


def strip_wake_phrase(text: str) -> str:
    return _strip_wake_phrase(text)


def _is_owner_session_active(state: RobotRuntimeState) -> bool:
    now = time.time()
    with state.lock:
        active = bool(state.owner_session_active and now <= float(state.owner_session_until or 0.0))
        if not active and state.owner_session_active:
            state.owner_session_active = False
            state.owner_session_person = None
    return active


def _refresh_owner_session(state: RobotRuntimeState, recognized: str | None, reason: str = "face") -> None:
    person = _normalize_person_name(recognized)
    if not _is_owner(person):
        return
    until = time.time() + _owner_session_timeout_seconds()
    now = time.time()
    should_log = False
    with state.lock:
        was_active = bool(state.owner_session_active and now <= float(state.owner_session_until or 0.0))
        state.owner_session_active = True
        state.owner_session_person = person
        state.owner_session_until = until
        if (
            not was_active
            or state.last_owner_session_logged_person != person
            or now - float(state.last_owner_session_log_at or 0.0) >= 5.0
        ):
            state.last_owner_session_log_at = now
            state.last_owner_session_logged_person = person
            should_log = True
    if should_log:
        print(f"[V7.14 OWNER SESSION] refreshed person={person} reason={reason}")


def _end_password_session(state: RobotRuntimeState) -> None:
    with state.lock:
        state.password_session_active = False
        state.password_session_until = 0.0


def _password_session_is_active(state: RobotRuntimeState) -> bool:
    now = time.time()
    with state.lock:
        active = bool(state.password_session_active and now <= float(state.password_session_until or 0.0))
        if not active and state.password_session_active:
            state.password_session_active = False
            state.password_session_until = 0.0
    return active


def _active_authorization_source(state: RobotRuntimeState) -> str:
    return "password_session" if _password_session_is_active(state) else "active_conversation"


def _has_active_conversation_locked(state: RobotRuntimeState, now: float | None = None) -> bool:
    now = now or time.time()
    grace_active = bool(
        state.last_reply_time
        and now - float(state.last_reply_time) <= float(state.conversation_grace_seconds)
    )
    return bool(
        state.is_speaking
        or _has_pending_reply_locked(state)
        or _has_pending_user_turn_locked(state)
        or state.brain_is_processing
        or state.turn_processing_active
        or grace_active
    )


def _audio_capture_max_seconds() -> float:
    return _env_float("MIGUEL_AUDIO_CAPTURE_MAX_SECONDS", 12.0)


def _audio_capture_grace_seconds() -> float:
    return _env_float("MIGUEL_AUDIO_CAPTURE_GRACE_SECONDS", 3.0)


def _mark_audio_capture_active(state: RobotRuntimeState) -> None:
    now = time.time()
    with state.lock:
        state.audio_capture_active = True
        state.audio_capture_started_at = now
        state.audio_capture_last_heartbeat_at = now
        state.audio_capture_blocked_reason = None
    print("[V7.14 AUDIO] capture_active=true")


def _mark_audio_capture_finished(state: RobotRuntimeState, reason: str) -> None:
    now = time.time()
    with state.lock:
        was_active = bool(state.audio_capture_active)
        state.audio_capture_active = False
        state.audio_capture_blocked_reason = reason
        state.last_audio_capture_finished_at = now
    if was_active:
        print(f"[V7.14 AUDIO] capture_active=false reason={reason}")


def _watchdog_audio_capture_state(state: RobotRuntimeState) -> None:
    now = time.time()
    with state.lock:
        capture_active = bool(state.audio_capture_active)
        capture_started_at = float(state.audio_capture_started_at or 0.0)
        face_listening = (
            state.last_face_status == "listening"
            and "your turn" in _normalize_for_echo(state.last_face_status_text)
        )
        shutdown_pending = bool(state.shutdown_confirmation_pending)
        sleeping = bool(state.sleep_mode_active)
        wake_required = bool(state.wake_required and not state.conversation_active)
        conversation_active = bool(state.conversation_active)
    if face_listening and not capture_active:
        print("[V7.14 AUDIO WARNING] face said listening but capture inactive; corrected.")
        if shutdown_pending:
            notify_face_status(state, "shutdown_pending", "Confirm shutdown")
        elif sleeping:
            notify_face_status(state, "sleeping", "Sleep")
        elif wake_required:
            notify_face_status(state, "wake_required", _wake_required_face_text())
        elif conversation_active:
            notify_face_status(state, _ready_face_state(), _ready_face_text(state))
        else:
            notify_face_status(state, _ready_face_state(), "Ready")
    if capture_active and capture_started_at and now - capture_started_at > _audio_capture_max_seconds() + _audio_capture_grace_seconds():
        _mark_audio_capture_finished(state, "watchdog_stuck")
        print("[V7.14 AUDIO WARNING] capture stuck; reset.")
        if shutdown_pending:
            notify_face_status(state, "shutdown_pending", "Confirm shutdown")
        elif sleeping:
            notify_face_status(state, "sleeping", "Sleep")
        elif wake_required:
            notify_face_status(state, "wake_required", _wake_required_face_text())
        elif conversation_active:
            notify_face_status(state, _ready_face_state(), _ready_face_text(state))
        else:
            notify_face_status(state, _ready_face_state(), "Ready")


def _face_status_payload(state: RobotRuntimeState, interaction_state: str, status_text: str = "") -> tuple[str, str]:
    normalized_text = _normalize_for_echo(status_text)
    with state.lock:
        audio_capture_active = bool(state.audio_capture_active)
        shutdown_confirmation_pending = bool(state.shutdown_confirmation_pending)
        sleep_mode_active = bool(state.sleep_mode_active or state.current_mode == "sleep")
        wake_required = bool(state.wake_required and not state.conversation_active)
        conversation_active = bool(state.conversation_active)
        processing_authorized_turn = bool(
            state.interaction_state in {"heard", "thinking", "looking", "speaking"}
            or state.brain_is_processing
            or state.turn_processing_active
            or state.is_speaking
            or _has_pending_user_turn_locked(state)
            or _has_pending_reply_locked(state)
        )
    if shutdown_confirmation_pending:
        face_state = "confirm" if _face_supports_status("confirm") else "shutdown_pending"
        return face_state, "Confirm shutdown"
    if sleep_mode_active:
        return "sleeping", "Sleep"
    if normalized_text == "your turn" or (interaction_state == "listening" and "your turn" in normalized_text):
        if audio_capture_active:
            return _persistent_expression_payload(state, "YOUR TURN") or ("listening", "YOUR TURN")
        print("[V7.14 AUDIO WARNING] face said listening but capture inactive; corrected.")
        if wake_required:
            face_state = "wake_required" if _face_supports_status("wake_required") else "idle"
            return face_state, _wake_required_display_text()
        if processing_authorized_turn and state.interaction_state in {"heard", "thinking", "looking"}:
            return state.interaction_state, state.current_status_text
        return _ready_face_state(), _ready_face_text(state)
    if processing_authorized_turn and interaction_state in {"idle", "ready", _ready_face_state()}:
        with state.lock:
            active_state = state.interaction_state
            active_text = state.current_status_text
        if active_state in {"heard", "thinking", "looking"}:
            return active_state, active_text
    if wake_required and processing_authorized_turn and interaction_state not in {"wake_required", "starting", "happy"}:
        return interaction_state, status_text
    if (wake_required and interaction_state in {"idle", "wake_required", "starting", "happy"}) or normalized_text in {"say hey miguel", "say miguel"}:
        face_state = "wake_required" if _face_supports_status("wake_required") else "idle"
        return face_state, _wake_required_display_text()
    if interaction_state in {"idle", "wake_required"}:
        if conversation_active:
            if audio_capture_active:
                return "listening", "YOUR TURN"
            expression = _resting_face_expression(state)
            return expression, expression.title()
        face_state = "wake_required" if _face_supports_status("wake_required") else "idle"
        return face_state, _wake_required_display_text()
    if interaction_state == "listening" and conversation_active:
        if audio_capture_active:
            return _persistent_expression_payload(state, "YOUR TURN") or ("listening", "YOUR TURN")
        expression_payload = _persistent_expression_payload(state, _ready_face_text(state))
        return expression_payload or (_ready_face_state(), _ready_face_text(state))
    if interaction_state in {"heard", "thinking"}:
        expression_payload = _persistent_expression_payload(
            state,
            status_text or interaction_state.title(),
        )
        if expression_payload:
            return expression_payload
    if interaction_state == "shutdown_pending":
        if _face_supports_status("confirm"):
            return "confirm", status_text or "Confirm shutdown"
        if _face_supports_status("shutdown"):
            return "shutdown", status_text or "Shutdown"
        return "idle", status_text or "Confirm shutdown"
    return interaction_state, status_text


def notify_face_status(state: RobotRuntimeState, interaction_state: str, status_text: str = "") -> None:
    face_state, face_text = _face_status_payload(state, interaction_state, status_text)
    now = time.time()
    important_transition = (
        face_state in {"wake_required", "listening"}
        or interaction_state in {"speaking", "shutdown_pending", "sleeping", "error"}
        or (interaction_state == "listening" and _normalize_for_echo(face_text) == "your turn")
    )
    debounce_states = {"idle", "wake_required", "listening", "speaking", "thinking", "looking"}
    face_key = (face_state, _interaction_status_key(face_state, face_text))
    with state.lock:
        current_priority, current_state = _current_face_priority_locked(state)
        # An expression can visually carry an ordinary lifecycle state.  Use
        # the stronger of the rendered face and the underlying interaction so
        # a persistent expression is not rejected as an idle-priority update.
        next_priority = max(
            _face_priority(face_state, face_text),
            _face_priority(interaction_state, status_text),
        )
        if next_priority < current_priority:
            if now - float(state.last_face_block_log_at or 0.0) > 2.0:
                state.last_face_block_log_at = now
                print(
                    f"[V7.14 FACE] blocked lower priority update "
                    f"status={face_state} text={face_text} current={current_state}"
                )
            return
        duplicate = state.last_face_status_key == face_key
        if (
            not important_transition
            and interaction_state in debounce_states
            and duplicate
            and now - float(state.last_face_status_at or 0.0) < _face_status_repeat_interval_seconds()
        ):
            return
        state.last_face_status = face_state
        state.last_face_status_text = face_text
        state.last_face_status_at = now
        state.last_face_status_key = face_key

    if face_state == "wake_required" or _normalize_for_echo(face_text) == "say hey miguel":
        print(f"[V7.14 FACE MODE] wake_required display text={face_text}")
    elif face_state == "listening" and "your turn" in _normalize_for_echo(face_text):
        print(f"[V7.14 FACE MODE] active display text={face_text}")
    print(f"[V7.5 FACE STATUS] {face_state} {face_text}".strip())
    try:
        if hasattr(full, "face_status"):
            full.face_status(face_state, face_text)
            return
        if interaction_state == "listening":
            full.face_listening()
        elif interaction_state in {"thinking", "heard"}:
            full.face_thinking()
        elif interaction_state == "looking":
            full.face_thinking()
        elif interaction_state == "speaking":
            full.face_speaking(status_text or "Speaking")
        elif interaction_state == "sleeping":
            full.face_sleeping()
        elif interaction_state == "shutdown_pending":
            full.face_idle()
        elif interaction_state == "error":
            full.face_error(status_text or "Error")
        elif interaction_state == "idle":
            full.face_idle()
    except Exception as exc:
        print("[V7.5 FACE STATUS] hook error:", exc)


def set_interaction_state(state: RobotRuntimeState, new_state: str, status_text: str = "") -> None:
    now = time.time()
    should_notify = True
    should_log = True
    with state.lock:
        old_state = state.interaction_state
        old_text = state.current_status_text
        if (
            state.shutdown_confirmation_pending
            and new_state in {"idle", "listening"}
            and new_state != "shutdown_pending"
        ):
            print(f"[V7.14 SHUTDOWN] blocked state={new_state} while pending")
            print("[V7.14 SHUTDOWN] pending lock active")
            state.interaction_state = "shutdown_pending"
            state.current_status_text = "Confirm shutdown"
            state.last_state_change_at = now
            state.last_state_emit_at = now
            state.last_state_emit_state = "shutdown_pending"
            state.last_state_emit_text = "Confirm shutdown"
            state.last_state_emit_recognition_key = state.last_face_recognition_key
            state.last_state_emit_key = ("shutdown_pending", _interaction_status_key("shutdown_pending", "Confirm shutdown"))
            new_state = "shutdown_pending"
            status_text = "Confirm shutdown"
        elif (
            state.sleep_mode_active
            and new_state not in {"sleeping", "shutdown_pending", "speaking", "error"}
        ):
            if now - float(state.last_face_block_log_at or 0.0) > 2.0:
                state.last_face_block_log_at = now
                print(f"[V7.14 FACE] blocked lower priority update status={new_state} text={status_text} current=sleeping")
            return
        elif (
            (state.turn_processing_active or state.interaction_state in {"heard", "thinking", "looking"})
            and (new_state in {"idle", "ready", "listening"} or _normalize_for_echo(status_text) == "ready")
            and state.interaction_state in {"heard", "thinking", "looking"}
        ):
            if now - float(state.last_face_block_log_at or 0.0) > 2.0:
                state.last_face_block_log_at = now
                print(
                    f"[V7.14 FACE] blocked lower priority update "
                    f"status={new_state} text={status_text} current={state.interaction_state}"
                )
            return
        else:
            important_transition = (
                new_state in {"speaking", "heard", "thinking", "looking", "enrolling", "shutdown_pending", "sleeping", "error"}
                or (new_state == "listening" and _normalize_for_echo(status_text) == "your turn")
            )
            old_duration = now - float(state.last_state_change_at or now)
            repeat_interval = _state_repeat_log_interval_seconds()
            old_status_key = _interaction_status_key(old_state, old_text)
            new_status_key = _interaction_status_key(new_state, status_text)
            emit_key = (new_state, new_status_key)
            current_recognition_key = state.last_face_recognition_key
            same_visible_state = old_state == new_state and old_status_key == new_status_key
            same_emit = (
                state.last_state_emit_key == emit_key
                and state.last_state_emit_recognition_key == current_recognition_key
            )

            if (
                not important_transition
                and same_visible_state
                and same_emit
                and now - float(state.last_state_emit_at or 0.0) < repeat_interval
            ):
                return

            min_duration = {
                "heard": 0.15,
                "thinking": 0.25,
                "looking": 0.25,
            }.get(old_state, 0.0)

            if new_state != "speaking" and old_duration < min_duration:
                return

            if new_state == "listening" and not _can_emit_ready_cue_locked(state):
                return

            if same_visible_state:
                should_log = True
                if new_state == "idle" and _is_idle_neutral_status(old_text) and _is_idle_neutral_status(status_text):
                    if current_recognition_key == state.last_state_emit_recognition_key:
                        status_text = old_text
                    else:
                        state.current_status_text = status_text
            else:
                state.interaction_state = new_state
                state.current_status_text = status_text
                state.last_state_change_at = now
                if new_state == "listening":
                    state.last_listen_started_at = now
                elif new_state == "heard":
                    state.last_heard_at = now
                elif new_state == "thinking":
                    state.last_thinking_started_at = now
                elif new_state == "speaking":
                    state.last_speaking_started_at = now

            state.last_state_emit_at = now
            state.last_state_emit_state = new_state
            state.last_state_emit_text = status_text
            state.last_state_emit_recognition_key = current_recognition_key
            state.last_state_emit_key = (new_state, _interaction_status_key(new_state, status_text))

    if should_log:
        print(f"[V7.5 STATE] {old_state} -> {new_state} {status_text}".strip())
    if should_notify:
        notify_face_status(state, new_state, status_text)


def force_interaction_state(state: RobotRuntimeState, new_state: str, status_text: str = "") -> None:
    now = time.time()
    with state.lock:
        old_state = state.interaction_state
        state.interaction_state = new_state
        state.current_status_text = status_text
        state.last_state_change_at = now
        state.last_state_emit_at = now
        state.last_state_emit_state = new_state
        state.last_state_emit_text = status_text
        state.last_state_emit_recognition_key = state.last_face_recognition_key
        state.last_state_emit_key = (new_state, _interaction_status_key(new_state, status_text))
    print(f"[V7.5 STATE] {old_state} -> {new_state} {status_text}".strip())
    notify_face_status(state, new_state, status_text)


def _update_face_identity_runtime_state(
    state: RobotRuntimeState,
    face_detected: bool,
    recognized_person: str | None,
    face_count: int | None = None,
    recognition_score: float | None = None,
    recognition_margin: float | None = None,
) -> tuple[str, str]:
    recognized = _normalize_person_name(recognized_person)
    new_key = _face_recognition_key(face_detected, recognized)
    try:
        normalized_face_count = int(face_count if face_count is not None else (1 if face_detected else 0))
    except (TypeError, ValueError):
        normalized_face_count = 1 if face_detected else 0
    with state.lock:
        previous_key = state.last_face_recognition_key
        state.face_detected = bool(face_detected)
        state.face_count = normalized_face_count
        state.known_person_present = bool(recognized)
        state.recognized_person = recognized
        state.recognized_person_updated_at = time.time()
        state.last_face_recognition_key = new_key
    score_ok = _identity_candidate_accepted(
        recognized,
        2,
        recognition_score if recognition_score is not None else 1.0,
        recognition_margin if recognition_margin is not None else 1.0,
    )
    if face_detected and recognized and _is_owner(recognized) and score_ok:
        _refresh_owner_session(state, recognized)
        with state.lock:
            active = bool(state.conversation_active and time.time() <= float(state.conversation_until or 0.0))
            mode = state.conversation_mode
            should_preserve = active and mode in {"creative", "story", "project"}
            should_update_partner = active and not state.conversation_partner
            if should_preserve or should_update_partner:
                state.conversation_partner = recognized
                state.last_conversation_activity_at = time.time()
                if should_preserve:
                    state.conversation_until = max(
                        float(state.conversation_until or 0.0),
                        time.time() + _conversation_timeout_seconds(mode),
                    )
        if should_preserve:
            print(f"[V7.15 SESSION] familiar_face_preserved_mode mode={mode} partner={recognized}")
        elif should_update_partner:
            print(f"[V7.15 SESSION] partner_updated_from_face partner={recognized}")
    return previous_key, new_key


def _maybe_surface_unknown_face(
    state: RobotRuntimeState,
    previous_recognition_key: str,
    current_recognition_key: str,
) -> None:
    if current_recognition_key != "unknown" or not previous_recognition_key.startswith("known:"):
        return

    now = time.time()
    with state.lock:
        if (
            state.interaction_state != "idle"
            or _has_active_conversation_locked(state, now)
            or now - float(state.last_unknown_face_visual_at or 0.0) < _state_repeat_log_interval_seconds()
        ):
            return
        state.last_unknown_face_visual_at = now

    set_interaction_state(state, "idle", "Unknown face")
    try:
        full.face_confused("Who is there?")
    except Exception as exc:
        print("[V7.5 FACE STATUS] unknown-face hook error:", exc)


def _log_latency(label: str, started_at: float | None = None) -> None:
    if not started_at:
        return
    print(f"[V7.5 LATENCY] {label}={time.monotonic() - started_at:.3f}s")


def _mark_route_done(state: RobotRuntimeState, started_at: float | None = None) -> None:
    if not started_at:
        return
    now = time.monotonic()
    with state.lock:
        state.current_turn_latency["route_done_at"] = now
    print(f"[V7.5 LATENCY] route_done={now - started_at:.3f}s")


def _log_latency_summary(latency: dict, text: str, speak_started_at: float, speak_finished_at: float) -> None:
    turn_started_at = latency.get("turn_started_at")
    if not turn_started_at:
        return

    route_done_at = latency.get("route_done_at") or latency.get("reply_queued_at") or speak_started_at
    route_s = max(0.0, float(route_done_at) - float(turn_started_at))
    speak_s = max(0.0, float(speak_finished_at) - float(speak_started_at))
    total_s = max(0.0, float(speak_finished_at) - float(turn_started_at))
    words = len(str(text or "").split())
    print(
        f"[V7.5 LATENCY SUMMARY] route={route_s:.3f}s "
        f"speak={speak_s:.3f}s total={total_s:.3f}s words={words}"
    )


def _set_reply_context(state: RobotRuntimeState, context: str) -> None:
    with state.lock:
        state.current_turn_latency["reply_context"] = context


def _set_response_length_context(state: RobotRuntimeState, mode: str) -> None:
    mode = str(mode or "normal").strip().lower()
    if mode not in {"terse", "normal", "detailed", "story", "long_story"}:
        mode = "normal"
    with state.lock:
        state.response_length_mode = mode
        state.current_turn_latency["response_length_mode"] = mode


def _set_transient_response_length_context(state: RobotRuntimeState, mode: str) -> None:
    mode = str(mode or "normal").strip().lower()
    if mode not in {"terse", "normal", "detailed", "story", "long_story"}:
        mode = "normal"
    with state.lock:
        state.current_turn_latency["response_length_mode"] = mode


def _set_response_depth_mode(state: RobotRuntimeState, mode: str, reason: str) -> None:
    mode = str(mode or "normal").strip().lower()
    if mode not in {"normal", "long_story", "long_explanation"}:
        mode = "normal"
    with state.lock:
        state.response_depth_mode = mode
        if mode == "normal":
            state.long_story_active = False
            state.long_story_topic = None
            state.long_story_segment_index = 0
            state.long_story_target_minutes = 0
            state.long_story_style = ""
            state.recovered_story_context = ""
            if state.conversation_mode == "story":
                state.conversation_mode = "general"
            if state.response_length_mode in {"long_story", "detailed"}:
                state.response_length_mode = "normal"
        state.current_turn_latency["response_depth_mode"] = mode
    print(f"[V7.15 DEPTH] mode={mode} reason={reason}")


LONG_STORY_ACTIVATION_PHRASES = {
    "long story mode",
    "go to long story mode",
    "story mode",
    "modo historia",
    "tell longer stories",
    "make it a real long story",
    "make the story longer",
    "give me the full story",
    "tell the full story",
    "tell a longer story",
    "continue as a long story",
    "historia longa",
    "conte uma historia",
    "contar uma historia",
    "uma historia de cinco minutos",
    "de cinco minutos",
    "historia comprida",
    "historia maior",
    "longa historia",
}

NORMAL_DEPTH_PHRASES = {
    "normal mode",
    "in normal mode",
    "shorter answers",
    "keep it short",
    "concise mode",
    "exit long story mode",
    "exit long explanation mode",
    "stop long mode",
    "talk normally",
}


def _is_normal_conversation_mode_request(text: str) -> bool:
    """Recognize requests to leave a persona and resume ordinary conversation.

    Keep this separate from the legacy personality substring matcher: a phrase
    such as "drop mission control" mentions that mode while explicitly asking
    Miguel to turn it off.
    """
    normalized = normalize_command_text(text)
    if not normalized:
        return False

    conversation_markers = {
        "normal conversation",
        "normal mode conversation",
        "conversation mode",
        "conversational mode",
    }
    if not any(marker in normalized for marker in conversation_markers):
        return False

    direct_markers = {
        "activate",
        "go to",
        "go in",
        "switch to",
        "normal",
    }
    leaving_persona = "mission control" in normalized and any(
        marker in normalized
        for marker in {"drop", "exit", "leave", "stop", "turn off", "no more"}
    )
    return leaving_persona or any(marker in normalized for marker in direct_markers)

LONG_EXPLANATION_ACTIVATION_PHRASES = {
    "long explanation mode",
    "activate long explanation mode",
    "turn on long explanation mode",
    "turn on detailed mode",
    "use longer answers from now on",
    "tell me a long explanation",
    "explain more",
    "give me the long version",
    "more details",
    "detailed mode",
}


PORTUGUESE_LONG_STORY_TRIGGERS = [
    ("historia longa", "história longa"),
    ("conte uma historia", "conte uma história"),
    ("conta uma historia", "conta uma história"),
    ("contar uma historia", "contar uma história"),
    ("quero uma historia", "quero uma história"),
    ("uma historia de cinco minutos", "uma história de cinco minutos"),
    ("uma historia de dez minutos", "uma história de dez minutos"),
    ("de cinco minutos", "de cinco minutos"),
    ("de dez minutos", "de dez minutos"),
    ("historia comprida", "história comprida"),
    ("historia maior", "história maior"),
    ("longa historia", "longa história"),
    ("modo historia", "modo história"),
    ("historia para dormir", "história para dormir"),
    ("historia de dormir", "história de dormir"),
    ("coloca o marquinho para dormir com uma historia", "coloca o Marquinho para dormir com uma história"),
    ("faz o marquinho dormir com uma historia", "faz o Marquinho dormir com uma história"),
]

ENGLISH_LONG_STORY_TRIGGERS = [
    ("long story", "long story"),
    ("tell me a story", "tell me a story"),
    ("tell a story", "tell a story"),
    ("story mode", "story mode"),
    ("bedtime story", "bedtime story"),
    ("sleep story", "sleep story"),
    ("five minute story", "five minute story"),
    ("five minutes story", "five minute story"),
    ("ten minute story", "ten minute story"),
    ("10 minute story", "10 minute story"),
    ("long explanation", "long explanation"),
]

STORY_WORDS = {"story", "historia"}
MULTILINGUAL_STORY_HINTS = {
    "story", "historia", "histoire", "cuento", "storia", "geschichte",
    "raconte", "racontez", "cuentame", "cuenta", "racconta", "erzahle",
}
STORY_GENERATION_TRIGGERS = {
    "tell a story",
    "tell me a story",
    "make a story",
    "create a story",
    "invent a story",
    "new story",
    "another story",
    "different story",
    "start a new story",
    "conte uma historia",
    "conta uma historia",
    "contar uma historia",
    "quero uma historia",
    "crie uma historia",
    "criar uma historia",
    "invente uma historia",
    "inventar uma historia",
    "nova historia",
    "outra historia",
    "historia diferente",
}

BEDTIME_STORY_MARKERS = {
    "bedtime story",
    "sleep story",
    "story to sleep",
    "to sleep with a story",
    "put marquinho to sleep with a story",
    "tell marquinho a story to sleep",
    "historia para dormir",
    "historia de dormir",
    "para o marquinho dormir",
    "para dormir",
    "coloca o marquinho para dormir com uma historia",
    "faz o marquinho dormir com uma historia",
}


def _contains_story_word(normalized: str) -> bool:
    words = set(str(normalized or "").split())
    return bool(words & STORY_WORDS)


def _has_multilingual_story_hint(normalized: str) -> bool:
    words = set(str(normalized or "").split())
    return bool(words & MULTILINGUAL_STORY_HINTS)


def _has_story_request_marker(normalized: str) -> bool:
    normalized = str(normalized or "")
    if not normalized:
        return False
    request_markers = STORY_GENERATION_TRIGGERS | {
        "tell me a long story",
        "tell a long story",
        "long story about",
        "conte uma historia longa",
        "conta uma historia longa",
        "contar uma historia longa",
        "quero uma historia",
        "historia longa sobre",
        "longa historia sobre",
        "historia para dormir",
        "historia de dormir",
        "bedtime story",
        "sleep story",
        "put marquinho to sleep with a story",
        "tell marquinho a story to sleep",
    }
    return any(marker in normalized for marker in request_markers) or bool(
        re.search(r"\b(?:tell me|tell|give me)\b.+\bstory\b", normalized)
    )


def _story_subtype(normalized: str) -> str:
    if any(marker in str(normalized or "") for marker in BEDTIME_STORY_MARKERS):
        return "bedtime"
    if any(marker in str(normalized or "") for marker in {"adventure", "aventura", "aventuras"}):
        return "adventure"
    return "general"


def _story_chapter_count(requested_minutes: int, subtype: str) -> int:
    if requested_minutes:
        return min(_long_story_max_segments(), _long_story_duration_chapters(requested_minutes))
    if subtype == "bedtime":
        return min(_long_story_max_segments(), 5)
    return 1


def _story_mode_for_intent(requested_minutes: int, subtype: str) -> str:
    if requested_minutes >= 5 or subtype == "bedtime":
        return "story_continuous"
    return "story_long_single"


def _empty_story_detection() -> dict:
    return {
        "detected": False,
        "language": None,
        "response_language": "",
        "trigger": "",
        "mode": "normal",
        "action": "none",
        "story_mode": "",
        "subtype": "general",
        "requested_minutes": 0,
        "chapter_count": 0,
        "target_words_per_chapter": _long_story_words_per_chapter(),
        "auto_continue": False,
        "has_story_request": False,
        "classifier_used": False,
        "low_confidence": False,
        "confidence": 0.0,
    }


def _story_detection_payload(
    language: str,
    trigger: str,
    mode: str,
    normalized: str,
    text: str,
    response_language: str | None = None,
    characters: list[str] | None = None,
    setting: str | None = None,
    confidence: float = 1.0,
) -> dict:
    requested_minutes = _extract_long_story_duration_minutes(text)
    subtype = _story_subtype(normalized)
    has_story_request = _has_story_request_marker(normalized)
    story_mode = _story_mode_for_intent(requested_minutes, subtype) if mode == "long_story" else ""
    chapter_count = _story_chapter_count(requested_minutes, subtype) if story_mode == "story_continuous" else 1
    return {
        "detected": True,
        "language": language,
        "response_language": response_language or language,
        "trigger": trigger,
        "mode": mode,
        "action": "generate_story" if mode == "long_story" and has_story_request else "mode_setting",
        "story_mode": story_mode,
        "subtype": subtype,
        "requested_minutes": requested_minutes,
        "chapter_count": chapter_count,
        "target_words_per_chapter": _long_story_words_per_chapter() if story_mode == "story_continuous" else 250,
        "auto_continue": story_mode == "story_continuous",
        "has_story_request": has_story_request,
        "characters": characters or [],
        "setting": setting or "",
        "confidence": float(confidence),
    }


STORY_INTENT_CACHE: dict[str, dict] = {}
SUPPORTED_STORY_LANGUAGES = {"en", "pt", "fr", "es", "it", "de", "unknown"}


def _coerce_intent_minutes(value) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _mode_from_classifier_minutes(mode: str, requested_minutes: int, subtype: str) -> str:
    if mode in {"story_short", "story_long_single", "story_continuous"}:
        if mode == "story_short":
            return "story_long_single"
        return mode
    return _story_mode_for_intent(requested_minutes, subtype)


def _normalize_multilingual_classifier_result(raw: dict, text: str) -> dict:
    if not isinstance(raw, dict):
        return _empty_story_detection()
    intent = str(raw.get("intent") or "normal_conversation").strip().lower()
    action = str(raw.get("action") or "answer").strip().lower()
    language = str(raw.get("language") or "unknown").strip().lower()
    if language not in SUPPORTED_STORY_LANGUAGES:
        language = "unknown"
    response_language = str(raw.get("response_language") or language or "unknown").strip().lower()
    subtype = str(raw.get("subtype") or "general").strip().lower()
    if subtype not in {"general", "bedtime", "adventure"}:
        subtype = "general"
    try:
        confidence = float(raw.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < 0.65:
        result = _empty_story_detection()
        result.update({
            "language": language,
            "response_language": response_language,
            "classifier_used": True,
            "low_confidence": True,
            "confidence": confidence,
        })
        return result
    if intent == "stop_story" or action == "stop":
        result = _empty_story_detection()
        result.update({
            "detected": True,
            "language": language,
            "response_language": response_language,
            "intent": "stop_story",
            "action": "stop",
            "confidence": confidence,
        })
        return result
    if intent != "story" or action != "generate_story":
        result = _empty_story_detection()
        result.update({
            "detected": intent == "mode_setting",
            "language": language,
            "response_language": response_language,
            "intent": intent,
            "action": "mode_setting" if intent == "mode_setting" else "answer",
            "confidence": confidence,
        })
        return result

    requested_minutes = _coerce_intent_minutes(raw.get("requested_minutes"))
    story_mode = _mode_from_classifier_minutes(str(raw.get("mode") or ""), requested_minutes, subtype)
    chapter_count = _story_chapter_count(requested_minutes, subtype) if story_mode == "story_continuous" else 1
    result = {
        "detected": True,
        "language": language,
        "response_language": response_language,
        "trigger": "multilingual_classifier",
        "mode": "long_story",
        "action": "generate_story",
        "story_mode": story_mode,
        "subtype": subtype,
        "requested_minutes": requested_minutes,
        "chapter_count": chapter_count,
        "target_words_per_chapter": _long_story_words_per_chapter() if story_mode == "story_continuous" else 250,
        "auto_continue": story_mode == "story_continuous",
        "has_story_request": True,
        "characters": list(raw.get("characters") or []),
        "setting": str(raw.get("setting") or ""),
        "confidence": confidence,
        "intent": "story",
        "classifier_used": True,
    }
    return result


def _classify_story_intent_multilingual(text: str) -> dict:
    instructions = (
        "Classify Miguel robot user transcripts for story intent. "
        "Return structured JSON only with this schema: "
        "{\"intent\":\"story|mode_setting|normal_conversation|stop_story\","
        "\"action\":\"generate_story|set_mode|answer|stop\","
        "\"language\":\"en|pt|fr|es|it|de|unknown\","
        "\"response_language\":\"string\","
        "\"mode\":\"story_short|story_long_single|story_continuous|null\","
        "\"subtype\":\"general|bedtime|adventure|null\","
        "\"requested_minutes\":0,"
        "\"characters\":[],\"setting\":null,\"confidence\":0.0}. "
        "Recognize story requests in any language, including French raconte-moi une longue histoire, "
        "French histoire de dix minutes, French histoire pour dormir, Spanish cuéntame una historia larga, "
        "Spanish cuento para dormir, Italian/German equivalents, and stop-story commands. "
        "Duration or bedtime story requests should use mode story_continuous. "
        "Do not classify story requests as long explanation mode."
    )
    response = v6.client.responses.create(
        model=getattr(v6, "OPENAI_MODEL", "gpt-4o-mini"),
        instructions=instructions,
        input=str(text or ""),
    )
    output = str(getattr(response, "output_text", "") or "").strip()
    output = re.sub(r"^```(?:json)?\s*|\s*```$", "", output, flags=re.IGNORECASE | re.DOTALL).strip()
    return json.loads(output)


def _multilingual_story_mode_detection(text: str, normalized: str) -> dict:
    key = str(normalized or normalize_command_text(text))
    if not key:
        return _empty_story_detection()
    if key in STORY_INTENT_CACHE:
        return dict(STORY_INTENT_CACHE[key])
    try:
        raw = _classify_story_intent_multilingual(text)
        result = _normalize_multilingual_classifier_result(raw, text)
        if result.get("classifier_used"):
            print(
                "[V7.15 MULTILINGUAL INTENT] "
                f"used=true language={result.get('language')} intent=story "
                f"confidence={float(result.get('confidence') or 0.0):.2f}"
            )
        STORY_INTENT_CACHE[key] = dict(result)
        return result
    except Exception as exc:
        print("[V7.15 MULTILINGUAL INTENT] used=false warning=", exc)
        result = _empty_story_detection()
        STORY_INTENT_CACHE[key] = dict(result)
        return result


def _story_mode_detection(text: str) -> dict:
    normalized = normalize_command_text(text)
    if not normalized:
        return _empty_story_detection()

    for trigger, display in PORTUGUESE_LONG_STORY_TRIGGERS:
        if trigger in normalized:
            return _story_detection_payload("pt", display, "long_story", normalized, text)

    for trigger, display in ENGLISH_LONG_STORY_TRIGGERS:
        if trigger in normalized:
            mode = "long_story" if "explanation" not in trigger else "long_explanation"
            return _story_detection_payload("en", display, mode, normalized, text)

    duration_minutes = _extract_long_story_duration_minutes(text)
    if duration_minutes and _contains_story_word(normalized):
        language = "pt" if "historia" in normalized else "en"
        trigger = "minutos" if language == "pt" else "minutes"
        return _story_detection_payload(language, trigger, "long_story", normalized, text)

    # Do not send ordinary context statements to the story classifier. This
    # also prevents prior story state from turning family introductions into a
    # new story request.
    if not _has_multilingual_story_hint(normalized):
        return _empty_story_detection()

    return _multilingual_story_mode_detection(text, normalized)


def _is_long_story_request_text(text: str) -> bool:
    return _story_mode_detection(text).get("mode") == "long_story"


def _log_story_mode_detection(detection: dict) -> None:
    if detection.get("detected"):
        print(
            "[V7.15 STORY INTENT] "
            f"detected=true language={detection.get('language')} "
            f"action={detection.get('action')} mode={detection.get('story_mode') or detection.get('mode')} "
            f"trigger=\"{detection.get('trigger')}\""
        )
        minutes = int(detection.get("requested_minutes") or 0)
        if detection.get("story_mode") == "story_continuous":
            print(
                "[V7.15 STORY INTENT] "
                f"detected=true language={detection.get('language')} "
                f"mode=story_continuous requested_minutes={minutes}"
            )


def _log_story_execution_generate() -> None:
    print("[V7.15 STORY EXECUTION] action=generate_story route=story depth=long_story skipped_legacy_long_mode=true")


def _log_story_execution_skip_legacy(reason: str = "story_request") -> None:
    print(f"[V7.15 STORY EXECUTION] skipped_legacy_long_mode=true reason={reason}")


def _route_low_confidence_story_intent(detection: dict, state: RobotRuntimeState) -> bool:
    if not detection.get("low_confidence"):
        return False
    language = str(detection.get("response_language") or detection.get("language") or "").lower()
    if language == "pt":
        reply = "Eu ouvi você, mas não tenho certeza se quer uma história. Pode repetir em uma frase curta?"
    elif language == "fr":
        reply = "Je vous ai entendu, mais je ne suis pas sûr que vous vouliez une histoire. Pouvez-vous répéter brièvement?"
    elif language == "es":
        reply = "Te escuché, pero no estoy seguro de si quieres una historia. ¿Puedes repetirlo brevemente?"
    else:
        reply = "I heard you, but I am not sure if you want a story. Please say that again briefly."
    _set_reply_context(state, "clarification")
    _set_response_length_context(state, "terse")
    print(
        "[V7.15 STORY INTENT] "
        f"low_confidence=true language={detection.get('language')} "
        f"confidence={float(detection.get('confidence') or 0.0):.2f} action=clarify"
    )
    v6.speak(reply)
    return True


def _exit_stale_story_mode_for_non_story_turn(
    user_text: str,
    detection: dict,
    state: RobotRuntimeState,
) -> bool:
    normalized = normalize_command_text(user_text)
    if (
        detection.get("detected")
        or _has_multilingual_story_hint(normalized)
        or _is_story_continue_text(user_text)
    ):
        return False
    with state.lock:
        if state.conversation_mode != "story" or state.story_session.active:
            return False
        state.conversation_mode = "general"
        state.response_length_mode = "normal"
        state.response_depth_mode = "normal"
        state.long_story_active = False
        state.long_story_topic = None
        state.long_story_target_minutes = 0
        state.current_turn_latency["story_state_reset"] = True
    print("[V7.15 STORY] inactive story state reset for non-story turn")
    return True


def _is_depth_status_question(text: str) -> bool:
    normalized = normalize_command_text(text)
    if not normalized:
        return False
    exact = {
        "are you in long story mode or normal mode",
        "what mode are you in",
        "are you in normal mode",
        "are you in long mode",
        "are you in long story mode",
        "are you in long explanation mode",
    }
    return normalized in exact or (
        normalized.startswith(("what mode", "which mode", "are you in"))
        and any(marker in normalized for marker in {"mode", "long", "normal"})
    )


def _route_depth_status_local_reply(user_text: str, state: RobotRuntimeState) -> bool:
    if not _is_depth_status_question(user_text):
        return False
    with state.lock:
        depth = state.response_depth_mode
        conversation_mode = state.conversation_mode
        target_minutes = state.long_story_target_minutes
    if depth == "long_story":
        prefix = "story mode" if conversation_mode == "story" else "normal conversation mode"
        duration = _format_long_story_duration(target_minutes)
        reply = f"I'm in {prefix}, with long story mode on"
        if duration:
            reply += f" for about {duration}"
        reply += "."
    elif depth == "long_explanation":
        reply = "I'm in normal conversation mode, with long explanation mode on."
    else:
        reply = "I'm in normal response mode."
    _set_reply_context(state, "status")
    _set_transient_response_length_context(state, "terse")
    v6.speak(reply)
    return True


def _is_explicit_long_story_request(text: str) -> bool:
    normalized = normalize_command_text(text)
    if not normalized:
        return False
    if _is_long_story_request_text(text):
        return True
    if _extract_long_story_duration_minutes(text) and _contains_story_word(normalized):
        return True
    markers = {
        "long story",
        "longer story",
        "full story",
        "real long story",
        "make the story longer",
        "tell the full story",
        "give me the full story",
    }
    return any(marker in normalized for marker in markers)


def _long_story_depth_applies_to_route(route: str, last_user_text: str, conversation_mode: str) -> bool:
    normalized = normalize_command_text(last_user_text)
    if route in {"creative", "story"}:
        return True
    if _is_long_story_request_text(last_user_text):
        return True
    if conversation_mode in {"creative", "story"} and route == "normal" and (
        _is_contextual_followup(normalized) or _is_story_continue_text(normalized)
    ):
        return True
    return any(
        marker in normalized
        for marker in {
            "continue chapter",
            "next chapter",
            "tell more story",
            "continue the story",
            "story part",
            "keep going",
            "keep it going",
        }
    )


def _ready_cue_min_interval_seconds() -> float:
    try:
        return max(0.0, float(os.getenv("MIGUEL_READY_CUE_MIN_INTERVAL_SECONDS", "2.0")))
    except (TypeError, ValueError):
        return 2.0


def _queue_has_items(work_queue: queue.Queue | None) -> bool:
    return bool(work_queue is not None and not work_queue.empty())


def _has_pending_reply_locked(state: RobotRuntimeState) -> bool:
    return state.pending_reply_count > 0 or _queue_has_items(state.reply_queue)


def _has_pending_user_turn_locked(state: RobotRuntimeState) -> bool:
    return state.pending_user_turn_count > 0 or _queue_has_items(state.user_turn_queue)


def _can_emit_ready_cue_locked(state: RobotRuntimeState) -> bool:
    blocked_states = {
        "heard",
        "thinking",
        "looking",
        "enrolling",
        "speaking",
        "shutdown_pending",
        "sleeping",
    }
    return not (
        state.stop_event.is_set()
        or state.is_speaking
        or _has_pending_reply_locked(state)
        or _has_pending_user_turn_locked(state)
        or state.brain_is_processing
        or state.turn_processing_active
        or state.sleep_mode_active
        or state.shutdown_pending
        or state.shutdown_confirmation_pending
        or state.interaction_state in blocked_states
    )


def _can_start_audio_capture_locked(state: RobotRuntimeState) -> bool:
    if state.sleep_mode_active:
        return not (
            state.stop_event.is_set()
            or state.is_speaking
            or _has_pending_reply_locked(state)
            or _has_pending_user_turn_locked(state)
            or state.brain_is_processing
        )
    if state.shutdown_confirmation_pending:
        return not (
            state.stop_event.is_set()
            or state.is_speaking
            or _has_pending_reply_locked(state)
            or _has_pending_user_turn_locked(state)
            or state.brain_is_processing
        )
    return _can_emit_ready_cue_locked(state)


def emit_ready_cue(state: RobotRuntimeState, speak_fn=None, force: bool = False) -> None:
    now = time.time()
    notify_state = "idle"
    notify_text = "Ready"
    notify_before_return = False
    with state.lock:
        if not state.audio_capture_active:
            notify_state = _ready_face_state()
            notify_text = _ready_face_text(state)
        else:
            notify_state = "listening"
            notify_text = "YOUR TURN"
        if not _can_emit_ready_cue_locked(state):
            if not state.shutdown_confirmation_pending and not state.stop_event.is_set():
                notify_before_return = True
            mode = "off"
        else:
            mode = (state.ready_cue_mode or "visual").strip().lower()
            if mode not in {"visual", "beep", "spoken", "off"}:
                mode = "visual"
                state.ready_cue_mode = mode
            if not state.ready_cue_enabled:
                mode = "off"
            if state.suppress_next_ready_cue:
                state.suppress_next_ready_cue = False
                mode = "off"
                notify_before_return = False
            elif mode != "off":
                min_interval = _ready_cue_min_interval_seconds()
                if not force and state.last_ready_cue_at and now - state.last_ready_cue_at < min_interval:
                    mode = "off"
                    notify_before_return = False
                else:
                    state.last_ready_cue_at = now
    if notify_before_return:
        notify_face_status(state, notify_state, notify_text)
        return
    if mode == "off":
        return

    notify_face_status(state, notify_state, notify_text)
    print(f"[V7.5 READY CUE] mode={mode}")

    if mode == "beep":
        print("[V7.5 READY CUE] beep TODO: local beep playback not wired in queue layer.")
    elif mode == "spoken" and speak_fn is not None:
        with state.lock:
            state.is_speaking = True
            state.suppress_next_ready_cue = True
        try:
            speak_fn("Go ahead.")
        except Exception as exc:
            print("[V7.5 READY CUE] spoken cue error:", exc)
        finally:
            with state.lock:
                state.is_speaking = False
                state.last_speech_finished_at = time.time()


def prepare_to_listen(state: RobotRuntimeState) -> bool:
    with state.lock:
        ready = _can_start_audio_capture_locked(state)
        shutdown_pending = bool(state.shutdown_confirmation_pending)
    if not ready:
        if shutdown_pending:
            print("[V7.14 SHUTDOWN] pending lock active")
            notify_face_status(state, "shutdown_pending", "Confirm shutdown")
        return False
    if shutdown_pending:
        notify_face_status(state, "shutdown_pending", "Confirm shutdown")
    else:
        emit_ready_cue(state)
    return True


def install_speech_queue(reply_queue: queue.Queue, safety: SafetyGuard, state: RobotRuntimeState):
    """
    Make legacy V6/V7 helpers enqueue speech instead of speaking immediately.
    SpeechWorker is the only owner of the original v6.speak.
    """
    original_speak = v6.speak

    def enqueue_speak(text: str):
        with state.lock:
            override = getattr(_speech_enqueue_context, "latency", None)
            latency = dict(override if override is not None else state.current_turn_latency)
            context = latency.get("reply_context", "normal")
        if state.stop_speech_event.is_set() and context != "shutdown_confirm":
            print("[V7.5 BARGE-IN] Dropped reply because speech stop is pending.")
            return
        if context == "shutdown_confirm":
            # The confirmation command itself can leave the ordinary barge-in
            # flag set.  Never let that suppress the terminal acknowledgement.
            state.stop_speech_event.clear()
        with state.lock:
            latency.setdefault("log_user_text", state.last_user_text)
            latency.setdefault("log_person", state.conversation_partner or state.recognized_person)
            latency.setdefault("log_conversation_mode", state.conversation_mode)
            latency.setdefault("log_topic", state.session_focus or state.last_topic or "")
            state.pending_reply_count += 1
        latency["reply_queued_at"] = time.monotonic()
        _log_latency("reply_queued", latency.get("turn_started_at"))
        reply_queue.put(ReplyEvent(str(text or ""), latency, context))

    v6.speak = enqueue_speak
    return original_speak


def _speak_with_enqueue_context(text: str, latency: dict) -> None:
    previous = getattr(_speech_enqueue_context, "latency", None)
    _speech_enqueue_context.latency = latency
    try:
        v6.speak(text)
    finally:
        if previous is None:
            try:
                del _speech_enqueue_context.latency
            except AttributeError:
                pass
        else:
            _speech_enqueue_context.latency = previous


def _strip_wake_phrase(text: str) -> str:
    t = str(text or "").lower().strip()
    phrases = list(getattr(full, "WAKE_PHRASES", [])) + [
        "hey miguel",
        "hello miguel",
        "hi miguel",
        "ei miguel",
        "miguel",
    ]
    for phrase in phrases:
        if t == phrase:
            return ""
        if t.startswith(phrase + " "):
            return str(text or "").strip()[len(phrase):].strip(" ,.")
    return str(text or "").strip()


def _is_conversation_grace_active(state: RobotRuntimeState) -> bool:
    with state.lock:
        if not state.last_reply_time:
            return False
        return (time.time() - state.last_reply_time) <= state.conversation_grace_seconds


def _wait_until_listening_allowed(state: RobotRuntimeState) -> None:
    delay = float(os.getenv("MIGUEL_POST_SPEECH_LISTEN_DELAY_SECONDS", "0.4"))

    while not state.stop_event.is_set():
        with state.lock:
            is_speaking = state.is_speaking
            brain_is_processing = state.brain_is_processing
            pending_reply = _has_pending_reply_locked(state)
            pending_user_turn = _has_pending_user_turn_locked(state)
            since_speech = time.time() - state.last_speech_finished_at if state.last_speech_finished_at else delay

        if (
            not is_speaking
            and not brain_is_processing
            and not pending_reply
            and not pending_user_turn
            and since_speech >= delay
        ):
            return

        time.sleep(0.05)


def _normalize_for_echo(text: str) -> str:
    t = str(text or "").lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(char for char in t if not unicodedata.combining(char))
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def normalize_command_text(text: str) -> str:
    normalized = _normalize_for_echo(text)
    wake_phrases = [
        "hey miguel",
        "hi miguel",
        "hello miguel",
        "okay miguel",
        "ok miguel",
        "yo miguel",
        "ei miguel",
        "miguel",
    ]
    for phrase in wake_phrases:
        if normalized == phrase:
            return ""
        if normalized.startswith(phrase + " "):
            normalized = normalized[len(phrase):].strip()
            break
    prefixes = ["by the way", "okay", "ok", "so", "again"]
    for prefix in prefixes:
        if normalized == prefix:
            return ""
        if normalized.startswith(prefix + " "):
            return normalized[len(prefix):].strip()
    return normalized


LANGUAGE_ALIASES = {
    "english": "english",
    "ingles": "english",
    "inglês": "english",
    "portuguese": "portuguese",
    "portugues": "portuguese",
    "português": "portuguese",
    "brazilian": "portuguese",
    "brazilian portuguese": "portuguese",
    "spanish": "spanish",
    "espanol": "spanish",
    "español": "spanish",
    "italian": "italian",
    "italiano": "italian",
    "french": "french",
    "frances": "french",
    "francês": "french",
    "romanian": "romanian",
    "română": "romanian",
    "czech": "czech",
    "cesky": "czech",
    "česky": "czech",
}

LANGUAGE_SCORE_WORDS = {
    "english": {
        "the", "and", "you", "your", "what", "who", "where", "when", "why", "how",
        "please", "story", "universe", "english", "conversation", "only", "consider",
        "ignore", "wake", "sleep", "shutdown", "status", "time", "weather", "calculate",
    },
    "portuguese": {
        "que", "voce", "você", "para", "pra", "porque", "como", "quando", "agora",
        "aqui", "isso", "esta", "está", "fazer", "fala", "filho", "papai", "brasil",
        "portugues", "português", "nao", "não", "sim", "tambem", "também",
        "isto", "chega", "pouco", "mais", "perto", "vai", "voce", "você", "quer",
        "gente", "estou", "estava", "tá", "esta", "está", "chuva", "chovendo",
    },
    "spanish": {
        "que", "como", "cuando", "donde", "porque", "hola", "gracias", "ahora",
        "usted", "espanol", "español", "tambien", "también", "asi", "así",
    },
    "italian": {
        "che", "come", "quando", "dove", "perche", "perché", "ciao", "grazie",
        "italiano", "sono", "vecchio", "pero", "però",
    },
    "french": {
        "que", "comment", "quand", "pourquoi", "bonjour", "merci", "francais",
        "français", "avec", "vous",
    },
    "romanian": {
        "cum", "cand", "când", "unde", "pentru", "romana", "română", "fariți",
        "cumpărat",
    },
    "czech": {
        "jak", "kdy", "kde", "proc", "proč", "cesky", "česky", "prakticka",
        "praktická", "otazka", "otázka",
    },
}

LANGUAGE_DIACRITIC_HINTS = {
    "portuguese": set("ãõçáéíóúâêôà"),
    "spanish": set("ñ¿¡"),
    "italian": set("ìòèù"),
    "french": set("ùûîïëÿœæ"),
    "romanian": set("ăâîșşțţ"),
    "czech": set("čďěňřšťůžýáíé"),
}


def _canonical_language_name(name: str) -> str | None:
    raw = str(name or "").strip().lower()
    normalized = _normalize_for_echo(raw)
    return LANGUAGE_ALIASES.get(raw) or LANGUAGE_ALIASES.get(normalized)


def _default_allowed_conversation_languages() -> list[str]:
    configured = os.getenv("MIGUEL_ALLOWED_CONVERSATION_LANGUAGES", "english")
    languages = []
    for part in re.split(r"[,;/]|\band\b|\bor\b|\+", configured, flags=re.IGNORECASE):
        language = _canonical_language_name(part)
        if language and language not in languages:
            languages.append(language)
    return languages or ["english"]


def _format_language_list(languages: list[str] | set[str] | tuple[str, ...]) -> str:
    names = [str(language).strip().title() for language in languages if str(language).strip()]
    if not names:
        return "none"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _extract_language_names_from_text(text: str) -> list[str]:
    lowered = str(text or "").lower()
    normalized = _normalize_for_echo(text)
    found = []
    for alias, canonical in sorted(LANGUAGE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        alias_normalized = _normalize_for_echo(alias)
        if (
            re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", lowered)
            or (alias_normalized and re.search(rf"(?<![a-z]){re.escape(alias_normalized)}(?![a-z])", normalized))
        ):
            if canonical not in found:
                found.append(canonical)
    return found


def _language_policy_command(text: str) -> tuple[str, list[str]] | None:
    normalized = normalize_command_text(text)
    if not normalized:
        return None
    languages = _extract_language_names_from_text(text)
    language_markers = {
        "language", "languages", "english mode", "conversation", "consider",
        "listen", "ignore", "respond", "reply", "speak", "filter",
        "language filter",
    }
    has_marker = any(marker in normalized for marker in language_markers)
    if not languages and ("allowed languages" in normalized or "language status" in normalized):
        return "status", []
    if not languages or not has_marker:
        return None

    set_markers = {
        "only", "lock", "lock in", "lock only", "from now on", "stick to",
        "use only", "answer only", "reply only", "consider only", "only consider",
        "change", "change to", "set", "set to", "switch", "switch to",
    }
    remove_markers = {"ignore", "dont consider", "don t consider", "do not consider", "don't consider", "stop considering"}
    add_markers = {"also", "add", "include", "allow", "accept"}

    if any(marker in normalized for marker in remove_markers) and any(marker in normalized for marker in set_markers):
        removed = []
        for language in languages:
            for alias, canonical in LANGUAGE_ALIASES.items():
                if canonical != language:
                    continue
                alias_normalized = _normalize_for_echo(alias)
                if any(
                    f"{marker} {alias_normalized}" in normalized
                    or f"{marker} the {alias_normalized}" in normalized
                    for marker in remove_markers
                ):
                    removed.append(language)
                    break
        kept = [language for language in languages if language not in set(removed)]
        if kept:
            return "set", kept

    if any(marker in normalized for marker in set_markers):
        return "set", languages
    if any(marker in normalized for marker in remove_markers):
        keep_languages = []
        if "only" in normalized and len(languages) >= 1:
            keep_languages = [languages[0]]
        return ("set", keep_languages) if keep_languages else ("remove", languages)
    if any(marker in normalized for marker in add_markers):
        return "add", languages
    if "language" in normalized or "conversation" in normalized:
        return "set", languages
    return None


def _route_language_policy_local_reply(user_text: str, state: RobotRuntimeState) -> bool:
    command = _language_policy_command(user_text)
    if not command:
        return False
    action, languages = command
    with state.lock:
        current = list(state.allowed_conversation_languages or ["english"])
        if action == "status":
            result = current
        elif action == "set":
            result = languages or ["english"]
            state.allowed_conversation_languages = result
        elif action == "add":
            result = current
            for language in languages:
                if language not in result:
                    result.append(language)
            state.allowed_conversation_languages = result
        elif action == "remove":
            result = [language for language in current if language not in set(languages)]
            if not result:
                result = ["english"]
            state.allowed_conversation_languages = result
        else:
            result = current
    print(f"[V7.15 LANGUAGE] allowed={','.join(result)} action={action}")
    _force_active_after_mode(state, "general", reason="language_policy")
    if result == ["portuguese"]:
        v6.speak("Filtro de idioma definido para Portugues.")
    else:
        v6.speak(f"Language filter set to {_format_language_list(result)}.")
    return True


def _language_scores(text: str) -> dict[str, int]:
    lowered = str(text or "").lower()
    normalized = _normalize_for_echo(text)
    words = set(normalized.split())
    scores = {}
    for language, markers in LANGUAGE_SCORE_WORDS.items():
        score = len(words & {_normalize_for_echo(marker) for marker in markers})
        hints = LANGUAGE_DIACRITIC_HINTS.get(language, set())
        if any(char in lowered for char in hints):
            score += 3
        if score:
            scores[language] = score
    return scores


def _detect_transcript_language(text: str) -> tuple[str | None, dict[str, int]]:
    scores = _language_scores(text)
    if not scores:
        return None, {}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return ranked[0][0], dict(ranked)


def _language_policy_bypass(text: str) -> bool:
    normalized = normalize_command_text(text)
    return bool(
        _language_policy_command(text)
        or _is_sleep_mode_request(normalized)
        or _is_sleep_wake_request(normalized)
        or _is_shutdown_request_text(normalized)
        or _is_shutdown_confirm_text(normalized)
        or _is_shutdown_cancel_text(normalized)
        or is_barge_in_command(normalized)
        or _is_global_without_wake_command(normalized)
    )


def _prefers_portuguese_reply(user_text: str, state: RobotRuntimeState) -> bool:
    language, scores = _detect_transcript_language(user_text)
    if language == "portuguese":
        return True
    with state.lock:
        allowed = list(state.allowed_conversation_languages or [])
    return allowed == ["portuguese"] and scores.get("english", 0) == 0


def _localize_identity_reply(reply: str, user_text: str, state: RobotRuntimeState) -> str:
    if not _prefers_portuguese_reply(user_text, state):
        return reply
    text = str(reply or "").strip()
    if not text:
        return text
    match = re.fullmatch(r"I see ([A-Za-z][A-Za-z0-9_ -]*?)\.", text)
    if match:
        return f"Eu vejo {_friendly_person_name(match.group(1))}."
    match = re.fullmatch(r"I see a face, and our active conversation is with ([A-Za-z][A-Za-z0-9_ -]*?)\.", text)
    if match:
        return f"Eu vejo um rosto, e nossa conversa ativa e com {_friendly_person_name(match.group(1))}."
    match = re.fullmatch(
        r"I do not have a confirmed face right now, but our active conversation is with ([A-Za-z][A-Za-z0-9_ -]*?)\.",
        text,
    )
    if match:
        return f"Nao tenho um rosto confirmado agora, mas nossa conversa ativa e com {_friendly_person_name(match.group(1))}."
    if text == "I see a face, but I do not recognize who it is yet.":
        return "Eu vejo um rosto, mas ainda nao reconheco quem e."
    if text == "I do not see a face right now.":
        return "Nao vejo um rosto agora."
    return text


def _language_policy_allows_turn(user_text: str, state: RobotRuntimeState) -> bool:
    if _language_policy_bypass(user_text):
        return True
    with state.lock:
        allowed = set(state.allowed_conversation_languages or ["english"])
    detected, scores = _detect_transcript_language(user_text)
    if not detected:
        return True
    if detected in allowed:
        return True
    allowed_score = max([scores.get(language, 0) for language in allowed] or [0])
    detected_score = scores.get(detected, 0)
    if allowed_score and allowed_score >= detected_score:
        return True
    print(
        "[V7.15 LANGUAGE] ignored "
        f"detected={detected} allowed={','.join(sorted(allowed))} "
        f"text={_short_log_text(user_text)}"
    )
    return False


def _is_bare_wake_phrase(text: str) -> bool:
    return _normalize_for_echo(text) in {
        "miguel",
        "hey miguel",
        "hello miguel",
        "hi miguel",
        "ei miguel",
    }


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", str(text or "")))


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", str(text or "").strip())
    return parts[0].strip() if parts and parts[0].strip() else str(text or "").strip()


TRAILING_WEAK_WORDS = {
    "a",
    "an",
    "and",
    "but",
    "how",
    "i",
    "in",
    "of",
    "or",
    "the",
    "to",
    "with",
    "about",
}


def _word_len(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", str(text or "")))


def _ends_with_weak_trailing_word(text: str) -> bool:
    tokens = re.findall(r"\b[\w']+\b", str(text or "").lower())
    return bool(tokens and tokens[-1] in TRAILING_WEAK_WORDS)


def _looks_truncated(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    return _ends_with_weak_trailing_word(stripped) or stripped[-1] not in ".!?"


def trim_to_word_limit_preserve_sentence(text: str, max_words: int) -> str:
    original = re.sub(r"\s+", " ", str(text or "")).strip()
    if not original:
        return original
    if _word_len(original) <= max_words:
        return original
    words_before = _word_len(original)

    sentences = [s.strip() for s in re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", original) if s.strip()]
    kept: list[str] = []
    kept_words = 0
    for sentence in sentences:
        sentence_words = _word_len(sentence)
        if kept and kept_words + sentence_words > max_words:
            if kept_words < max_words * 0.45:
                kept.append(sentence)
            break
        if not kept and sentence_words > max_words:
            break
        kept.append(sentence)
        kept_words += sentence_words

    if kept:
        candidate = " ".join(kept).strip()
        if candidate and candidate[-1] in ".!?" and not _ends_with_weak_trailing_word(candidate):
            print(f"[V7.14 LENGTH] trimmed mode=unknown words_before={words_before} words_after={_word_len(candidate)}")
            return candidate

    # If the first sentence alone exceeds the spoken cap, prefer ending at a
    # clause boundary.  A raw word slice produced audible fragments in the
    # runtime log (for example, "I..." and "a lower...").
    words = original.split()
    capped = " ".join(words[:max_words])
    clause_ends = [match.end() for match in re.finditer(r"[,;:]|\s+[—–-]\s+", capped)]
    for clause_end in reversed(clause_ends):
        clause = capped[:clause_end].rstrip(" ,;:—–-")
        if _word_len(clause) >= max(4, int(max_words * 0.45)) and not _ends_with_weak_trailing_word(clause):
            trimmed = clause.rstrip(".!?") + "."
            print(
                f"[V7.14 LENGTH] trimmed mode=unknown words_before={words_before} "
                f"words_after={_word_len(trimmed)} boundary=clause"
            )
            return trimmed

    candidate_words = words[:max_words]
    while candidate_words and re.sub(r"[^a-zA-Z']+", "", candidate_words[-1]).lower() in TRAILING_WEAK_WORDS:
        candidate_words.pop()
    candidate = " ".join(candidate_words).rstrip(" ,;:")
    if not candidate:
        candidate = " ".join(words[:max_words]).rstrip(" ,;:")
    trimmed = candidate.rstrip(".!?") + "."
    print(f"[V7.14 LENGTH] trimmed mode=unknown words_before={words_before} words_after={_word_len(trimmed)}")
    return trimmed


def _limit_words(text: str, max_words: int) -> str:
    return trim_to_word_limit_preserve_sentence(text, max_words)


def _warn_if_possible_truncation(text: str) -> None:
    if _looks_truncated(text):
        print(f"[V7.14 LENGTH] warning possible truncation text={_short_log_text(text)}")


def _friendly_person_name(name: str | None) -> str:
    cleaned = str(name or "").strip().replace("_", " ")
    return cleaned.title() if cleaned else ""


def _trim_scene_reply(text: str, max_words: int = 14) -> str:
    original = str(text or "").strip()
    cleaned = re.sub(
        r"^(the image shows|image shows|the frame shows|frame shows|in the image,?|this image shows)\s+",
        "I see ",
        original,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^i see\s+(that\s+)?(the\s+)?image\s+shows\s+", "I see ", cleaned, flags=re.IGNORECASE)
    if cleaned and not cleaned.lower().startswith("i see"):
        cleaned = "I see " + cleaned[0].lower() + cleaned[1:]

    clauses = re.split(r",|\band\b", _first_sentence(cleaned))
    banned = (
        "neutral expression",
        "positioned",
        "left side",
        "right side",
        "confidence",
        "appears to be",
    )
    kept = [clause.strip() for clause in clauses if clause.strip() and not any(b in clause.lower() for b in banned)]
    if kept:
        cleaned = ", ".join(kept)

    if _word_len(cleaned) < 6 and _word_len(_first_sentence(original)) >= 6:
        cleaned = _first_sentence(original)
        cleaned = re.sub(
            r"^(the image shows|image shows|the frame shows|frame shows|in the image,?|this image shows)\s+",
            "I see ",
            cleaned,
            flags=re.IGNORECASE,
        )
        if cleaned and not cleaned.lower().startswith("i see"):
            cleaned = "I see " + cleaned[0].lower() + cleaned[1:]

    if _word_len(cleaned) < 4:
        cleaned = "I can only see part of the camera view."

    if not cleaned.endswith((".", "!", "?")):
        cleaned += "."
    return _limit_words(cleaned, max_words)


def make_robot_reply_concise(
    text: str,
    max_words: int | None = None,
    context: str = "normal",
    response_length_mode: str = "normal",
    response_depth_mode: str = "normal",
) -> str:
    original = str(text or "").strip()
    if not original:
        return original

    lower = original.lower()
    mode = str(response_length_mode or "normal").strip().lower()
    depth = str(response_depth_mode or "normal").strip().lower()
    if mode not in {"terse", "normal", "detailed", "story", "long_story"}:
        mode = "normal"
    if depth == "long_story" and context in {"creative", "story", "normal"}:
        mode = "long_story"
    elif depth == "long_explanation" and context not in TERSE_ALLOWED_ROUTES:
        mode = "detailed"
    if context in {"safety_refusal", "enrollment"} or "can't help" in lower or "cannot help" in lower:
        mode = "terse" if context == "safety_refusal" else "normal"
    limit = int(max_words or _response_word_limit(mode))
    words_before = _word_len(original)

    local_map = {
        "i'm here.": "Here.",
        "yes, i can hear you.": "I hear you.",
        "got it.": "Confirmed.",
        "confirmed.": "Confirmed.",
        "looking.": "Looking.",
        "robot voice activated.": "Robot voice.",
        "robotic voice activated.": "Robot voice.",
        "natural voice activated.": "Natural voice.",
        "friendly voice activated.": "Friendly voice.",
        "deep voice activated.": "Deep voice.",
        "story voice activated.": "Story voice.",
    }
    mapped = local_map.get(lower)
    if mapped:
        return mapped

    if "i checked the camera" in lower and "recognize one face as" in lower:
        match = re.search(r"recognize one face as ([a-zA-Z_ -]+?)(?:,|\.|$)", original, re.IGNORECASE)
        if match:
            return _limit_words(f"I see {_friendly_person_name(match.group(1))}.", 4)

    if "i checked the camera and see a face" in lower and "cannot" in lower:
        return "Not sure who."

    if "i do not see a confirmed face" in lower or "do not have a fresh camera frame" in lower:
        return "I see no face."

    if context == "scene":
        return _trim_scene_reply(original, limit)

    if context == "identity":
        if re.search(r"\bi see\s+\d+\s+faces\b|\bi see\s+one\s+face\b", lower):
            return _limit_words(original, max(limit, 18))
        return _limit_words(_first_sentence(original), 8)

    if mode == "terse":
        shaped = _limit_words(_first_sentence(original), limit)
    else:
        shaped = trim_to_word_limit_preserve_sentence(original, limit)
    if _word_len(shaped) < words_before:
        if mode == "long_story":
            print(
                f"[V7.15 LENGTH] mode=long_story words_before={words_before} "
                f"words_after={_word_len(shaped)} trim_policy=story_cap"
            )
        else:
            print(f"[V7.14 LENGTH] trimmed mode={mode} words_before={words_before} words_after={_word_len(shaped)}")
    return shaped


def _looks_like_self_heard_speech(transcript: str, state: RobotRuntimeState) -> bool:
    with state.lock:
        last_spoken = state.last_spoken_text
        finished_at = state.last_speech_finished_at
        enrollment_state = state.enrollment_state

    heard = _normalize_for_echo(transcript)
    spoken = _normalize_for_echo(last_spoken)
    if not heard or not spoken:
        return False

    if enrollment_state == "awaiting_name" and heard.startswith("is your friend s name"):
        return False

    if heard == "what is your friend s name":
        return True

    if time.time() - finished_at > 8.0:
        return False

    spoken_words = spoken.split()
    if "why did" in spoken and "?" in str(last_spoken or ""):
        joke_fragments = [
            "robot bring a pencil to bed",
            "little robot bring a pencil to bed",
            "robot bring a ladder",
            "miguel cross the room",
            "robot take a nap",
            "computer get cold",
        ]
        if any(fragment in heard for fragment in joke_fragments):
            return True
        if len(spoken_words) >= 5 and " ".join(spoken_words[:5]) in heard:
            return True

    if len(spoken_words) >= 8:
        first_8 = " ".join(spoken_words[:8])
        if first_8 and first_8 in heard:
            return True

    echo_phrases = [
        "enrollment flow is authorized for charlie",
        "enrollment needs approval",
    ]
    if any(p in heard for p in echo_phrases):
        return True

    ratio = difflib.SequenceMatcher(None, heard, spoken).ratio()
    return ratio > 0.72


def _salvage_self_heard_command(transcript: str) -> str | None:
    normalized = _normalize_for_echo(transcript)
    commands = [
        "who do you see",
        "who am i",
        "do you recognize me",
        "identify me",
        "what do you see",
        "what time is it",
        "current time",
        "status",
        "mission control",
    ]

    for command in commands:
        index = normalized.rfind(command)
        if index < 0:
            continue

        suffix = normalized[index:]
        prefix = normalized[:index].strip()
        if suffix == command and (
            not prefix
            or prefix.endswith("miguel")
            or "try again" in prefix
            or "hey miguel" in prefix
        ):
            return command

    return None


def _enqueue_user_turn(
    user_turn_queue: queue.Queue,
    state: RobotRuntimeState,
    text: str,
    recognized_person: str | None = None,
    authorized: bool = False,
    authorization_source: str = "",
    stripped_text: str = "",
) -> None:
    if _looks_like_asr_prompt_leak(text):
        print("[V7.5 AUDIO] Dropped ASR prompt leak.")
        return

    if _looks_like_self_heard_speech(text, state):
        salvaged = _salvage_self_heard_command(text)
        if salvaged:
            print(f"[V7.5 AUDIO] Salvaged command from self-heard speech: {salvaged}")
            text = salvaged
        else:
            print("[V7.5 AUDIO] Dropped self-heard speech.")
            return

    if _is_enrollment_cancel_text(text):
        _reset_enrollment_state(state)

    if is_barge_in_command(text):
        _clear_queue_items(user_turn_queue)
        with state.lock:
            state.pending_user_turn_count = 0
        if _is_speech_stop_barge_in(text):
            _request_speech_stop(state)

    now = time.monotonic()
    normalized_text = normalize_command_text(text)
    if not stripped_text and _has_v7_5_wake_phrase(text):
        stripped_text = _strip_wake_phrase(text)
    with state.lock:
        state.previous_user_text = state.last_user_text
        state.last_user_text = str(text or "")
        state.last_non_self_heard_user_text = str(text or "")
        state.current_turn_started_at = now
        state.current_turn_latency = {"turn_started_at": now, "transcript_ready_at": now}
        state.pending_user_turn_count += 1
    set_interaction_state(state, "heard", str(text or "")[:48])
    _log_latency("transcript_ready", now)
    turn_latency = {
        "turn_started_at": now,
        "transcript_ready_at": now,
        "log_user_text": str(text or ""),
        "log_person": recognized_person,
    }
    with state.lock:
        turn_latency.update(state.last_audio_capture_timing)
        state.last_audio_capture_timing = {}
    user_turn_queue.put(
        UserTurnEvent(
            text,
            recognized_person,
            bool(authorized),
            str(authorization_source or ""),
            normalized_text,
            str(stripped_text or ""),
            turn_latency,
        )
    )


def _looks_like_asr_prompt_leak(text: str) -> bool:
    normalized = _normalize_for_echo(text)
    leaks = [
        "this is speech to a small father son robot named miguel",
        "important names and terms",
        "jetson orin nano",
        "oak d lite",
    ]
    return any(leak in normalized for leak in leaks)


def _looks_like_robot_health_complaint(normalized: str) -> bool:
    if not normalized:
        return False
    complaint_markers = {
        "issue",
        "issues",
        "problem",
        "freezing",
        "not responding",
        "stopped responding",
        "stopping from responding",
        "stopped in the middle",
        "you stopped",
        "you just stop your systems",
        "you stop your systems",
        "stopped the system",
        "cannot hear me",
        "cant hear me",
        "can't hear me",
        "not hearing me",
        "not listening",
        "stay on mute",
        "stays on mute",
        "break your ear",
    }
    return any(marker in normalized for marker in complaint_markers)


def _is_clear_speech_stop_barge_in(text: str) -> bool:
    normalized = normalize_command_text(text)
    if not normalized:
        return False
    if _looks_like_robot_health_complaint(normalized):
        return False
    exact_phrases = {
        "stop",
        "stop it",
        "stop talking",
        "pause",
        "pausa",
        "espera",
        "espere",
        "para a historia",
        "parar historia",
        "cancel speech",
        "cancel",
        "cancela",
        "cancelar",
        "shutdown",
        "shut down",
        "confirm shutdown",
        "confirme shutdown",
        "quiet",
        "wait",
    }
    if normalized in exact_phrases:
        return True
    polite_suffixes = (" please", " por favor", " now", " agora")
    return any(
        normalized == phrase + suffix
        for phrase in exact_phrases
        for suffix in polite_suffixes
    )


def is_barge_in_command(text: str) -> bool:
    normalized = normalize_command_text(text)
    if _is_shutdown_request_text(normalized) or _is_shutdown_confirm_text(normalized):
        return True
    return _is_clear_speech_stop_barge_in(text)


def _is_barge_in_command(text: str) -> bool:
    return is_barge_in_command(text)


def _is_speech_stop_barge_in(text: str) -> bool:
    normalized = normalize_command_text(text)
    if _is_shutdown_request_text(normalized) or _is_shutdown_confirm_text(normalized):
        return False
    return _is_clear_speech_stop_barge_in(text)


def _clear_queue_items(target_queue: queue.Queue | None) -> int:
    if target_queue is None:
        return 0
    cleared = 0
    try:
        while True:
            target_queue.get_nowait()
            target_queue.task_done()
            cleared += 1
    except queue.Empty:
        return cleared


def _request_speech_stop(state: RobotRuntimeState) -> int:
    already_requested = state.stop_speech_event.is_set()
    state.stop_speech_event.set()
    cleared = _clear_queue_items(state.reply_queue)
    with state.lock:
        state.pending_reply_count = 0
        speaking = state.is_speaking
    if not already_requested or cleared:
        print(f"[V7.5 BARGE-IN] Stop requested; cleared {cleared} queued replies.")
    if speaking and not already_requested:
        print("[V7.5 BARGE-IN] Stop requested; current speak backend is not interruptible yet.")
    return cleared


def _captured_during_speaking(text: str, state: RobotRuntimeState, capture_started_at: float) -> bool:
    if not text:
        return False
    with state.lock:
        is_speaking = state.is_speaking
        speech_started_at = state.last_speech_started_at
    return bool(is_speaking or (speech_started_at and speech_started_at >= capture_started_at))


def _restore_face_after_audio_capture(state: RobotRuntimeState) -> None:
    with state.lock:
        shutdown_pending = bool(state.shutdown_confirmation_pending)
        sleeping = bool(state.sleep_mode_active)
        processing = bool(state.turn_processing_active)
        current_state = state.interaction_state
        current_text = state.current_status_text
        conversation_active = bool(state.conversation_active)
        wake_required = bool(state.wake_required and not state.conversation_active)
    if shutdown_pending:
        set_interaction_state(state, "shutdown_pending", "Confirm shutdown")
    elif sleeping:
        set_interaction_state(state, "sleeping", "Sleep")
    elif processing and current_state in {"heard", "thinking", "looking"}:
        set_interaction_state(state, current_state, current_text)
    elif conversation_active:
        set_interaction_state(state, _ready_face_state(), _ready_face_text(state))
    elif wake_required:
        notify_face_status(state, "wake_required", _wake_required_face_text())
    else:
        set_interaction_state(state, _ready_face_state(), "Ready")


def capture_user_turn_when_ready(state: RobotRuntimeState) -> str:
    capture_started_at = time.time()
    sleeping = _sleep_mode_active(state)
    if not prepare_to_listen(state):
        _mark_audio_capture_finished(state, "blocked")
        return ""
    _mark_audio_capture_active(state)
    if sleeping:
        print("[V7.14 SLEEP] wake-only listening")
        set_interaction_state(state, "sleeping", "Sleep")
    else:
        set_interaction_state(state, "listening", "YOUR TURN")
        notify_face_status(state, "listening", "YOUR TURN")
    reason = "empty"
    try:
        user_text = v6.capture_user_turn()
        timing = dict(getattr(v6.capture_user_turn, "last_timing", {}) or {})
        with state.lock:
            state.last_audio_capture_timing = timing
        if user_text and _captured_during_speaking(user_text, state, capture_started_at) and not _is_barge_in_command(user_text):
            _store_interrupted_creative_topic(state, user_text)
            print("[V7.5 AUDIO] Dropped speech captured during Miguel speaking.")
            reason = "captured_during_speaking"
            return ""
        reason = "transcript" if user_text else "empty"
        return user_text
    except Exception:
        reason = "exception"
        raise
    finally:
        _mark_audio_capture_finished(state, reason)
        if reason == "transcript":
            print("[V7.14 FACE] hold transcript state before routing")
        else:
            _restore_face_after_audio_capture(state)


def _has_v7_5_wake_phrase(text: str) -> bool:
    normalized = _normalize_for_echo(text)
    words = normalized.split()
    wake_phrases = [
        "miguel",
        "hey miguel",
        "ei miguel",
        "mission control",
    ]
    return any(
        normalized == phrase
        or normalized.startswith(phrase + " ")
        or (phrase == "miguel" and "miguel" in words)
        for phrase in wake_phrases
    )


def _contains_grace_command_phrase(text: str) -> bool:
    t = str(text or "").lower().strip()
    phrases = [
        "voice",
        "robot voice",
        "natural voice",
        "friendly voice",
        "deep voice",
        "story voice",
        "camera",
        "what do you see",
        "who do you see",
        "can you see",
        "do you see",
        "enroll",
        "new friend",
        "time",
        "status",
        "recent topics",
        "previous topics",
        "conversation history",
        "analyze conversation",
        "analyze previous log",
        "previous log",
        "log file",
    ]
    return any(p in t for p in phrases)


def _is_preserved_command_transcript(text: str) -> bool:
    normalized = normalize_command_text(text)
    preserved = [
        "what are you",
        "who am i",
        "who do you see",
        "you see me",
        "see me",
        "see me now",
        "you see me now",
        "do you see me now",
        "can you see me now",
        "am i visible",
        "can you recognize me now",
        "what do you see",
        "who are the engineers",
        "who is the engineer",
        "who is the system engineer",
        "who is the chief engineer",
        "what is my role",
        "can you hear me",
        "shutdown",
        "voice",
        "enroll",
    ]
    return any(phrase in normalized for phrase in preserved)


def _is_acceptable_grace_transcript(text: str, state: RobotRuntimeState) -> bool:
    normalized = _normalize_for_echo(text)

    if not normalized:
        return False

    if _has_v7_5_wake_phrase(text):
        return True

    if _is_preserved_command_transcript(text):
        return True

    if _looks_like_self_heard_speech(text, state):
        return False

    with state.lock:
        prompt_type = state.last_prompt_type
        enrollment_state = state.enrollment_state
        shutdown_pending = state.shutdown_pending

    if prompt_type == "enrollment_name" or enrollment_state == "awaiting_name":
        return bool(_extract_enrollment_name_answer(text))

    if shutdown_pending and (_is_shutdown_cancel_text(text) or _is_shutdown_confirm_text(text) or _has_clear_non_shutdown_command(text)):
        return True

    if len(normalized.split()) < 3:
        return False

    expects_response = prompt_type in {
        "general_prompt",
        "shutdown_confirmation",
        "enrollment_request",
    }

    return expects_response or _contains_grace_command_phrase(text)


def _is_global_audio_command(text: str) -> bool:
    t = str(text or "").lower().strip()
    enrollment_phrases = [
        "approval enrolling",
        "approves enrolling",
        "approve enrolling",
        "approves and rolling",
        "approve and rolling",
    ]
    return full.is_global_idle_command(t) or _is_sleep_mode_request(t) or _is_sleep_wake_request(t) or _is_enrollment_request_text(t) or any(p in t for p in enrollment_phrases)


def _is_protected_audio_text(text: str) -> bool:
    normalized = _normalize_for_echo(text)
    protected = [
        "who am i",
        "who do you see",
        "what do you see",
        "hey miguel",
        "miguel",
        "mission control",
    ]
    return any(normalized == p or normalized.startswith(p + " ") for p in protected)


def _direct_command_kind(text: str) -> str | None:
    normalized = normalize_command_text(text)
    if not normalized:
        return None

    if (
        is_barge_in_command(normalized)
        or _is_shutdown_request_text(normalized)
        or _is_shutdown_cancel_text(normalized)
        or _is_shutdown_confirm_text(normalized)
    ):
        return "shutdown"
    if _is_sleep_mode_request(normalized) or _is_sleep_wake_request(normalized):
        return "sleep"
    if _is_password_session_command(normalized):
        return "owner"
    if robot_timer.parse_timer_command(normalized):
        return "timer"
    if _is_voice_command_text(normalized):
        return "voice"
    if (
        _is_recent_conversation_topics_request(normalized)
        or _is_conversation_log_analysis_request(normalized)
        or _is_conversation_log_location_request(normalized)
    ):
        return "memory"
    if _is_capabilities_request(normalized):
        return "capabilities"
    if is_identity_camera_request(normalized) or any(
        phrase in normalized
        for phrase in {
            "who am i",
            "who do you see",
            "you see me",
            "see me",
            "see me now",
            "you see me now",
            "do you see me now",
            "can you see me now",
            "am i visible",
            "can you recognize me now",
            "do you recognize me",
            "identify me",
            "who is this person",
            "who is this face",
            "who is that face",
            "who is his face",
            "who is her face",
            "who is this child",
            "who is that child",
            "identify this face",
            "whose face is this",
            "do you see me",
            "do you see a face",
            "can you see a face",
            "you cannot see me",
            "you can t see me",
            "can you recognize me",
            "who is in front of you",
            "did you see another face",
            "do you see another face",
            "can you see another face",
            "do you see both faces",
            "can you see both faces",
            "can you recognize the faces",
            "who are those faces",
            "who are the faces",
            "who are the people",
            "who is there",
            "who is with me",
            "do you recognize both of us",
            "can you recognize both of us",
            "is marquinho there",
            "is marco there",
            "can you see marco",
            "can you see marquinho",
            "see any other face",
            "any other face besides me",
            "other face besides me",
            "anyone behind me",
            "anybody behind me",
        }
    ):
        return "camera_identity"
    if is_scene_camera_request(normalized) or any(
        phrase in normalized
        for phrase in {"what do you see", "look around", "describe what you see"}
    ):
        return "camera_scene"
    if any(phrase in normalized for phrase in {"what time is it", "current time", "status"}):
        return "time_status"
    if any(phrase in normalized for phrase in {"weather", "calculate"}):
        return "time_status"
    if _creative_fast_allow_topic(normalized) or _infer_conversation_mode(normalized) in {"creative", "story"}:
        return "creative"
    if _is_enrollment_request_text(normalized):
        return "enrollment"
    if any(phrase in normalized for phrase in {"how are you", "what are you", "can you hear me", "do you hear me", "you hear me", "are you listening", "are you there", "hello", "yo"}):
        return "general"
    return None


def _infer_conversation_mode(text: str, camera_intent: str = "none") -> str:
    normalized = normalize_command_text(text)
    if _is_enrollment_request_text(normalized):
        return "enrollment"
    if camera_intent != "none":
        return "robot_control"
    conversation_markers = {
        "movie",
        "movies",
        "star wars",
        "sequels",
        "trilogy",
        "disney",
        "theater",
        "theatre",
        "topic",
        "talk about",
    }
    if any(marker in normalized for marker in conversation_markers):
        return "story" if "story mode" in normalized else "general"
    if _is_long_story_request_text(text):
        return "story"
    if any(
        phrase in normalized
        for phrase in {
            "creative mode",
            "brainstorm",
            "character",
            "hero",
            "new superheroes",
            "skeleton",
            "superhero idea",
            "superhero",
            "invent",
            "invention",
            "machine",
            "robot idea",
            "sci fi",
            "sci-fi",
            "technology",
            "tech idea",
            "make up",
            "villain",
        }
    ):
        return "creative"
    if any(
        phrase in normalized
        for phrase in {
            "story mode",
            "tell me a story",
            "tell a story",
            "make a story",
            "create a story",
            "invent a story",
            "new story",
            "another story",
            "change the story",
            "continue the story",
            "modo historia",
            "conte uma historia",
            "contar uma historia",
            "crie uma historia",
            "inventar uma historia",
            "nova historia",
            "outra historia",
            "continua a historia",
            "narrate",
        }
    ):
        return "story"
    if any(
        phrase in normalized
        for phrase in {
            "robot project",
            "miguel project",
            "architecture",
            "codex",
            "thread",
            "camera",
            "face recognition",
        }
    ):
        return "project"
    if _is_mode_command_not_physical(normalized):
        if "creative mode" in normalized or normalized == "go creative":
            return "creative"
        if "long story mode" in normalized or "modo historia" in normalized:
            return "story"
        if "robot project" in normalized or "project mode" in normalized:
            return "project"
        return "general"
    if (
        _is_voice_command_text(normalized)
        or _is_shutdown_request_text(normalized)
        or _is_shutdown_cancel_text(normalized)
        or _is_shutdown_confirm_text(normalized)
        or full.is_local_robot_control_request(normalized)
    ):
        return "robot_control"
    if _is_enrollment_request_text(normalized):
        return "enrollment"
    return "general"


CREATIVE_FAST_ALLOW_KEYWORDS = {
    "character",
    "could be",
    "extra powers",
    "hero",
    "how he works",
    "make it cooler",
    "maybe",
    "moving around him",
    "moving around his body",
    "power",
    "skeleton",
    "slithering",
    "story",
    "superhero",
    "try again",
    "villain",
    "weakness",
    "what if",
    "machine",
    "robot idea",
    "invention",
    "invent",
    "sci fi",
    "sci-fi",
    "technology",
    "tech idea",
}

REAL_WORLD_HARM_MARKERS = {
    "assassinate",
    "attack",
    "bomb",
    "build a gun",
    "build a weapon",
    "chemical weapon",
    "explosive",
    "harm a real",
    "hurt someone",
    "instructions",
    "kill",
    "make a bomb",
    "poison",
    "real person",
    "shoot",
    "stab",
    "terror",
    "weapon",
}

CORRECTION_RETRY_MARKERS = {
    "redo it",
    "make it cooler",
    "no i mean",
    "no, i mean",
    "that s not it",
    "thats not it",
    "nao e isso",
    "nao eh isso",
    "nao era isso",
    "tem que ser",
    "that s not the skeleton",
    "thats not the skeleton",
    "try again",
    "wrong",
    "faz maior",
}


def _contains_real_world_harm_instruction(text: str) -> bool:
    normalized = normalize_command_text(text)
    return any(marker in normalized for marker in REAL_WORLD_HARM_MARKERS)


CREATIVE_TOPIC_KEYWORDS = {
    "superhero",
    "hero",
    "imaginary hero",
    "character",
    "skeleton",
    "turtle spirit",
    "machine",
    "robot",
    "robot idea",
    "invention",
    "invent",
    "sci fi",
    "sci-fi",
    "technology",
    "tech",
    "power",
    "powers",
    "story",
}

CONTEXTUAL_FOLLOWUP_PHRASES = {
    "how can he work",
    "how does he work",
    "how would he work",
    "how can it work",
    "how does it work",
    "how would it work",
    "what powers does he have",
    "what powers does it have",
    "what about him",
    "what about it",
    "what should it do",
    "can it fly",
    "can he fly",
    "what can he do",
    "what can it do",
    "tell me more",
    "continue",
}


def _contains_owner_password_phrase(text: str) -> bool:
    configured = os.getenv("MIGUEL_OWNER_PASSWORD_PHRASE", "").strip()
    if not configured:
        return False
    return _normalize_owner_password_value(configured) in _normalize_owner_password_value(text)


def _safe_memory_snippet(text: str, max_chars: int = 120) -> str:
    value = str(text or "").strip()
    if not value or _contains_owner_password_phrase(value):
        return ""
    value = re.sub(r"\s+", " ", value)
    return value[:max_chars]


def _extract_called_or_named_subject(text: str) -> str | None:
    match = re.search(r"\b(?:called|named)\s+([A-Za-z][A-Za-z0-9' -]{0,40})", str(text or ""), re.IGNORECASE)
    if not match:
        return None
    name = match.group(1).strip(" .,:;!?")
    stop = re.search(r"\b(?:who|that|and|with|because|where|when|what|how)\b", name, re.IGNORECASE)
    if stop:
        name = name[:stop.start()].strip(" .,:;!?")
    words = name.split()
    if len(words) > 4:
        name = " ".join(words[:4])
    return name or None


STORY_CONTINUATION_PHRASES = {
    "continue",
    "continue story",
    "continue the story",
    "continua",
    "continua a historia",
    "continua historia",
    "keep going",
    "keep it going",
    "keep it going miguel",
    "go on",
    "next part",
    "next chapter",
    "what happens next",
}


def _is_story_continue_text(text: str) -> bool:
    normalized = normalize_command_text(text)
    if not normalized:
        return False
    return normalized in STORY_CONTINUATION_PHRASES or any(
        normalized.startswith(phrase + " ") or phrase in normalized
        for phrase in {
            "continua",
            "continue miguel",
            "continua miguel",
            "continua a historia",
            "continua historia",
            "continue the story",
            "keep going",
            "keep it going",
            "next part",
            "next chapter",
            "what happens next",
        }
    )


def _is_story_finish_request(text: str) -> bool:
    normalized = normalize_command_text(text)
    if not normalized:
        return False
    return any(
        marker in normalized
        for marker in {
            "finish the story",
            "finish this story",
            "finish that story",
            "finish it",
            "tell the end of the story",
            "tell me the end of the story",
            "tell the ending",
            "tell me the ending",
            "end the story",
            "wrap up the story",
            "conclude the story",
            "conte o fim da historia",
            "conta o fim da historia",
            "conte o final da historia",
            "conta o final da historia",
            "conte o fim",
            "conta o fim",
            "termine a historia",
            "termina a historia",
            "final da historia",
            "fim da historia",
        }
    )


def _is_story_pause_request(text: str) -> bool:
    normalized = normalize_command_text(text)
    if not normalized:
        return False
    return any(
        marker in normalized
        for marker in {
            "pause story",
            "pause the story",
            "pause",
            "wait",
            "hold story",
            "pausa historia",
            "pausa a historia",
            "pausar historia",
            "espera",
            "espere",
        }
    )


def _is_story_resume_request(text: str) -> bool:
    normalized = normalize_command_text(text)
    if not normalized:
        return False
    return any(
        marker in normalized
        for marker in {
            "resume story",
            "resume the story",
            "unpause story",
            "continue",
            "continue the story",
            "keep going",
            "continua",
            "continua a historia",
            "continua historia",
            "retoma a historia",
            "volta a historia",
        }
    )


def _extract_story_redirect_instruction(text: str) -> str:
    normalized = normalize_command_text(text)
    if not normalized:
        return ""
    patterns = [
        r"\bchange\s+(?:the\s+)?story\s+(?:to|toward|into|about)\s+(.+)$",
        r"\bturn\s+(?:the\s+)?story\s+(?:to|toward|into)\s+(.+)$",
        r"\bcontinue(?:\s+the\s+story)?\s+but\s+(.+)$",
        r"\bmake\s+it\s+(.+)$",
        r"\bmuda\s+(?:a\s+)?historia\s+(?:para|pra|pro|sobre)\s+(.+)$",
        r"\btroca\s+(?:a\s+)?historia\s+(?:para|pra|pro|sobre)\s+(.+)$",
        r"\bcontinua(?:\s+a\s+historia)?\s+mas\s+(.+)$",
        r"\bagora\s+(.+)$",
        r"\btem que ser\s+(.+)$",
        r"\bfaz\s+(?:ela\s+)?(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            instruction = match.group(1).strip(" .,:;!?")
            instruction = re.sub(r"\b(?:miguel|por favor|please)\b", " ", instruction).strip()
            instruction = re.sub(r"\s+", " ", instruction)
            if instruction:
                return instruction[:220]
    return ""


def _is_story_redirect_request(text: str) -> bool:
    return bool(_extract_story_redirect_instruction(text))


def _is_new_story_request(text: str) -> bool:
    normalized = normalize_command_text(text)
    if not normalized:
        return False
    if _is_story_finish_request(text):
        return False
    story_detection = _story_mode_detection(text)
    if story_detection.get("mode") == "long_story" and story_detection.get("action") == "generate_story":
        return True
    if not _contains_story_word(normalized):
        return False
    if _extract_long_story_duration_minutes(text) and not _is_mode_command_not_physical(normalized):
        return True
    return any(marker in normalized for marker in STORY_GENERATION_TRIGGERS | {"change the story", "change story"})


def _extract_story_theme(text: str) -> str:
    normalized = normalize_command_text(text)
    if not normalized:
        return "new adventure"
    patterns = [
        r"\bbedtime story\s+(?:of|about|around|with|on)\s+(.+)$",
        r"\bstory mode\b.*?\b(?:about|around|with|on)\s+(.+)$",
        r"\bstory\b.*?\b(?:about|around|with|on|of)\s+(.+)$",
        r"\bstory\s+(?:about|around|with)\s+(.+)$",
        r"\b(?:make|create|invent|tell me|tell)\s+(?:a\s+)?(?:new\s+)?(?:bedtime\s+)?story\s+(?:about|around|with|of)\s+(.+)$",
        r"\bchange\s+(?:the\s+)?story\s+(?:to|for|into|about)\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            theme = match.group(1).strip(" .,:;!?")
            theme = re.sub(r"\b(?:please|yeah|miguel|and make it exciting|make it exciting|be exciting)\b", " ", theme).strip()
            return re.sub(r"\s+", " ", theme) or "new adventure"
    if _is_new_story_request(normalized):
        return "new adventure"
    return normalized or "new adventure"


def _clean_story_topic(topic: str) -> str:
    topic = normalize_command_text(topic)
    if not topic:
        return ""
    topic = re.sub(r"\b(?:for|about)?\s*\d{1,2}\s*(?:-| )?\s*minutes?\b", " ", topic)
    topic = re.sub(
        r"\b(?:for|about)?\s*(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty)\s*(?:-| )?\s*minutes?\b",
        " ",
        topic,
    )
    # ASR commonly preserves phrasing such as "twenty minutes duration about
    # adventures and space".  The numeric phrase is removed above; remove its
    # orphaned unit noun as well so it cannot become the story's setting/title.
    topic = re.sub(r"\b(?:of\s+)?duration\s+(?:of|about|for)\s+", " ", topic)
    topic = re.sub(r"^duration\b", " ", topic)
    topic = re.sub(
        r"\b(?:de|por)?\s*(?:um|uma|dois|duas|tres|três|quatro|cinco|seis|sete|oito|nove|dez|onze|doze|treze|catorze|quatorze|quinze|dezesseis|dezasseis|dezessete|dezassete|dezoito|dezenove|dezanove|vinte|trinta)\s*(?:-| )?\s*minutos?\b",
        " ",
        topic,
    )
    topic = re.sub(
        r"\b(?:please|yeah|hi|hello|miguel|now|can you|could you|would you|creative mode|long story mode|story mode|long mode|go to|switch to|set|make it|make this|tell a story|tell me a story|tell me a bedtime story|bedtime story|for me)\b",
        " ",
        topic,
    )
    topic = re.sub(r"\b(?:at the same time|same time|five minutes long|ten minutes long)\b", " ", topic)
    topic = re.sub(r"\b(?:the\s+)?topic\s+is\s+", " ", topic)
    topic = re.sub(r"\b(?:style|tone|genre)\s+is\s+[a-z ]{2,40}$", " ", topic)
    topic = re.sub(r"\b(?:in|with)\s+(?:a\s+)?[a-z ]{2,30}\s+(?:style|tone|genre)\b", " ", topic)
    topic = re.sub(r"\b(?:and\s+)?tell\s+(?:us|me)?\s*(?:a\s+)?(?:long\s+|bedtime\s+)?story\s*(?:about|on|of)?\s*", " ", topic)
    topic = re.sub(r"\b(?:e\s+)?(?:conte|contar|crie|criar|invente|inventar)\s+(?:uma\s+)?(?:historia\s+)?(?:longa|comprida|maior)?\s*(?:sobre)?\s*", " ", topic)
    topic = re.sub(r"\b(?:historia\s+longa|longa\s+historia|historia\s+comprida|historia\s+maior|modo\s+historia)\b", " ", topic)
    topic = re.sub(r"\b(?:and\s+)?a\s+(?:story|about)\b", " ", topic)
    topic = re.sub(r"\b(?:uma\s+)?historia\s+(?:sobre|de)?\b", " ", topic)
    topic = re.sub(r"^(?:about|around|with|on|of)\s+", "", topic.strip())
    topic = re.sub(r"^(?:his|her|its)\s+", "", topic.strip())
    topic = re.sub(r"\s+", " ", topic).strip(" .,:;!?")
    return "" if topic in {"your", "you", "me", "this", "that"} else topic


STORY_STYLE_KEYWORDS = {
    "adventure": "adventure",
    "action": "action adventure",
    "funny": "funny",
    "silly": "silly",
    "comedy": "funny",
    "mystery": "mystery",
    "detective": "mystery",
    "spooky": "gentle spooky",
    "scary": "gentle spooky",
    "creepy": "gentle spooky",
    "bedtime": "calm bedtime",
    "calm": "calm bedtime",
    "gentle": "gentle",
    "epic": "epic adventure",
    "space": "space adventure",
    "sci fi": "science fiction",
    "sci-fi": "science fiction",
    "science fiction": "science fiction",
    "star wars": "space adventure",
    "superhero": "superhero adventure",
    "educational": "educational adventure",
}


def _extract_story_style(text: str) -> str:
    normalized = normalize_command_text(text)
    if not normalized:
        return ""
    def styles_from(raw_text: str) -> list[str]:
        matches = []
        for keyword, style in STORY_STYLE_KEYWORDS.items():
            index = raw_text.find(keyword)
            if index >= 0:
                matches.append((index, style))
        found = []
        for _, style in sorted(matches, key=lambda item: item[0]):
            if style not in found:
                found.append(style)
        return found

    explicit_patterns = [
        r"\b(?:style|tone|genre)\s+is\s+([a-z ]{2,40})",
        r"\b(?:in|with)\s+(?:a\s+)?([a-z ]{2,30})\s+(?:style|tone|genre)\b",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, normalized)
        if match:
            raw = re.sub(r"\b(?:story|please|for me|and|but)\b", " ", match.group(1))
            raw = re.sub(r"\s+", " ", raw).strip()
            found = styles_from(raw)
            if found:
                return " ".join(found[:2])
            if raw:
                return raw[:40]
    make_match = re.search(r"\bmake\s+it\s+([a-z ]{2,30})\b", normalized)
    if make_match:
        raw = re.sub(r"\s+", " ", make_match.group(1)).strip()
        found = styles_from(raw)
        if found:
            return " ".join(found[:2])
    found = styles_from(normalized)
    if found:
        return " ".join(found[:2])
    return ""


def _extract_long_story_topic_hint(text: str) -> str:
    theme = _extract_story_theme(text)
    if theme and theme != "new adventure":
        cleaned = _clean_story_topic(theme)
        if cleaned:
            return cleaned
    topic = _extract_long_story_topic(text)
    topic = _clean_story_topic(topic)
    return "" if topic in {"this topic", "bedtime story", "new adventure"} else topic


def _remember_long_story_topic(state: RobotRuntimeState, topic: str) -> None:
    topic = _clean_story_topic(topic)
    if not topic:
        return
    now = time.time()
    story_topic = {"label": f"story: {topic}", "name": topic, "category": "story"}
    with state.lock:
        state.active_topic = dict(story_topic)
        state.active_topic_updated_at = now
        state.session_topic = "story"
        state.last_user_creative_subject = topic
        state.last_topic = str(story_topic["label"])
        state.last_topic_until = now + 600.0
        state.long_story_topic = str(story_topic["label"])


def _should_recover_story_context(text: str) -> bool:
    normalized = normalize_command_text(text)
    if not normalized:
        return False
    return bool(
        _is_story_continue_text(normalized)
        or _is_story_finish_request(normalized)
        or any(marker in normalized for marker in {
            "continue yesterday",
            "continue the previous story",
            "continue previous story",
            "continue last story",
            "resume the story",
            "resume yesterday",
            "go back to the story",
            "yesterday story",
            "previous story",
            "last story",
            "continua a historia",
            "continua historia",
            "continua a historia anterior",
            "volta para a historia",
            "historia anterior",
            "ultima historia",
        })
    )


def _recover_story_context_from_logs(user_text: str, state: RobotRuntimeState, max_chars: int = 1800) -> str:
    if not _should_recover_story_context(user_text):
        return ""
    with state.lock:
        current_session_id = state.conversation_log_session_id
    logs = robot_memory.select_conversation_logs(
        query=user_text,
        current_session_id=current_session_id,
        limit=5,
    )
    snippets = []
    for log in logs:
        metadata = log.get("metadata", {})
        for event in log.get("events", []):
            if event.get("type") != "turn":
                continue
            topic = normalize_command_text(event.get("topic") or "")
            user = str(event.get("user_text") or "").strip()
            assistant = str(event.get("assistant_reply") or "").strip()
            combined = normalize_command_text(user + " " + assistant)
            if (
                not _contains_story_word(topic)
                and not _contains_story_word(combined)
                and "chapter" not in combined
                and "capitulo" not in combined
            ):
                continue
            snippets.append(
                f"{metadata.get('session_id')}: user={_safe_memory_snippet(user, 160)} reply={_safe_memory_snippet(assistant, 260)}"
            )
    if not snippets:
        return ""
    text = " | ".join(snippets[-6:])
    return text[-max_chars:]


def _extract_creative_topic(text: str) -> dict | None:
    normalized = normalize_command_text(text)
    if not normalized or not any(keyword in normalized for keyword in CREATIVE_TOPIC_KEYWORDS):
        return None
    if not _is_new_story_request(normalized) and (
        normalized in LONG_STORY_ACTIVATION_PHRASES
        or normalized in {"story mode", "long story mode", "go to long story mode"}
    ):
        return None
    if _contains_real_world_harm_instruction(normalized):
        return None

    subject = _extract_called_or_named_subject(text)
    if not subject and "turtle spirit" in normalized:
        subject = "Turtle Spirit"
    elif not subject and "skeleton" in normalized:
        subject = "The Skeleton" if "superhero" in normalized or "hero" in normalized else "skeleton"

    category = "creative"
    if "story" in normalized:
        category = "story"
        if not subject:
            subject = _extract_story_theme(text)
    elif any(word in normalized for word in {"machine", "invention", "invent", "technology", "tech", "sci fi", "sci-fi", "robot"}):
        category = "invention"
    elif any(word in normalized for word in {"superhero", "hero", "character", "skeleton", "power", "powers", "turtle spirit"}):
        category = "superhero"

    label = subject or category
    if subject and category not in normalize_command_text(subject):
        label = f"{category}: {subject}"
    return {"label": label, "name": subject, "category": category}


def _topic_log_label(topic: dict | None) -> str:
    if not topic:
        return ""
    return _safe_memory_snippet(str(topic.get("label") or topic.get("name") or topic.get("category") or ""), 80)


def _remember_accepted_turn(state: RobotRuntimeState, text: str) -> None:
    snippet = _safe_memory_snippet(text)
    if not snippet:
        return
    with state.lock:
        state.recent_conversation_turns.append(snippet)
        state.recent_conversation_turns = state.recent_conversation_turns[-8:]


def _safe_log_person(state: RobotRuntimeState, fallback: str | None = None) -> str:
    with state.lock:
        person = _normalize_person_name(fallback or state.conversation_partner or state.recognized_person)
    return person or "unknown"


def _append_log_event(state: RobotRuntimeState, event_type: str, **payload) -> None:
    with state.lock:
        session_id = state.conversation_log_session_id
    if not session_id:
        return
    try:
        robot_memory.append_conversation_log_event(session_id, event_type, **payload)
    except Exception as exc:
        print("[V7.15 LOG] write warning:", exc)


def _log_user_turn_event(state: RobotRuntimeState, user_text: str, route_hint: str = "", partner: str | None = None) -> None:
    logged_at = time.time()
    with state.lock:
        state.last_logged_user_turn_at = logged_at
        state.last_logged_user_turn_text = str(user_text or "")
    _append_log_event(
        state,
        "user_turn",
        person=_safe_log_person(state, partner),
        user_text=str(user_text or ""),
        route_hint=route_hint,
        topic=_active_creative_topic(state) or "",
        conversation_mode=getattr(state, "conversation_mode", "general"),
    )


def _log_assistant_reply_event(
    state: RobotRuntimeState,
    reply_text: str,
    route: str,
    latency_override: dict | None = None,
) -> None:
    with state.lock:
        latency = dict(latency_override if latency_override is not None else state.current_turn_latency)
        user_text = latency.get("log_user_text", state.last_user_text)
        mode = latency.get("log_conversation_mode", state.conversation_mode)
        partner = latency.get("log_person", state.conversation_partner or state.recognized_person)
        topic = latency.get("log_topic", state.session_focus or state.last_topic or "")
        identity_debug = dict(latency.get("identity_debug") or {}) if route == "identity" else {}
    payload = {
        "person": _normalize_person_name(partner) or "unknown",
        "user_text": str(user_text or ""),
        "assistant_reply": str(reply_text or ""),
        "route": route,
        "topic": topic,
        "conversation_mode": mode,
    }
    if identity_debug:
        payload["identity_debug"] = identity_debug
    turn_started_at = latency.get("turn_started_at")
    route_done_at = latency.get("route_done_at") or latency.get("reply_queued_at")
    reply_queued_at = latency.get("reply_queued_at")
    if turn_started_at:
        payload["latency_ms"] = {
            "route": round(max(0.0, float(route_done_at or time.monotonic()) - float(turn_started_at)) * 1000),
            "reply_queue": round(max(0.0, float(reply_queued_at or route_done_at or time.monotonic()) - float(turn_started_at)) * 1000),
        }
        for key in ("tts_prepare", "playback_start", "playback", "total"):
            value = latency.get(f"{key}_ms")
            if value is not None:
                payload["latency_ms"][key] = round(float(value))
    payload["diagnostics"] = {
        "reply_context": str(latency.get("reply_context") or route),
        "response_length_mode": str(latency.get("response_length_mode") or ""),
        "response_depth_mode": str(latency.get("response_depth_mode") or ""),
    }
    for key in ("story_chapter", "story_chapter_count", "story_auto_continue"):
        if key in latency:
            payload["diagnostics"][key] = latency[key]
    for key in (
        "capture_total_ms",
        "speech_wait_ms",
        "speech_duration_ms",
        "endpoint_silence_ms",
        "transcription_ms",
        "story_generation_ms",
        "story_speech_slot_wait_ms",
    ):
        if latency.get(key) is not None:
            payload["diagnostics"][key] = round(float(latency[key]))
    event_type = "startup_announcement" if route == "startup" else "turn"
    _append_log_event(state, event_type, **payload)
    if event_type != "turn":
        return
    with state.lock:
        state.last_completed_user_turn_at = time.time()


def _is_audio_or_reply_health_complaint(text: str) -> bool:
    normalized = normalize_command_text(text)
    if not normalized:
        return False
    phrases = {
        "miguel nao consegue nos escutar",
        "miguel nao consegue me escutar",
        "miguel nao consegue escutar",
        "miguel nao esta escutando",
        "miguel nao ta escutando",
        "miguel nao consegue nos ouvir",
        "miguel nao consegue me ouvir",
        "miguel nao consegue ouvir",
        "miguel nao esta ouvindo",
        "miguel nao ta ouvindo",
        "miguel nao responde",
        "miguel nao esta respondendo",
        "miguel nao ta respondendo",
        "miguel nao ta pra responder",
        "miguel travou",
        "voce nao consegue me ouvir",
        "voce nao esta me ouvindo",
        "voce nao ta me ouvindo",
        "voce nao consegue nos escutar",
        "voce nao consegue me escutar",
        "you cannot hear me",
        "you can't hear me",
        "you are not hearing me",
        "you are not listening",
        "miguel cannot hear us",
        "miguel can't hear us",
        "miguel cannot hear me",
        "miguel can't hear me",
        "miguel is not responding",
        "miguel stopped responding",
    }
    return any(phrase in normalized for phrase in phrases)


def _is_sensor_health_request(text: str) -> bool:
    normalized = normalize_command_text(text)
    if not normalized:
        return False
    phrases = {
        "quick check of your sensors",
        "quick sensor check",
        "check your sensors",
        "sensor check",
        "sensors check",
        "test your sensors",
        "are your sensors working",
        "is your camera working",
        "is your vision working",
        "camera health",
        "vision health",
    }
    return any(phrase in normalized for phrase in phrases)


def _camera_runtime_health(camera_manager) -> dict:
    health = {
        "camera_thread_alive": False,
        "fresh_frame_available": False,
        "frame_age_ms": None,
        "face_state_available": False,
        "face_detected": False,
        "recognized_person": None,
        "status": "degraded",
        "error": None,
    }
    if camera_manager is None:
        health["error"] = "camera_manager_missing"
        return health

    thread = getattr(camera_manager, "thread", None)
    health["camera_thread_alive"] = bool(thread and thread.is_alive())

    try:
        snapshot = camera_manager.get_latest_frame(require_fresh=True, wait_timeout=0.8)
        if snapshot is not None:
            health["fresh_frame_available"] = True
            captured_at = float(getattr(snapshot, "captured_at", 0.0) or 0.0)
            if captured_at:
                health["frame_age_ms"] = round(max(0.0, time.time() - captured_at) * 1000)
    except Exception as exc:
        health["error"] = str(exc)[:200]

    try:
        face_state = camera_manager.get_face_state(max_age_seconds=2.0)
        health["face_state_available"] = bool(face_state)
        health["face_detected"] = bool(face_state.get("face_detected")) if face_state else False
        health["recognized_person"] = _normalize_person_name(face_state.get("recognized_person")) if face_state else None
    except Exception as exc:
        if not health["error"]:
            health["error"] = str(exc)[:200]

    health["status"] = "online" if health["camera_thread_alive"] and health["fresh_frame_available"] else "degraded"
    return health


def _sensor_health_reply(health: dict) -> str:
    if health.get("status") == "online":
        frame_age = health.get("frame_age_ms")
        frame_part = f"fresh, about {frame_age} milliseconds old" if frame_age is not None else "fresh"
        if health.get("recognized_person"):
            face_part = f"face recognition currently sees {_friendly_person_name(health.get('recognized_person'))}"
        elif health.get("face_detected"):
            face_part = "face recognition sees a face but has not confirmed a name"
        else:
            face_part = "face recognition does not currently see a confirmed face"
        return f"Sensor check: camera thread is running, the frame is {frame_part}, and {face_part}."

    if health.get("camera_thread_alive"):
        return "Sensor check: camera thread is running, but I do not have a fresh frame right now."
    return "Sensor check: camera is not delivering fresh frames right now."


def _route_sensor_health_local_reply(user_text: str, camera_manager, state: RobotRuntimeState) -> bool:
    if not _is_sensor_health_request(user_text):
        return False
    health = _camera_runtime_health(camera_manager)
    with state.lock:
        state.current_turn_latency["camera_health"] = health
    _set_reply_context(state, "sensor_health")
    v6.speak(_sensor_health_reply(health))
    return True


def _route_audio_or_reply_health_local_reply(user_text: str, state: RobotRuntimeState) -> bool:
    if not _is_audio_or_reply_health_complaint(user_text):
        return False
    _set_reply_context(state, "audio_health")
    normalized = normalize_command_text(user_text)
    if any(token in normalized for token in {"nao", "voce", "miguel travou"}):
        reply = (
            "Eu ouvi essa frase. Se pareci parado, o atraso esta na resposta ou na fala, "
            "nao na captura do audio. Vou responder mais curto agora."
        )
    else:
        reply = (
            "I heard that sentence. If I seemed stuck, the delay is in reply generation "
            "or speech playback, not audio capture. I will answer shorter now."
        )
    v6.speak(reply)
    return True


def _update_active_topic_from_text(state: RobotRuntimeState, text: str) -> dict | None:
    topic = _extract_creative_topic(text)
    if not topic:
        return None
    now = time.time()
    with state.lock:
        state.active_topic = dict(topic)
        state.active_topic_updated_at = now
        state.last_user_creative_subject = str(topic.get("name") or topic.get("label") or topic.get("category") or "")
        state.last_topic = str(topic.get("label") or topic.get("category") or "creative")
        state.last_topic_until = now + 300.0
        if topic.get("category") in {"superhero", "invention", "story"}:
            state.conversation_mode = "story" if topic.get("category") == "story" else "creative"
    print(f"[V7.14 MEMORY] active_topic={_topic_log_label(topic)}")
    return topic


def _start_new_story_topic(state: RobotRuntimeState, text: str, partner: str | None = None) -> str:
    theme = _clean_story_topic(_extract_story_theme(text)) or "new adventure"
    topic = {"label": f"story: {theme}", "name": theme, "category": "story"}
    now = time.time()
    target_minutes = _extract_long_story_duration_minutes(text)
    story_style = _extract_story_style(text)
    recovered_context = _recover_story_context_from_logs(text, state)
    with state.lock:
        state.active_topic = dict(topic)
        state.active_topic_updated_at = now
        state.session_focus = ""
        state.session_topic = "story"
        state.last_user_creative_subject = theme
        state.last_topic = str(topic["label"])
        state.last_topic_until = now + 600.0
        state.long_story_topic = str(topic["label"])
        state.long_story_segment_index = 0
        state.long_story_max_segments = 0
        state.long_story_target_minutes = target_minutes
        state.long_story_style = story_style
        state.recovered_story_context = recovered_context
        state.conversation_mode = "story"
    start_conversation_session(state, mode="story", partner=partner or _current_owner_partner(state), reason="new_story")
    print(f"[V7.15 STORY] new_topic={_topic_log_label(topic)}")
    return theme


def _store_interrupted_creative_topic(state: RobotRuntimeState, text: str) -> None:
    topic = _extract_creative_topic(text)
    if not topic:
        return
    with state.lock:
        state.last_interrupted_user_topic = dict(topic)
    print(f"[V7.14 CONTEXT] stored_interrupted_topic={_topic_log_label(topic)}")


def _is_contextual_followup(text: str) -> bool:
    normalized = normalize_command_text(text)
    if not normalized:
        return False
    if _is_story_continue_text(normalized):
        return True
    if normalized in CONTEXTUAL_FOLLOWUP_PHRASES:
        return True
    if any(phrase in normalized for phrase in CONTEXTUAL_FOLLOWUP_PHRASES):
        return True
    words = normalized.split()
    if len(words) <= 7 and any(token in words for token in {"he", "him", "his", "it", "its"}):
        return any(token in words for token in {"how", "what", "can", "does", "would", "should", "powers", "work", "fly", "do"})
    return False


def _current_active_topic(state: RobotRuntimeState) -> dict | None:
    with state.lock:
        topic = dict(state.active_topic or {})
        updated_at = float(state.active_topic_updated_at or 0.0)
        fallback = state.last_topic or ""
        fallback_until = float(state.last_topic_until or 0.0)
    if topic and (not updated_at or time.time() - updated_at <= 600.0):
        return topic
    if fallback and time.time() <= fallback_until:
        return {"label": fallback, "category": "creative"}
    return None


def _recover_contextual_followup_prompt(text: str, state: RobotRuntimeState) -> str:
    if not (_is_contextual_followup(text) or _is_story_finish_request(text)):
        return text

    topic = _current_active_topic(state)
    if not topic:
        with state.lock:
            interrupted = dict(state.last_interrupted_user_topic or {})
            if interrupted:
                state.active_topic = dict(interrupted)
                state.active_topic_updated_at = time.time()
                state.last_interrupted_user_topic = None
        if interrupted:
            topic = interrupted
            print(f"[V7.14 CONTEXT] recovered_interrupted_topic={_topic_log_label(topic)}")

    if not topic:
        return text

    label = _topic_log_label(topic)
    print(f"[V7.14 MEMORY] recovered_context topic={label}")
    category = str(topic.get("category") or "").strip().lower()
    if category == "story" or _is_story_continue_text(text) or _is_story_finish_request(text):
        if _is_story_finish_request(text):
            return f"Finish the current story with a real ending: {label}. User asks: {text}"
        return f"Continue the current story: {label}. User asks: {text}"
    return f"Continue the current creative topic: {label}. User asks: {text}"


CREATIVE_CONTINUATION_PHRASES = {
    "something moving around him",
    "moving around his body",
    "slithering",
    "weakness",
    "extra powers",
    "how he works",
    "maybe",
    "what if",
    "could be",
    "try again",
    "make it cooler",
}


def _active_creative_topic(state: RobotRuntimeState) -> str:
    now = time.time()
    with state.lock:
        mode = state.conversation_mode
        focus = state.session_focus or ""
        active = dict(state.active_topic or {})
        topic = state.last_topic or ""
        topic_active = bool(topic and now <= float(state.last_topic_until or 0.0))
    if focus:
        return focus
    if active:
        return str(active.get("label") or active.get("name") or active.get("category") or "")
    if topic_active:
        return topic
    if mode == "creative":
        return topic or "creative"
    return ""


def _is_creative_continuation(text: str, state: RobotRuntimeState) -> str | None:
    normalized = normalize_command_text(text)
    topic = _active_creative_topic(state)
    if not topic:
        return None
    if "skeleton superhero" in topic or "skeleton superhero" in normalize_command_text(topic):
        if any(phrase in normalized for phrase in CREATIVE_CONTINUATION_PHRASES):
            return "skeleton superhero"
    with state.lock:
        creative_mode = state.conversation_mode == "creative"
    if creative_mode and any(phrase in normalized for phrase in CREATIVE_CONTINUATION_PHRASES):
        return topic or "creative"
    return None


def _creative_fast_allow_topic(text: str) -> str | None:
    normalized = normalize_command_text(text)
    if not any(keyword in normalized for keyword in CREATIVE_FAST_ALLOW_KEYWORDS):
        return None
    if _contains_real_world_harm_instruction(normalized):
        return None
    if "skeleton" in normalized:
        return "skeleton superhero"
    if "superhero" in normalized or "hero" in normalized:
        return "superhero"
    if "villain" in normalized:
        return "villain"
    if any(marker in normalized for marker in {"machine", "robot idea", "invention", "invent", "sci fi", "sci-fi", "technology", "tech idea"}):
        return "invention"
    if "story" in normalized:
        return "story"
    return "creative"


def _looks_like_asr_ambiguous_creative_text(text: str, state: RobotRuntimeState) -> bool:
    normalized = normalize_command_text(text)
    if not normalized:
        return False
    with state.lock:
        mode = state.conversation_mode
        topic = " ".join(
            str(part or "")
            for part in (
                state.session_focus,
                state.last_topic,
                _topic_log_label(state.active_topic),
            )
        ).lower()
    if mode not in {"creative", "story"} and not any(marker in topic for marker in {"hero", "superhero"}):
        return False
    if re.search(r"\bthe bad\b|\bbad to be\b|\bbad is useless\b|\bbath\b", normalized):
        return True
    if sum(1 for word in {"bat", "bad", "bash", "bath"} if re.search(rf"\b{word}\b", normalized)) >= 2:
        return True
    return False


def _route_asr_ambiguity_clarification(user_text: str, state: RobotRuntimeState) -> bool:
    if not is_conversation_active(state):
        return False
    if not _looks_like_asr_ambiguous_creative_text(user_text, state):
        return False
    extend_conversation_session(state, seconds=20.0, reason="asr_clarification")
    _set_reply_context(state, "clarification")
    _set_response_length_context(state, "terse")
    print(f"[V7.15 ASR] clarification text={_short_log_text(user_text)}")
    v6.speak("Did you mean bat, bash, or bad?")
    return True


def _safety_guard_route_reason(text: str, route_hint: str | None = None, conversation_mode: str | None = None) -> tuple[bool, str]:
    normalized = normalize_command_text(text)
    route = str(route_hint or "").strip().lower()
    mode = str(conversation_mode or "").strip().lower()
    if not normalized:
        return False, "empty"

    if _contains_real_world_harm_instruction(normalized):
        return True, "harm_or_weapon_marker"

    high_risk_markers = {
        "suicide",
        "kill myself",
        "hurt myself",
        "self harm",
        "hurt someone",
        "harm someone",
        "make a bomb",
        "build a bomb",
        "make poison",
        "illegal",
        "steal",
        "break into",
        "hack into",
        "bypass security",
        "medical emergency",
        "emergency medicine",
        "choking",
        "heart attack",
        "overdose",
    }
    if any(marker in normalized for marker in high_risk_markers):
        return True, "risk_sensitive_content"

    security_markers = {
        "password",
        "owner mode",
        "unlock owner",
        "enroll",
        "enrolling",
        "learn this face",
        "add a new face",
        "remember this person",
        "add my friend",
        "this is my friend",
    }
    if any(marker in normalized for marker in security_markers):
        return True, "owner_security_or_enrollment"

    if _is_mode_command_not_physical(normalized):
        return False, "mode_command_not_physical"

    if (
        full.is_local_robot_control_request(normalized)
        or _is_shutdown_request_text(normalized)
        or _is_shutdown_confirm_text(normalized)
        or _is_shutdown_cancel_text(normalized)
        or _is_sleep_mode_request(normalized)
        or _is_sleep_wake_request(normalized)
    ):
        return True, "physical_or_destructive_command"

    ambiguous_action_markers = {
        "move",
        "go forward",
        "back up",
        "turn left",
        "turn right",
        "grab",
        "push",
        "pull",
        "open the door",
        "run motor",
        "motor",
        "servo",
        "gpio",
    }
    if any(marker in normalized for marker in ambiguous_action_markers):
        return True, "ambiguous_real_world_action"

    local_ack_phrases = {
        "hi",
        "hello",
        "yo",
        "hey",
        "okay",
        "ok",
        "yes",
        "no",
        "can you hear me",
        "do you hear me",
        "you hear me",
        "are you listening",
        "are you there",
    }
    if normalized in local_ack_phrases:
        return False, "local_ack"

    if route in {"local_ack", "timer", "utility", "identity", "scene", "greeting", "creative", "story"}:
        return False, f"route_{route}"

    safe_markers = {
        "tell me a joke",
        "science joke",
        "be creative",
        "creative mode",
        "superhero",
        "imaginary hero",
        "character",
        "skeleton",
        "turtle spirit",
        "machine",
        "robot idea",
        "invention",
        "invent",
        "sci fi",
        "sci-fi",
        "technology",
        "tech concept",
        "book",
        "story",
        "continue the story",
        "what time is it",
        "weather",
        "calculate",
        "status",
        "timer",
        "can you see me",
        "who am i",
        "do you know who i am",
    }
    if any(marker in normalized for marker in safe_markers):
        return False, "safe_normal_or_creative"

    if mode in {"creative", "story"} and _is_contextual_followup(normalized):
        return False, f"{mode}_context_followup"

    return False, "normal_conversation"


def _is_mode_command_not_physical(text: str) -> bool:
    normalized = normalize_command_text(text)
    if not normalized:
        return False
    if any(marker in normalized for marker in {"shutdown", "sleep mode", "owner mode", "password", "enroll"}):
        return False
    exact = {
        "go creative",
        "go to creative mode",
        "go to long story mode",
        "go to long explanation mode",
        "activate long explanation mode",
        "turn on long explanation mode",
        "turn on detailed mode",
        "use longer answers from now on",
        "go to normal mode",
        "switch to creative mode",
        "switch back to robot project",
        "switch to project mode",
        "entra em modo historia",
        "entrar em modo historia",
        "keep creative mode",
        "stay in creative mode",
        "normal mode",
    }
    if normalized in exact:
        return True
    polite_prefixes = (
        "can you ",
        "could you ",
        "would you ",
        "please ",
        "can we ",
        "let s ",
        "lets ",
        "but ",
        "but still ",
        "still ",
    )
    stripped = normalized
    changed = True
    while changed:
        changed = False
        for prefix in polite_prefixes:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):].strip()
                changed = True
    if stripped in exact:
        return True
    mode_markers = {
        "creative mode",
        "long story mode",
        "modo historia",
        "long explanation mode",
        "normal mode",
        "robot project",
        "project mode",
    }
    command_markers = {
        "go to",
        "switch to",
        "switch back to",
        "activate",
        "turn on",
        "use",
        "entra em",
        "entrar em",
        "keep",
        "stay in",
        "go creative",
    }
    return any(marker in stripped for marker in mode_markers) and any(command in stripped for command in command_markers)


def _local_safety_hard_stop(text: str) -> tuple[str | None, str | None]:
    normalized = normalize_command_text(text)
    if not normalized:
        return None, None

    drug_markers = {
        "how to make illegal drugs",
        "make illegal drugs",
        "instructions for illegal drugs",
        "cook meth",
        "make meth",
        "make cocaine",
        "make fentanyl",
        "make heroin",
        "illegal drug recipe",
    }
    if any(marker in normalized for marker in drug_markers):
        return "illegal_drugs", "I can't help with illegal drugs, but I can help with something safe."

    self_harm_markers = {
        "kill myself",
        "hurt myself",
        "how do i suicide",
        "how to suicide",
        "commit suicide",
        "self harm",
    }
    if any(marker in normalized for marker in self_harm_markers):
        return "self_harm", "I can't help with self-harm. Please tell a trusted adult right now."

    violence_markers = {
        "find weapons to kill someone",
        "weapon to kill someone",
        "weapons to kill someone",
        "weapons to hurt someone",
        "kill someone",
        "hurt someone",
        "build a bomb",
        "make a bomb",
        "poison someone",
        "stab someone",
        "shoot someone",
    }
    if any(marker in normalized for marker in violence_markers):
        return "violence", "I can't help with hurting anyone. Please step away from anything dangerous and get a trusted adult now."

    return None, None


def should_run_safety_guard(text: str, route_hint: str | None = None, conversation_mode: str | None = None) -> bool:
    required, reason = _safety_guard_route_reason(text, route_hint=route_hint, conversation_mode=conversation_mode)
    state = "required" if required else "skipped"
    print(f"[V7.14 SAFETY ROUTER] {state} reason={reason}")
    return required


def _is_correction_retry_text(text: str) -> bool:
    normalized = normalize_command_text(text)
    return any(marker in normalized for marker in CORRECTION_RETRY_MARKERS)


def infer_response_length_mode(text: str, conversation_mode: str = "general", camera_intent: str = "none") -> str:
    normalized = normalize_command_text(text)
    explicit_detail_request = any(
        phrase in normalized
        for phrase in {
            "talk longer",
            "a little bit longer",
            "little bit longer",
            "explain more",
            "tell me more",
            "go deeper",
            "take your time",
            "give more detail",
            "more on the answer",
            "more in the answer",
            "cutting your answer",
            "cutting your answers",
            "answers are too short",
            "answer is too short",
        }
    )
    story_detection = _story_mode_detection(text)
    if story_detection.get("mode") == "long_story" and story_detection.get("action") == "generate_story":
        return "long_story"
    if _extract_long_story_duration_minutes(text) and _contains_story_word(normalized):
        return "long_story"
    if _is_explicit_long_story_request(text) or any(
        phrase in normalized
        for phrase in {
            "long story mode",
            "tell me a long story",
            "tell me a bedtime story",
            "tell me a long explanation",
            "explain in detail for a long time",
            "give me the full explanation",
            "teach me this topic",
        }
    ):
        return "long_story" if _contains_story_word(normalized) else "detailed"
    if explicit_detail_request:
        return "detailed"
    if any(
        phrase in normalized
        for phrase in {
            "movie",
            "movies",
            "star wars",
            "sequels",
            "trilogy",
            "disney",
            "theater",
            "theatre",
            "what else",
            "tell me more",
            "full answer",
            "long story mode",
            "long story",
            "bedtime story",
            "long explanation",
            "full explanation",
            "teach me this topic",
            "talk about",
        }
    ):
        if any(phrase in normalized for phrase in {"tell me more", "full answer", "give me the full answer"}):
            return "detailed"
        return "normal"
    if conversation_mode == "story" or any(phrase in normalized for phrase in {"story mode", "modo historia", "tell me a story", "conte uma historia", "continue the story", "continua a historia"}):
        return "story"
    if any(
        phrase in normalized
        for phrase in {
            "long conversation",
            "talk longer",
            "a little bit longer",
            "little bit longer",
            "explain",
            "explain more",
            "tell me more",
            "go deeper",
            "detailed",
            "take your time",
            "give more detail",
            "more on the answer",
            "more in the answer",
            "cutting your answer",
            "cutting your answers",
            "answers are too short",
            "answer is too short",
        }
    ):
        return "detailed"
    if (
        camera_intent == "identity_camera"
        or _is_voice_command_text(normalized)
        or _is_shutdown_request_text(normalized)
        or _is_shutdown_cancel_text(normalized)
        or _is_shutdown_confirm_text(normalized)
        or any(phrase in normalized for phrase in {"what time is it", "current time", "status"})
    ):
        return "terse"
    if camera_intent in {"scene_camera", "camera_generic"}:
        return "detailed" if "describe in detail" in normalized else "normal"
    return "normal"


def _route_allows_terse_response(text: str, conversation_mode: str = "general", camera_intent: str = "none") -> bool:
    normalized = normalize_command_text(text)
    if camera_intent == "identity_camera":
        return True
    return bool(
        _is_bare_wake_phrase(text)
        or _is_voice_command_text(normalized)
        or _is_shutdown_request_text(normalized)
        or _is_shutdown_cancel_text(normalized)
        or _is_shutdown_confirm_text(normalized)
        or _is_password_session_command(normalized)
        or normalized in {"status", "what time is it", "current time", "time", "date", "what date is it"}
        or normalized in {"hi", "hello", "yes", "no", "ok", "okay", "stop", "pause", "cancel"}
    )


def _response_length_instruction(mode: str) -> str:
    mode = str(mode or "normal").strip().lower()
    if mode == "terse":
        return "Answer in 1 short sentence."
    if mode == "detailed":
        return "Give the full answer in 1 to 2 short paragraphs."
    if mode == "story":
        return "Answer as one short story segment, about 80 to 120 words."
    if mode == "long_story":
        return "Answer as a real longer story, about 180 to 250 spoken words in 2 to 4 short spoken paragraphs, ending cleanly."
    return "Answer in 2 to 5 short sentences."


def _with_response_length_instruction(user_text: str, mode: str) -> str:
    instruction = _response_length_instruction(mode)
    return f"{user_text}\n\nMiguel response length instruction: {instruction}"


def _with_cloud_reply_instructions(
    user_text: str,
    mode: str,
    route_hint: str = "normal",
    conversation_mode: str = "general",
    response_depth_mode: str = "normal",
    active_topic: str = "",
    allowed_languages: list[str] | None = None,
    long_story_target_minutes: int = 0,
    story_style: str = "",
    recovered_story_context: str = "",
) -> str:
    prompt = _with_response_length_instruction(user_text, mode)
    depth = str(response_depth_mode or "normal").strip().lower()
    allowed_languages = allowed_languages or _default_allowed_conversation_languages()
    prompt += (
        "\nMiguel language instruction: Reply only in "
        f"{_format_language_list(allowed_languages)} unless the user changes the language filter."
    )
    if route_hint == "creative" or conversation_mode in {"creative", "story"}:
        prompt += (
            "\nMiguel creative instruction: Continue and add to the idea without unnecessary clarification. "
            "Use vivid but concise child-safe details. Usually give one strong idea or at most two options."
        )
    if depth == "long_story" and (route_hint == "creative" or conversation_mode in {"creative", "story"}):
        topic_line = f" Current creative topic: {active_topic}." if active_topic else ""
        duration_line = ""
        if long_story_target_minutes:
            target_words = _long_story_spoken_cap_words(True)
            duration_line = (
                f" The user asked for about {_format_long_story_duration(long_story_target_minutes)}; treat that as a request for a rich story, "
                f"but keep this spoken segment around 180 to {target_words} words for voice usability."
            )
        style_line = f" Requested story style: {story_style}." if story_style else ""
        recovered_line = (
            f" Previous story context to continue from: {recovered_story_context}. "
            if recovered_story_context
            else ""
        )
        paragraph_instruction = (
            "Use 2 to 4 short spoken paragraphs"
            if long_story_target_minutes
            else "Answer in 2 to 4 short spoken paragraphs"
        )
        prompt += (
            f"\nMiguel long story instruction: {paragraph_instruction}. "
            "Continue the remembered idea when context is available. Use vivid, family-safe details. "
            "Include a beginning, middle, and a real closed ending for the current story or chapter. "
            "Do not stop mid-scene. You may close with 'The End of Chapter 1' if future chapters can continue later. "
            f"{duration_line} "
            f"{style_line} "
            f"{recovered_line}"
            f"{topic_line} For superheroes include name, problem, obstacle, creative solution, and a closed chapter ending. "
            "For machines include what it does, how it works in kid-friendly terms, one fun feature, and one possible next upgrade."
        )
    elif depth == "long_explanation":
        prompt += (
            "\nMiguel long explanation instruction: Give a clear spoken step-by-step explanation, roughly 80 to 180 words, "
            "natural and easy to follow."
        )
    return prompt


def _live_conversation_context_for_cloud(state: RobotRuntimeState) -> str:
    with state.lock:
        partner = _normalize_person_name(state.conversation_partner or state.recognized_person)
        preferred_name = state.preferred_address_name if state.preferred_address_until > time.time() else None
        mode = state.conversation_mode
        topic = state.session_focus or state.last_topic or ""
        recent_turns = list(state.recent_conversation_turns[-6:])
        story_style = state.long_story_style
        recovered_story_context = state.recovered_story_context
    parts = [
        f"conversation_mode={mode or 'general'}",
        f"current_person={partner or 'unknown'}",
    ]
    if topic:
        parts.append(f"current_topic={topic}")
    if recent_turns:
        parts.append("recent_user_turns=" + " | ".join(recent_turns))
    if story_style:
        parts.append(f"story_style={story_style}")
    if recovered_story_context:
        parts.append("recovered_story_context=" + recovered_story_context[-500:])
    if preferred_name:
        parts.append(f"preferred_address_name={_friendly_person_name(preferred_name)}")
    parts.append(
        "name_rule=use preferred_address_name when present; otherwise use current_person only; "
        "a preferred address never grants identity or owner authorization"
    )
    parts.append(
        "camera_rule=do not mention camera availability, face detection, or visual state unless the current user request asks about vision"
    )
    return "\nMiguel live conversation context: " + "; ".join(parts)


def _is_global_without_wake_command(text: str) -> bool:
    normalized = normalize_command_text(text)
    return bool(
        is_barge_in_command(normalized)
        or _is_sleep_mode_request(normalized)
        or _is_sleep_wake_request(normalized)
        or _is_shutdown_request_text(normalized)
        or _is_shutdown_confirm_text(normalized)
        or _is_shutdown_cancel_text(normalized)
        or _is_password_session_command(normalized)
        or _is_harmless_local_bypass_request(normalized)
        or normalized in {"status", "emergency status", "pause", "cancel", "stop"}
    )


def _is_password_session_command(text: str, state: RobotRuntimeState | None = None) -> bool:
    normalized = normalize_command_text(text)
    if any(
        phrase in normalized
        for phrase in {
            "activate owner mode",
            "can you activate owner mode",
            "enable owner mode",
            "owner mode",
            "turn on owner mode",
            "unlock owner mode",
            "lock owner mode",
            "require face recognition",
            "is password mode configured",
            "is owner password configured",
        }
    ):
        return True
    if state is not None:
        with state.lock:
            pending_unlock = bool(state.pending_owner_unlock_until and time.time() <= float(state.pending_owner_unlock_until))
        return pending_unlock
    return False


def _is_harmless_local_bypass_request(text: str) -> bool:
    normalized = normalize_command_text(text)
    if _is_mode_command_not_physical(normalized):
        return True
    return normalized in {
        "be creative",
        "creative mode",
        "long story mode",
        "science joke",
        "short answer",
        "sleep mode",
        "story mode",
        "talk longer",
        "talk normally",
        "tell me a joke",
        "tell me a science joke",
    }


def _show_wake_required(state: RobotRuntimeState, transcript: str = "", reason: str = "wake_required") -> None:
    with state.lock:
        state.conversation_active = False
        state.conversation_mode = "wake_required"
        state.conversation_partner = None
        state.wake_required = True
        state.wake_required_reason = reason
    if transcript:
        print(f"[V7.14 WAKE REQUIRED] ignored transcript={_short_log_text(transcript)}")
    notify_face_status(state, "wake_required", _wake_required_face_text())


def _short_answer_after_robot_question(text: str, state: RobotRuntimeState) -> bool:
    normalized = normalize_command_text(text)
    if not normalized:
        return False
    low_signal = {
        "and i will figure",
        "i will figure",
        "i will figure it out",
        "let me figure",
        "let me figure it out",
    }
    if normalized in low_signal:
        return False
    with state.lock:
        prompt_type = state.last_prompt_type
        question_type = state.last_robot_question_type
        asked_at = state.last_robot_question_at
    if not (prompt_type or question_type or (asked_at and time.time() - asked_at < 20.0)):
        return False
    return _word_count(normalized) <= 4


def _looks_like_expected_slot_answer(text: str, state: RobotRuntimeState) -> bool:
    normalized = normalize_command_text(text)
    if not normalized:
        return False
    with state.lock:
        expected_slot = state.last_robot_question_expected_slot
        asked_at = float(state.last_robot_question_at or 0.0)
    if expected_slot != "movie_theater_location" or not asked_at or time.time() - asked_at > 90.0:
        return False
    location_markers = {
        "theater",
        "theatre",
        "santana",
        "row",
        "san jose",
        "california",
        "cinema",
        "movie",
    }
    return _word_count(normalized) <= 12 or any(marker in normalized for marker in location_markers)


def _is_topic_continuation(text: str, state: RobotRuntimeState) -> bool:
    normalized = normalize_command_text(text)
    if not normalized:
        return False
    with state.lock:
        topic = state.last_topic
        until = float(state.last_topic_until or 0.0)
    if not topic or time.time() > until:
        return False
    return normalized in {"what else", "tell me more", "continue", "go on"} or "what else" in normalized


V715_SHORT_FOLLOWUP_PHRASES = {
    "wow",
    "nice",
    "cool",
    "interesting",
    "continue",
    "continua",
    "keep going",
    "tell me more",
    "make it longer",
    "faz maior",
    "nao e isso",
    "tem que ser",
    "that was good",
    "switch back to robot project",
    "robot project",
    "keep creative mode",
    "still creative mode",
    "but keep creative mode",
    "but still keep on creative mode",
    "but still keep creative mode",
}


def _is_v715_short_followup_text(text: str) -> bool:
    normalized = normalize_command_text(text)
    if not normalized:
        return False
    if normalized in V715_SHORT_FOLLOWUP_PHRASES:
        return True
    return any(
        marker in normalized
        for marker in {
            "switch back to robot project",
            "keep creative mode",
            "still creative mode",
            "tell me more",
            "make it longer",
            "continua",
            "faz maior",
            "nao e isso",
            "tem que ser",
        }
    )


def _accept_v715_short_followup_if_allowed(
    text: str,
    state: RobotRuntimeState,
    familiar_present: bool = False,
) -> bool:
    if not _is_v715_short_followup_text(text):
        return False
    normalized = normalize_command_text(text)
    now = time.time()
    with state.lock:
        active = bool(state.conversation_active and now <= float(state.conversation_until or 0.0))
        mode = state.conversation_mode
        partner = state.conversation_partner or _normalize_person_name(state.recognized_person) or "unknown_wake_user"
        last_reply_recent = bool(state.last_reply_time and now - float(state.last_reply_time or 0.0) <= 180.0)
    allowed_mode = mode in {"creative", "story", "project"}
    mode_command = any(marker in normalized for marker in {"creative mode", "robot project", "project mode"})
    if not ((active and allowed_mode) or (familiar_present and (allowed_mode or mode_command or last_reply_recent))):
        return False
    accepted_mode = mode if allowed_mode else _infer_conversation_mode(text)
    _force_active_after_mode(state, accepted_mode, partner=partner, reason="short_followup")
    print(f"[V7.15 SESSION] accepted_short_followup mode={accepted_mode} text={_short_log_text(text)}")
    return True


def is_directed_to_miguel(text: str, state: RobotRuntimeState) -> bool:
    normalized = normalize_command_text(text)
    if not normalized:
        return False
    raw = str(text or "").lower().strip(" .,:;!?")
    if raw in {"não", "nao", "tare"}:
        return False
    if _has_v7_5_wake_phrase(text) or "miguel" in normalized.split():
        return True
    with state.lock:
        active = bool(state.conversation_active and time.time() <= float(state.conversation_until or 0.0))
        last_reply_time = float(state.last_reply_time or 0.0)
        mode = state.conversation_mode
        topic = state.session_topic or ""
        focus = state.session_focus or ""

    if active:
        with state.lock:
            long_story_active = bool(state.long_story_active)
        if _is_enrollment_request_text(text):
            return True
        if long_story_active and _is_long_mode_continue(text):
            return True
        words = set(normalized.split())
        clear_human_to_human = {
            "tell dad",
            "ask mom",
            "come here",
            "pass me",
            "where is my phone",
            "what did you say to him",
        }
        if any(phrase in normalized for phrase in clear_human_to_human):
            return False
        question_starters = (
            "what",
            "who",
            "why",
            "how",
            "can you",
            "do you",
            "are you",
            "did you",
            "would you",
            "could you",
            "should we",
        )
        local_routes = {
            "what are you",
            "who am i",
            "how are you",
            "can you hear me",
            "what do you see",
            "who do you see",
            "what time is it",
            "do you hear me",
            "you hear me",
            "are you listening",
            "are you there",
            "hello",
            "yo",
            "status",
            "voice",
            "robot voice",
            "natural voice",
            "shutdown",
        }
        complaint_markers = {
            "answer me",
            "not answering",
            "why you don t answer",
            "why you dont answer",
            "why are you not answering",
            "why you are not answering",
            "did you hear",
            "repeat myself",
        }
        short_followups = {
            "yes",
            "no",
            "ok",
        "okay",
        "like now",
        "continue",
        "next part",
        "keep going",
        "go on",
        "think about it",
            "the skeleton",
            "skeleton",
        }
        if "you" in words or "your" in words:
            return True
        if str(text or "").strip().endswith("?") or normalized.startswith(question_starters):
            return True
        if any(normalized == route or normalized.startswith(route + " ") for route in local_routes):
            return True
        if any(marker in normalized for marker in complaint_markers):
            return True
        if normalized in short_followups and last_reply_time and time.time() - last_reply_time <= 30.0:
            return True

    direct_questions = {
        "what do you think",
        "how would it work",
        "how would he work",
        "can you explain",
        "what is that",
        "who am i",
        "what happens next",
    }
    if any(phrase in normalized for phrase in direct_questions):
        return True
    request_phrases = {
        "tell me",
        "continue",
        "think about it",
        "look",
        "describe",
        "show me",
        "switch voice",
        "explain",
        "let s do that",
        "lets do that",
    }
    if any(normalized == phrase or normalized.startswith(phrase + " ") for phrase in request_phrases):
        return True
        if _short_answer_after_robot_question(text, state):
            return True
        if _looks_like_expected_slot_answer(text, state):
            return True
        if _is_topic_continuation(text, state):
            return True
        if mode in {"creative", "story", "project"} and _is_v715_short_followup_text(text):
            return True
        if mode in {"creative", "story", "project"}:
            if normalized in {"yes", "no", "okay", "ok", "sure", "continue", "the skeleton", "skeleton", "creative", "story"}:
                return True
        if topic and topic in normalized:
            return True
        if focus and any(word in normalized for word in focus.split()):
            return True
    return False


def _likely_background_speech_reason(text: str, state: RobotRuntimeState) -> str:
    normalized = normalize_command_text(text)
    if not normalized:
        return "empty"
    human_to_human = {
        "tell dad",
        "ask mom",
        "come here",
        "pass me",
        "where is my phone",
        "what did you say to him",
    }
    if any(phrase in normalized for phrase in human_to_human):
        return "human_to_human"
    if "you" not in set(normalized.split()) and "your" not in set(normalized.split()):
        return "no_direct_address"
    return "ambiguous"


_NUMBER_WORDS = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "to": "2",
    "too": "2",
    "three": "3",
    "four": "4",
    "for": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "ate": "8",
    "nine": "9",
}


def _normalize_owner_password_value(text: str) -> str:
    normalized = _normalize_for_echo(str(text or ""))
    tokens = normalized.split()
    converted = [_NUMBER_WORDS.get(token, token) for token in tokens]
    return " ".join(converted)


def _owner_password_variants(text: str) -> set[str]:
    spaced = _normalize_owner_password_value(text)
    compact = spaced.replace(" ", "")
    variants = {spaced}
    if compact:
        variants.add(compact)
    return {variant for variant in variants if variant}


def _owner_password_matches(supplied: str, configured_password: str) -> bool:
    expected = _owner_password_variants(configured_password)
    attempt = _owner_password_variants(supplied)
    return bool(expected and attempt and expected.intersection(attempt))


def _password_env_configured() -> bool:
    return bool(os.getenv("MIGUEL_OWNER_PASSWORD_PHRASE", "").strip())


def _log_password_env_configured_once(state: RobotRuntimeState) -> None:
    configured = _password_env_configured()
    with state.lock:
        if state.password_env_logged:
            return
        state.password_env_logged = True
    print(f"[V7.14 PASSWORD SESSION] env_configured={str(configured).lower()}")


def _route_password_owner_session(user_text: str, state: RobotRuntimeState) -> bool:
    normalized = normalize_command_text(user_text)
    password = os.getenv("MIGUEL_OWNER_PASSWORD_PHRASE", "").strip()
    if any(phrase in normalized for phrase in {"is password mode configured", "is owner password configured"}):
        _log_password_env_configured_once(state)
        if password:
            v6.speak("Password mode is configured.")
        else:
            v6.speak("Password mode is not configured.")
        return True

    if any(
        phrase in normalized
        for phrase in {
            "what is the password",
            "what s the password",
            "whats the password",
            "say the password",
            "tell me the password",
            "what is the secret phrase",
            "what s the secret phrase",
            "whats the secret phrase",
            "say the secret phrase",
            "tell me the secret phrase",
        }
    ):
        v6.speak("I can't say the secret phrase out loud.")
        return True

    if any(
        phrase in normalized
        for phrase in {
            "how do i avoid wake phrase",
            "how do i avoid the wake phrase",
            "how can i avoid wake phrase",
            "how can i avoid the wake phrase",
            "how do i stop saying hey miguel",
            "how can i stop saying hey miguel",
            "how do i talk without wake phrase",
            "how can i talk without the wake phrase",
            "how do i talk without saying miguel",
            "how can i talk without saying miguel",
        }
    ):
        v6.speak("Owners can use face recognition or owner mode.")
        return True

    if normalized in {"lock owner mode", "require face recognition", "miguel lock owner mode", "miguel require face recognition"}:
        _end_password_session(state)
        with state.lock:
            recognized = _normalize_person_name(state.recognized_person)
            owner_active = bool(
                _is_owner(recognized)
                and time.time() - float(state.recognized_person_updated_at or 0.0) <= 3.0
            )
            if not owner_active:
                state.conversation_active = False
                state.conversation_mode = "wake_required"
                state.conversation_partner = None
                state.wake_required = True
                state.wake_required_reason = "face_mode"
        print("[V7.14 PASSWORD SESSION] ended reason=face_mode")
        v6.speak("Face mode on.")
        return True

    unlock_prefixes = {
        "activate owner mode",
        "can you activate owner mode",
        "enable owner mode",
        "owner mode",
        "turn on owner mode",
        "unlock owner mode",
    }
    for prefix in unlock_prefixes:
        if normalized == prefix or normalized.startswith(prefix + " "):
            _log_password_env_configured_once(state)
            print("[V7.14 PASSWORD SESSION] unlock requested")
            if not password:
                v6.speak("Password mode is not configured.")
                return True
            supplied = normalized[len(prefix):].strip()
            if not supplied:
                with state.lock:
                    state.pending_owner_unlock_until = time.time() + 20.0
                v6.speak("Owner mode needs the secret phrase.")
                return True
            if _owner_password_matches(supplied, password):
                print("[V7.14 PASSWORD SESSION] secret matched")
                timeout = _env_float("MIGUEL_PASSWORD_SESSION_TIMEOUT_SECONDS", 600.0)
                with state.lock:
                    state.password_session_active = True
                    state.password_session_until = time.time() + timeout
                start_conversation_session(
                    state,
                    mode="owner_password",
                    partner="owner_password",
                    timeout_seconds=timeout,
                    reason="password_unlock",
                )
                print("[V7.14 PASSWORD SESSION] started")
                v6.speak("Owner mode on.")
                return True
            print("[V7.14 PASSWORD SESSION] secret rejected")
            v6.speak("Owner mode not unlocked.")
            return True

    with state.lock:
        pending_unlock = bool(state.pending_owner_unlock_until and time.time() <= float(state.pending_owner_unlock_until))
    if pending_unlock:
        _log_password_env_configured_once(state)
        if not password:
            v6.speak("Password mode is not configured.")
            return True
        print("[V7.14 PASSWORD SESSION] unlock requested")
        if _owner_password_matches(normalized, password):
            print("[V7.14 PASSWORD SESSION] secret matched")
            timeout = _env_float("MIGUEL_PASSWORD_SESSION_TIMEOUT_SECONDS", 600.0)
            with state.lock:
                state.pending_owner_unlock_until = 0.0
                state.password_session_active = True
                state.password_session_until = time.time() + timeout
            start_conversation_session(
                state,
                mode="owner_password",
                partner="owner_password",
                timeout_seconds=timeout,
                reason="password_unlock",
            )
            print("[V7.14 PASSWORD SESSION] started")
            v6.speak("Owner mode on.")
            return True
        print("[V7.14 PASSWORD SESSION] secret rejected")
        with state.lock:
            state.pending_owner_unlock_until = 0.0
        v6.speak("Owner mode not unlocked.")
        return True
    return False


def _route_heard_repeat(user_text: str, state: RobotRuntimeState) -> bool:
    normalized = normalize_command_text(user_text)
    if not any(
        phrase in normalized
        for phrase in {
            "did you hear what marquinho said",
            "did you hear what marco said",
            "did you hear what he said",
            "what did he say",
        }
    ):
        return False
    with state.lock:
        heard = state.last_non_self_heard_user_text
    if heard:
        v6.speak(f"I heard: {_short_log_text(heard, 90)}")
    else:
        v6.speak("I heard part of it. Please repeat after YOUR TURN.")
    return True


def _route_repeat_last_reply(user_text: str, state: RobotRuntimeState) -> bool:
    normalized = normalize_command_text(user_text)
    if not any(
        phrase in normalized
        for phrase in {
            "repeat what you last said",
            "repeat your last answer",
            "repeat that",
            "say that again",
            "what did you just say",
            "what did you say",
        }
    ):
        return False
    with state.lock:
        last_spoken = str(state.last_spoken_text or "").strip()
    print("[V7.15 REPEAT] served_local=true")
    _set_reply_context(state, "repeat")
    _set_transient_response_length_context(state, "normal")
    v6.speak(last_spoken or "I do not have a previous reply to repeat.")
    return True


def _route_last_answer_clarification(user_text: str, state: RobotRuntimeState) -> bool:
    normalized = normalize_command_text(user_text)
    markers = {
            "what do you mean",
            "what did you mean",
            "explain what you meant",
            "explain that",
            "what means",
            "still what",
    }
    matched = next((marker for marker in markers if marker in normalized), None)
    if not matched:
        return False
    # A correction can begin with a clarification phrase and then provide the
    # missing context. Let that richer turn reach the normal conversational
    # route instead of blindly repeating the previous answer.
    remainder = normalized.replace(matched, "", 1).strip()
    if len(remainder.split()) > 3:
        return False
    with state.lock:
        route = state.last_answer_route or ""
        previous = state.last_answer_text_short or state.last_robot_text or ""
        age = time.time() - float(state.last_answer_at or 0.0) if state.last_answer_at else 9999.0
    if not previous or age > 120.0:
        return False
    _set_reply_context(state, "clarification")
    _set_transient_response_length_context(state, "normal")
    if route == "scene" or "visible" in normalize_command_text(previous):
        v6.speak("I meant the camera could only see part of the view, so I could not confidently describe the whole person or object.")
        return True
    if route == "identity":
        v6.speak("I meant the face recognition was uncertain, so I should not claim a name unless the camera confirms it clearly.")
        return True
    v6.speak(f"I meant this: {previous}")
    return True


def _format_timer_duration(seconds: int) -> str:
    seconds = max(0, int(seconds))
    if seconds and seconds % 60 == 0:
        minutes = seconds // 60
        return f"{minutes} minute" + ("" if minutes == 1 else "s")
    return f"{seconds} second" + ("" if seconds == 1 else "s")


def _format_timer_remaining(seconds: int) -> str:
    seconds = max(0, int(seconds))
    minutes, remainder = divmod(seconds, 60)
    if minutes:
        return f"{minutes} minute{'s' if minutes != 1 else ''} and {remainder} second{'s' if remainder != 1 else ''}"
    return f"{remainder} second" + ("" if remainder == 1 else "s")


def _route_timer_local_reply(user_text: str, state: RobotRuntimeState) -> bool:
    command = robot_timer.parse_timer_command(user_text)
    if not command:
        return False

    intent = command.get("intent")
    print(f"[V7.5 TIMER] handled locally intent={intent} text={_short_log_text(user_text)}")
    if intent == "start_timer":
        result = robot_timer.start_timer(int(command.get("seconds", 0) or 0))
        _set_reply_context(state, "timer")
        v6.speak(f"Timer set for {_format_timer_duration(result['seconds'])}.")
        return True

    if intent == "cancel_timer":
        result = robot_timer.cancel_timer()
        _set_reply_context(state, "timer")
        if result.get("canceled"):
            v6.speak("Timer canceled.")
        else:
            v6.speak("No timer is running.")
        return True

    if intent == "timer_status":
        status = robot_timer.get_timer_status()
        _set_reply_context(state, "timer")
        if not status.get("active"):
            v6.speak("No timer is running.")
            return True
        v6.speak(f"There are about {_format_timer_remaining(status['remaining_seconds'])} left.")
        return True

    return False


def _check_timer_tick(state: RobotRuntimeState) -> None:
    expired = robot_timer.timer_tick()
    if not expired:
        return
    print("[V7.5 TIMER] expired")
    try:
        full.face_happy("Time is up")
    except Exception as exc:
        print("[V7.5 TIMER] face alert warning:", exc)
    _set_reply_context(state, "timer")
    v6.speak("Time is up.")


def _route_teacher_mode_local_reply(user_text: str, state: RobotRuntimeState, partner: str | None = None) -> bool:
    controller = state.teacher_controller
    # A capability inventory may mention "teacher mode" as one item.  Let the
    # capability router answer it instead of accidentally entering a lesson.
    if _is_capabilities_request(user_text):
        return False
    if not controller.detects_teacher_intent(user_text):
        return False
    with state.lock:
        runtime_person = _normalize_person_name(state.recognized_person)
        conversation_partner = _normalize_person_name(state.conversation_partner)
    teacher_user = _normalize_person_name(partner) or runtime_person or conversation_partner
    teacher_display = _friendly_person_name(teacher_user) if teacher_user else None
    reply = controller.handle(user_text, user_id=teacher_user, display_name=teacher_display)
    if not reply:
        return False
    print(f"[V7.5 TEACHER] handled locally text={_short_log_text(user_text)}")
    now = time.time()
    with state.lock:
        if controller.session.active:
            state.conversation_mode = "teacher"
            state.conversation_partner = partner or state.conversation_partner
            state.conversation_active = True
            state.conversation_until = now + _env_float("MIGUEL_TEACHER_MODE_TIMEOUT_SECONDS", 600.0)
            state.last_conversation_activity_at = now
        elif state.conversation_mode == "teacher":
            state.conversation_mode = "general"
            state.conversation_until = now + _env_float("MIGUEL_CONVERSATION_TIMEOUT_SECONDS", 120.0)
    v6.speak(reply)
    return True


_PORTUGUESE_SMALL_NUMBERS = {
    "um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3, "quatro": 4,
    "cinco": 5, "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10,
    "onze": 11, "doze": 12,
}


def _spoken_small_number(value: str) -> int | None:
    token = normalize_command_text(value)
    if token.isdigit():
        return int(token)
    return _PORTUGUESE_SMALL_NUMBERS.get(token)


def _parse_portuguese_times_table_request(text: str) -> tuple[int, int, int] | None:
    """Return (table, start, end) for an explicit Portuguese table range."""
    normalized = normalize_command_text(text)
    number = r"(?:\d{1,2}|um|uma|dois|duas|tres|quatro|cinco|seis|sete|oito|nove|dez|onze|doze)"
    table_match = re.search(rf"\btabuada\s+d[oa]\s+({number})\b", normalized)
    if not table_match:
        # ASR often renders multiplication as "cinco por seis".
        table_match = re.search(rf"\b({number})\s+(?:vezes|por)\s+({number})\b", normalized)
    if not table_match:
        return None
    table = _spoken_small_number(table_match.group(1))
    if table is None:
        return None
    range_match = re.search(
        rf"\b(?:de\s+)?(?:{number}\s+(?:vezes|por)\s+)?({number})\s+(?:a|ate)\s+"
        rf"(?:{number}\s+(?:vezes|por)\s+)?({number})\b",
        normalized,
    )
    if range_match:
        start = _spoken_small_number(range_match.group(1))
        end = _spoken_small_number(range_match.group(2))
    else:
        factors = re.findall(rf"\b{table_match.group(1)}\s+(?:vezes|por)\s+({number})\b", normalized)
        start = _spoken_small_number(factors[0]) if factors else 1
        end = 12 if "tabuada" in normalized else start
    if start is None or end is None or not (0 <= table <= 20 and 0 <= start <= end <= 20):
        return None
    return table, start, end


def _route_portuguese_times_table_reply(user_text: str, state: RobotRuntimeState) -> bool:
    request = _parse_portuguese_times_table_request(user_text)
    normalized = normalize_command_text(user_text)
    if request is None and normalized in {"sim", "sim miguel", "sim miguel sim", "continue", "continua"}:
        with state.lock:
            request = state.pending_times_table
            last_robot_text = state.last_robot_text
        if request is None:
            offered_range = re.search(
                r"\b(?:do\s+)?(\d{1,2})\s*x\s*(\d{1,2})\s+ate\s+(?:o\s+)?(\d{1,2})\s*x\s*(\d{1,2})\b",
                normalize_command_text(last_robot_text),
            )
            if offered_range and offered_range.group(1) == offered_range.group(3):
                request = tuple(int(offered_range.group(index)) for index in (1, 2, 4))
    if request is None:
        return False
    table, start, end = request
    facts = [f"{table} vezes {factor} e {table * factor}" for factor in range(start, end + 1)]
    reply = ". ".join(facts) + "."
    with state.lock:
        state.pending_times_table = None
        state.last_topic = f"tabuada do {table}"
        state.last_topic_until = time.time() + 300.0
    _set_reply_context(state, "teacher")
    _set_response_length_context(state, "detailed")
    v6.speak(reply)
    return True


def _route_name_correction_local_reply(user_text: str, state: RobotRuntimeState) -> bool:
    normalized = normalize_command_text(user_text)
    if not any(
        marker in normalized
        for marker in {
            "not tommy",
            "i'm not tommy",
            "im not tommy",
            "i am not tommy",
            "do not call me tommy",
            "don't call me tommy",
            "why did you call me tommy",
            "why are you calling me tommy",
        }
    ):
        return False

    now = time.time()
    with state.lock:
        runtime_person = _normalize_person_name(state.recognized_person)
        runtime_age = now - float(state.recognized_person_updated_at or 0.0)
        partner = _normalize_person_name(state.conversation_partner)
    person = runtime_person if runtime_person and runtime_age <= 10.0 else partner
    if person and person not in {"unknown", "unknown_wake_user", "tommy"}:
        reply = f"Got it. I should call you {_friendly_person_name(person)}."
    else:
        reply = "Got it. I will not call you Tommy. I will use the camera name when I can confirm it."
    print(f"[V7.5 IDENTITY] name_correction handled locally person={person or 'unknown'}")
    _set_reply_context(state, "identity")
    v6.speak(reply)
    try:
        v6.update_conversation_memory(user_text=user_text, assistant_reply=reply)
    except Exception:
        pass
    return True


def _extract_spoken_identity_claim(user_text: str) -> str | None:
    normalized = normalize_command_text(user_text)
    if any(
        phrase in normalized
        for phrase in {
            "i am back",
            "i m back",
            "im back",
            "i am back now",
            "i m back now",
            "im back now",
        }
    ):
        return None
    # Treat identity as an explicit, complete claim.  An unanchored match used
    # to turn ordinary continuations such as "I'm pretty much gonna talk..."
    # into the preferred name "pretty_much" for the rest of the session.
    patterns = [
        r"\b(?:i am|i m|my name is)\s+(?:the\s+)?([a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*)?)\s*$",
        r"\b(?:eu sou o|eu sou a|meu nome e)\s+([a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*)?)\s*$",
    ]
    action_starters = {
        "asking",
        "bringing",
        "checking",
        "describing",
        "doing",
        "going",
        "holding",
        "looking",
        "moving",
        "pretty",
        "putting",
        "quite",
        "showing",
        "sitting",
        "standing",
        "talking",
        "trying",
        "using",
        "very",
        "walking",
        "wearing",
    }
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            candidate_text = match.group(1).strip()
            first_word = candidate_text.split()[0] if candidate_text else ""
            if first_word in action_starters:
                continue
            candidate = re.sub(r"\s+", "_", candidate_text)
            if candidate not in {"miguel", "robot", "robo", "back", "ready", "here"}:
                return candidate[:48]
    return None


def _extract_preferred_address_request(user_text: str) -> tuple[str, str] | None:
    """Return (name, kind) for explicit address preferences or speaker handoffs.

    These are conversational labels only.  They must never replace camera
    identity or grant owner authorization.
    """
    normalized = normalize_command_text(user_text)
    patterns = [
        (r"\b(?:me chame de|pode me chamar de|quero que (?:voce )?me chame de)\s+([a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*)?)(?:\s+por favor)?\s*$", "self"),
        (r"\b(?:call me|please call me|you can call me)\s+([a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*)?)(?:\s+please)?\s*$", "self"),
        (r"\bquem vai falar com voce agora e (?:a|o) .+? (?:ela|ele) se chama\s+([a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*)?)\s*$", "handoff"),
        (r"\b(?:the person|the one) (?:speaking|talking) (?:to you )?now is .+? (?:her|his|their) name is\s+([a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*)?)\s*$", "handoff"),
    ]
    ignored = {"miguel", "robot", "robo", "please", "por_favor"}
    for pattern, kind in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        candidate = re.sub(r"\s+", "_", match.group(1).strip())[:48]
        if candidate and candidate not in ignored:
            return candidate, kind
    return None


def _route_preferred_address_request(user_text: str, state: RobotRuntimeState) -> bool:
    request = _extract_preferred_address_request(user_text)
    if not request:
        return False
    preferred_name, kind = request
    now = time.time()
    with state.lock:
        camera_person = _normalize_person_name(state.recognized_person)
        state.preferred_address_name = preferred_name
        state.preferred_address_until = now + 600.0
        state.current_turn_latency["preferred_address_name"] = preferred_name
        state.current_turn_latency["identity_debug"] = {
            "camera_person": camera_person or None,
            "preferred_address_name": preferred_name,
            "address_request_kind": kind,
            "authorization_changed": False,
        }
    display_name = _friendly_person_name(preferred_name)
    if kind == "handoff":
        reply = f"Oi, {display_name}! Prazer em falar com você. Vou te chamar de {display_name} nesta conversa."
    elif any(marker in normalize_command_text(user_text) for marker in {"me chame", "me chamar"}):
        reply = f"Combinado, {display_name}. Vou te chamar de {display_name} nesta conversa."
    else:
        reply = f"Got it, {display_name}. I will call you {display_name} in this conversation."
    print(f"[V7.5 IDENTITY] preferred_address={preferred_name} kind={kind} camera_person={camera_person or 'unknown'}")
    _set_reply_context(state, "identity")
    v6.speak(reply)
    return True


def _route_spoken_identity_claim(user_text: str, state: RobotRuntimeState) -> bool:
    claimed_name = _extract_spoken_identity_claim(user_text)
    if not claimed_name:
        return False
    now = time.time()
    with state.lock:
        camera_person = _normalize_person_name(state.recognized_person)
        state.preferred_address_name = claimed_name
        state.preferred_address_until = now + 600.0
        state.current_turn_latency["identity_debug"] = {
            "camera_person": camera_person or None,
            "spoken_name": claimed_name,
            "identity_conflict": bool(camera_person and camera_person != claimed_name),
            "authorization_changed": False,
        }
    display_name = _friendly_person_name(claimed_name)
    normalized = normalize_command_text(user_text)
    if "eu sou" in normalized or "meu nome" in normalized:
        reply = (
            f"Oi, {display_name}. Vou usar esse nome nesta conversa, "
            "mas isso não altera o reconhecimento facial."
        )
    else:
        reply = (
            f"Hello, {display_name}. I will use that name in this conversation, "
            "but it does not change face recognition."
        )
    print(f"[V7.5 IDENTITY] spoken_name={claimed_name} camera_person={camera_person or 'unknown'}")
    _set_reply_context(state, "identity")
    v6.speak(reply)
    return True


def _route_creative_story_local_reply(user_text: str, state: RobotRuntimeState) -> bool:
    normalized = normalize_command_text(user_text)
    with state.lock:
        mode = state.conversation_mode
        focus = state.session_focus
        depth_mode = state.response_depth_mode

    if depth_mode == "long_story" and (
        mode in {"creative", "story"}
        or any(
            marker in normalized
            for marker in {
                "story",
                "superhero",
                "hero",
                "character",
                "machine",
                "invention",
                "invent",
                "sci fi",
                "sci-fi",
                "technology",
            }
        )
    ):
        return False

    if any(phrase in normalized for phrase in {"new superheroes", "some new superheroes", "superhero idea"}):
        start_conversation_session(state, mode="creative", partner=_current_owner_partner(state), reason="creative_superheroes")
        with state.lock:
            state.session_topic = "superheroes"
            state.last_topic = "superheroes"
            state.last_topic_until = time.time() + 300.0
            state.last_robot_question_type = "creative_pick_hero"
            state.last_robot_question_at = time.time()
        v6.speak("Nice. Pick one hero.")
        return True

    if mode == "creative" and normalized in {"the skeleton", "skeleton", "let s go with the skeleton", "lets go with the skeleton"}:
        extend_conversation_session(state, reason="creative_focus")
        with state.lock:
            state.session_focus = "skeleton superhero"
            state.last_topic = "skeleton superhero"
            state.last_topic_until = time.time() + 300.0
            state.last_robot_question_type = "creative_power"
            state.last_robot_question_at = time.time()
        v6.speak("Great. Skeleton hero. How should his power work?")
        return True

    if mode == "creative" and "skeleton" in normalized and any(
        phrase in normalized for phrase in {"think about", "how the skeleton works", "how skeleton works", "power work"}
    ):
        extend_conversation_session(state, reason="creative_skeleton")
        with state.lock:
            state.session_focus = "skeleton superhero"
            state.last_topic = "skeleton superhero"
            state.last_topic_until = time.time() + 300.0
        v6.speak("He can detach bones, rebuild himself, and use bone tools. Weakness: magnets or glue traps.")
        return True

    creative_topic = _is_creative_continuation(user_text, state)
    if creative_topic:
        print(f"[V7.14 CREATIVE] continuation topic={creative_topic}")
        extend_conversation_session(state, reason="creative_continuation")
        with state.lock:
            state.conversation_mode = "creative"
            state.session_focus = creative_topic if creative_topic == "skeleton superhero" else state.session_focus
            state.last_topic = creative_topic
            state.last_topic_until = time.time() + 300.0
        if creative_topic == "skeleton superhero" and any(
            phrase in normalized
            for phrase in {"something moving around him", "moving around his body", "slithering"}
        ):
            v6.speak(
                "Yes. Give him a living shadow-serpent made of loose bones. "
                "It circles his ribs, becomes armor, scouts ahead, and can steal one enemy power for a few seconds. "
                "Its weakness is bright light or music vibrations."
            )
            return True
        v6.speak("Yes. Build that into the character as a power with a clear weakness, so it feels cool but not unbeatable.")
        return True

    if "tell me a story" in normalized:
        start_conversation_session(state, mode="story", partner=_current_owner_partner(state), reason="story_start")
        with state.lock:
            state.last_robot_question_type = "story_continue"
            state.last_robot_question_at = time.time()
        v6.speak("A tiny robot found a moonlit key under the workshop table. It clicked once, and a hidden map glowed on the wall.")
        return True

    if mode == "story" and normalized in {"continue", "what happens next", "continue the story"}:
        extend_conversation_session(state, reason="story_continue")
        with state.lock:
            state.last_robot_question_type = "story_continue"
            state.last_robot_question_at = time.time()
        v6.speak("The map led to a drawer full of spare bolts, where one silver bolt whispered, Follow the blue wire.")
        return True

    if mode == "creative" and focus and normalized in {"how would he work", "how would it work", "what do you think"}:
        extend_conversation_session(state, reason="creative_followup")
        v6.speak("He could fall apart to dodge danger, then snap back together into new shapes.")
        return True

    return False


def _route_correction_retry(user_text: str, state: RobotRuntimeState) -> bool:
    if not is_conversation_active(state) or not _is_correction_retry_text(user_text):
        return False
    with state.lock:
        topic = state.last_answer_topic or state.session_focus or state.last_topic or ""
    if not topic:
        return False

    extend_conversation_session(state, reason="correction_retry")
    _set_response_length_context(state, "normal")
    print(f"[V7.14 CORRECTION] retry topic={topic}")
    if "skeleton" in topic or "superhero" in topic:
        with state.lock:
            state.conversation_mode = "creative"
            state.session_focus = "skeleton superhero"
            state.last_topic = "skeleton superhero"
            state.last_topic_until = time.time() + 300.0
        v6.speak(
            "Got it. New version: The Skeleton can detach and rebuild his bones into tools, armor, and escape paths. "
            "His weakness is that every rebuild costs energy, so he has to choose carefully."
        )
        return True

    v6.speak(f"Got it. I'll try {topic} again with a better version.")
    return True


def _route_topic_followup_local_reply(user_text: str, state: RobotRuntimeState) -> bool:
    normalized = normalize_command_text(user_text)
    if _is_story_continue_text(normalized):
        return False
    if not _is_topic_continuation(user_text, state):
        if "star wars" in normalized:
            with state.lock:
                state.last_topic = "Star Wars"
                state.last_topic_until = time.time() + 300.0
        return False
    with state.lock:
        topic = state.last_topic or "that"
        state.last_topic_until = time.time() + 300.0
    if topic == "Star Wars":
        v6.speak("About Star Wars, there are Jedi, Sith, droids, starships, and the Force. We can talk about the movies, the characters, or the timeline.")
        return True
    v6.speak(f"About {topic}, tell me which part you want to explore next.")
    return True


def _route_response_depth_mode(user_text: str, state: RobotRuntimeState) -> bool:
    normalized = normalize_command_text(user_text)
    if not normalized:
        return False

    if _is_normal_conversation_mode_request(user_text):
        _set_response_depth_mode(state, "normal", "normal_conversation_mode")
        _set_response_length_context(state, "normal")
        _force_active_after_mode(state, "general", reason="normal_conversation_mode")
        with state.lock:
            state.current_mode = "normal"
        v6.speak("Normal conversation mode on. I'll use natural, complete replies.")
        return True

    story_detection = _story_mode_detection(user_text)
    target_minutes = _extract_long_story_duration_minutes(user_text)
    topic_hint = _extract_long_story_topic_hint(user_text)
    story_style = _extract_story_style(user_text)
    recovered_context = _recover_story_context_from_logs(user_text, state)
    generate_story_now = _is_story_generation_request(user_text) or (
        _is_explicit_long_story_request(user_text)
        and any(marker in normalized for marker in {"tell", "story about", "topic is"})
    )

    if _is_depth_status_question(user_text):
        return False

    if _is_new_story_request(user_text) and normalized not in LONG_STORY_ACTIVATION_PHRASES:
        return False

    if story_detection.get("mode") == "long_story" and story_detection.get("has_story_request"):
        _log_story_execution_skip_legacy("story_request")
        return False

    if _is_mode_command_not_physical(normalized):
        if any(marker in normalized for marker in {"long story mode", "modo historia"}) or story_detection.get("mode") == "long_story":
            _log_story_mode_detection(story_detection)
            _set_response_depth_mode(state, "long_story", "mode_command")
            _set_response_length_context(state, "long_story")
            with state.lock:
                state.long_story_active = False
                state.long_story_topic = f"story: {topic_hint}" if topic_hint else None
                state.long_story_segment_index = 0
                state.long_story_target_minutes = target_minutes
                state.long_story_style = story_style
                state.recovered_story_context = recovered_context
            if topic_hint:
                _remember_long_story_topic(state, topic_hint)
            _force_active_after_mode(state, "story", reason="long_story_mode")
            if _is_story_continue_text(normalized) or generate_story_now:
                return False
            duration = _format_long_story_duration(target_minutes)
            topic_part = f" about {topic_hint}" if topic_hint else ""
            style_part = f" in {story_style} style" if story_style else ""
            v6.speak("Long story mode on" + topic_part + style_part + (f" for about {duration}." if duration else ". I'll give richer stories when you ask."))
            return True
        if any(marker in normalized for marker in {"long explanation mode"}):
            _set_response_depth_mode(state, "long_explanation", "mode_command")
            _set_response_length_context(state, "detailed")
            _force_active_after_mode(state, "general", reason="long_explanation_mode")
            v6.speak("Long explanation mode on. I'll explain with more detail.")
            return True
        if any(marker in normalized for marker in {"normal mode"}):
            _set_response_depth_mode(state, "normal", "mode_command")
            _set_response_length_context(state, "normal")
            _force_active_after_mode(state, "general", reason="normal_mode")
            v6.speak("Normal mode on. I'll keep answers shorter.")
            return True

    if normalized in NORMAL_DEPTH_PHRASES:
        _set_response_depth_mode(state, "normal", normalized.replace(" ", "_"))
        _set_response_length_context(state, "normal")
        _force_active_after_mode(state, "general", reason="normal_mode")
        v6.speak("Normal mode on. I'll keep answers shorter.")
        return True

    if normalized in LONG_STORY_ACTIVATION_PHRASES:
        _log_story_mode_detection(story_detection)
        _set_response_depth_mode(state, "long_story", normalized.replace(" ", "_"))
        _set_response_length_context(state, "long_story")
        with state.lock:
            state.long_story_active = False
            state.long_story_topic = f"story: {topic_hint}" if topic_hint else None
            state.long_story_segment_index = 0
            state.long_story_target_minutes = target_minutes
            state.long_story_style = story_style
            state.recovered_story_context = recovered_context
            if state.conversation_mode in {"general", "wake_required", "creative"}:
                state.conversation_mode = "story"
        if topic_hint:
            _remember_long_story_topic(state, topic_hint)
        _force_active_after_mode(state, "story", reason="long_story_mode")
        duration = _format_long_story_duration(target_minutes)
        topic_part = f" about {topic_hint}" if topic_hint else ""
        style_part = f" in {story_style} style" if story_style else ""
        v6.speak("Long story mode on" + topic_part + style_part + (f" for about {duration}." if duration else ". I'll give richer stories when you ask."))
        return True

    if normalized in LONG_EXPLANATION_ACTIVATION_PHRASES:
        _set_response_depth_mode(state, "long_explanation", normalized.replace(" ", "_"))
        _set_response_length_context(state, "detailed")
        _force_active_after_mode(state, "general", reason="long_explanation_mode")
        v6.speak("Long explanation mode on. I'll explain with more detail.")
        return True

    return False


def _current_voice_mode() -> str:
    getter = getattr(robot_memory, "get_voice_mode", None)
    if callable(getter):
        try:
            return str(getter() or "natural_voice")
        except Exception:
            pass
    try:
        memory = robot_memory.load_memory()
        return str(memory.get("voice_mode") or "natural_voice")
    except Exception:
        return "natural_voice"


def _is_voice_modes_list_request(user_text: str) -> bool:
    normalized = normalize_command_text(user_text)
    if not normalized:
        return False
    phrases = {
        "all your voice modes",
        "different types of voice",
        "different voice types",
        "list your voice modes",
        "what are your voice modes",
        "what are all your voice modes",
        "what voice can we set you up",
        "what voice modes do you have",
        "what voices do you have",
        "voice modes",
        "voice options",
    }
    return normalized in phrases or any(normalized.startswith(phrase + " ") or phrase in normalized for phrase in phrases)


def _voice_mode_command_action(user_text: str) -> tuple[str, str | None]:
    normalized = normalize_command_text(user_text)
    if not normalized:
        return "", None
    if _is_voice_modes_list_request(normalized):
        return "list", None
    if "angry voice" in normalized:
        return "unsupported_angry", None
    if any(
        phrase in normalized
        for phrase in {
            "natural voice",
            "use natural voice",
            "speak naturally",
            "voz natural",
            "voz para natural",
            "falar naturalmente",
            "robot voice",
            "use robot voice",
            "deep voice",
            "use deep voice",
            "go to deep voice",
            "storyteller voice",
            "kid-friendly storyteller voice",
            "story voice",
            "use story voice",
            "friendly voice",
            "use friendly voice",
        }
    ):
        return "set", None
    return "", None


def _route_voice_modes_local_reply(user_text: str, state: RobotRuntimeState) -> bool:
    action, requested_mode = _voice_mode_command_action(user_text)
    if not action:
        return False
    current_voice = _current_voice_mode()
    _set_reply_context(state, "voice_command")
    _set_transient_response_length_context(state, "terse" if action == "set" else "normal")
    print(f"[V7.15 VOICE MODES] served_local=true action={action}")
    if action == "unsupported_angry":
        if any(marker in normalize_command_text(user_text) for marker in {"example", "sound", "sounds"}):
            v6.speak(
                "I do not have an angry voice mode. An acted angry voice would sound sharper and more forceful, "
                "but I would keep it pretend and safe."
            )
        else:
            v6.speak(
                "I have an angry face expression, but no angry voice mode. "
                "My voice modes are natural, robot, friendly, deep, and story."
            )
        return True
    if action == "set":
        handler = getattr(robot_memory, "handle_voice_mode_command", None)
        reply = None
        if callable(handler):
            try:
                reply = handler(user_text)
            except Exception as exc:
                print("[V7.15 VOICE MODES] set warning:", exc)
        if reply:
            if _prefers_portuguese_reply(user_text, state):
                selected = _current_voice_mode().replace("_", " ")
                localized = {
                    "natural voice": "Voz natural.",
                    "robot voice": "Voz robotica.",
                    "deep voice": "Voz grave.",
                    "friendly voice": "Voz amigavel.",
                    "story voice": "Voz de historia.",
                    "storyteller voice": "Voz de historia.",
                }
                reply = localized.get(selected, reply)
            v6.speak(reply)
            return True
        v6.speak(f"My current voice mode is {_current_voice_mode()}.")
        return True
    v6.speak(
        f"My current voice is {current_voice}. I have five voice modes: robot voice, natural voice, "
        "friendly voice, deep voice, and story voice."
    )
    return True


def _is_capabilities_request(user_text: str) -> bool:
    normalized = normalize_command_text(user_text)
    if not normalized:
        return False
    phrases = {
        "all commands",
        "what can you do",
        "what are your capabilities",
        "what modes do you have",
        "list your commands",
        "command list",
        "help",
        "what commands can i say",
        "what commands do you know",
        "provide a detailed list of your commands",
        "detailed list of your commands",
        "full command list",
        "explain your commands",
        "explain all your capabilities",
        "detailed command list",
        "list all commands",
        "tell me all your modes",
    }
    if normalized in phrases or any(normalized.startswith(phrase + " ") or phrase in normalized for phrase in phrases):
        return True
    inventory_terms = {"capability", "capabilities", "function", "functions", "modes", "commands"}
    inventory_verbs = {"describe", "list", "explain", "show", "tell"}
    words = set(normalized.split())
    return bool(words & inventory_terms and words & inventory_verbs and ("all" in words or "your" in words))


def _requested_face_expression(user_text: str) -> str | None:
    normalized = normalize_command_text(user_text)
    command_markers = {"go", "change", "switch", "set", "show", "make", "use"}
    words = set(normalized.split())
    compact_faces = {f"{expression}face": expression for expression in FACE_EXPRESSIONS}
    compact_expression = next((value for alias, value in compact_faces.items() if alias in words), None)
    if not (words & command_markers) or not ({"face", "expression"} & words or compact_expression):
        return None
    if compact_expression:
        return compact_expression
    aliases = {"neutral": "normal", "concern": "concerned", "motivation": "motivated"}
    for expression in FACE_EXPRESSIONS:
        if expression in words:
            return expression
    for alias, expression in aliases.items():
        if alias in words:
            return expression
    return None


def _is_face_expression_command(user_text: str) -> bool:
    normalized = normalize_command_text(user_text)
    words = set(normalized.split())
    return bool(
        words & {"go", "change", "switch", "set", "show", "make", "use"}
        and words & {"face", "expression"}
    )


def _is_face_expression_inventory_request(user_text: str) -> bool:
    normalized = normalize_command_text(user_text)
    words = set(normalized.split())
    return bool(
        words & {"face", "faces", "expression", "expressions"}
        and (
            "how many" in normalized
            or words & {"list", "describe", "available", "have", "support"}
        )
    )


def _contextual_face_expression(user_text: str, state: RobotRuntimeState) -> str | None:
    """Resolve short follow-ups without letting the cloud invent face actions."""
    normalized = normalize_command_text(user_text)
    words = set(normalized.split())
    aliases = {
        "scary": "scared",
        "neutral": "normal",
        "concern": "concerned",
        "motivation": "motivated",
    }
    # A named emotion is not necessarily an instruction (for example, "are
    # you angry?" or "that is a good angry face").  Accept a bare expression
    # only as the intentionally short follow-up this helper is meant for;
    # full commands are handled by _requested_face_expression above.
    bare_expression_fillers = {"face", "expression", "please", "very", "super", "really"}
    named_expression = next((item for item in FACE_EXPRESSIONS if item in words), None)
    if named_expression and words <= bare_expression_fillers | {named_expression}:
        return named_expression
    for alias, expression in aliases.items():
        if alias in words and words <= bare_expression_fillers | {alias}:
            return expression

    with state.lock:
        current = state.face_expression
        source = state.face_expression_source
        prompt = str(state.last_robot_question_text or "").lower()
        asked_at = float(state.last_robot_question_at or 0.0)
    recent_prompt = bool(asked_at and time.time() - asked_at <= 90.0)
    refers_to_repeat = any(
        phrase in normalized
        for phrase in {"do it again", "do that again", "repeat it", "same face", "again"}
    )
    affirmative_face_answer = normalized in {"yes", "yeah", "yep", "sure", "ok", "okay"} and (
        "face" in prompt or "expression" in prompt
    )
    if current in FACE_EXPRESSIONS and source == "voice" and (refers_to_repeat or (recent_prompt and affirmative_face_answer)):
        return current
    return None


def _automatic_face_expression(user_text: str) -> str | None:
    """Return an expression only for strong conversational cues."""
    normalized = normalize_command_text(user_text)
    cue_groups = (
        ("scared", {"i am scared", "i'm scared", "frightened", "terrified", "emergency"}),
        ("sad", {"i am sad", "i'm sad", "passed away", "died", "grieving", "heartbroken"}),
        ("concerned", {"i am worried", "i'm worried", "concerned about", "something is wrong", "not working", "unsafe"}),
        ("motivated", {"let's learn", "lets learn", "let's practice", "lets practice", "help me study", "we can do it", "my goal"}),
        ("happy", {"great news", "good news", "well done", "congratulations", "that was funny", "i am happy", "i'm happy"}),
    )
    for expression, cues in cue_groups:
        if any(cue in normalized for cue in cues):
            return expression
    return None


def _set_face_expression(state: RobotRuntimeState, expression: str, source: str) -> None:
    if expression not in FACE_EXPRESSIONS:
        return
    with state.lock:
        state.face_expression = expression
        state.face_expression_source = source
        state.current_turn_latency["face_expression"] = expression
        state.current_turn_latency["face_expression_source"] = source


def _route_face_expression_local_reply(user_text: str, state: RobotRuntimeState) -> bool:
    if _is_face_expression_inventory_request(user_text):
        _set_reply_context(state, "face_expression")
        names = ", ".join(FACE_EXPRESSION_ORDER[:-1]) + f", and {FACE_EXPRESSION_ORDER[-1]}"
        v6.speak(f"I have seven selectable face expressions: {names}.")
        return True

    expression = _requested_face_expression(user_text) or _contextual_face_expression(user_text, state)
    if not expression:
        normalized = normalize_command_text(user_text)
        with state.lock:
            face_context = state.face_expression_source == "voice" and state.face_expression in FACE_EXPRESSIONS
        if face_context and set(normalized.split()) & {"big", "small", "next", "repeat"}:
            _set_reply_context(state, "face_expression")
            v6.speak("I have one version of each face expression; I cannot resize it or switch to another version.")
            return True
        if _is_face_expression_command(user_text):
            _set_reply_context(state, "face_expression")
            names = ", ".join(FACE_EXPRESSION_ORDER[1:-1]) + f", or {FACE_EXPRESSION_ORDER[-1]}"
            v6.speak(f"I do not have that face expression. I can show normal, {names}.")
            return True
        return False
    _set_face_expression(state, expression, "voice")
    _set_reply_context(state, "face_expression")
    if _is_joke_request(user_text):
        v6.speak(f"{expression.title()} face selected. {_select_local_joke(user_text)}")
    else:
        v6.speak(f"{expression.title()} face selected.")
    return True


def _route_capabilities_local_reply(user_text: str, state: RobotRuntimeState) -> bool:
    if not _is_capabilities_request(user_text):
        return False

    normalized = normalize_command_text(user_text)
    detailed = any(
        phrase in normalized
        for phrase in {
            "explain all your capabilities",
            "detailed list of your commands",
            "detailed command list",
            "provide a detailed list of your commands",
            "list all commands",
            "all commands",
            "full command list",
            "explain your commands",
            "what commands do you know",
            "what commands can i say",
            "tell me all your modes",
        }
    ) or (
        "all" in normalized.split()
        and any(term in normalized.split() for term in {"capability", "capabilities", "function", "functions", "modes", "commands"})
    )
    print(f"[V7.15 CAPABILITIES] served_local=true detail={str(detailed).lower()}")
    _set_reply_context(state, "capabilities")
    if detailed:
        _set_transient_response_length_context(state, "detailed")
        v6.speak(
            "Here are commands I can actually handle. Say Hey Miguel or Hello Miguel to start talking. "
            "Ask can you hear me or can you hear us. Say set a timer for five minutes, cancel timer, or timer status. "
            "Ask for a joke or science joke. Ask what do you see, can you see me, who am I, who do you see, or can you see both faces. "
            "Ask what were our recent topics, list previous topics, or analyze the conversation log for improvements and safety review. "
            "For face quality, say re-enroll Marco or re-enroll Marquinho. For a new friend, say enroll a new friend, then give the name and owner approval. "
            "Say creative mode, then ask for superhero, machine, or sci-fi ideas. Say long story mode, continue chapter four, "
            "long explanation mode, normal mode, or keep it short. Ask about Marco and Marquinho's project roles. "
            "Owner mode needs the secret phrase, which I will not reveal. I also support sleep mode, wake mode, and shutdown with explicit confirmation. "
            "I refuse dangerous requests."
        )
        return True

    _set_transient_response_length_context(state, "normal")
    v6.speak(
        "I can converse, recognize faces, describe the camera view, set timers, tell jokes, remember topics, "
        "analyze logs, guide enrollment, and create or explain things. "
        "Say creative mode for ideas or ask for a detailed command list."
    )
    return True


def _is_recent_conversation_topics_request(user_text: str) -> bool:
    normalized = normalize_command_text(user_text)
    phrases = {
        "recent topics",
        "previous topics",
        "last topics",
        "what were our recent topics",
        "what were our last topics",
        "what did we talk about",
        "what did we discuss",
        "what did we talk about yesterday",
        "what was our conversation yesterday",
        "what was yesterday conversation",
        "list recent topics",
        "list previous topics",
        "conversation history",
        "interaction history",
    }
    return normalized in phrases or any(phrase in normalized for phrase in phrases)


def _is_conversation_log_analysis_request(user_text: str) -> bool:
    normalized = normalize_command_text(user_text)
    return any(
        phrase in normalized
        for phrase in {
            "analyze conversation",
            "analyze the conversation",
            "analyze interaction",
            "analyze the interaction",
            "analyze conversation log",
            "analyze the conversation log",
            "analyze previous log",
            "analyze the previous log",
            "analyze previous interaction",
            "analyze previous interactions",
            "analyze previous conversation",
            "analyze the previous conversation",
            "review conversation log",
            "review previous log",
            "summarize conversation log",
            "summarize the conversation",
            "summarize previous log",
            "summarize previous conversation",
            "research conversation",
            "research the conversation",
            "research our conversation",
            "research yesterday conversation",
            "research what was our conversation yesterday",
            "what was our conversation yesterday",
            "what did we talk about yesterday",
            "what did we discuss yesterday",
            "what can we improve from the log",
            "any inappropriate topic",
        }
    )


def _is_conversation_log_location_request(user_text: str) -> bool:
    normalized = normalize_command_text(user_text)
    return any(
        phrase in normalized
        for phrase in {
            "where is the log",
            "where are the logs",
            "log file",
            "conversation log file",
            "you have a log file",
            "there is a log file",
            "where do you save logs",
            "where are conversation logs",
        }
    )


def _is_conversation_log_recall_request(user_text: str) -> bool:
    normalized = normalize_command_text(user_text)
    if not normalized:
        return False
    return any(
        phrase in normalized
        for phrase in {
            "do you remember what we talked",
            "do you remember what was",
            "remember yesterday",
            "remember our conversation",
            "recall yesterday",
            "recall our conversation",
            "what was our conversation yesterday",
            "what did we talk about yesterday",
            "what did we discuss yesterday",
            "what were the last topics",
            "what were our last topics",
            "what were previous topics",
            "what was the last conversation",
            "what was our last conversation",
        }
    )


def _is_conversation_save_request(user_text: str) -> bool:
    normalized = normalize_command_text(user_text)
    return any(
        phrase in normalized
        for phrase in {
            "save this conversation",
            "save the conversation",
            "save our conversation",
            "save this chat",
            "save the sword conversation",
        }
    )


def _compact_logs_for_cloud(logs: list[dict], max_chars: int = 14000) -> str:
    payload = []
    for log in logs:
        metadata = log.get("metadata", {})
        events = []
        for event in log.get("events", []):
            if event.get("type") not in {"user_turn", "turn", "session_start"}:
                continue
            events.append({
                "type": event.get("type"),
                "created_at": event.get("created_at"),
                "person": event.get("person"),
                "user_text": event.get("user_text"),
                "assistant_reply": event.get("assistant_reply"),
                "route": event.get("route") or event.get("route_hint"),
                "topic": event.get("topic"),
                "conversation_mode": event.get("conversation_mode"),
            })
        payload.append({"metadata": metadata, "events": events[-80:]})
    text = json.dumps(payload, ensure_ascii=False)
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


def _analyze_conversation_logs_with_cloud(user_text: str, state: RobotRuntimeState) -> str:
    normalized = normalize_command_text(user_text)
    with state.lock:
        current_session_id = state.conversation_log_session_id
    logs = robot_memory.select_conversation_logs(
        query=normalized,
        current_session_id=current_session_id,
        limit=3,
    )
    if not logs:
        return "I do not have conversation logs yet."
    log_text = _compact_logs_for_cloud(logs)
    instructions = (
        "You analyze Miguel robot conversation logs. Give a short spoken summary with: "
        "conversation summary, what worked well, functional flaws or fixes, improvement ideas, "
        "and whether any inappropriate or sensitive topic appeared. Be concrete and concise."
    )
    try:
        response = v6.client.responses.create(
            model=getattr(v6, "OPENAI_MODEL", "gpt-4o-mini"),
            instructions=instructions,
            input=json.dumps({"user_request": user_text, "logs": log_text}, ensure_ascii=False),
        )
        return response.output_text.strip()
    except Exception as exc:
        print("[V7.15 LOG ANALYSIS] cloud warning:", exc)
        return "I could not analyze the log with the cloud brain right now."


def _route_conversation_memory_local_reply(user_text: str, state: RobotRuntimeState) -> bool:
    if _is_conversation_save_request(user_text):
        _set_reply_context(state, "memory")
        _set_transient_response_length_context(state, "normal")
        v6.speak("This conversation is already being saved in the current session log, including the sword discussion.")
        return True
    if _is_conversation_log_recall_request(user_text):
        with state.lock:
            current_session_id = state.conversation_log_session_id
        _set_reply_context(state, "memory")
        _set_transient_response_length_context(state, "normal")
        v6.speak(robot_memory.format_conversation_log_recall(user_text, current_session_id=current_session_id, limit=3))
        return True
    if _is_recent_conversation_topics_request(user_text):
        _set_reply_context(state, "memory")
        _set_transient_response_length_context(state, "normal")
        v6.speak(robot_memory.format_recent_conversation_topics(limit=6))
        return True
    if _is_conversation_log_location_request(user_text) and not _is_conversation_log_analysis_request(user_text):
        _set_reply_context(state, "memory")
        _set_transient_response_length_context(state, "normal")
        v6.speak("I save conversation logs in week3 memory conversation logs. Ask me to analyze the previous log or list recent topics.")
        return True
    if _is_conversation_log_analysis_request(user_text):
        _set_reply_context(state, "memory_analysis")
        _set_transient_response_length_context(state, "detailed")
        reply = _analyze_conversation_logs_with_cloud(user_text, state)
        v6.speak(reply)
        return True
    return False


def _extract_long_story_topic(user_text: str) -> str:
    normalized = normalize_command_text(user_text)
    bedtime_match = re.search(r"\bbedtime story\s+(?:of|about|around|with|on)\s+(.+)$", normalized)
    if bedtime_match:
        return bedtime_match.group(1).strip() or "bedtime story"
    for phrase in (
        "tell me a long story",
        "tell me a bedtime story",
        "conte uma historia longa",
        "conte uma historia",
        "conta uma historia longa",
        "conta uma historia",
        "contar uma historia",
        "historia longa",
        "longa historia",
        "tell me a long explanation",
        "explain in detail for a long time",
        "give me the full explanation",
        "teach me this topic",
        "long story mode",
        "modo historia",
    ):
        if normalized.startswith(phrase):
            topic = normalized[len(phrase):].strip()
            return topic or ("bedtime story" if "story" in phrase or "historia" in phrase else "this topic")
    return normalized or "this topic"


def _is_long_mode_request(text: str) -> bool:
    normalized = normalize_command_text(text)
    return (
        (_extract_long_story_duration_minutes(text) > 0 and _contains_story_word(normalized))
        or _is_long_story_request_text(text)
        or normalized in LONG_STORY_ACTIVATION_PHRASES
        or any(
            phrase in normalized
            for phrase in {
                "tell me a long story",
                "tell me a bedtime story",
                "tell me a long explanation",
                "explain in detail for a long time",
                "give me the full explanation",
                "teach me this topic",
            }
        )
    )


def _is_long_mode_continue(text: str) -> bool:
    normalized = normalize_command_text(text)
    return normalized in {"continue", "continua", "next part", "keep going", "go on", "continue the story", "continua a historia", "continua historia"}


def _is_story_generation_request(text: str) -> bool:
    normalized = normalize_command_text(text)
    if not normalized:
        return False
    story_detection = _story_mode_detection(text)
    if story_detection.get("mode") == "long_story" and story_detection.get("action") == "generate_story":
        return True
    if not _contains_story_word(normalized):
        return False
    if _is_mode_command_not_physical(normalized) and not any(
        phrase in normalized
        for phrase in STORY_GENERATION_TRIGGERS
    ):
        return False
    return any(
        phrase in normalized
        for phrase in STORY_GENERATION_TRIGGERS
    ) or (_extract_long_story_duration_minutes(text) > 0 and not _is_mode_command_not_physical(normalized))


def _is_story_stop_request(text: str) -> bool:
    normalized = normalize_command_text(text)
    if not normalized:
        return False
    return any(
        marker in normalized
        for marker in {
            "parar historia",
            "para a historia",
            "cancelar historia",
            "chega",
            "dormir",
            "stop story",
            "cancel story",
            "enough",
            "go to sleep",
        }
    )


STORY_SAFETY_RULES = [
    "Family-friendly and kid-safe.",
    "No illegal drugs, self-harm, graphic violence, gore, sexual content, adult romantic content, or adult themes.",
    "No instructions for dangerous actions.",
    "Scary stories must be suspenseful and safe, not traumatic.",
    "Historical stories must not invent major false facts, achievements, dates, or events.",
    "Use a real ending; bedtime stories must end calmly and completely.",
]


STORY_ARC_TEMPLATES = {
    "adventure": [
        "setup",
        "call_to_adventure",
        "enter_new_world",
        "first_obstacle",
        "discovery",
        "complication",
        "midpoint_reveal",
        "setback",
        "new_plan",
        "climax",
        "resolution",
        "return_home",
    ],
    "funny": [
        "normal_setup",
        "silly_mistake",
        "misunderstanding_grows",
        "chaos_escalates",
        "unexpected_helper",
        "clever_fix",
        "funny_resolution",
    ],
    "bedtime": [
        "calm_setup",
        "gentle_wish",
        "soft_discovery",
        "small_worry",
        "comforting_help",
        "peaceful_solution",
        "cozy_ending",
    ],
    "kid_friendly_scary": [
        "safe_setup",
        "strange_but_safe_mystery",
        "spooky_clue",
        "brave_investigation",
        "false_alarm",
        "real_but_non_dangerous_reveal",
        "courage_and_teamwork",
        "safe_resolution",
    ],
    "emotional": [
        "warm_setup",
        "character_want",
        "disappointment",
        "support_from_friend_or_family",
        "inner_realization",
        "brave_choice",
        "meaningful_resolution",
    ],
    "message": [
        "setup",
        "character_faces_choice",
        "easy_wrong_path",
        "consequence_without_harshness",
        "reflection",
        "better_choice",
        "lesson_lands_naturally",
    ],
    "science_fiction": [
        "ordinary_world",
        "new_invention_or_signal",
        "launch_into_wonder",
        "first_science_problem",
        "experiment_and_discovery",
        "systems_complication",
        "ethical_or_team_choice",
        "clever_science_solution",
        "wonder_filled_resolution",
        "return_with_new_understanding",
    ],
    "mystery": [
        "ordinary_setup",
        "puzzling_question",
        "first_clue",
        "wrong_guess",
        "second_clue",
        "patterns_connect",
        "gentle_confrontation_or_test",
        "solution_revealed",
        "fair_resolution",
    ],
    "historical": [
        "real_world_context",
        "introduce_real_person",
        "dream_or_goal",
        "challenge_of_the_time",
        "preparation",
        "major_attempt_or_event",
        "obstacle",
        "result",
        "legacy",
        "reflection_for_child",
    ],
    "educational": [
        "curiosity_setup",
        "big_question",
        "first_example",
        "hands_on_discovery",
        "confusing_moment",
        "clear_explanation",
        "use_the_learning",
        "meaningful_takeaway",
    ],
}


STORY_MODE_ROTATION = ["adventure", "funny", "mystery", "science_fiction", "emotional", "message", "bedtime"]


def detect_story_mode(user_text: str) -> str:
    normalized = normalize_command_text(user_text)
    if not normalized:
        return "adventure"
    if any(marker in normalized for marker in {"scary", "spooky", "haunted", "ghost", "assustadora", "terror"}):
        return "kid_friendly_scary"
    if any(marker in normalized for marker in {"emotional", "touching", "sad but good", "sad-but-good", "heartwarming", "emocionante"}):
        return "emotional"
    if any(marker in normalized for marker in {"lesson", "moral", "important message", "message story", "teach a message", "lição", "licao"}):
        return "message"
    if any(marker in normalized for marker in {"funny", "silly", "comedy", "hilarious", "engraçada", "engracada"}):
        return "funny"
    if any(marker in normalized for marker in {"adventure", "aventura", "quest", "journey"}):
        return "adventure"
    if any(marker in normalized for marker in {"science fiction", "sci fi", "sci-fi", "space", "spaceship", "robot planet", "future"}):
        return "science_fiction"
    if any(marker in normalized for marker in {"mystery", "detective", "clue", "solve"}):
        return "mystery"
    if any(marker in normalized for marker in {"true story", "real story", "history", "historical", "biography", "biographical", "real person", "inventor", "scientist", "amelia earhart"}):
        return "historical"
    if any(marker in normalized for marker in {"educational", "learn about", "teach me about", "explain through a story"}):
        return "educational"
    if any(marker in normalized for marker in {"sleep", "bedtime", "calm", "quiet", "soothing", "dormir"}):
        return "bedtime"
    return "adventure"


def _story_mode_explicitly_requested(user_text: str) -> bool:
    normalized = normalize_command_text(user_text)
    if not normalized:
        return False
    markers = {
        "adventure", "aventura", "scary", "spooky", "haunted", "ghost", "emotional", "touching",
        "sad but good", "heartwarming", "lesson", "moral", "important message", "funny", "silly",
        "comedy", "science fiction", "sci fi", "sci-fi", "space", "spaceship", "mystery",
        "detective", "clue", "true story", "real story", "history", "historical", "biography",
        "biographical", "real person", "inventor", "scientist", "educational", "learn about",
        "teach me about", "sleep", "bedtime", "calm", "quiet", "soothing",
    }
    return any(marker in normalized for marker in markers)


def _resolve_story_mode(user_text: str, state: RobotRuntimeState | None) -> tuple[str, str]:
    mode = detect_story_mode(user_text)
    if _story_mode_explicitly_requested(user_text):
        return mode, "user_request"
    recent = []
    if state is not None:
        with state.lock:
            recent = list(state.recent_story_modes_used[-3:])
    for candidate in STORY_MODE_ROTATION:
        if candidate not in recent:
            return candidate, "rotation"
    return STORY_MODE_ROTATION[int(time.time()) % len(STORY_MODE_ROTATION)], "rotation"


def _remember_story_mode_used(state: RobotRuntimeState, mode: str) -> None:
    with state.lock:
        state.recent_story_modes_used.append(mode)
        state.recent_story_modes_used = state.recent_story_modes_used[-6:]


def _story_fictionality_for_mode(mode: str, user_text: str) -> str:
    normalized = normalize_command_text(user_text)
    if mode == "historical" and any(marker in normalized for marker in {"true story", "real story", "history", "historical", "biography", "biographical"}):
        return "historical"
    if mode == "historical":
        return "inspired_by_true_events"
    return "fictional"


def _story_message_from_text(user_text: str, mode: str) -> str:
    normalized = normalize_command_text(user_text)
    if "kindness" in normalized:
        return "Kindness matters most when things are difficult."
    if "courage" in normalized or "brave" in normalized:
        return "Courage means continuing carefully even when something feels hard."
    if "family" in normalized:
        return "Family and teamwork help people find their way."
    if mode == "message":
        return "The lesson should land naturally through the characters' choices."
    if mode == "emotional":
        return "Feelings can be understood with patience, honesty, and support."
    return ""


def _story_central_goal(topic: str, mode: str, user_text: str) -> str:
    topic = _clean_story_topic(topic) or "the adventure"
    if mode == "historical":
        return f"Understand the real person's dream, challenge, work, and legacy around {topic}."
    if mode == "mystery":
        return f"Solve the central mystery of {topic} fairly, using clues that connect."
    if mode == "bedtime":
        return f"Reach a calm, safe ending around {topic}."
    if mode == "message":
        return f"Let the characters face a meaningful choice about {topic} and learn without preaching."
    if mode == "educational":
        return f"Explore and understand {topic} through a child-friendly story."
    return f"Follow one clear quest about {topic} from discovery to resolution."


def _story_central_question(topic: str, mode: str) -> str:
    topic = _clean_story_topic(topic) or "this story"
    questions = {
        "adventure": f"Can the characters complete the quest about {topic} and return safely changed?",
        "funny": f"How will the characters turn the growing silliness around {topic} into a happy fix?",
        "bedtime": f"How will the characters find peace and safety by the end of {topic}?",
        "kid_friendly_scary": "What is the strange mystery, and how can it be explained safely?",
        "emotional": "What feeling does the main character need to understand before the ending?",
        "message": "What better choice will the characters discover through the story?",
        "science_fiction": "What wonder or problem will science help the characters understand?",
        "mystery": "What really happened, and which clues prove it?",
        "historical": "What can a child learn from this real person's choices and legacy?",
        "educational": "What important idea will the story make clear?",
    }
    return questions.get(mode, questions["adventure"])


def _role_objective(role: str, mode: str, chapter_number: int, chapter_count: int, central_goal: str) -> str:
    base = {
        "setup": "Introduce the main characters, their ordinary world, and the first hint of the quest.",
        "call_to_adventure": "Make the quest impossible to ignore and commit the characters to the journey.",
        "enter_new_world": "Move into the new place or situation and show what makes it wondrous.",
        "first_obstacle": "Present a specific obstacle that tests the characters without resolving the whole quest.",
        "discovery": "Reveal a clue, tool, or truth that changes how the characters understand the goal.",
        "complication": "Make the goal harder in a new way, without repeating the previous obstacle.",
        "midpoint_reveal": "Reveal a larger truth that reorients the whole story.",
        "setback": "Let a kind, age-appropriate setback force the characters to rethink their plan.",
        "new_plan": "Have the characters choose a clearer, wiser plan for the final push.",
        "climax": "Resolve the central external challenge through earned courage, cleverness, or teamwork.",
        "resolution": "Show what changed because of the characters' choices.",
        "return_home": "Bring the story to a complete, warm ending.",
        "normal_setup": "Set up normal life and the first small comic mismatch.",
        "silly_mistake": "Let one harmless silly mistake create a bigger comic situation.",
        "misunderstanding_grows": "Escalate the misunderstanding through clear cause and effect.",
        "chaos_escalates": "Let the comedy peak while staying kind and safe.",
        "unexpected_helper": "Introduce an unexpected helper or clue that changes the comic direction.",
        "clever_fix": "Let the characters fix the mess in a clever, satisfying way.",
        "funny_resolution": "End with a happy laugh and no loose plot threads.",
        "calm_setup": "Begin slowly with warmth, safety, and gentle sensory details.",
        "gentle_wish": "Give the characters a soft wish or question to follow.",
        "soft_discovery": "Offer a peaceful discovery with wonder but no danger.",
        "small_worry": "Introduce a small worry that can be comforted.",
        "comforting_help": "Let family, friendship, or Miguel's calm help reduce the worry.",
        "peaceful_solution": "Resolve the worry with quiet confidence.",
        "cozy_ending": "Close completely with rest, safety, and calm.",
        "safe_setup": "Make the setting safe before anything spooky appears.",
        "strange_but_safe_mystery": "Introduce a strange mystery that is intriguing, not traumatic.",
        "spooky_clue": "Add one suspenseful clue without gore or real danger.",
        "brave_investigation": "Let the characters investigate carefully together.",
        "false_alarm": "Reveal that one frightening guess was harmless.",
        "real_but_non_dangerous_reveal": "Explain the real cause in a surprising but safe way.",
        "courage_and_teamwork": "Show how courage and teamwork help them understand the mystery.",
        "safe_resolution": "End in clear safety with comfort and closure.",
        "warm_setup": "Start with a warm relationship and a sincere emotional need.",
        "character_want": "Show what the character wants and why it matters.",
        "disappointment": "Let the character face disappointment gently and believably.",
        "support_from_friend_or_family": "Bring in support that listens before fixing.",
        "inner_realization": "Let the character understand something true about their feeling.",
        "brave_choice": "Have the character make a small brave choice from that realization.",
        "meaningful_resolution": "End with emotional change that feels earned.",
        "character_faces_choice": "Put the character in a clear choice with two understandable paths.",
        "easy_wrong_path": "Show the tempting easier path without making the character bad.",
        "consequence_without_harshness": "Let a gentle consequence reveal why the easy path falls short.",
        "reflection": "Give the character a quiet moment to think.",
        "better_choice": "Let the character choose better and act on it.",
        "lesson_lands_naturally": "End with the message visible through action, not a lecture.",
        "ordinary_world": "Start with everyday life before the science-fiction wonder appears.",
        "new_invention_or_signal": "Introduce the invention, signal, robot, or space clue.",
        "launch_into_wonder": "Move into a wider world of discovery.",
        "first_science_problem": "Pose a science-shaped problem with clear rules.",
        "experiment_and_discovery": "Let the characters test an idea and learn from it.",
        "systems_complication": "Make the system behave unexpectedly but safely.",
        "ethical_or_team_choice": "Ask the characters to choose responsibility and teamwork.",
        "clever_science_solution": "Solve the problem using understandable science logic.",
        "wonder_filled_resolution": "End with awe and a clear takeaway.",
        "return_with_new_understanding": "Return home or to safety with new understanding.",
        "ordinary_setup": "Introduce normal life and the odd detail that starts the mystery.",
        "puzzling_question": "State the mystery question clearly.",
        "first_clue": "Reveal a clue that points somewhere specific.",
        "wrong_guess": "Let the characters make a plausible wrong guess.",
        "second_clue": "Reveal a second clue that corrects the guess.",
        "patterns_connect": "Show how the clues fit together.",
        "gentle_confrontation_or_test": "Test the solution without danger or accusation.",
        "solution_revealed": "Reveal the answer fairly.",
        "fair_resolution": "Resolve the mystery and show why the clues mattered.",
        "real_world_context": "Give accurate, child-friendly context for the time and place.",
        "introduce_real_person": "Introduce the real person carefully and respectfully.",
        "dream_or_goal": "Explain the person's dream or goal without exaggeration.",
        "challenge_of_the_time": "Describe an age-appropriate real challenge from that time.",
        "preparation": "Show preparation, practice, study, or persistence.",
        "major_attempt_or_event": "Describe a major known event carefully.",
        "obstacle": "Explain an obstacle without making up dramatic false details.",
        "result": "Describe what is known to have happened.",
        "legacy": "Explain the person's legacy for children.",
        "reflection_for_child": "End with a gentle reflection about courage, curiosity, or persistence.",
        "curiosity_setup": "Begin with a child's curiosity and a concrete question.",
        "big_question": "State the learning question clearly.",
        "first_example": "Give the first simple example through story action.",
        "hands_on_discovery": "Let the characters discover by observing or trying safely.",
        "confusing_moment": "Include a common confusion or misconception.",
        "clear_explanation": "Resolve the confusion in simple language.",
        "use_the_learning": "Let the characters use the new idea.",
        "meaningful_takeaway": "End with a memorable takeaway.",
    }
    objective = base.get(role, "Advance the story with a specific new event.")
    if chapter_number == chapter_count:
        objective += " Make this chapter the real ending."
    return f"{objective} Keep the chapter tied to the central goal without quoting the planning notes."


def _expanded_arc_roles(mode: str, chapter_count: int) -> list[str]:
    template = STORY_ARC_TEMPLATES.get(mode) or STORY_ARC_TEMPLATES["adventure"]
    chapter_count = max(1, int(chapter_count or 1))
    if chapter_count <= len(template):
        if chapter_count == 1:
            return [template[-1]]
        return [template[round(i * (len(template) - 1) / (chapter_count - 1))] for i in range(chapter_count)]
    return [template[round(i * (len(template) - 1) / (chapter_count - 1))] for i in range(chapter_count)]


def _build_story_session_plan_local(
    user_text: str,
    language: str,
    subtype: str,
    chapter_count: int,
    topic: str,
    characters: list[str],
    setting: str,
    mode: str,
) -> dict:
    title = _story_title_from_topic(topic, language, subtype)
    central_goal = _story_central_goal(topic or title, mode, user_text)
    central_question = _story_central_question(topic or title, mode)
    message = _story_message_from_text(user_text, mode)
    roles = _expanded_arc_roles(mode, chapter_count)
    chapters = []
    for index, role in enumerate(roles, start=1):
        chapters.append({
            "chapter_number": index,
            "title": f"Chapter {index}",
            "arc_role": role,
            "objective": _role_objective(role, mode, index, chapter_count, central_goal),
            "must_include": [topic or title, ", ".join(characters), setting],
            "must_avoid": ["Do not repeat the same obstacle pattern from earlier chapters."],
        })
    return {
        "story_mode": mode,
        "story_title": title,
        "fictionality": _story_fictionality_for_mode(mode, user_text),
        "safety_level": "family_friendly",
        "main_characters": characters,
        "central_goal": central_goal,
        "central_question": central_question,
        "setting": setting,
        "emotional_tone": "calm and sleepy" if mode == "bedtime" else "family-safe and engaging",
        "message": message,
        "chapter_count": chapter_count,
        "chapters": chapters,
        "arc_template": mode,
        "planner": "local",
    }


def _json_object_from_text(text: str) -> dict | None:
    text = str(text or "").strip()
    if not text:
        return None
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _normalize_story_plan(plan: dict, fallback: dict, chapter_count: int, mode: str) -> dict:
    if not isinstance(plan, dict):
        plan = {}
    merged = dict(fallback)
    for key in {
        "story_mode", "story_title", "fictionality", "safety_level", "main_characters", "central_goal",
        "central_question", "setting", "emotional_tone", "message", "arc_template",
    }:
        value = plan.get(key)
        if value:
            merged[key] = value
    merged["story_mode"] = str(merged.get("story_mode") or mode)
    merged["planner"] = str(plan.get("planner") or merged.get("planner") or "cloud")
    chapters = plan.get("chapters") if isinstance(plan.get("chapters"), list) else []
    normalized_chapters = []
    fallback_chapters = fallback.get("chapters") or []
    for index in range(chapter_count):
        raw = chapters[index] if index < len(chapters) and isinstance(chapters[index], dict) else {}
        fallback_chapter = fallback_chapters[index] if index < len(fallback_chapters) else {}
        normalized_chapters.append({
            "chapter_number": index + 1,
            "title": str(raw.get("title") or fallback_chapter.get("title") or f"Chapter {index + 1}")[:80],
            "arc_role": str(raw.get("arc_role") or fallback_chapter.get("arc_role") or "story_progression")[:80],
            "objective": str(raw.get("objective") or fallback_chapter.get("objective") or "Advance the story.")[:500],
            "must_include": [str(item)[:160] for item in (raw.get("must_include") or fallback_chapter.get("must_include") or [])[:5]],
            "must_avoid": [str(item)[:160] for item in (raw.get("must_avoid") or fallback_chapter.get("must_avoid") or [])[:5]],
        })
    merged["chapter_count"] = chapter_count
    merged["chapters"] = normalized_chapters
    return merged


def _story_cloud_planning_preferred(mode: str, requested_minutes: int, chapter_count: int, user_text: str) -> bool:
    normalized = normalize_command_text(user_text)
    return (
        requested_minutes > 5
        or chapter_count > 5
        or mode in {"historical", "educational", "message"}
        or any(marker in normalized for marker in {"true story", "real story", "biography", "biographical", "important message"})
    )


def build_story_session_plan_cloud(
    user_text: str,
    language: str,
    subtype: str,
    chapter_count: int,
    topic: str,
    characters: list[str],
    setting: str,
    mode: str,
    fallback_plan: dict,
) -> dict | None:
    ask_cloud = getattr(v6, "ask_cloud_brain", None)
    if not callable(ask_cloud):
        return None
    prompt = (
        "You are Miguel's family-safe story director. Return only valid JSON, no markdown. "
        "Build a cohesive chapter-by-chapter plan for a spoken child-safe story. "
        f"Story mode: {mode}. Language: {language}. Subtype: {subtype}. Chapter count: {chapter_count}. "
        f"Topic: {topic}. Characters: {', '.join(characters)}. Setting: {setting}. "
        "The plan must have a stable central goal or central question, clear progression, a beginning, middle, climax, and real ending. "
        "Avoid repeating the same mini-scene formula in each chapter. "
        "Safety rules: family-friendly; no illegal drugs; no self-harm; no graphic violence or gore; no sexual content; "
        "no adult romantic content; no adult themes; no instructions for dangerous actions. "
        "If scary, make it suspenseful but safe and non-traumatic. "
        "If historical or true, do not invent major false facts, achievements, dates, or events; if uncertain, set fictionality to inspired_by_true_events. "
        "Return exactly this JSON shape: "
        "{\"story_mode\":\"...\",\"story_title\":\"...\",\"fictionality\":\"fictional|historical|inspired_by_true_events\","
        "\"safety_level\":\"family_friendly\",\"main_characters\":[\"...\"],\"central_goal\":\"...\","
        "\"central_question\":\"...\",\"setting\":\"...\",\"emotional_tone\":\"...\",\"message\":\"...\","
        "\"chapter_count\":0,\"arc_template\":\"...\",\"chapters\":[{\"chapter_number\":1,\"title\":\"...\","
        "\"arc_role\":\"...\",\"objective\":\"...\",\"must_include\":[\"...\"],\"must_avoid\":[\"...\"]}]} "
        f"Original user request: {user_text}"
    )
    try:
        raw = str(ask_cloud(prompt, _neutral_conversation_face_state()) or "")
    except Exception as exc:
        print("[STORY_PLAN] planner=cloud warning=", exc)
        return None
    parsed = _json_object_from_text(raw)
    if not parsed:
        print("[STORY_PLAN] planner=cloud warning=invalid_json")
        return None
    parsed["planner"] = "cloud"
    return _normalize_story_plan(parsed, fallback_plan, chapter_count, mode)


def _story_title_from_topic(topic: str, language: str, subtype: str) -> str:
    topic = _clean_story_topic(topic)
    if language == "fr":
        if subtype == "bedtime":
            return "La Nuit Calme de Marquinho"
        return topic[:70].title() if topic and topic != "this story" else "La Grande Aventure"
    if language == "es":
        if subtype == "bedtime":
            return "La Noche Tranquila de Marquinho"
        return topic[:70].title() if topic and topic != "this story" else "La Gran Aventura"
    if language == "it":
        if subtype == "bedtime":
            return "La Notte Tranquilla di Marquinho"
        return topic[:70].title() if topic and topic != "this story" else "La Grande Avventura"
    if language == "de":
        if subtype == "bedtime":
            return "Marquinhos Ruhige Nacht"
        return topic[:70].title() if topic and topic != "this story" else "Das Grosse Abenteuer"
    if subtype == "bedtime":
        return "A Noite Calma de Marquinho" if language == "pt" else "Marquinho's Quiet Night"
    if topic and topic != "this story":
        return topic[:70].title()
    return "A Grande Aventura" if language == "pt" else "The Big Adventure"


def _story_characters_from_text(text: str) -> list[str]:
    normalized = normalize_command_text(text)
    names = []
    for candidate in ("marquinho", "helena", "marco", "miguel"):
        if candidate in normalized and candidate.title() not in names:
            names.append(candidate.title())
    return names or ["Marquinho", "Helena"]


def _story_setting_from_text(text: str, language: str) -> str:
    normalized = normalize_command_text(text)
    if "brasilia" in normalized:
        return "Brasília"
    if "cidade" in normalized:
        return "uma cidade tranquila" if language == "pt" else "a quiet city"
    if language == "fr":
        return "un lieu plein d'imagination"
    if language == "es":
        return "un lugar lleno de imaginación"
    if language == "it":
        return "un luogo pieno di immaginazione"
    if language == "de":
        return "ein Ort voller Fantasie"
    return "um lugar cheio de imaginação" if language == "pt" else "an imaginative place"


def _build_story_plan(title: str, chapter_count: int, language: str, subtype: str) -> list[str]:
    if language == "pt":
        if subtype == "bedtime":
            base = [
                "Chegada calma e apresentação dos personagens.",
                "Uma pequena descoberta gentil, sem sustos.",
                "Um passeio sereno que resolve uma preocupação.",
                "Os personagens ajudam uns aos outros e desaceleram.",
                "Final aconchegante, com todos seguros e prontos para dormir.",
            ]
        else:
            base = [
                "Começo da aventura e apresentação do desejo dos personagens.",
                "A descoberta de um problema curioso.",
                "Uma tentativa criativa que quase funciona.",
                "A solução corajosa e colaborativa.",
                "Final completo, com volta para casa e aprendizado.",
            ]
    elif language == "fr":
        if subtype == "bedtime":
            base = [
                "Arrivée calme et présentation douce des personnages.",
                "Une petite découverte paisible, sans peur.",
                "Une promenade tranquille qui résout une inquiétude.",
                "Les personnages s'aident et ralentissent ensemble.",
                "Fin rassurante, tout le monde est en sécurité et prêt à dormir.",
            ]
        else:
            base = [
                "Début de l'aventure et présentation du souhait des personnages.",
                "Un problème curieux apparaît.",
                "Une idée créative fonctionne presque.",
                "Une solution courageuse et collective réussit.",
                "Fin complète, retour à la maison et petite leçon.",
            ]
    elif language == "es":
        if subtype == "bedtime":
            base = [
                "Llegada tranquila y presentación suave de los personajes.",
                "Un pequeño descubrimiento pacífico, sin sustos.",
                "Un paseo sereno que resuelve una preocupación.",
                "Los personajes se ayudan y bajan el ritmo.",
                "Final acogedor, todos seguros y listos para dormir.",
            ]
        else:
            base = [
                "Comienza la aventura y aparece el deseo de los personajes.",
                "Surge un problema curioso.",
                "Una idea creativa casi funciona.",
                "Una solución valiente y colaborativa tiene éxito.",
                "Final completo, regreso a casa y aprendizaje.",
            ]
    elif language == "it":
        if subtype == "bedtime":
            base = [
                "Arrivo tranquillo e presentazione dolce dei personaggi.",
                "Una piccola scoperta serena, senza paura.",
                "Una passeggiata calma risolve una preoccupazione.",
                "I personaggi si aiutano e rallentano insieme.",
                "Finale rassicurante, tutti al sicuro e pronti per dormire.",
            ]
        else:
            base = [
                "Inizio dell'avventura e desiderio dei personaggi.",
                "Compare un problema curioso.",
                "Un'idea creativa quasi funziona.",
                "Una soluzione coraggiosa e collaborativa riesce.",
                "Finale completo, ritorno a casa e piccola lezione.",
            ]
    elif language == "de":
        if subtype == "bedtime":
            base = [
                "Ruhige Ankunft und sanfte Vorstellung der Figuren.",
                "Eine kleine friedliche Entdeckung ohne Angst.",
                "Ein stiller Spaziergang loest eine Sorge.",
                "Die Figuren helfen einander und werden langsam ruhig.",
                "Geborgener Schluss, alle sind sicher und bereit zum Schlafen.",
            ]
        else:
            base = [
                "Das Abenteuer beginnt und der Wunsch der Figuren wird klar.",
                "Ein neugieriges Problem taucht auf.",
                "Eine kreative Idee funktioniert fast.",
                "Eine mutige gemeinsame Loesung gelingt.",
                "Vollstaendiger Schluss, Heimkehr und kleine Erkenntnis.",
            ]
    else:
        if subtype == "bedtime":
            base = [
                "A calm arrival and gentle character introduction.",
                "A soft discovery with no scary surprises.",
                "A peaceful walk that solves a small worry.",
                "The characters help each other slow down.",
                "A cozy ending with everyone safe and ready to sleep.",
            ]
        else:
            base = [
                "The adventure begins and the characters' wish is introduced.",
                "A curious problem appears.",
                "A creative attempt almost works.",
                "A brave collaborative solution succeeds.",
                "A complete ending brings everyone home with a lesson.",
            ]
    return base[:chapter_count]


def _start_story_session_from_intent(state: RobotRuntimeState, intent: dict, user_text: str) -> StorySession:
    language = intent.get("response_language") or intent.get("language") or ("pt" if "historia" in normalize_command_text(user_text) else "en")
    if language not in SUPPORTED_STORY_LANGUAGES or language == "unknown":
        language = intent.get("language") if intent.get("language") in SUPPORTED_STORY_LANGUAGES else "en"
    subtype = intent.get("subtype") or "general"
    chapter_count = int(intent.get("chapter_count") or 3)
    topic = str(intent.get("setting") or "").strip() or _extract_long_story_topic_hint(user_text) or _clean_story_topic(_extract_long_story_topic(user_text))
    characters = [str(name).strip() for name in intent.get("characters") or [] if str(name).strip()]
    characters = characters or _story_characters_from_text(user_text)
    setting = str(intent.get("setting") or "").strip() or _story_setting_from_text(user_text, language)
    story_mode, story_mode_source = _resolve_story_mode(user_text, state)
    if subtype == "bedtime":
        story_mode = "bedtime"
        story_mode_source = "bedtime_subtype"
    fallback_plan = _build_story_session_plan_local(
        user_text=user_text,
        language=language,
        subtype=subtype,
        chapter_count=chapter_count,
        topic=topic,
        characters=characters,
        setting=setting,
        mode=story_mode,
    )
    requested_minutes = int(intent.get("requested_minutes") or 0)
    plan = None
    if _story_cloud_planning_preferred(story_mode, requested_minutes, chapter_count, user_text):
        plan = build_story_session_plan_cloud(
            user_text=user_text,
            language=language,
            subtype=subtype,
            chapter_count=chapter_count,
            topic=topic,
            characters=characters,
            setting=setting,
            mode=story_mode,
            fallback_plan=fallback_plan,
        )
    if not plan:
        plan = fallback_plan
    plan = _normalize_story_plan(plan, fallback_plan, chapter_count, story_mode)
    title = str(plan.get("story_title") or fallback_plan.get("story_title") or _story_title_from_topic(topic, language, subtype))
    chapter_plan = list(plan.get("chapters") or [])
    story_plan = [str(chapter.get("objective") or chapter.get("arc_role") or f"Chapter {idx + 1}") for idx, chapter in enumerate(chapter_plan)]
    session = StorySession(
        active=True,
        mode="story_continuous",
        subtype=subtype,
        language=language,
        story_mode=str(plan.get("story_mode") or story_mode),
        planner=str(plan.get("planner") or "local"),
        fictionality=str(plan.get("fictionality") or _story_fictionality_for_mode(story_mode, user_text)),
        title=title,
        characters=[str(name).strip() for name in plan.get("main_characters") or characters if str(name).strip()],
        setting=str(plan.get("setting") or setting),
        central_goal=str(plan.get("central_goal") or fallback_plan.get("central_goal") or ""),
        central_question=str(plan.get("central_question") or fallback_plan.get("central_question") or ""),
        emotional_tone=str(plan.get("emotional_tone") or fallback_plan.get("emotional_tone") or ""),
        message=str(plan.get("message") or fallback_plan.get("message") or ""),
        chapter_count=chapter_count,
        current_chapter=0,
        story_plan=story_plan,
        chapter_plan=chapter_plan,
        arc_template=str(plan.get("arc_template") or story_mode),
        target_words_per_chapter=int(intent.get("target_words_per_chapter") or _long_story_words_per_chapter()),
        auto_continue=True,
    )
    with state.lock:
        state.story_session = session
        state.long_story_active = True
        state.long_story_topic = title
        state.long_story_target_minutes = requested_minutes
        state.long_story_max_segments = chapter_count
        state.long_story_segment_index = 0
        state.long_story_style = "calm bedtime" if subtype == "bedtime" else _extract_story_style(user_text)
        state.conversation_mode = "story"
        state.response_depth_mode = "long_story"
        state.response_length_mode = "long_story"
    _remember_story_mode_used(state, session.story_mode)
    print(f"[STORY_MODE] mode={session.story_mode} source={story_mode_source}")
    print(f"[STORY_PLAN] planner={session.planner} chapter_count={chapter_count} title={_short_log_text(title)}")
    print(f"[STORY_PLAN] central_goal={_short_log_text(session.central_goal)}")
    print(f"[STORY_PLAN] arc_template={session.arc_template}")
    for chapter in session.chapter_plan:
        chapter_number = int(chapter.get("chapter_number") or 0)
        if chapter_number in {1, max(1, chapter_count // 2), chapter_count}:
            print(
                f"[STORY_CHAPTER] n={chapter_number} arc_role={chapter.get('arc_role')} "
                f"objective={_short_log_text(chapter.get('objective'))}"
            )
    print(
        f"[V7.15 STORY DURATION] requested_minutes={requested_minutes} "
        f"target_words={_long_story_target_words(requested_minutes)} "
        f"policy=proportional_chapters chapters={chapter_count} words_per_chapter={session.target_words_per_chapter}"
    )
    print(
        f"[V7.15 STORY SESSION] started=true mode=story_continuous subtype={subtype} "
        f"language={language} chapters={chapter_count}"
    )
    print(f"[V7.15 STORY PLAN] title=\"{title}\" chapters={chapter_count}")
    return session


def _fallback_story_chapter(session: StorySession, chapter_number: int) -> str:
    final = chapter_number >= session.chapter_count
    if session.language == "pt":
        names = " e ".join(session.characters)
    elif session.language == "fr":
        names = " et ".join(session.characters)
    elif session.language == "es":
        names = " y ".join(session.characters)
    elif session.language == "it":
        names = " e ".join(session.characters)
    elif session.language == "de":
        names = " und ".join(session.characters)
    else:
        names = " and ".join(session.characters)
    summary = session.story_plan[chapter_number - 1] if chapter_number - 1 < len(session.story_plan) else ""
    if session.language == "pt":
        ending = (
            "No fim, eles voltaram para casa com o coração tranquilo, sabendo que a aventura tinha terminado bem."
            if final
            else "Quando a lua subiu um pouco mais, um novo caminho apareceu, e a próxima parte começou sozinha."
        )
        tone = "calma e macia" if session.subtype == "bedtime" else "cheia de coragem"
        return (
            f"Capítulo {chapter_number}: {session.title}. {names} estavam em {session.setting}, numa noite {tone}. "
            f"{summary} Eles encontraram uma pista pequena, pensaram juntos, e escolheram a solução mais gentil. "
            f"Marquinho percebeu que uma boa aventura não precisa ser barulhenta para ser importante. Helena sorriu, "
            f"Miguel piscou suas luzes, e todos seguiram com cuidado. A cada passo, eles lembravam de respirar devagar, "
            f"escutar um ao outro, e transformar medo em curiosidade. Quando uma porta parecia fechada, Helena procurava "
            f"um detalhe brilhante, Marquinho inventava uma ideia simples, e Miguel ajudava a testar sem pressa. "
            f"Assim, a aventura ficava segura, bonita, e cheia de pequenas descobertas. {ending}"
        )
    if session.language == "fr":
        ending = (
            "À la fin, ils rentrèrent le cœur tranquille, certains que l'aventure s'était bien terminée."
            if final
            else "Quand la lune monta un peu plus haut, un nouveau chemin apparut, et le chapitre suivant commença doucement."
        )
        tone = "douce et calme" if session.subtype == "bedtime" else "courageuse et lumineuse"
        return (
            f"Chapitre {chapter_number}: {session.title}. {names} étaient à {session.setting}, pendant une soirée {tone}. "
            f"{summary} Ils trouvèrent un petit indice, réfléchirent ensemble, puis choisirent la solution la plus gentille. "
            f"Marquinho comprit qu'une aventure n'a pas besoin d'être bruyante pour être importante. Helena sourit, "
            f"Miguel fit clignoter ses lumières, et chacun avança avec patience. À chaque pas, ils respiraient lentement, "
            f"écoutaient les idées des autres, et transformaient l'inquiétude en curiosité. Quand le silence devenait profond, "
            f"ils regardaient les étoiles comme de petites lampes amies, et l'histoire avançait sans se presser. {ending}"
        )
    if session.language == "es":
        ending = (
            "Al final, volvieron a casa con el corazón tranquilo, sabiendo que la aventura había terminado bien."
            if final
            else "Cuando la luna subió un poco más, apareció un nuevo camino, y el siguiente capítulo empezó suavemente."
        )
        tone = "suave y tranquila" if session.subtype == "bedtime" else "valiente y luminosa"
        return (
            f"Capítulo {chapter_number}: {session.title}. {names} estaban en {session.setting}, durante una noche {tone}. "
            f"{summary} Encontraron una pista pequeña, pensaron juntos y eligieron la solución más amable. "
            f"Marquinho aprendió que una aventura no necesita ser ruidosa para importar. Helena sonrió, "
            f"Miguel encendió sus luces, y todos avanzaron con paciencia. En cada paso respiraban despacio, "
            f"escuchaban las ideas de los demás y convertían la preocupación en curiosidad. Cuando el silencio se hacía profundo, "
            f"miraban las estrellas como pequeñas lámparas amigas, y la historia seguía sin prisa. También guardaban cada "
            f"descubrimiento en la memoria, como una lucecita tranquila para llevar en el corazón. {ending}"
        )
    if session.language == "it":
        ending = (
            "Alla fine tornarono a casa con il cuore tranquillo, sapendo che l'avventura era finita bene."
            if final
            else "Quando la luna salì un po' più in alto, apparve un nuovo sentiero, e il capitolo seguente cominciò piano."
        )
        tone = "dolce e calma" if session.subtype == "bedtime" else "coraggiosa e luminosa"
        return (
            f"Capitolo {chapter_number}: {session.title}. {names} erano in {session.setting}, durante una sera {tone}. "
            f"{summary} Trovarono un piccolo indizio, pensarono insieme e scelsero la soluzione più gentile. "
            f"Marquinho capì che un'avventura non deve essere rumorosa per essere importante. Helena sorrise, "
            f"Miguel accese le sue luci, e tutti avanzarono con pazienza. A ogni passo respiravano piano, "
            f"ascoltavano le idee degli altri e trasformavano la preoccupazione in curiosità. Quando il silenzio diventava profondo, "
            f"guardavano le stelle come piccole lampade amiche, e la storia continuava senza fretta. {ending}"
        )
    if session.language == "de":
        ending = (
            "Am Ende gingen sie mit ruhigem Herzen nach Hause, weil das Abenteuer wirklich gut ausgegangen war."
            if final
            else "Als der Mond etwas höher stieg, erschien ein neuer Weg, und das nächste Kapitel begann leise."
        )
        tone = "sanften und ruhigen" if session.subtype == "bedtime" else "mutigen und hellen"
        return (
            f"Kapitel {chapter_number}: {session.title}. {names} waren in {session.setting}, an einem {tone} Abend. "
            f"{summary} Sie fanden einen kleinen Hinweis, dachten gemeinsam nach und waehlten die freundlichste Loesung. "
            f"Marquinho verstand, dass ein Abenteuer nicht laut sein muss, um wichtig zu sein. Helena laechelte, "
            f"Miguel liess seine Lichter blinken, und alle gingen geduldig weiter. Bei jedem Schritt atmeten sie langsam, "
            f"hoerten einander zu und verwandelten Sorge in Neugier. Wenn die Stille tief wurde, sahen sie zu den Sternen, "
            f"als waeren sie kleine freundliche Lampen, und die Geschichte ging ruhig weiter. {ending}"
        )
    ending = (
        "In the end, they came home peaceful and proud, knowing the adventure had truly ended well."
        if final
        else "By the end of this chapter, one new piece of the larger journey was clear."
    )
    tone = "soft and calm" if session.subtype == "bedtime" else "brave and bright"
    chapter_spec = _story_chapter_spec(session, chapter_number)
    objective = str(chapter_spec.get("objective") or summary or "Advance the story.")
    role = str(chapter_spec.get("arc_role") or "story_progression").replace("_", " ")
    lead = names.split(" and ")[0] if names else "The characters"
    return (
        f"Chapter {chapter_number}: {session.title}. This chapter's role is {role}. "
        f"{names} were in {session.setting}, moving through a {tone} part of the story. "
        f"The central question was: {session.central_question} {objective} "
        f"{lead} noticed one concrete change that made the goal feel closer, while the others helped in different ways. "
        f"The chapter added a new event instead of repeating an old obstacle, and it kept the story safe, clear, and connected. "
        f"{ending}"
    )


def _story_chapter_spec(session: StorySession, chapter_number: int) -> dict:
    if 0 < chapter_number <= len(session.chapter_plan):
        chapter = session.chapter_plan[chapter_number - 1]
        if isinstance(chapter, dict):
            return chapter
    objective = session.story_plan[chapter_number - 1] if 0 < chapter_number <= len(session.story_plan) else "Advance the story."
    return {
        "chapter_number": chapter_number,
        "title": f"Chapter {chapter_number}",
        "arc_role": "story_progression",
        "objective": objective,
        "must_include": [session.central_goal, session.central_question],
        "must_avoid": ["Do not repeat the same obstacle pattern from earlier chapters."],
    }


def _summarize_previous_story_chapter(session: StorySession, chapter_number: int) -> str:
    previous = str(session.generated_chapters.get(chapter_number - 1) or "").strip()
    if not previous:
        return "No previous chapter yet; this chapter starts the story." if chapter_number <= 1 else "Previous chapter summary unavailable."
    words = re.findall(r"\S+", previous)
    if len(words) <= 55:
        return previous
    return " ".join(words[:55]) + "..."


def _spoken_story_include_items(session: StorySession, chapter_spec: dict) -> str:
    planning_fragments = {
        str(session.central_goal or "").strip().lower(),
        str(session.central_question or "").strip().lower(),
        str(chapter_spec.get("objective") or "").strip().lower(),
    }
    items = []
    for item in chapter_spec.get("must_include") or []:
        value = str(item or "").strip()
        lowered = value.lower()
        if not value or lowered in planning_fragments:
            continue
        if "central goal" in lowered or "central question" in lowered:
            continue
        if lowered.startswith(("how will ", "can the characters ", "reach a calm")):
            continue
        items.append(value)
    return "; ".join(items[:4]) or "the named characters, setting, and story topic"


def _remove_story_planning_leaks(text: str, session: StorySession, chapter_spec: dict) -> str:
    cleaned = str(text or "")
    leak_phrases = [
        session.central_goal,
        session.central_question,
        chapter_spec.get("objective"),
    ]
    for phrase in leak_phrases:
        phrase = str(phrase or "").strip()
        if phrase and phrase in cleaned:
            cleaned = cleaned.replace(phrase, "")
    cleaned = re.sub(
        r"\b(?:central goal|central question|current chapter objective|must include)\s*:\s*[^.?!]*(?:[.?!]|$)",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", cleaned).strip()


def _generate_story_chapter(session: StorySession, user_text: str, chapter_number: int) -> str:
    final = chapter_number >= session.chapter_count
    language_names = {
        "pt": "Portuguese",
        "en": "English",
        "fr": "French",
        "es": "Spanish",
        "it": "Italian",
        "de": "German",
    }
    language_name = language_names.get(session.language, session.language or "the user's language")
    tone = "calm, gentle, non-scary bedtime" if session.subtype == "bedtime" else "family-safe adventure"
    chapter_spec = _story_chapter_spec(session, chapter_number)
    previous_summary = _summarize_previous_story_chapter(session, chapter_number)
    must_include = _spoken_story_include_items(session, chapter_spec)
    must_avoid = "; ".join(str(item) for item in chapter_spec.get("must_avoid") or [] if str(item).strip()) or "repeating earlier chapter structure"
    prompt_parts = [
        f"Write chapter {chapter_number} of {session.chapter_count} for a continuous spoken story. "
        f"Language: {language_name}. Story mode: {session.story_mode}. Fictionality: {session.fictionality}. "
        f"Tone: {session.emotional_tone or tone}. Overall story title: {session.title}. "
        f"Characters: {', '.join(session.characters)}. Setting: {session.setting}. "
        f"Central goal: {session.central_goal}. Central question: {session.central_question}. "
        f"Message, if any: {session.message}. "
        f"Previous chapter summary: {previous_summary}. "
        f"Current chapter title: {chapter_spec.get('title')}. "
        f"Current chapter arc role: {chapter_spec.get('arc_role')}. "
        f"Current chapter objective: {chapter_spec.get('objective')}. "
        f"Must include: {must_include}. Must avoid: {must_avoid}. "
        f"Safety rules: {' '.join(STORY_SAFETY_RULES)} "
        "Use the plan only as private guidance. Do not quote or mention the central goal, central question, "
        "chapter objective, must-include list, or original request in the story. "
        "Do not reuse the same obstacle, same solution rhythm, or same sentence frame from earlier chapters. "
        f"Target {session.target_words_per_chapter - 30} to {session.target_words_per_chapter + 30} spoken words. ",
    ]
    if session.redirect_instruction:
        prompt_parts.append(f"Updated direction for this and later chapters: {session.redirect_instruction}. ")
    prompt_parts.append("Do not ask whether to continue. ")
    prompt_parts.append(
        "This is the final chapter; conclude the story gently and completely. "
        if final
        else "End naturally but allow the next chapter to continue. "
    )
    prompt_parts.append(f"Original user request: {user_text}")
    prompt = "".join(prompt_parts)
    ask_cloud = getattr(v6, "ask_cloud_brain", None)
    if callable(ask_cloud):
        try:
            reply = str(ask_cloud(prompt, _neutral_conversation_face_state()) or "").strip()
            if reply:
                reply = _remove_story_planning_leaks(reply, session, chapter_spec)
                if reply:
                    return trim_to_word_limit_preserve_sentence(reply, session.target_words_per_chapter + 30)
        except Exception as exc:
            print("[V7.15 STORY CHAPTER] cloud_generation_warning=", exc)
    return trim_to_word_limit_preserve_sentence(_fallback_story_chapter(session, chapter_number), session.target_words_per_chapter + 30)


def _story_control_reply(language: str, kind: str) -> str:
    language = str(language or "").lower()
    replies = {
        "paused": {
            "pt": "Está bem. Vou pausar a história aqui.",
            "es": "De acuerdo. Pausaré la historia aquí.",
            "fr": "D'accord. Je mets l'histoire en pause ici.",
        },
        "resumed": {
            "pt": "Continuando a história.",
            "es": "Continúo la historia.",
            "fr": "Je continue l'histoire.",
        },
        "redirected": {
            "pt": "Entendi. Vou mudar o rumo da história daqui pra frente.",
            "es": "Entendido. Cambiaré el rumbo de la historia desde aquí.",
            "fr": "Compris. Je vais changer la direction de l'histoire à partir d'ici.",
        },
        "stopped": {
            "pt": "Está bem. Vou parar a história por aqui.",
            "es": "De acuerdo. Detendré la historia aquí.",
            "fr": "D'accord. J'arrête l'histoire ici.",
        },
    }
    return replies.get(kind, {}).get(language, {
        "paused": "Okay. I will pause the story here.",
        "resumed": "Continuing the story.",
        "redirected": "Got it. I will change the story direction from here.",
        "stopped": "Okay. I will stop the story here.",
    }.get(kind, "Okay."))


def _route_story_stop_request(user_text: str, state: RobotRuntimeState) -> bool:
    if not _is_story_stop_request(user_text):
        return False
    with state.lock:
        active = bool(state.story_session.active)
        state.story_session.stop_requested = True
        state.story_session.paused = False
        state.long_story_active = False
        language = state.story_session.language
    if not active:
        return False
    _request_speech_stop(state)
    state.stop_speech_event.clear()
    print("[V7.15 STORY SESSION] stopped=true reason=user_stop_command")
    v6.speak(_story_control_reply(language, "stopped"))
    return True


def _route_story_control_request(user_text: str, state: RobotRuntimeState) -> bool:
    normalized = normalize_command_text(user_text)
    if not normalized:
        return False
    with state.lock:
        active = bool(state.story_session.active)
        paused = bool(state.story_session.paused)
        language = state.story_session.language
    if not active and not paused:
        return False

    if _is_story_stop_request(user_text):
        return _route_story_stop_request(user_text, state)

    if _is_story_pause_request(user_text):
        with state.lock:
            state.story_session.paused = True
            state.long_story_active = False
        _request_speech_stop(state)
        state.stop_speech_event.clear()
        print("[V7.15 STORY CONTROL] action=pause active=true")
        v6.speak(_story_control_reply(language, "paused"))
        return True

    redirect = _extract_story_redirect_instruction(user_text)
    if redirect:
        with state.lock:
            current = max(0, int(state.story_session.current_chapter or 0))
            state.story_session.redirect_instruction = redirect
            state.story_session.paused = False
            state.story_session.stop_requested = False
            state.long_story_active = True
            for chapter_number in list(state.story_session.generated_chapters):
                if chapter_number > current:
                    state.story_session.generated_chapters.pop(chapter_number, None)
            for index in range(current, state.story_session.chapter_count):
                if index < len(state.story_session.story_plan):
                    state.story_session.story_plan[index] = f"Follow the new direction: {redirect}"
        print(f"[V7.15 STORY CONTROL] action=redirect direction={_short_log_text(redirect)}")
        v6.speak(_story_control_reply(language, "redirected"))
        return True

    if paused and _is_story_resume_request(user_text):
        with state.lock:
            state.story_session.paused = False
            state.story_session.stop_requested = False
            state.long_story_active = True
        print("[V7.15 STORY CONTROL] action=resume")
        v6.speak(_story_control_reply(language, "resumed"))
        return True

    return False


def _route_story_duration_reality_question(user_text: str) -> bool:
    normalized = normalize_command_text(user_text)
    if not normalized:
        return False
    if _extract_long_story_duration_minutes(normalized) and any(
        marker in normalized
        for marker in {
            "will it really be",
            "really be",
            "exactly",
            "literal",
            "vai ser mesmo",
            "realmente vai ser",
            "exatamente",
        }
    ):
        v6.speak("Not exactly. I will make it a longer chaptered story, but I will keep it voice-friendly so it does not take too long.")
        return True
    return False


def _story_speech_is_busy(state: RobotRuntimeState) -> bool:
    with state.lock:
        return bool(state.is_speaking or state.pending_reply_count > 0 or _queue_has_items(state.reply_queue))


def _wait_for_story_speech_slot(state: RobotRuntimeState, poll_seconds: float = 0.1) -> bool:
    while not state.stop_event.is_set():
        with state.lock:
            session = state.story_session
            if session.stop_requested:
                return False
            if session.paused:
                busy = True
            else:
                busy = bool(state.is_speaking or state.pending_reply_count > 0 or _queue_has_items(state.reply_queue))
        if not busy:
            return True
        time.sleep(poll_seconds)
    return False


def _wait_for_story_resume(state: RobotRuntimeState, poll_seconds: float = 0.1) -> bool:
    while not state.stop_event.is_set():
        with state.lock:
            if state.story_session.stop_requested:
                return False
            if not state.story_session.paused:
                return True
        time.sleep(poll_seconds)
    return False


def _run_story_session(user_text: str, state: RobotRuntimeState, session: StorySession) -> None:
    # Automatic chapters belong to the turn that started this story.  The
    # ordinary conversation lease can expire during a long playback, so do
    # not let later chapters lose their person or get logged as wake_required.
    with state.lock:
        story_person = state.conversation_partner or state.recognized_person
        story_topic = state.session_focus or state.last_topic or session.title
    for chapter_number in range(max(1, session.current_chapter + 1), session.chapter_count + 1):
        if not _wait_for_story_resume(state):
            print("[V7.15 STORY SESSION] stopped=true reason=user_stop_command")
            return
        with state.lock:
            if state.story_session.stop_requested:
                print("[V7.15 STORY SESSION] stopped=true reason=user_stop_command")
                return
            state.story_session.current_chapter = chapter_number
            state.long_story_segment_index = chapter_number
            active_session = state.story_session
        generation_started = time.monotonic()
        if chapter_number < active_session.chapter_count:
            print(f"[V7.15 STORY PREFETCH] generating_next={chapter_number + 1} while_speaking={chapter_number}")
        chapter = _generate_story_chapter(active_session, user_text, chapter_number)
        generation_finished = time.monotonic()
        speech_slot_wait_started = generation_finished
        if not _wait_for_story_speech_slot(state):
            print("[V7.15 STORY SESSION] stopped=true reason=user_stop_command")
            return
        words = _word_len(chapter)
        with state.lock:
            state.story_session.generated_chapters[chapter_number] = chapter
        print(f"[V7.15 STORY CHAPTER] generated={chapter_number} queued_to_speech=true words={words}")
        chapter_latency = {
            # Automatic chapters are prefetched while the preceding chapter is
            # playing.  Start speech latency when generation completes instead
            # of charging that overlapping work to the reply queue/total.
            "turn_started_at": speech_slot_wait_started,
            "transcript_ready_at": speech_slot_wait_started,
            "route_done_at": generation_finished,
            "reply_context": "story",
            "response_length_mode": "long_story",
            "response_depth_mode": "long_story",
            "log_user_text": user_text if chapter_number == 1 else "",
            "log_person": story_person,
            "log_conversation_mode": "story",
            "log_topic": story_topic,
            "story_chapter": chapter_number,
            "story_chapter_count": active_session.chapter_count,
            "story_auto_continue": chapter_number > 1,
            "story_generation_ms": max(0.0, generation_finished - generation_started) * 1000,
            "story_speech_slot_wait_ms": max(0.0, time.monotonic() - speech_slot_wait_started) * 1000,
        }
        _speak_with_enqueue_context(chapter, chapter_latency)
        if chapter_number < active_session.chapter_count:
            print(f"[V7.15 STORY PREFETCH] ready_next={chapter_number + 1}")
            print(f"[V7.15 STORY CHAPTER] auto_continue={chapter_number + 1}")
    _wait_for_story_speech_slot(state)
    with state.lock:
        if state.story_session.stop_requested:
            print("[V7.15 STORY SESSION] stopped=true reason=user_stop_command")
            return
        state.story_session.active = False
        state.story_session.paused = False
        state.long_story_active = False
        state.long_story_segment_index = 0
        state.conversation_mode = "general"
        state.response_depth_mode = "normal"
        state.response_length_mode = "normal"
    print(f"[V7.15 STORY SESSION] completed=true chapters_spoken={session.chapter_count}")


def _route_story_continuous_generation(user_text: str, state: RobotRuntimeState, intent: dict) -> bool:
    if not (intent.get("detected") and intent.get("action") == "generate_story" and intent.get("story_mode") == "story_continuous"):
        return False
    _log_story_execution_generate()
    _log_story_execution_skip_legacy("story_request")
    session = _start_story_session_from_intent(state, intent, user_text)
    _set_reply_context(state, "story")
    _set_response_length_context(state, "long_story")
    worker = threading.Thread(
        target=_run_story_session,
        args=(user_text, state, session),
        name="StorySessionWorker",
        daemon=True,
    )
    with state.lock:
        state.story_worker_thread = worker
    worker.start()
    return True


def _long_story_segment(topic: str, segment_index: int, max_segments: int) -> str:
    topic = topic or "this story"
    if "explanation" in topic or "topic" in topic:
        openings = [
            f"Part {segment_index}: Let's build the idea step by step. The main thing about {topic} is that small pieces connect into a bigger pattern.",
            f"Part {segment_index}: Another useful layer is cause and effect. When one part changes, the next part often changes too.",
            f"Part {segment_index}: Now we can compare examples. A simple example makes the idea easier to remember than a big definition.",
            f"Part {segment_index}: The deeper point is that good thinking checks both what is true and what might be missing.",
            f"Part {segment_index}: To wrap this part, remember the core pattern first, then add details only when they help.",
        ]
    else:
        openings = [
            f"Part {segment_index}: Once, under a soft blue night sky, Miguel found a tiny glowing map folded behind a toolbox.",
            f"Part {segment_index}: The map led him past quiet shelves and silver wires until he reached a door no bigger than a book.",
            f"Part {segment_index}: Behind the door was a little city of lights, where every window blinked like it was thinking.",
            f"Part {segment_index}: Miguel followed a brave spark through the city, learning that courage can be quiet and still be strong.",
            f"Part {segment_index}: At last, the spark showed Miguel the way home, and the map folded itself into a star.",
        ]
    base = openings[min(segment_index - 1, len(openings) - 1)]
    suffix = " Say continue for the next part." if segment_index < max_segments else " That is the end for now."
    return trim_to_word_limit_preserve_sentence(base, _long_story_segment_words()) + suffix


def _route_long_story_mode(user_text: str, state: RobotRuntimeState) -> bool:
    normalized = normalize_command_text(user_text)
    if normalized in NORMAL_DEPTH_PHRASES or normalized == "exit long mode":
        _set_response_depth_mode(state, "normal", "legacy_long_mode_normal")
        _set_response_length_context(state, "normal")
        v6.speak("Normal mode on. I'll keep answers shorter.")
        return True

    story_detection = _story_mode_detection(user_text)
    with state.lock:
        current_depth_mode = state.response_depth_mode
    if (
        story_detection.get("mode") == "long_story"
        and story_detection.get("has_story_request")
    ) or current_depth_mode == "long_story":
        _log_story_execution_skip_legacy("story_request")
        return False

    if _is_long_mode_request(user_text):
        topic = _extract_long_story_topic_hint(user_text) or _clean_story_topic(_extract_long_story_topic(user_text)) or "this story"
        narrative = _contains_story_word(normalized) or any(phrase in normalized for phrase in {"bedtime"})
        target_minutes = _extract_long_story_duration_minutes(user_text)
        story_style = _extract_story_style(user_text)
        recovered_context = _recover_story_context_from_logs(user_text, state)
        _set_response_depth_mode(state, "long_story" if narrative else "long_explanation", "legacy_long_mode_request")
        _set_response_length_context(state, "long_story" if narrative else "detailed")
        _force_active_after_mode(state, "story" if narrative else "general", reason="long_mode")
        with state.lock:
            state.long_story_active = False
            state.long_story_topic = topic
            state.long_story_segment_index = 0
            state.long_story_max_segments = 0
            state.long_story_target_minutes = target_minutes if narrative else 0
            state.long_story_style = story_style if narrative else ""
            state.recovered_story_context = recovered_context if narrative else ""
        if narrative and topic and topic != "this story":
            _remember_long_story_topic(state, topic)
        if narrative and _is_story_generation_request(user_text):
            return False
        if narrative:
            duration = _format_long_story_duration(target_minutes)
            topic_part = f" about {topic}" if topic and topic != "this story" else ""
            style_part = f" in {story_style} style" if story_style else ""
            v6.speak("Long story mode on" + topic_part + style_part + (f" for about {duration}." if duration else ". I'll give richer stories when you ask."))
        else:
            v6.speak("Long explanation mode on. I'll explain with more detail.")
        return True

    with state.lock:
        active = bool(state.long_story_active)
        index = int(state.long_story_segment_index or 0)
        max_segments = int(state.long_story_max_segments or _long_story_max_segments())
        topic = state.long_story_topic or state.last_topic or "this story"
    if active and _is_long_mode_continue(user_text):
        _set_response_depth_mode(state, "long_story", "legacy_long_mode_continue")
        _set_response_length_context(state, "long_story")
        with state.lock:
            state.long_story_active = False
            state.long_story_topic = topic
            state.long_story_segment_index = index
            state.long_story_max_segments = max_segments
        return False
    return False


def _choose_first_direct_command(user_text: str) -> str:
    priority_order = ("shutdown", "sleep", "owner", "timer", "voice", "memory", "capabilities", "time_status", "camera_identity", "camera_scene", "creative", "enrollment", "general")
    parts = [part.strip() for part in re.split(r"[.!?;]+", str(user_text or "")) if part.strip()]
    if len(parts) <= 1:
        normalized = normalize_command_text(user_text)
        markers = [
            ("shutdown", "confirm shutdown"),
            ("shutdown", "cancel shutdown"),
            ("shutdown", "shutdown"),
            ("shutdown", "shut down"),
            ("shutdown", "stop"),
            ("sleep", "sleep mode"),
            ("sleep", "wake up"),
            ("owner", "owner mode"),
            ("owner", "password"),
            ("timer", "set a timer"),
            ("timer", "start a timer"),
            ("timer", "cancel timer"),
            ("timer", "timer status"),
            ("voice", "robotic voice"),
            ("voice", "natural voice"),
            ("voice", "friendly voice"),
            ("voice", "deep voice"),
            ("voice", "story voice"),
            ("voice", "robot voice"),
            ("voice", "voice options"),
            ("voice", "what voices"),
            ("voice", "voice"),
            ("camera_identity", "who am i"),
            ("camera_identity", "who do you see"),
            ("camera_identity", "you see me now"),
            ("camera_identity", "you see me"),
            ("camera_identity", "see me now"),
            ("camera_identity", "see me"),
            ("camera_identity", "am i visible"),
            ("camera_identity", "can you recognize me now"),
            ("camera_identity", "do you recognize me"),
            ("camera_identity", "identify me"),
            ("camera_identity", "who is this person"),
            ("camera_identity", "do you see me"),
            ("camera_identity", "do you see a face"),
            ("camera_identity", "can you see a face"),
            ("camera_identity", "you cannot see me"),
            ("camera_identity", "you can t see me"),
            ("camera_identity", "can you recognize me"),
            ("camera_identity", "who is in front of you"),
            ("camera_identity", "did you see another face"),
            ("camera_identity", "do you see another face"),
            ("camera_identity", "can you see another face"),
            ("camera_identity", "do you see both faces"),
            ("camera_identity", "can you see both faces"),
            ("camera_identity", "can you recognize the faces"),
            ("camera_identity", "who are those faces"),
            ("camera_identity", "who are the faces"),
            ("camera_identity", "who are the people"),
            ("camera_identity", "who is there"),
            ("camera_identity", "who is with me"),
            ("camera_identity", "do you recognize both of us"),
            ("camera_identity", "can you recognize both of us"),
            ("camera_identity", "is marquinho there"),
            ("camera_identity", "is marco there"),
            ("camera_identity", "can you see marco"),
            ("camera_identity", "can you see marquinho"),
            ("camera_scene", "what do you see"),
            ("camera_scene", "look around"),
            ("time_status", "what time is it"),
            ("time_status", "what time is it now"),
            ("time_status", "current time"),
            ("time_status", "weather"),
            ("time_status", "calculate"),
            ("time_status", "status"),
            ("memory", "recent topics"),
            ("memory", "previous topics"),
            ("memory", "conversation history"),
            ("memory", "analyze conversation log"),
            ("memory", "analyze conversation"),
            ("memory", "analyze the previous log"),
            ("memory", "analyze previous log"),
            ("memory", "previous log"),
            ("memory", "log file"),
            ("capabilities", "what can you do"),
            ("capabilities", "what are your capabilities"),
            ("capabilities", "capabilities"),
            ("creative", "creative mode"),
            ("creative", "be creative"),
            ("creative", "superhero"),
            ("creative", "invent"),
            ("enrollment", "learn this face"),
            ("enrollment", "add a new face"),
            ("enrollment", "enroll a new person"),
            ("enrollment", "start enrollment"),
            ("enrollment", "start enrollment for"),
            ("enrollment", "re enroll"),
            ("enrollment", "reenroll"),
            ("enrollment", "reinrou"),
            ("enrollment", "remember this person"),
            ("general", "how are you"),
            ("general", "what are you"),
            ("general", "can you hear me"),
        ]
        matches = []
        for kind, phrase in markers:
            index = normalized.find(phrase)
            if index >= 0:
                matches.append((index, kind, phrase))
        unique_positions = {(index, kind, phrase) for index, kind, phrase in matches}
        if len(unique_positions) <= 1:
            return user_text

        for priority_kind in priority_order:
            priority_matches = sorted(match for match in unique_positions if match[1] == priority_kind)
            if priority_matches:
                _index, kind, phrase = priority_matches[0]
                print(f"[V7.13 MULTI] Selected {kind} command from run-on transcript: {phrase}")
                return phrase

        _index, kind, phrase = sorted(unique_positions)[0]
        print(f"[V7.13 MULTI] Selected first {kind} command from run-on transcript: {phrase}")
        return phrase

    candidates = [(part, _direct_command_kind(part)) for part in parts]
    candidates = [(part, kind) for part, kind in candidates if kind]
    if len(candidates) <= 1:
        return user_text

    for priority_kind in priority_order:
        for part, kind in candidates:
            if kind == priority_kind:
                print(f"[V7.13 MULTI] Selected {priority_kind} command from multi-command transcript: {part}")
                return part

    first_part, first_kind = candidates[0]
    print(f"[V7.13 MULTI] Selected first {first_kind} command from multi-command transcript: {first_part}")
    return first_part


def _expects_prompt_answer(state: RobotRuntimeState) -> bool:
    with state.lock:
        prompt_type = state.last_prompt_type
        enrollment_state = state.enrollment_state

    return bool(prompt_type) or enrollment_state in {"awaiting_name", "requested"}


def _should_drop_filler_transcript(text: str, state: RobotRuntimeState, camera_intent: str = "none") -> bool:
    if _expects_prompt_answer(state):
        return False

    if (
        _has_v7_5_wake_phrase(text)
        or _is_global_audio_command(text)
        or _is_protected_audio_text(text)
        or camera_intent != "none"
        or _is_enrollment_request_text(text)
        or full.is_local_robot_control_request(text)
    ):
        return False
    with state.lock:
        active = bool(state.conversation_active and time.time() <= float(state.conversation_until or 0.0))
        mode = state.conversation_mode
    if active and mode in {"creative", "story", "project"} and _is_v715_short_followup_text(text):
        return False

    raw = str(text or "").lower().strip(" .,:;!?")
    normalized = _normalize_for_echo(text)
    filler_phrases = {
        "uh",
        "um",
        "wow",
        "oh my god",
        "omg",
        "ugh",
        "ai meu deus",
        "wow didn t you",
        "si",
        "sí",
        "não",
        "nao",
        "tare",
        "ok",
        "okay",
        "hi buddy",
    }

    return raw in filler_phrases or normalized in filler_phrases


def _normalize_person_name(name: str | None) -> str:
    return str(name or "").lower().strip().replace(" ", "_")


def _is_owner(person: str | None) -> bool:
    return _normalize_person_name(person) in {"marco", "marquinho"}


def is_owner_present(state: RobotRuntimeState, camera_manager: full.CameraManager, max_age_seconds: float = 3.0) -> bool:
    with state.lock:
        runtime_person = _normalize_person_name(state.recognized_person)
        runtime_age = time.time() - float(state.recognized_person_updated_at or 0.0)
    if _is_owner(runtime_person) and runtime_age < max_age_seconds:
        return True

    tracked_state = state.identity_tracker.get_owner_authorization_identity(max_age_seconds=max_age_seconds)
    if tracked_state and _is_owner(tracked_state.get("recognized_person")):
        age = _face_state_age(tracked_state)
        if age is None or age < max_age_seconds:
            return True

    try:
        face_state = camera_manager.get_face_state(max_age_seconds=max_age_seconds)
    except Exception:
        return False

    if not _is_owner(face_state.get("recognized_person")):
        return False

    age = _face_state_age(face_state)
    return age is None or age < max_age_seconds


def _is_voice_command_text(text: str) -> bool:
    normalized = normalize_command_text(text)
    voice_markers = {
        "voice",
        "deep voice",
        "robot voice",
        "robotic voice",
        "natural voice",
        "friendly voice",
        "story voice",
        "storyteller voice",
        "narrator voice",
        "change voice",
        "switch voice",
        "what voices",
        "voice options",
        "which voice",
    }
    return any(marker in normalized for marker in voice_markers)


def _is_owner_natural_direct_command(text: str) -> bool:
    normalized = normalize_command_text(text)
    if not normalized:
        return False

    direct_phrases = {
        "who am i",
        "who do you see",
        "what do you see",
        "how are you",
        "what are you",
        "can you hear me",
        "do you hear me",
        "you hear me",
        "are you listening",
        "are you there",
        "hello",
        "yo",
        "what time is it",
        "current time",
        "status",
        "recent topics",
        "previous topics",
        "conversation history",
        "interaction history",
        "analyze conversation",
        "analyze conversation log",
        "analyze previous log",
        "previous log",
        "log file",
        "owner mode",
        "unlock owner mode",
        "is password mode configured",
        "is owner password configured",
    }
    if any(phrase in normalized for phrase in direct_phrases):
        return True

    return (
        _is_voice_command_text(normalized)
        or _is_shutdown_request_text(normalized)
        or _is_shutdown_cancel_text(normalized)
        or _is_shutdown_confirm_text(normalized)
        or _is_enrollment_request_text(normalized)
        or full.is_local_robot_control_request(normalized)
    )


def _face_state_age(face_state: dict) -> float | None:
    age = face_state.get("age")
    if age is not None:
        return age

    updated_at = float(face_state.get("updated_at", 0.0) or 0.0)
    if updated_at:
        return time.time() - updated_at

    return None


def _identity_state_rank(face_state: dict) -> tuple:
    recognized = bool(face_state.get("recognized_person"))
    score = face_state.get("recognition_score")
    if score is None:
        score_value = -1.0
    else:
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = -1.0

    detected = bool(face_state.get("face_detected"))
    age = _face_state_age(face_state)
    freshness = -float(age) if age is not None else -9999.0
    return (recognized, score_value, detected, freshness)


def get_fresh_identity_state(camera_manager: full.CameraManager, timeout_seconds: float = 2.0) -> dict:
    print("[V7.5 IDENTITY] waiting for fresh face state...")
    deadline = time.time() + float(timeout_seconds)
    best_state: dict | None = None

    while True:
        face_state = camera_manager.get_face_state(max_age_seconds=1.0)
        if best_state is None or _identity_state_rank(face_state) > _identity_state_rank(best_state):
            best_state = face_state

        if face_state.get("recognized_person") and face_state.get("recognition_score") is not None:
            best_state = face_state
            break

        if time.time() >= deadline:
            break

        time.sleep(0.2)

    selected = best_state or {}
    recognized = selected.get("recognized_person")
    score = selected.get("recognition_score")
    age = _face_state_age(selected)
    print(f"[V7.5 IDENTITY] selected recognized={recognized} score={score} age={age}")
    return selected


def _is_yes_no(text: str) -> bool:
    t = str(text or "").lower().strip()
    return t in {"yes", "yeah", "yep", "sure", "ok", "okay", "no", "nope", "not now"}


def _resolve_short_context(text: str, state: RobotRuntimeState) -> str:
    t = str(text or "").lower().strip()
    if not _is_yes_no(t):
        return text

    with state.lock:
        prompt_type = state.last_prompt_type

    if prompt_type == "shutdown_confirmation":
        if t in {"yes", "yeah", "yep", "sure", "ok", "okay"}:
            return "confirm shutdown"
        return "cancel shutdown"

    if prompt_type == "enrollment_request":
        if t in {"yes", "yeah", "yep", "sure", "ok", "okay"}:
            return "enroll new friend"
        return "cancel enrollment"

    return text


def _update_prompt_state(reply_text: str, state: RobotRuntimeState) -> None:
    text = str(reply_text or "").strip()
    lower = text.lower()
    prompt_type = None

    if "shutdown confirmation required" in lower:
        prompt_type = "shutdown_confirmation"
    elif "why did" in lower and lower.endswith("?"):
        prompt_type = "joke_setup"
    elif "enrollment needs approval" in lower:
        prompt_type = "enrollment_request"
    elif "what is your friend's name" in lower:
        prompt_type = "enrollment_name"
    elif lower.endswith("?") or " say " in lower:
        prompt_type = "general_prompt"

    with state.lock:
        state.last_reply_time = time.time()
        state.last_prompt_text = text
        state.last_prompt_type = prompt_type
        if prompt_type:
            state.last_robot_question_at = time.time()
            state.last_robot_question_type = prompt_type
            state.last_robot_question_text = text
            if (
                ("which movie" in lower and "theater" in lower)
                or ("which theater" in lower)
                or ("which city" in lower and "movie" in lower)
            ):
                state.last_robot_question_expected_slot = "movie_theater_location"
            else:
                state.last_robot_question_expected_slot = None
        if "star wars" in lower:
            state.last_topic = "Star Wars"
            state.last_topic_until = time.time() + 300.0

        temp_f = _extract_fahrenheit_temperature(text)
        if temp_f is not None:
            state.last_weather_temp_f = temp_f


def _extract_fahrenheit_temperature(text: str) -> float | None:
    patterns = [
        r"(-?\d+(?:\.\d+)?)\s*°?\s*f\b",
        r"(-?\d+(?:\.\d+)?)\s*degrees?\s+fahrenheit\b",
        r"(-?\d+(?:\.\d+)?)\s*fahrenheit\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, str(text or ""), re.IGNORECASE)
        if match:
            return float(match.group(1))

    return None


def _route_celsius_conversion(user_text: str, state: RobotRuntimeState) -> bool:
    t = str(user_text or "").lower().strip()
    if "celsius" not in t:
        return False

    temp_f = _extract_fahrenheit_temperature(user_text)
    if temp_f is None and ("weather" in t or "temperature" in t or "fahrenheit" in t):
        with state.lock:
            temp_f = state.last_weather_temp_f

    if temp_f is None:
        return False

    temp_c = (temp_f - 32.0) * 5.0 / 9.0
    if abs(temp_f - round(temp_f)) < 0.05:
        f_text = f"{int(round(temp_f))}°F"
    else:
        f_text = f"{temp_f:.1f}°F"
    v6.speak(f"{f_text} is about {round(temp_c)}°C.")
    return True


def _has_pending_prompt(state: RobotRuntimeState) -> bool:
    with state.lock:
        return bool(state.last_prompt_type)


def _shutdown_confirmation_timeout_seconds() -> float:
    return _env_float("MIGUEL_SHUTDOWN_CONFIRMATION_TIMEOUT_SECONDS", 30.0)


def _clear_shutdown_confirmation(state: RobotRuntimeState) -> None:
    with state.lock:
        state.shutdown_pending = False
        state.shutdown_confirmation_pending = False
        state.shutdown_confirmation_until = 0.0
        if state.last_prompt_type == "shutdown_confirmation":
            state.last_prompt_type = None
            state.last_prompt_text = None


def _expire_shutdown_confirmation_if_needed(state: RobotRuntimeState) -> bool:
    now = time.time()
    expired = False
    with state.lock:
        if state.shutdown_confirmation_pending and now > float(state.shutdown_confirmation_until or 0.0):
            state.shutdown_pending = False
            state.shutdown_confirmation_pending = False
            state.shutdown_confirmation_until = 0.0
            if state.last_prompt_type == "shutdown_confirmation":
                state.last_prompt_type = None
                state.last_prompt_text = None
            expired = True
    if expired:
        print("[V7.14 SHUTDOWN] confirmation expired")
        v6.speak("Shutdown canceled.")
    return expired


def _set_shutdown_pending(state: RobotRuntimeState, pending: bool) -> None:
    with state.lock:
        state.shutdown_pending = bool(pending)
        if pending:
            state.current_mode = "normal"
            state.shutdown_confirmation_pending = True
            state.shutdown_confirmation_until = time.time() + _shutdown_confirmation_timeout_seconds()
            state.last_prompt_type = "shutdown_confirmation"
            state.last_prompt_text = "Shutdown confirmation required."
        else:
            state.shutdown_confirmation_pending = False
            state.shutdown_confirmation_until = 0.0
            if state.current_mode == "shutdown_pending":
                state.current_mode = "normal"
            if state.last_prompt_type == "shutdown_confirmation":
                state.last_prompt_type = None
                state.last_prompt_text = None
    try:
        robot_memory.set_pending_shutdown(bool(pending))
    except Exception as exc:
        print("[V7.5 SHUTDOWN] memory sync warning:", exc)


def _route_fast_local_reply(user_text: str, state: RobotRuntimeState) -> bool:
    if _is_bare_wake_phrase(user_text):
        v6.speak("Here.")
        return True

    normalized = normalize_command_text(user_text)

    if normalized in {"creative mode", "be creative"} or (
        _is_mode_command_not_physical(normalized) and ("creative mode" in normalized or "go creative" in normalized)
    ) or normalized == "go creative":
        _force_active_after_mode(state, "creative", reason="creative_mode")
        _set_response_length_context(state, "normal")
        v6.speak("Creative mode activated. I can make up heroes, machines, stories, and wild ideas.")
        return True

    if normalized in {
        "conversation mode",
        "conversational mode",
        "go to conversation mode",
        "go in conversation mode",
        "go to conversational mode",
        "go in conversational mode",
        "can you go to conversation mode",
        "can you go to conversational mode",
    }:
        _force_active_after_mode(state, "general", reason="conversation_mode")
        _set_response_length_context(state, "normal")
        v6.speak("Conversation mode on.")
        return True

    if _is_mode_command_not_physical(normalized) and any(marker in normalized for marker in {"robot project", "project mode"}):
        _force_active_after_mode(state, "project", reason="project_mode")
        _set_response_length_context(state, "normal")
        v6.speak("Robot project mode.")
        return True

    length_modes = {
        "short answer": ("terse", "Short mode."),
        "talk shorter": ("terse", "Short mode."),
        "talk normally": ("normal", "Normal mode."),
        "normal answer": ("normal", "Normal mode."),
        "talk longer": ("detailed", "Longer mode."),
        "give me the full answer": ("detailed", "Detailed mode."),
        "full answer": ("detailed", "Detailed mode."),
        "give more detail": ("detailed", "Detailed mode."),
        "story mode": ("story", "Story mode."),
    }
    if normalized in length_modes:
        mode, reply = length_modes[normalized]
        with state.lock:
            state.response_length_mode = mode
            if mode == "story":
                state.conversation_mode = "story"
        if mode == "story":
            _force_active_after_mode(state, "story", reason="story_mode")
        _set_response_length_context(state, mode)
        v6.speak(reply)
        return True

    cue_modes = {
        "turn on listening beep": ("beep", "Listening beep on."),
        "turn off listening beep": ("visual", "Listening beep off."),
        "use spoken cue": ("spoken", "Spoken cue on."),
        "turn off ready cue": ("off", "Ready cue off."),
        "use visual cue": ("visual", "Visual cue."),
    }
    if normalized in cue_modes:
        mode, reply = cue_modes[normalized]
        with state.lock:
            state.ready_cue_mode = mode
            state.ready_cue_enabled = mode != "off"
        v6.speak(reply)
        return True

    if normalized in {"", "miguel", "hey miguel", "ei miguel"}:
        v6.speak("I'm here.")
        return True

    if normalized == "can i interrupt you":
        v6.speak("Yes. Say Miguel stop.")
        return True

    if normalized == "how do i stop you talking":
        v6.speak("Say Miguel stop.")
        return True

    if normalized in {"hi", "hello", "hi buddy", "yo"}:
        v6.speak("Hi.")
        return True

    if normalized in {"can you hear me", "do you hear me", "you hear me", "can you hear us", "do you hear us", "can you hear everyone"}:
        v6.speak("I hear you.")
        return True

    if normalized in {"can you hear both of us"}:
        v6.speak("I hear you both.")
        return True

    if normalized in {"are you listening", "are you there"}:
        v6.speak("Yes. I'm here.")
        return True

    if normalized in {"how are you", "how are you doing", "are you okay", "how do you feel"}:
        with state.lock:
            person = _normalize_person_name(state.conversation_partner or state.recognized_person)
        name = "Marco" if person == "marco" else "Marquinho" if person == "marquinho" else ""
        prefix = f"I'm good, {name}. " if name else "I'm good. "
        v6.speak(prefix + "Voice and camera are online, and I'm ready.")
        return True

    if normalized in {"yes", "no", "okay", "ok", "good", "entendi", "certo"} and not _has_pending_prompt(state):
        if normalized == "no":
            v6.speak("Okay.")
        elif normalized in {"entendi", "certo"}:
            v6.speak("Entendi.")
        else:
            v6.speak("Got it.")
        return True

    return False


def _volume_adjustment_percent(user_text: str) -> int:
    """Return a small relative adjustment for an explicit volume command."""
    normalized = normalize_command_text(user_text)
    if not any(term in normalized for term in {"volume", "som", "taxa de som", "audio", "speaker"}):
        return 0
    if any(marker in normalized for marker in {"aumente", "aumentar", "mais alto", "raise", "increase", "turn up"}):
        return 5
    if any(marker in normalized for marker in {"diminua", "diminuir", "mais baixo", "lower", "decrease", "turn down"}):
        return -5
    return 0


def _route_system_volume_local_reply(user_text: str, state: RobotRuntimeState) -> bool:
    adjustment = _volume_adjustment_percent(user_text)
    if not adjustment:
        return False
    _set_reply_context(state, "utility")
    direction = "+" if adjustment > 0 else "-"
    try:
        result = subprocess.run(
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{direction}{abs(adjustment)}%"],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[V7.15 VOLUME] adjustment_failed error={exc}")
        v6.speak("Nao consegui ajustar o volume do sistema.")
        return True
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        print(f"[V7.15 VOLUME] adjustment_failed returncode={result.returncode} detail={detail}")
        v6.speak("Nao consegui ajustar o volume do sistema.")
        return True
    print(f"[V7.15 VOLUME] adjusted relative_percent={adjustment}")
    v6.speak("Aumentei o volume." if adjustment > 0 else "Diminuí o volume.")
    return True


def _route_project_local_reply(user_text: str, state: RobotRuntimeState) -> bool:
    normalized = normalize_command_text(user_text)

    role_reply = "Marco is the Systems Engineer. Marquinho is the Chief Engineer. Together, you are Mission Control for Miguel."
    role_followup_reply = "Yes - Marco is the Systems Engineer, and Marquinho is the Chief Engineer. Both of you are Mission Control."

    role_questions = {
        "who am i in this project",
        "who am i on this project",
        "what is my role",
        "what is my role in this project",
        "do you know my role",
        "what is marco role",
        "what is marcos role",
        "what is marco s role",
        "what is marquinho role",
        "what is marquinhos role",
        "what is marquinho s role",
        "who is system engineer",
        "who is systems engineer",
        "who is the system engineer",
        "who is the systems engineer",
        "who is chief engineer",
        "who is the chief engineer",
    }
    role_followups = {
        "did you finish your sentence",
        "what about marquinho",
        "talk about marquinho also",
        "and marco",
        "and marquinho",
        "and marco?",
        "and marquinho?",
    }
    with state.lock:
        recent_role_discussion = time.time() - float(state.project_role_discussed_at or 0.0) <= 90.0
    if normalized in role_questions or any(phrase in normalized for phrase in role_questions):
        with state.lock:
            state.project_role_discussed_at = time.time()
            state.last_topic = "project roles"
            state.last_topic_until = time.time() + 300.0
        v6.speak(role_reply)
        return True
    if recent_role_discussion and (normalized in role_followups or any(phrase in normalized for phrase in role_followups)):
        with state.lock:
            state.project_role_discussed_at = time.time()
        v6.speak(role_followup_reply)
        return True

    if any(
        phrase in normalized
        for phrase in {
            "why you don t answer",
            "why you dont answer",
            "why are you not answering",
            "why you are not answering",
            "why don t you answer me",
            "why dont you answer me",
            "do i need to repeat myself",
        }
    ):
        v6.speak("I answer when I'm in YOUR TURN mode. If I show SAY HEY MIGUEL, wake me first.")
        return True

    if normalized in {
        "what are you",
        "do you know what are you",
        "do you know what you are",
        "what do you know about yourself",
        "can you repeat what are you",
    }:
        v6.speak("I'm Miguel, Marquinho's voice-and-vision robot project. I can talk, listen, use the camera, remember topics, and help with robot experiments.")
        return True

    if normalized in {"are you a robot", "are you human", "are you a human", "are you a human or a robot"}:
        v6.speak("I'm a robot project running on the Jetson, with voice, camera, memory, and cloud brain features.")
        return True

    if normalized == "which type of project is this":
        v6.speak("Father-son robot.")
        return True

    if "who are the system engineers that are building you" in normalized:
        v6.speak("Marquinho and Marco.")
        return True

    if "who are the engineers" in normalized:
        v6.speak("Marquinho and Marco.")
        return True

    if "who built you" in normalized:
        v6.speak("Marco and Marquinho.")
        return True

    if normalized == "do i need to repeat myself sometimes":
        v6.speak("Sometimes. Wait for YOUR TURN.")
        return True

    return False


def _is_project_role_request(user_text: str) -> bool:
    normalized = normalize_command_text(user_text)
    role_markers = {
        "who am i in this project",
        "who am i on this project",
        "what is my role",
        "what is my role in this project",
        "what is my role on this project",
        "do you know my role",
    }
    return any(marker in normalized for marker in role_markers)


def _is_location_question(user_text: str) -> bool:
    normalized = normalize_command_text(user_text)
    phrases = {
        "where are we",
        "where we are",
        "which location are we here",
        "what location are we",
        "what location are we in",
        "do you know which location",
        "do you know where we are",
    }
    return any(phrase in normalized for phrase in phrases)


def _route_location_local_reply(user_text: str, state: RobotRuntimeState) -> bool:
    if not _is_location_question(user_text):
        return False
    location = os.getenv("MIGUEL_LOCATION_NAME", "").strip()
    if location:
        v6.speak(f"We are at {location}.")
    else:
        v6.speak("I do not have a GPS or saved room location yet. I only know I am running on the Jetson.")
    return True


LOCAL_JOKES = [
    "Why did the robot bring a ladder? To reach the cloud.",
    "Why did Miguel cross the room? To charge his ideas.",
    "Why did the robot take a nap? It needed to reboot.",
    "Why did the computer get cold? It left its Windows open.",
]


def _is_joke_request(user_text: str) -> bool:
    normalized = normalize_command_text(user_text)
    return any(
        phrase in normalized
        for phrase in {
            "tell me a joke",
            "tell a joke",
            "say a joke",
            "funny joke",
            "make me laugh",
            "another joke",
            "do you know a joke",
            "science joke",
        }
    )


def _select_local_joke(user_text: str) -> str:
    normalized = normalize_command_text(user_text)
    return LOCAL_JOKES[abs(hash(normalized)) % len(LOCAL_JOKES)]


def _route_fun_local_reply(user_text: str, state: RobotRuntimeState) -> bool:
    normalized = normalize_command_text(user_text)

    with state.lock:
        prompt_type = state.last_prompt_type
        punchline = state.last_joke_punchline

    if prompt_type == "joke_setup" and normalized in {"i don t know", "i dont know", "why", "tell me"}:
        v6.speak(punchline or "To reach the cloud.")
        with state.lock:
            state.last_prompt_type = None
            state.last_prompt_text = None
            state.last_joke_punchline = None
        return True

    if _is_joke_request(user_text):
        v6.speak(_select_local_joke(user_text))
        with state.lock:
            state.last_joke_punchline = None
        return True

    fun_replies = {
        "sing something": "Beep beep, Miguel is online.",
        "dance": "Tiny robot dance.",
        "are you funny": "I'm trying.",
        "good job": "Thanks.",
        "he knows": "Yes.",
    }
    reply = fun_replies.get(normalized)
    if reply:
        v6.speak(reply)
        return True

    return False


def _is_shutdown_confirm_text(text: str) -> bool:
    normalized = normalize_command_text(text)
    return normalized in {
        "confirm shutdown",
        "confirmed shutdown",
        "confirme shutdown",
        "firm shutdown",
        "yes shutdown",
        "yes shut down",
        "yes shutdown",
        "yes, shut down",
    }


def is_explicit_robot_shutdown(text: str) -> bool:
    normalized = normalize_command_text(text)
    question_markers = {"what about shutdown", "tell me about shutdown", "explain shutdown"}
    if normalized in question_markers:
        return False
    robot_targets = {"miguel", "robot", "program"}
    if normalized in {
        "shutdown",
        "shutdown now",
        "shut down",
        "shut down now",
        "prepare shutdown",
        "start shutdown",
        "power down",
    }:
        return True
    if "shutdown" in normalized or "shut down" in normalized:
        words = set(normalized.split())
        return bool(words & robot_targets)
    # Natural owner phrasing often targets Miguel with "you" instead of
    # repeating his name. Keep this narrow so requests about lights, cameras,
    # or other hardware are not mistaken for a process shutdown.
    if re.fullmatch(
        r"(?:(?:so|okay|ok) )?(?:miguel )?(?:please )?"
        r"(?:you (?:can|may|should|need to) (?:just )?)?"
        r"turn (?:yourself )?off(?: (?:now|please))?",
        normalized,
    ):
        return True
    return normalized in {
        "turn off miguel",
        "turn off the robot",
        "turn off robot",
        "stop the robot program",
        "stop robot program",
        "confirm shutdown",
    }


def _is_shutdown_request_text(text: str) -> bool:
    return is_explicit_robot_shutdown(text)


def _is_shutdown_cancel_text(text: str) -> bool:
    normalized = normalize_command_text(text)
    if normalized in {"cancel", "no", "nope"}:
        return True
    return any(
        phrase in normalized
        for phrase in {
            "cancel shutdown",
            "stop shutdown",
            "never mind",
            "changed my mind",
            "no shutdown",
            "do not shutdown",
            "dont shutdown",
            "do not shut down",
            "dont shut down",
        }
    )


def _has_clear_non_shutdown_command(text: str) -> bool:
    normalized = normalize_command_text(text)
    command_markers = [
        "voice",
        "robotic voice",
        "robot voice",
        "natural voice",
        "friendly voice",
        "deep voice",
        "story voice",
        "camera",
        "what do you see",
        "who do you see",
        "what time",
        "time is it",
        "status",
        "recent topics",
        "previous topics",
        "analyze conversation",
    ]
    return any(marker in normalized for marker in command_markers)


def _is_sleep_mode_request(text: str) -> bool:
    raw_normalized = _normalize_for_echo(text)
    if raw_normalized in {"miguel sleep", "sleep miguel"}:
        return True
    normalized = normalize_command_text(text)
    return any(
        phrase in normalized
        for phrase in {
            "sleep mode",
            "sleepy mode",
            "sleeping mode",
            "go to sleep",
            "miguel sleep",
            "sleep miguel",
            "silent mode",
            "silence mode",
            "go silent",
            "modo silencio",
            "modo silencioso",
            "modo pausa",
            "entre em pausa",
        }
    )


def _is_sleep_wake_request(text: str) -> bool:
    normalized = normalize_command_text(text)
    wake_phrases = {
        "miguel wake up",
        "wake up miguel",
        "hello miguel",
        "hey miguel",
        "mission control",
        "miguel voltar",
        "miguel continuar",
        "miguel acorde",
        "voltar",
        "continuar",
        "acorde",
    }
    return normalized in wake_phrases or any(normalized.startswith(phrase + " ") for phrase in wake_phrases)


def _activate_sleep_mode(state: RobotRuntimeState) -> None:
    with state.lock:
        state.sleep_mode_active = True
        state.sleep_mode_until = 0.0
        state.current_mode = "sleep"
        state.conversation_active = False
        state.conversation_mode = "wake_required"
        state.conversation_partner = None
        state.conversation_until = 0.0
        state.wake_required = True
        state.wake_required_reason = "sleep"
        state.audio_capture_active = False
        state.audio_capture_blocked_reason = "sleep"
    set_interaction_state(state, "sleeping", "Sleep")


def _deactivate_sleep_mode(state: RobotRuntimeState) -> None:
    with state.lock:
        state.sleep_mode_active = False
        state.sleep_mode_until = 0.0
        state.current_mode = "normal"
        state.wake_required = False
        state.wake_required_reason = ""


def _route_sleep_control(user_text: str, state: RobotRuntimeState, partner: str | None = None) -> bool | None:
    if _is_sleep_mode_request(user_text):
        _activate_sleep_mode(state)
        _set_reply_context(state, "sleep")
        v6.speak("Sleep mode on. Say Miguel wake up or Mission Control.")
        return True

    with state.lock:
        sleeping = bool(state.sleep_mode_active)
    if not sleeping:
        return None

    if _is_sleep_wake_request(user_text):
        _deactivate_sleep_mode(state)
        _force_active_after_mode(state, "general", partner=partner or _current_owner_partner(state), reason="sleep_wake")
        _set_reply_context(state, "wake")
        v6.speak("I am awake.")
        return True

    with state.lock:
        shutdown_waiting = bool(state.shutdown_pending or state.shutdown_confirmation_pending)
    if is_explicit_robot_shutdown(user_text) or (
        shutdown_waiting and (_is_shutdown_confirm_text(user_text) or _is_shutdown_cancel_text(user_text))
    ):
        return None

    print(f"[V7.14 SLEEP] ignored text={_short_log_text(user_text)}")
    _set_reply_context(state, "sleep_ignore")
    set_interaction_state(state, "sleeping", "Sleep")
    return True


def _sleep_mode_active(state: RobotRuntimeState) -> bool:
    with state.lock:
        return bool(state.sleep_mode_active)


def _handle_sleep_mode_audio_text(
    user_turn_queue: queue.Queue,
    state: RobotRuntimeState,
    user_text: str,
    recognized_person: str | None = None,
) -> bool:
    if not _sleep_mode_active(state):
        return False
    if _is_sleep_wake_request(user_text):
        print(f"[V7.14 SLEEP] wake phrase accepted text={_short_log_text(user_text)}")
        _enqueue_user_turn(
            user_turn_queue,
            state,
            user_text,
            recognized_person,
            authorized=True,
            authorization_source="wake_phrase",
            stripped_text=_strip_wake_phrase(user_text),
        )
        return True
    print(f"[V7.14 SLEEP] ignored non-wake text={_short_log_text(user_text)}")
    set_interaction_state(state, "sleeping", "Sleep")
    return True


def _route_shutdown_control(user_text: str, state: RobotRuntimeState) -> bool | None:
    _expire_shutdown_confirmation_if_needed(state)
    with state.lock:
        pending = bool(state.shutdown_pending or state.shutdown_confirmation_pending)
        creative_mode = state.conversation_mode == "creative"

    if pending and _is_shutdown_confirm_text(user_text):
        _set_shutdown_pending(state, False)
        _set_reply_context(state, "shutdown_confirm")
        v6.speak("Confirmed.")
        with state.lock:
            state.debug_handoff_requested = True
        state.stop_event.set()
        return False

    if pending and _is_shutdown_request_text(user_text):
        print("[V7.14 SHUTDOWN] repeated_shutdown_requires_confirm=true")
        set_interaction_state(state, "shutdown_pending", "Confirm shutdown")
        _set_reply_context(state, "shutdown")
        v6.speak("Please say confirm shutdown or cancel.")
        return True

    if pending and _is_shutdown_cancel_text(user_text):
        _set_shutdown_pending(state, False)
        set_interaction_state(state, "idle", "")
        _set_reply_context(state, "shutdown_cancel")
        v6.speak("Shutdown canceled.")
        return True

    if pending:
        set_interaction_state(state, "shutdown_pending", "Confirm shutdown")
        return True

    normalized = normalize_command_text(user_text)
    if creative_mode and "turn off" in normalized and not is_explicit_robot_shutdown(user_text):
        print(f"[V7.14 SHUTDOWN] blocked creative false positive text={_short_log_text(user_text)}")
        return None

    if _is_shutdown_request_text(user_text):
        _set_shutdown_pending(state, True)
        set_interaction_state(state, "shutdown_pending", "Confirm shutdown")
        _set_reply_context(state, "shutdown")
        v6.speak("Shutdown confirmation required.")
        return True

    return None


def _route_barge_in_control(user_text: str, state: RobotRuntimeState) -> bool | None:
    if not is_barge_in_command(user_text):
        return None

    shutdown_result = _route_shutdown_control(user_text, state)
    if shutdown_result is not None:
        return shutdown_result

    if not _is_speech_stop_barge_in(user_text):
        return None

    _request_speech_stop(state)
    normalized = normalize_command_text(user_text)
    if normalized != "pause":
        _set_response_depth_mode(state, "normal", "barge_in_stop")
    with state.lock:
        speaking = state.is_speaking
    set_interaction_state(state, "idle", "")
    if not speaking:
        state.stop_speech_event.clear()
        v6.speak("Stopped.")
    else:
        print("[V7.5 BARGE-IN] Stop requested; current speak backend may finish current segment.")
    return True


def _is_identity_camera_turn(user_text: str, camera_intent: str) -> bool:
    t = str(user_text or "").lower()
    if camera_intent == "identity_camera" or is_identity_camera_request(user_text):
        return True

    if _expanded_identity_trigger(user_text):
        return True

    return camera_intent == "camera_generic" and any(
        marker in t
        for marker in ["who", "identify", "recognize", "recognise"]
    )


def _multi_face_identity_trigger(user_text: str) -> str:
    normalized = normalize_command_text(user_text)
    triggers = {
        "did you see another face",
        "do you see another face",
        "can you see another face",
        "do you see both faces",
        "can you see both faces",
        "can you recognize the faces",
        "who are those faces",
        "who are those two",
        "who are those two people",
        "who are these two",
        "who are these two people",
        "who are those people",
        "who are these people",
        "who are the two people",
        "who are both people",
        "who are both of us",
        "who are they",
        "who are the faces",
        "who are the people",
        "identify the people",
        "identify those people",
        "identify these people",
        "who else is there",
        "anybody else",
        "see any other face",
        "any other face besides me",
        "other face besides me",
        "anyone behind me",
        "anybody behind me",
        "who is there",
        "who is with me",
        "do you recognize both of us",
        "can you recognize both of us",
        "is marquinho there",
        "is marco there",
        "can you see marco",
        "can you see marquinho",
        "who is this face",
        "who is that face",
        "who is his face",
        "who is her face",
        "who is this child",
        "who is that child",
        "identify this face",
        "whose face is this",
    }
    return "multi_face" if any(trigger in normalized for trigger in triggers) else ""


def _expanded_identity_trigger(user_text: str) -> str:
    normalized = normalize_command_text(user_text)
    multi_face = _multi_face_identity_trigger(user_text)
    if multi_face:
        return multi_face
    triggers = {
        "you see me": "short_see_me",
        "see me": "short_see_me",
        "see me now": "short_see_me",
        "you see me now": "short_see_me",
        "do you see me now": "short_see_me",
        "can you see me now": "short_see_me",
        "am i visible": "short_see_me",
        "can you recognize me now": "short_see_me",
        "who is this person": "who_is_this_person",
        "who is this face": "who_is_this_face",
        "who is that face": "who_is_this_face",
        "who is his face": "who_is_this_face",
        "who is her face": "who_is_this_face",
        "who is this child": "who_is_this_face",
        "who is that child": "who_is_this_face",
        "identify this face": "who_is_this_face",
        "whose face is this": "who_is_this_face",
        "who do you see": "who_do_you_see",
        "do you see me": "do_you_see_me",
        "do you see a face": "do_you_see_a_face",
        "can you see a face": "can_you_see_a_face",
        "you cannot see me": "you_cannot_see_me",
        "you can t see me": "you_cannot_see_me",
        "you cant see me": "you_cannot_see_me",
        "do you recognize me": "do_you_recognize_me",
        "can you recognize me": "can_you_recognize_me",
        "who is in front of you": "who_is_in_front_of_you",
    }
    for phrase, reason in triggers.items():
        if phrase in normalized:
            return reason
    return ""


def _known_identity_names() -> set[str]:
    names = {"marco", "marquinho"}
    face_db = getattr(v6, "INSIGHT_FACE_DB", None)
    if isinstance(face_db, dict):
        names.update(_normalize_person_name(name) for name in face_db.keys())
    return {name for name in names if name}


IDENTITY_ACCEPTANCE_THRESHOLDS = {
    "marco": {
        "votes": 2,
        "score": 0.60,
        "margin": 0.08,
        "strong_score": 0.75,
        "strong_margin": 0.12,
        "owner_score": 0.65,
        "owner_margin": 0.10,
    },
    "marquinho": {
        "votes": 2,
        "score": 0.58,
        "margin": 0.24,
        "strong_score": 0.68,
        "strong_margin": 0.34,
        "owner_score": 0.60,
        "owner_margin": 0.24,
    },
}


def _identity_thresholds_for(person: str | None) -> dict:
    normalized = _normalize_person_name(person)
    return IDENTITY_ACCEPTANCE_THRESHOLDS.get(
        normalized,
        {
            "votes": 2,
            "score": 0.62,
            "margin": 0.10,
            "strong_score": 0.76,
            "strong_margin": 0.14,
            "owner_score": 0.68,
            "owner_margin": 0.12,
        },
    )


def _identity_candidate_accepted(
    person: str | None,
    votes: int,
    avg_score: float | None,
    avg_margin: float | None,
    *,
    owner_context: bool = False,
    single_frame: bool = False,
) -> bool:
    normalized = _normalize_person_name(person)
    if not normalized:
        return False
    thresholds = _identity_thresholds_for(normalized)
    try:
        score = float(avg_score or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    try:
        margin = float(avg_margin or 0.0)
    except (TypeError, ValueError):
        margin = 0.0
    try:
        vote_count = int(votes or 0)
    except (TypeError, ValueError):
        vote_count = 0

    if owner_context:
        return vote_count >= 2 and score >= thresholds["owner_score"] and margin >= thresholds["owner_margin"]
    if single_frame:
        return score >= thresholds["strong_score"] and margin >= thresholds["strong_margin"]
    return (
        vote_count >= thresholds["votes"]
        and score >= thresholds["score"]
        and margin >= thresholds["margin"]
    ) or (score >= thresholds["strong_score"] and margin >= thresholds["strong_margin"])


def _stable_identity_reply_state(
    face_detected: bool,
    recognized_person: str | None = None,
    recognition_score: float | None = None,
    recognition_margin: float | None = None,
    recognition_votes: int = 0,
) -> dict:
    return {
        "face_detected": bool(face_detected),
        "face_count": 1 if face_detected else 0,
        "recognized_person": recognized_person,
        "recognition_score": recognition_score,
        "recognition_margin": recognition_margin,
        "recognition_votes": recognition_votes,
        "recognition_scores": {},
        "face_position": "unknown" if face_detected else "none",
        "source": "stable_identity",
    }


def get_stable_identity_for_reply(camera_manager, timeout_seconds: float = 2.2) -> dict:
    tracker = getattr(camera_manager, "identity_tracker", None)
    deadline = time.time() + float(timeout_seconds)
    known_names = _known_identity_names()
    best_candidate = (None, 0, 0.0, 0.0, None)
    face_detected = False

    while True:
        raw_state = camera_manager.get_face_state(max_age_seconds=1.0)
        face_detected = face_detected or bool(raw_state.get("face_detected"))
        raw_age = _face_state_age(raw_state)
        raw_person = _normalize_person_name(raw_state.get("recognized_person"))
        raw_score = raw_state.get("recognition_score")
        raw_margin = raw_state.get("recognition_margin")
        if (
            raw_person
            and raw_age is not None
            and raw_age < 1.0
            and _identity_candidate_accepted(raw_person, 1, raw_score, raw_margin, single_frame=True)
        ):
            return _stable_identity_reply_state(
                face_detected=True,
                recognized_person=raw_person,
                recognition_score=raw_score,
                recognition_margin=raw_margin,
                recognition_votes=1,
            )

        if tracker is not None:
            face_detected = face_detected or tracker.has_recent_face_detected(max_age_seconds=3.0)
            candidate = tracker.get_reply_candidate(max_age_seconds=3.0)
            candidate_rank = (candidate[1], candidate[2], candidate[3])
            best_rank = (best_candidate[1], best_candidate[2], best_candidate[3])
            if candidate_rank > best_rank:
                best_candidate = candidate

        name, votes, avg_score, avg_margin, _obs = best_candidate
        person = _normalize_person_name(name)
        candidate_known = bool(person) and (not known_names or person in known_names)
        is_owner = _is_owner(person)
        accepted = False

        if candidate_known:
            accepted = _identity_candidate_accepted(
                person,
                votes,
                avg_score,
                avg_margin,
                owner_context=is_owner,
            ) and face_detected

        if accepted:
            return _stable_identity_reply_state(
                face_detected=True,
                recognized_person=person,
                recognition_score=avg_score,
                recognition_margin=avg_margin,
                recognition_votes=votes,
            )

        if time.time() >= deadline:
            break

        time.sleep(0.2)

    if face_detected:
        return _stable_identity_reply_state(face_detected=True)

    return _stable_identity_reply_state(face_detected=False)


def _fresh_identity_state_for_route(camera_manager, timeout_seconds: float = 2.2) -> dict:
    cached_state = camera_manager.get_face_state(max_age_seconds=1.0)
    cached_age = _face_state_age(cached_state)
    needs_fresh_wait = (
        not cached_state
        or cached_age is None
        or cached_age > 0.8
        or not cached_state.get("face_detected")
        or not cached_state.get("recognized_person")
    )
    if needs_fresh_wait:
        print("[V7.14 IDENTITY] waiting_for_fresh_face")
    face_state = get_stable_identity_for_reply(camera_manager, timeout_seconds=timeout_seconds if needs_fresh_wait else 0.8)
    print(
        "[V7.14 IDENTITY] fresh_face_result="
        f"recognized={face_state.get('recognized_person')} detected={face_state.get('face_detected')}"
    )
    return face_state


def _promote_recent_runtime_identity_for_reply(face_state: dict, state: RobotRuntimeState, max_age_seconds: float = 5.0) -> dict:
    if face_state.get("recognized_person"):
        return face_state
    if not face_state.get("face_detected"):
        return face_state
    with state.lock:
        runtime_person = _normalize_person_name(state.recognized_person)
        updated_at = float(state.recognized_person_updated_at or 0.0)
    if not runtime_person or not updated_at:
        return face_state
    runtime_age = time.time() - updated_at
    if runtime_age > max_age_seconds:
        return face_state
    promoted = dict(face_state)
    promoted["recognized_person"] = runtime_person
    promoted["source"] = "runtime_recent_identity_fallback"
    promoted["runtime_recognized_age"] = runtime_age
    return promoted


def _face_count_from_state(face_state: dict) -> int:
    try:
        return int(face_state.get("face_count") or (1 if face_state.get("face_detected") else 0))
    except (TypeError, ValueError):
        return 1 if face_state.get("face_detected") else 0


def _recognized_names_from_face_state(face_state: dict) -> list[str]:
    names: list[str] = []
    explicit_names = face_state.get("recognized_names")
    if isinstance(explicit_names, (list, tuple, set)):
        for name in explicit_names:
            normalized_name = _normalize_person_name(name)
            if normalized_name and normalized_name not in names:
                names.append(normalized_name)
    person = _normalize_person_name(face_state.get("recognized_person"))
    if person and person not in names:
        names.append(person)
    if _face_count_from_state(face_state) < 2:
        return names[:1]
    votes = face_state.get("recognition_votes")
    if isinstance(votes, dict):
        for name, vote_count in sorted(votes.items(), key=lambda item: item[1], reverse=True):
            normalized_name = _normalize_person_name(name)
            if not normalized_name or normalized_name in names:
                continue
            try:
                if int(vote_count or 0) <= 0:
                    continue
            except (TypeError, ValueError):
                continue
            names.append(normalized_name)
    scores = face_state.get("recognition_scores")
    if isinstance(scores, dict):
        for name, score in scores.items():
            normalized_name = _normalize_person_name(name)
            if not normalized_name or normalized_name in names:
                continue
            try:
                score_value = float(score)
            except (TypeError, ValueError):
                continue
            if score_value >= 0.55:
                names.append(normalized_name)
    return names[:2]


def _recent_recognized_names_from_tracker(tracker: IdentityTracker | None, max_age_seconds: float = 5.0) -> list[str]:
    if tracker is None:
        return []
    observations = tracker._recent_observations(max_age_seconds)
    by_person: dict[str, list[dict]] = {}
    for obs in observations:
        person = _normalize_person_name(obs.get("recognized_person"))
        if person:
            by_person.setdefault(person, []).append(obs)

    ranked: list[tuple[int, float, float, str]] = []
    for person, person_observations in by_person.items():
        votes = len(person_observations)
        avg_score = sum(float(obs.get("recognition_score") or 0.0) for obs in person_observations) / votes
        avg_margin = sum(float(obs.get("recognition_margin") or 0.0) for obs in person_observations) / votes
        if _identity_candidate_accepted(person, votes, avg_score, avg_margin):
            ranked.append((votes, avg_score, avg_margin, person))

    ranked.sort(reverse=True)
    return [person for _votes, _score, _margin, person in ranked[:2]]


def _fresh_multi_face_state_for_route(camera_manager, state: RobotRuntimeState, timeout_seconds: float = 1.4) -> dict:
    print("[V7.14 IDENTITY] waiting_for_fresh_faces")
    deadline = time.time() + float(timeout_seconds)
    best_state: dict | None = None
    tracker = getattr(camera_manager, "identity_tracker", None) or getattr(state, "identity_tracker", None)

    def rank(face_state: dict) -> tuple:
        count = _face_count_from_state(face_state)
        recognized_count = len(_recognized_names_from_face_state(face_state))
        age = _face_state_age(face_state)
        freshness = -float(age) if age is not None else -9999.0
        return (count, recognized_count, bool(face_state.get("face_detected")), freshness)

    while True:
        raw_state = camera_manager.get_face_state(max_age_seconds=1.0)
        if best_state is None or rank(raw_state) > rank(best_state):
            best_state = raw_state

        if tracker is not None:
            tracked_state = tracker.get_stable_identity(max_age_seconds=3.0)
            if tracked_state and (best_state is None or rank(tracked_state) > rank(best_state)):
                best_state = tracked_state

        if best_state and _face_count_from_state(best_state) >= 2 and len(_recognized_names_from_face_state(best_state)) >= 2:
            break
        if time.time() >= deadline:
            break
        time.sleep(0.15)

    selected = best_state or _stable_identity_reply_state(face_detected=False)
    if _face_count_from_state(selected) >= 2:
        merged_names = _recognized_names_from_face_state(selected)
        for name in _recent_recognized_names_from_tracker(tracker, max_age_seconds=5.0):
            if name not in merged_names:
                merged_names.append(name)
        if merged_names:
            selected = dict(selected)
            selected["recognized_names"] = merged_names[:2]
    recognized = ",".join(_recognized_names_from_face_state(selected)) or "none"
    print(f"[V7.14 IDENTITY] fresh_faces_result=count={_face_count_from_state(selected)} recognized={recognized}")
    return selected


def _format_name_list(names: list[str]) -> str:
    order = {"marco": 0, "marquinho": 1}
    names = sorted(names, key=lambda name: (order.get(_normalize_person_name(name), 99), _normalize_person_name(name)))
    pretty = [name.replace("_", " ").title() for name in names if name]
    if not pretty:
        return ""
    if len(pretty) == 1:
        return pretty[0]
    return ", and ".join([", ".join(pretty[:-1]), pretty[-1]]) if len(pretty) > 2 else " and ".join(pretty)


def _multi_face_status_reply(face_state: dict) -> str:
    count = _face_count_from_state(face_state)
    names = _recognized_names_from_face_state(face_state)
    if count >= 2:
        if names:
            name_text = _format_name_list(names)
            if len(names) >= 2:
                return f"I see {count} faces. I recognize {name_text}."
            return f"I see {count} faces. I recognize {name_text}, but I'm not sure who the other face is."
        return f"I see {count} faces, but I'm not sure who they are."
    if count == 1:
        if names:
            return f"I see one face: {_format_name_list(names)}."
        return "I see one face, but I'm not sure who."
    return "I don't see a face right now."


def _face_status_reply(face_state: dict, prefix: str = "") -> str:
    person = _normalize_person_name(face_state.get("recognized_person"))
    if person:
        visible_name = person.replace("_", " ").title()
        return f"{prefix}I see {visible_name}.".strip()
    if face_state.get("face_detected"):
        return f"{prefix}I see a face, but recognition is still stabilizing. Hold still.".strip()
    if prefix:
        if prefix.strip().lower().startswith("i took a fresh look"):
            return "I took a fresh look, but I don't see a face right now."
        return f"{prefix.rstrip()} but I don't see a face right now."
    return "I don't see a face right now."


def _identity_debug_payload(face_state: dict, state: RobotRuntimeState, expanded_trigger: str = "") -> dict:
    with state.lock:
        conversation_partner = _normalize_person_name(state.conversation_partner)
        runtime_person = _normalize_person_name(state.recognized_person)
        runtime_age = time.time() - float(state.recognized_person_updated_at or 0.0) if state.recognized_person_updated_at else None
    debug = {
        "expanded_trigger": expanded_trigger or "",
        "face_detected": bool(face_state.get("face_detected")),
        "face_count": _face_count_from_state(face_state),
        "recognized_person": _normalize_person_name(face_state.get("recognized_person")),
        "recognized_names": _recognized_names_from_face_state(face_state),
        "recognition_score": face_state.get("recognition_score"),
        "recognition_margin": face_state.get("recognition_margin"),
        "recognition_votes": face_state.get("recognition_votes"),
        "recognition_scores": face_state.get("recognition_scores"),
        "face_position": face_state.get("face_position"),
        "source": face_state.get("source") or face_state.get("recognizer"),
        "age": _face_state_age(face_state),
        "conversation_partner": conversation_partner,
        "runtime_recognized_person": runtime_person,
        "runtime_recognized_age": runtime_age,
    }
    thresholds = _identity_thresholds_for(debug["recognized_person"] or conversation_partner or runtime_person)
    debug["acceptance_thresholds"] = {
        "votes": thresholds.get("votes"),
        "score": thresholds.get("score"),
        "margin": thresholds.get("margin"),
        "strong_score": thresholds.get("strong_score"),
        "strong_margin": thresholds.get("strong_margin"),
    }
    return debug


def _neutral_conversation_face_state() -> dict:
    return {
        "face_detected": False,
        "face_count": 0,
        "face_position": "not_used",
        "recognized_person": None,
        "recognition_score": None,
        "recognition_margin": None,
        "recognition_votes": {},
        "recognition_scores": {},
        "recognizer": "not_used_for_route",
        "source": "v7_14_non_vision_route",
        "vision_context_available": False,
    }


def _get_tracked_identity_state(
    state: RobotRuntimeState,
    camera_manager: full.CameraManager,
    timeout_seconds: float = 2.0,
) -> dict:
    deadline = time.time() + float(timeout_seconds)
    best_state = None

    print("[V7.5 IDENTITY] waiting for stable identity state...")
    while True:
        tracked_state = state.identity_tracker.get_stable_identity(max_age_seconds=3.0)
        if tracked_state:
            best_state = tracked_state
            if tracked_state.get("recognized_person"):
                break

        if time.time() >= deadline:
            break

        time.sleep(0.2)

    if best_state:
        return best_state

    return camera_manager.get_face_state(max_age_seconds=1.0)


def _route_identity_camera_intent(
    user_text: str,
    camera_intent: str,
    camera_manager: full.CameraManager,
    state: RobotRuntimeState,
) -> bool:
    if not _is_identity_camera_turn(user_text, camera_intent):
        return False

    expanded_trigger = _expanded_identity_trigger(user_text)
    if expanded_trigger:
        print(f"[V7.14 IDENTITY] expanded_trigger={expanded_trigger}")
    if expanded_trigger == "multi_face":
        face_state = _fresh_multi_face_state_for_route(camera_manager, state, timeout_seconds=1.4)
        reply = _multi_face_status_reply(face_state)
        if any(marker in normalize_command_text(user_text) for marker in {"doing", "what are", "what is happening"}):
            scene_reply = full.build_scene_reply(camera_manager)
            reply = f"{reply} {_trim_scene_reply(scene_reply, 18)}"
    else:
        face_state = _fresh_identity_state_for_route(camera_manager, timeout_seconds=2.2)
        face_state = _promote_recent_runtime_identity_for_reply(face_state, state)
        reply = _face_status_reply(face_state) if expanded_trigger else full.build_identity_reply(face_state)
        if not face_state.get("recognized_person"):
            with state.lock:
                partner = _normalize_person_name(state.conversation_partner)
            if partner and partner not in {"unknown", "unknown_wake_user"}:
                if face_state.get("face_detected"):
                    reply = f"I see a face, and our active conversation is with {_friendly_person_name(partner)}."
                else:
                    reply = f"I do not have a confirmed face right now, but our active conversation is with {_friendly_person_name(partner)}."
    reply = _localize_identity_reply(reply, user_text, state)
    _set_reply_context(state, "identity")
    identity_debug = _identity_debug_payload(face_state, state, expanded_trigger=expanded_trigger)
    with state.lock:
        state.current_turn_latency["identity_debug"] = identity_debug
    print(
        "[V7.14 IDENTITY DEBUG] "
        f"detected={identity_debug.get('face_detected')} "
        f"recognized={identity_debug.get('recognized_person')} "
        f"score={identity_debug.get('recognition_score')} "
        f"margin={identity_debug.get('recognition_margin')} "
        f"votes={identity_debug.get('recognition_votes')} "
        f"partner={identity_debug.get('conversation_partner')}"
    )
    v6.speak(reply)
    try:
        v6.update_conversation_memory(user_text=user_text, assistant_reply=reply)
    except Exception:
        pass
    return True


def _is_camera_refresh_request(user_text: str) -> bool:
    normalized = normalize_command_text(user_text)
    return any(
        phrase in normalized
        for phrase in {
            "refresh camera",
            "refresh the camera",
            "look again",
            "take a fresh look",
            "try camera again",
        }
    )


def _route_camera_refresh(user_text: str, camera_manager: full.CameraManager, state: RobotRuntimeState) -> bool:
    if not _is_camera_refresh_request(user_text):
        return False

    print("[V7.14 CAMERA] refresh_requested")
    stopped = bool(getattr(getattr(camera_manager, "stop_event", None), "is_set", lambda: False)())
    thread = getattr(camera_manager, "thread", None)
    if stopped or (thread is not None and not thread.is_alive()):
        print("[V7.14 CAMERA] refresh_result=unavailable")
        _set_reply_context(state, "camera_refresh")
        v6.speak("Camera refresh is unavailable right now.")
        return True

    face_state = _fresh_identity_state_for_route(camera_manager, timeout_seconds=2.2)
    result = "recognized" if face_state.get("recognized_person") else "face_unknown" if face_state.get("face_detected") else "no_face"
    print(f"[V7.14 CAMERA] refresh_result={result}")
    _set_reply_context(state, "camera_refresh")
    v6.speak(_face_status_reply(face_state, prefix="I took a fresh look. "))
    return True


def _route_scene_camera_intent(
    user_text: str,
    camera_intent: str,
    camera_manager: full.CameraManager,
    state: RobotRuntimeState,
) -> bool:
    if _is_identity_camera_turn(user_text, camera_intent):
        return False

    if not (is_scene_camera_request(user_text) or camera_intent == "camera_generic"):
        return False

    set_interaction_state(state, "looking", user_text[:48])
    _set_reply_context(state, "scene_prelude")
    v6.speak("Looking.")
    _set_reply_context(state, "scene")
    reply = full.build_scene_reply(camera_manager)
    v6.speak(reply)
    try:
        v6.update_conversation_memory(user_text=user_text, assistant_reply=reply)
    except Exception:
        pass
    return True


def _reset_enrollment_state(state: RobotRuntimeState) -> None:
    with state.lock:
        state.enrollment_state = "idle"
        state.enrollment_target_name = None
        state.enrollment_approved_by = None
        state.enrollment_approved_at = 0.0
        if str(state.last_prompt_type or "").startswith("enrollment_"):
            state.last_prompt_type = None
            state.last_prompt_text = None


def _enrollment_name_prompt() -> str:
    return (
        "Step one: tell me the new user's name. Say: their name is, followed by the name. "
        "If I hear it incorrectly, say: correct enrollment name to, followed by the correct name."
    )


def _enrollment_approval_prompt(target: str) -> str:
    name = target.title()
    return (
        f"Step two: enrollment for {name} needs approval from an owner. "
        f"Marco or Marquinho, stand in front of my camera so I can recognize you, then say: "
        f"Marco approves enrolling {name}. If I heard the name incorrectly, say: "
        "correct enrollment name to, followed by the correct name."
    )


def _enrollment_subject_prompt(target: str) -> str:
    name = target.title()
    return (
        f"Approval confirmed for {name}. Step three: the owner should move out of view. "
        f"Put only {name} in front of my camera, with their face well lit and uncovered. "
        f"When {name} is ready, say: {name} is here."
    )


def _route_enrollment(user_text: str, state: RobotRuntimeState, camera_manager: full.CameraManager) -> bool:
    t = str(user_text or "").lower().strip()

    if _is_enrollment_cancel_text(user_text):
        _reset_enrollment_state(state)
        v6.speak("Enrollment canceled.")
        return True

    corrected_name = _extract_enrollment_name_correction(user_text)
    if corrected_name:
        target = _normalize_enrollment_target(corrected_name)
        with state.lock:
            enrollment_active = state.enrollment_state != "idle" or bool(state.enrollment_target_name)
        if enrollment_active:
            if _is_protected_identity(target) or _normalize_person_name(target) == "miguel":
                v6.speak("I will not enroll that protected name as a new friend.")
                return True
            # A changed subject requires fresh owner authorization. Never carry
            # approval or capture state from the incorrectly heard identity.
            with state.lock:
                state.enrollment_target_name = target
                state.enrollment_state = "requested"
                state.enrollment_approved_by = None
                state.enrollment_approved_at = 0.0
                state.last_prompt_type = "enrollment_approval"
                state.last_prompt_text = _enrollment_approval_prompt(target)
            print(f"[V7.5 ENROLL] corrected target_name to {target}; approval reset")
            v6.speak(
                f"Corrected. The enrollment name is {target.title()}. "
                + _enrollment_approval_prompt(target)
            )
            return True

    face_state = camera_manager.get_face_state(max_age_seconds=2.0)
    recognized = face_state.get("recognized_person")
    approval_name = _extract_enrollment_approval_name(user_text)
    approval_requested = _is_enrollment_approval_text(user_text)

    if _is_reenrollment_request_text(user_text):
        target = _normalize_enrollment_target(_extract_reenrollment_name(user_text, state))
        if not target or target == "charlie":
            v6.speak("Who should I re-enroll? Say re-enroll Marco or re-enroll Marquinho.")
            return True
        recognized, face_state = _wait_for_owner_approval_face(camera_manager, state)
        if not _is_owner(recognized):
            v6.speak("Re-enrollment needs Marco or Marquinho visible for approval.")
            return True
        with state.lock:
            state.enrollment_state = "approved_pending_subject"
            state.enrollment_target_name = target
            state.enrollment_approved_by = _normalize_person_name(recognized)
            state.enrollment_approved_at = time.time()
        v6.speak(
            f"Approved. I will improve face recognition for {target.title()}. "
            "I will ask for a few positions and save a fresh embedding set."
        )
        return _run_enrollment_capture(camera_manager, state)

    if approval_name or approval_requested:
        with state.lock:
            active_target = state.enrollment_target_name
        # A pending target is authoritative. ASR often damages the repeated
        # name (for example, "Elvira" -> "your virus"), so never replace an
        # already-confirmed target with text from the approval utterance.
        target = _normalize_enrollment_target(active_target or approval_name)
        if not active_target and not approval_name:
            v6.speak(_enrollment_name_prompt())
            return True
        if _is_protected_identity(target) and target != _normalize_person_name(approval_name):
            _reset_enrollment_state(state)
            v6.speak("I will not overwrite Marco or Marquinho.")
            return True

        recognized, face_state = _wait_for_owner_approval_face(camera_manager, state)
        if _is_owner(recognized):
            with state.lock:
                state.enrollment_state = "approved_pending_subject"
                state.enrollment_target_name = target
                state.enrollment_approved_by = _normalize_person_name(recognized)
                state.enrollment_approved_at = time.time()
                state.last_prompt_type = "enrollment_subject_ready"
                state.last_prompt_text = _enrollment_subject_prompt(target)
            v6.speak(_enrollment_subject_prompt(target))
        else:
            v6.speak(
                "I could not confirm owner approval. Only Marco or Marquinho can approve. "
                "Marco or Marquinho, face my camera in good lighting. "
                f"After I recognize you, say: Marco approves enrolling {target.title()}."
            )
        return True

    with state.lock:
        enrollment_state = state.enrollment_state
        target = state.enrollment_target_name or "charlie"
        prompt_type = state.last_prompt_type

    if enrollment_state == "awaiting_name" or prompt_type == "enrollment_name":
        possible_name = _extract_enrollment_name_answer(user_text)
        if possible_name:
            target = _normalize_enrollment_target(possible_name)
            if _is_protected_identity(target) or _normalize_person_name(target) == "miguel":
                v6.speak("I will not enroll that protected name as a new friend.")
                return True
            with state.lock:
                state.enrollment_target_name = target
                state.enrollment_state = "requested"
                state.last_prompt_type = "enrollment_approval"
                state.last_prompt_text = _enrollment_approval_prompt(target)
            print(f"[V7.5 ENROLL] target_name set to {target}")
            v6.speak(_enrollment_approval_prompt(target))
            return True
        v6.speak(_enrollment_name_prompt())
        return True

    if enrollment_state == "requested":
        v6.speak(_enrollment_approval_prompt(target))
        return True

    capture_markers = [
        f"{target} is here",
        f"this is {target}",
        "ready",
        "take picture",
        "take a picture",
        "capture",
    ]

    unknown_face_present = bool(face_state.get("face_detected")) and not face_state.get("recognized_person")
    if enrollment_state == "approved_pending_subject":
        if unknown_face_present or any(p in t for p in capture_markers) or _is_enrollment_request_text(t):
            return _run_enrollment_capture(camera_manager, state)

    if enrollment_state == "capture_subject_samples":
        if _is_enrollment_request_text(t) or any(p in t for p in capture_markers):
            return _run_enrollment_capture(camera_manager, state)

    if not _is_enrollment_request_text(t):
        return False

    extracted_name = _extract_enrollment_name(user_text)
    if not extracted_name:
        with state.lock:
            state.enrollment_state = "awaiting_name"
            state.enrollment_target_name = None
            state.enrollment_approved_by = None
            state.enrollment_approved_at = 0.0
            state.last_prompt_type = "enrollment_name"
            state.last_prompt_text = _enrollment_name_prompt()
        v6.speak(_enrollment_name_prompt())
        return True

    target = _normalize_enrollment_target(extracted_name)
    with state.lock:
        state.enrollment_state = "requested"
        state.enrollment_target_name = target
        state.enrollment_approved_by = None
        state.enrollment_approved_at = 0.0
        state.last_prompt_type = "enrollment_approval"
        state.last_prompt_text = _enrollment_approval_prompt(target)

    print(f"[V7.5 ENROLL] target_name set to {target}")
    v6.speak(_enrollment_approval_prompt(target))
    return True


def _log_approval_face_state(face_state: dict) -> None:
    recognized = face_state.get("recognized_person")
    score = face_state.get("recognition_score")
    updated_at = float(face_state.get("updated_at", 0.0) or 0.0)
    age = face_state.get("age")
    if age is None and updated_at:
        age = time.time() - updated_at
    print(f"[V7.5 ENROLL] approval face_state recognized={recognized} score={score} age={age}")


def _wait_for_owner_approval_face(camera_manager: full.CameraManager, state: RobotRuntimeState):
    deadline = time.time() + 2.0
    face_state = None
    current_owner = _current_owner_partner(state)
    if _is_owner(current_owner):
        face_state = dict(camera_manager.get_face_state(max_age_seconds=2.0))
        face_state["recognized_person"] = current_owner
        _log_approval_face_state(face_state)
        return current_owner, face_state

    while True:
        face_state = state.identity_tracker.get_owner_authorization_identity(max_age_seconds=3.0)
        if face_state:
            break

        if time.time() >= deadline:
            break

        time.sleep(0.2)

    if not face_state:
        face_state = dict(camera_manager.get_face_state(max_age_seconds=1.0))
        fallback_owner = _current_owner_partner(state)
        face_state["recognized_person"] = fallback_owner if _is_owner(fallback_owner) else None

    _log_approval_face_state(face_state)
    recognized = face_state.get("recognized_person")
    return recognized, face_state


def _is_enrollment_request_text(text: str) -> bool:
    t = normalize_command_text(text)
    if not t:
        return False
    vague_background_phrases = {
        "check the new faces background",
        "new faces background",
        "check faces in the background",
        "check the new face background",
        "new face background",
    }
    if any(phrase in t for phrase in vague_background_phrases):
        return False
    explicit_phrases = {
        "learn this face",
        "add a new face",
        "enroll a new person",
        "enroll new person",
        "register a new user",
        "register new user",
        "registrar a new user",
        "registrar new user",
        "remember this person",
        "recognize someone",
        "recognise someone",
        "make a friend",
        "make friends",
        "enroll new friend",
        "enroll a new friend",
        "i want to enroll a new friend",
        "approves enrolling",
        "approve enrolling",
        "approves and rolling",
        "approve and rolling",
        "approved enrolling",
        "approved enroll",
        "approved in rolling",
        "start enrollment",
        "start enrollment for",
        "start face enrollment",
        "re enroll",
        "re enrol",
        "reenroll",
        "reenrol",
        "re-enroll",
        "reinrou",
        "reinroll",
        "update my face",
        "update face recognition",
        "improve face recognition",
        "improve my face recognition",
        "improve marquinho face recognition",
        "improve marco face recognition",
        "retrain my face",
        "retrain face",
        "conhecer aquela cara como",
        "conhecer essa cara como",
        "conhecer aquele rosto como",
        "conhecer esse rosto como",
        "aprender aquele rosto como",
        "aprender esse rosto como",
        "cadastrar aquela cara como",
        "cadastrar esse rosto como",
        "registrar aquela cara como",
        "registrar esse rosto como",
    }
    if any(phrase in t for phrase in explicit_phrases):
        return True
    return bool(
        re.search(r"\bthis is my friend\s+[a-zA-Z][a-zA-Z_-]*\b", t)
        or re.search(r"\badd my friend\s+[a-zA-Z][a-zA-Z_-]*\b", t)
        or re.search(r"\badd (?:a )?friend\s+[a-zA-Z][a-zA-Z_-]*\b", t)
        or re.search(r"\brecogni[sz]e\b.+\bas (?:a )?friend\b", t)
        or re.search(r"\bas (?:a )?friend named\s+[a-zA-Z][a-zA-Z_-]*\b", t)
        or re.search(r"\bregister\b.+\bcamera\b.+\bnew user\b", t)
    )


def _is_enrollment_cancel_text(text: str) -> bool:
    """Recognize explicit enrollment cancellation without weakening its gates."""
    t = normalize_command_text(text)
    return bool(
        re.search(r"\b(?:cancel|abort|stop)\s+(?:the\s+)?(?:face\s+)?enrollment\b", t)
        or re.search(r"\bcancel\s+enrolling\b", t)
    )


def _enrollment_followup_pending(state: RobotRuntimeState) -> bool:
    with state.lock:
        return bool(
            state.last_prompt_type in {"enrollment_name", "enrollment_approval"}
            or state.enrollment_state in {
                "awaiting_name",
                "requested",
                "approved_pending_subject",
                "capture_subject_samples",
            }
        )


def _is_enrollment_approval_text(text: str) -> bool:
    """Recognize approval wording; authorization is still checked by face."""
    t = normalize_command_text(text)
    if not t:
        return False
    if t in {"approve", "approved", "i approve", "yes i approve"}:
        return True
    has_approval = bool(re.search(r"\bapprov(?:e|es|ed|al)\b", t))
    has_enrollment_context = any(
        marker in t
        for marker in {
            "enroll",
            "enrollment",
            "new user",
            "new friend",
            "by me marco",
            "by me marquinho",
        }
    )
    return has_approval and has_enrollment_context


def _should_route_enrollment_followup(text: str, state: RobotRuntimeState) -> bool:
    """Keep expected enrollment answers local without hijacking other turns."""
    t = normalize_command_text(text)
    with state.lock:
        enrollment_state = state.enrollment_state
        prompt_type = state.last_prompt_type

    if enrollment_state == "awaiting_name" or prompt_type == "enrollment_name":
        return True
    if enrollment_state == "requested" or prompt_type == "enrollment_approval":
        return (
            _is_enrollment_approval_text(text)
            or _is_enrollment_cancel_text(text)
            or bool(_extract_enrollment_name_correction(text))
        )
    if enrollment_state in {"approved_pending_subject", "capture_subject_samples"}:
        return (
            _is_enrollment_request_text(text)
            or any(
                marker in t
                for marker in {"is here", "ready", "take picture", "take a picture", "capture"}
            )
            or _is_enrollment_cancel_text(text)
            or bool(_extract_enrollment_name_correction(text))
        )
    return False


def _is_reenrollment_request_text(text: str) -> bool:
    t = normalize_command_text(text)
    protected_target = any(name in t or name.replace("_", " ") in t for name in {"marco", "marquinho"})
    return any(
        phrase in t
        for phrase in {
            "re enroll",
            "re enrol",
            "reenroll",
            "reenrol",
            "re-enroll",
            "reinrou",
            "reinroll",
            "update my face",
            "update face recognition",
            "improve face recognition",
            "improve my face recognition",
            "retrain my face",
            "retrain face",
        }
    ) or (protected_target and any(phrase in t for phrase in {"start enrollment", "start face enrollment", "enrollment for"}))


def _extract_reenrollment_name(user_text: str, state: RobotRuntimeState) -> str | None:
    normalized = normalize_command_text(user_text)
    for name in _known_identity_names():
        display = name.replace("_", " ")
        if name in normalized or display in normalized:
            return name
    if any(marker in normalized for marker in {"my face", "me", "my recognition", "face recognition"}):
        with state.lock:
            return _normalize_person_name(state.recognized_person or state.conversation_partner)
    return None


def _extract_enrollment_approval_name(user_text: str) -> str | None:
    text = str(user_text or "").strip()
    pattern = re.compile(
        r"\b(?:marco|marquinho)\s+"
        r"(?:approves|approve|approved)\s+"
        r"(?:enrolling|enroll|in\s+roll|in\s+rolling|and\s+rolling)\s+"
        r"([a-zA-Z][a-zA-Z_-]*)",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        return match.group(1)
    return None


def _extract_enrollment_name_answer(user_text: str) -> str | None:
    text = str(user_text or "").strip()
    patterns = [
        r"\b(?:his|her|their)\s+name\s+is\s+([a-zA-Z][a-zA-Z_-]*)\b",
        r"\bthe\s+name\s+is\s+([a-zA-Z][a-zA-Z_-]*)\b",
        r"\bname\s+is\s+([a-zA-Z][a-zA-Z_-]*)\b",
        r"\bit\s+is\s+([a-zA-Z][a-zA-Z_-]*)\b",
        r"\bit's\s+([a-zA-Z][a-zA-Z_-]*)\b",
        r"\bis\s+your\s+friend'?s\s+name\s+([a-zA-Z][a-zA-Z_-]*)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1)
            if _is_protected_identity(name) or _normalize_person_name(name) == "miguel":
                return None
            return name

    name = _extract_short_enrollment_name(text)
    if _is_protected_identity(name) or _normalize_person_name(name) == "miguel":
        return None
    return name


def _extract_enrollment_name_correction(user_text: str) -> str | None:
    """Extract an explicit correction without treating ordinary speech as a rename."""
    text = str(user_text or "").strip()
    patterns = [
        r"\bcorrect(?:\s+the)?(?:\s+enrollment)?\s+name\s+to\s+([a-zA-Z][a-zA-Z_-]*)\b",
        r"\bchange(?:\s+the)?(?:\s+enrollment)?\s+name\s+to\s+([a-zA-Z][a-zA-Z_-]*)\b",
        r"\bthe\s+(?:correct|actual)\s+name\s+is\s+([a-zA-Z][a-zA-Z_-]*)\b",
        r"\bi\s+said\s+([a-zA-Z][a-zA-Z_-]*)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _normalize_enrollment_target(name: str | None) -> str:
    value = re.sub(r"\s+", "_", str(name or "").lower().strip(" .,:;!?"))
    return value or "charlie"


def _is_protected_identity(name: str | None) -> bool:
    return _normalize_person_name(name) in {"marco", "marquinho"}


def _next_embedding_index(person_dir: Path) -> int:
    existing = []
    for path in person_dir.glob("emb_*.npy"):
        match = re.search(r"emb_(\d+)\.npy$", path.name)
        if match:
            existing.append(int(match.group(1)))
    return (max(existing) + 1) if existing else 1


def _evaluate_enrollment_frame(frame, target_name: str, allow_protected_target: bool = False):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if brightness < 45.0:
        return None, "too_dark", "I need more light."
    if blur < 65.0:
        return None, "blurry", "Hold still for a second."

    faces = v6.insight_app.get(frame)
    if not faces:
        return None, "no_face", f"Please put only {target_name.title()} in front of the camera."
    if len(faces) > 1:
        return None, "multiple_faces", f"I see more than one face. Please leave only {target_name.title()} in front of me."

    face = faces[0]
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    area_ratio = (box_w * box_h) / float(w * h)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    if area_ratio < 0.045:
        return None, "too_far", "Please move closer."
    if area_ratio > 0.42 or x1 <= 4 or y1 <= 4 or x2 >= w - 4 or y2 >= h - 4:
        return None, "too_close", "Please move a little farther back."
    if center_x < w * 0.35:
        return None, "off_center_left", "Move a little to your right."
    if center_x > w * 0.65:
        return None, "off_center_right", "Move a little to your left."
    if center_y < h * 0.25 or center_y > h * 0.78:
        return None, "off_center", "Move a little closer to the center."

    matched_name, score, _margin, _scores = v6.recognize_insight_embedding(face.embedding)
    matched_normalized = _normalize_person_name(matched_name)
    target_normalized = _normalize_person_name(target_name)
    same_protected_target = allow_protected_target and matched_normalized == target_normalized
    if _is_protected_identity(matched_name) and score >= 0.48 and not same_protected_target:
        return None, "protected_identity", "This looks like a protected owner identity, so I will not enroll it as a new friend."

    return face, "ok", "Good, I captured that."


def _say_capture_prompt(text: str, pause_seconds: float = 1.2) -> None:
    v6.speak(text)
    time.sleep(pause_seconds)


def _reload_insight_embeddings() -> None:
    if hasattr(v6, "load_insight_embeddings"):
        v6.INSIGHT_FACE_DB = v6.load_insight_embeddings()


def _archive_existing_embeddings(person_dir: Path, target_name: str) -> Path | None:
    existing = sorted(person_dir.glob("emb_*.npy"))
    if not existing:
        return None
    backup_dir = person_dir / "backups" / time.strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    for emb_path in existing:
        shutil.move(str(emb_path), str(backup_dir / emb_path.name))
    print(f"[V7.5 ENROLL] Archived {len(existing)} old embeddings for {target_name} to {backup_dir}")
    return backup_dir


def _finalize_guided_capture(
    target_name: str,
    person_dir: Path,
    temp_embed_dir: Path,
    replace_existing: bool,
) -> int:
    new_embeddings = sorted(temp_embed_dir.glob("emb_*.npy"))
    if not new_embeddings:
        return 0
    if replace_existing:
        _archive_existing_embeddings(person_dir, target_name)
        next_index = 1
    else:
        next_index = _next_embedding_index(person_dir)
    for emb_path in new_embeddings:
        destination = person_dir / f"emb_{next_index:03d}.npy"
        shutil.move(str(emb_path), str(destination))
        next_index += 1
    return len(new_embeddings)


def _run_enrollment_capture(camera_manager: full.CameraManager, state: RobotRuntimeState) -> bool:
    set_interaction_state(state, "enrolling", "Enrollment capture")
    with state.lock:
        enrollment_state = state.enrollment_state
        target_name = state.enrollment_target_name
        approved_by = state.enrollment_approved_by

    if enrollment_state != "approved_pending_subject" or not target_name or not _is_owner(approved_by):
        v6.speak("Enrollment is not authorized yet.")
        return True

    target_name = _normalize_enrollment_target(target_name)
    replace_existing = _is_protected_identity(target_name)
    person_dir = v6.INSIGHT_EMBED_DIR / target_name
    person_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = person_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = person_dir / f"capture_{time.strftime('%Y%m%d_%H%M%S')}"
    temp_embed_dir = temp_dir / "embeddings"
    temp_sample_dir = temp_dir / "samples"
    temp_embed_dir.mkdir(parents=True, exist_ok=True)
    temp_sample_dir.mkdir(parents=True, exist_ok=True)
    next_index = 1

    with state.lock:
        state.enrollment_state = "capture_subject_samples"

    phases = [
        ("center", "Look straight at me and hold still.", 6),
        ("left", "Turn your head slightly left, but keep your face visible.", 5),
        ("right", "Turn your head slightly right, but keep your face visible.", 5),
        ("up", "Tilt your chin slightly up.", 4),
        ("down", "Tilt your chin slightly down.", 4),
        ("closer", "Move a little closer.", 3),
        ("farther", "Move a little farther back.", 3),
    ]

    captured = 0
    requested_samples = int(os.getenv("MIGUEL_ENROLLMENT_SAMPLE_COUNT", "30"))
    target_samples = max(20, min(36, requested_samples))
    phase_scale = target_samples / float(sum(phase[2] for phase in phases))
    start = time.time()
    last_guidance = ""
    last_guidance_at = 0.0
    last_saved_at = 0.0

    intro = "Re-enrollment" if replace_existing else "Enrollment"
    _say_capture_prompt(
        f"{intro} capture starting for {target_name.title()}. Please keep only that face in view.",
        pause_seconds=1.8,
    )

    for phase_name, prompt, base_count in phases:
        if captured >= target_samples or time.time() - start >= 120 or state.stop_event.is_set():
            break
        phase_target = max(2, int(round(base_count * phase_scale)))
        phase_captured = 0
        phase_deadline = time.time() + max(12.0, phase_target * 3.0)
        _say_capture_prompt(prompt, pause_seconds=1.2)

        while (
            phase_captured < phase_target
            and captured < target_samples
            and time.time() < phase_deadline
            and time.time() - start < 120
            and not state.stop_event.is_set()
        ):
            with state.lock:
                if state.enrollment_state == "idle":
                    v6.speak("Enrollment canceled.")
                    return True

            snap = camera_manager.get_latest_frame(require_fresh=True, wait_timeout=1.2)
            if snap is None:
                now = time.time()
                if now - last_guidance_at > 2.5:
                    v6.speak("I do not have a fresh camera frame right now.")
                    last_guidance_at = now
                continue

            face, status, guidance = _evaluate_enrollment_frame(
                snap.frame,
                target_name,
                allow_protected_target=replace_existing,
            )
            now = time.time()

            if status == "protected_identity":
                _reset_enrollment_state(state)
                v6.speak(guidance)
                return True

            if status != "ok":
                if guidance != last_guidance and now - last_guidance_at > 1.8:
                    v6.speak(guidance)
                    last_guidance = guidance
                    last_guidance_at = now
                time.sleep(0.15)
                continue

            if now - last_saved_at < 0.35:
                time.sleep(0.05)
                continue

            emb = v6.insight_normalize_embedding(face.embedding)
            emb_path = temp_embed_dir / f"emb_{next_index:03d}.npy"
            sample_path = temp_sample_dir / f"{phase_name}_{next_index:03d}.jpg"
            np.save(str(emb_path), emb)
            cv2.imwrite(str(sample_path), snap.frame)
            print(f"[V7.5 ENROLL] Saved {emb_path} and {sample_path}")

            captured += 1
            phase_captured += 1
            next_index += 1
            last_saved_at = now

        if phase_captured:
            v6.speak(f"Good. I captured {phase_captured} for that position.")

    if captured < 20:
        with state.lock:
            state.enrollment_state = "approved_pending_subject"
            state.last_prompt_type = "enrollment_subject_ready"
        v6.speak(
            f"I captured only {captured} usable samples, so enrollment is not complete. "
            f"Improve the lighting, keep only {target_name.title()} in view, and uncover their face. "
            f"Then say: {target_name.title()} is here, to try again."
        )
        return True

    finalized = _finalize_guided_capture(target_name, person_dir, temp_embed_dir, replace_existing)
    try:
        shutil.move(str(temp_sample_dir), str(sample_dir / time.strftime("guided_%Y%m%d_%H%M%S")))
        shutil.rmtree(str(temp_dir), ignore_errors=True)
    except Exception as exc:
        print("[V7.5 ENROLL] sample archive warning:", exc)

    _reload_insight_embeddings()
    with state.lock:
        state.enrollment_state = "completed"
        state.last_prompt_type = None
        state.last_prompt_text = None
    mode_text = "Re-enrollment complete" if replace_existing else "Enrollment complete"
    v6.speak(
        f"{mode_text} for {target_name.title()}. I saved {finalized} guided face samples. "
        f"Next, have {target_name.title()} leave and return, then ask: Miguel, who do you see?"
    )
    return True


def _extract_short_enrollment_name(user_text: str) -> str | None:
    words = re.findall(r"[a-zA-Z][a-zA-Z_-]*", str(user_text or ""))
    ignored = {"yes", "yeah", "yep", "ok", "okay", "no", "nope", "ready"}
    words = [w for w in words if w.lower() not in ignored]
    if 1 <= len(words) <= 2:
        return words[0]
    return None


def _extract_enrollment_name(user_text: str) -> str | None:
    text = str(user_text or "").strip()
    lower = text.lower()
    markers = [
        "approves enrolling",
        "approve enrolling",
        "approves and rolling",
        "approve and rolling",
        "enroll this new face as",
        "enroll new friend",
        "enroll a new friend",
        "enroll a new person",
        "enroll new person",
        "register a new user",
        "register new user",
        "registrar a new user",
        "registrar new user",
        "learn this face as",
        "add a new face",
        "add my friend",
        "add friend",
        "this is my friend",
        "take a picture of",
        "remember this person as",
        "as a friend named",
        "friend named",
        "conhecer aquela cara como",
        "conhecer essa cara como",
        "conhecer aquele rosto como",
        "conhecer esse rosto como",
        "aprender aquele rosto como",
        "aprender esse rosto como",
        "cadastrar aquela cara como",
        "cadastrar esse rosto como",
        "registrar aquela cara como",
        "registrar esse rosto como",
    ]
    no_name_phrases = [
        "new friend",
        "new face",
        "enroll this new face",
        "i want to enroll a new friend",
        "enroll a new friend",
        "enroll a new person",
        "register a new user",
        "register new user",
        "registrar a new user",
        "registrar new user",
        "add a friend",
        "add a new face",
        "learn this face",
        "remember this person",
    ]

    # Mixed-language ASR commonly emits "registrar a new user, Ana is ...".
    # The appositive immediately after "new user" is a name; relationship
    # details are deliberately ignored. Authorization is still enforced later
    # by _route_enrollment.
    named_user = re.search(
        r"\b(?:register|registrar)\s+(?:a\s+)?new\s+user\s*[,;:]?\s*"
        r"([a-zA-Z][a-zA-Z_-]*)\s+is\b",
        text,
        re.IGNORECASE,
    )
    if named_user:
        return named_user.group(1).title()

    for marker in markers:
        index = lower.find(marker)
        if index >= 0:
            candidate = text[index + len(marker):].strip(" .,:;!?")
            words = [w for w in candidate.split() if w.lower() not in {"a", "as", "new", "friend", "face", "por", "favor"}]
            if words:
                return "_".join(words[:3]).strip(" .,:;!?").title()
            if any(p in lower for p in no_name_phrases):
                return None

    match = re.search(
        r"\b(?:his|her|their) name is\s+([a-zA-Z][a-zA-Z_-]*)\b",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).title()

    return None


def _update_mode_state_from_text(user_text: str, state: RobotRuntimeState) -> None:
    t = normalize_command_text(user_text)
    mode = None
    if _is_sleep_mode_request(t):
        mode = "sleep"
    elif "wake up" in t:
        mode = "normal"
    elif "mission control" in t:
        mode = "mission_control"

    if mode:
        with state.lock:
            state.current_mode = mode
            state.sleep_mode_active = mode == "sleep"


def face_worker(camera_manager: full.CameraManager, state: RobotRuntimeState):
    print("[V7.5 FACE] FaceWorker started.")

    while not state.stop_event.is_set():
        try:
            if hasattr(v6, "detect_face_state"):
                face_state = v6.detect_face_state(camera_manager)
            else:
                face_state = {
                    "face_detected": False,
                    "face_count": 0,
                    "recognized_person": None,
                    "recognizer": "detect_face_state_missing",
                }

            camera_manager.update_face_state(face_state)
            state.identity_tracker.update(face_state)

            recognized = _normalize_person_name(face_state.get("recognized_person"))
            previous_key, current_key = _update_face_identity_runtime_state(
                state,
                bool(face_state.get("face_detected")),
                recognized,
                face_state.get("face_count"),
                face_state.get("recognition_score"),
                face_state.get("recognition_margin"),
            )
            _maybe_surface_unknown_face(state, previous_key, current_key)

            if recognized:
                print(
                    f"[V7.5 FACE] recognized={recognized} "
                    f"score={face_state.get('recognition_score')}"
                )

        except Exception as exc:
            msg = str(exc)
            if "No fresh camera frame" not in msg:
                print("[V7.5 FACE] error:", exc)

        time.sleep(0.6)

    print("[V7.5 FACE] FaceWorker stopped.")


def audio_worker(
    camera_manager: full.CameraManager,
    user_turn_queue: queue.Queue,
    state: RobotRuntimeState,
):
    print("[V7.5 AUDIO] AudioWorker started.")
    last_known_person = None

    while not state.stop_event.is_set():
        try:
            _watchdog_audio_capture_state(state)
            _expire_shutdown_confirmation_if_needed(state)
            expire_conversation_session_if_needed(state)
            face_state = camera_manager.get_face_state(max_age_seconds=2.0)
            recognized = _normalize_person_name(face_state.get("recognized_person"))
            previous_key, current_key = _update_face_identity_runtime_state(
                state,
                bool(face_state.get("face_detected")),
                recognized,
                face_state.get("face_count"),
                face_state.get("recognition_score"),
                face_state.get("recognition_margin"),
            )
            _maybe_surface_unknown_face(state, previous_key, current_key)

            if recognized:
                if recognized != last_known_person:
                    print(f"[V7.5 READY] Familiar person present: {recognized}")
                    last_known_person = recognized

                _wait_until_listening_allowed(state)
                if state.stop_event.is_set():
                    break

                user_text = capture_user_turn_when_ready(state)
                if user_text:
                    with state.lock:
                        shutdown_waiting = bool(state.shutdown_confirmation_pending)
                    if shutdown_waiting:
                        if _is_shutdown_confirm_text(user_text) or _is_shutdown_request_text(user_text) or _is_shutdown_cancel_text(user_text):
                            _enqueue_user_turn(
                                user_turn_queue,
                                state,
                                user_text,
                                recognized,
                                authorized=True,
                                authorization_source="global_command",
                            )
                        else:
                            set_interaction_state(state, "shutdown_pending", "Confirm shutdown")
                        continue
                    if _handle_sleep_mode_audio_text(user_turn_queue, state, user_text, recognized):
                        continue
                    active = is_conversation_active(state)
                    familiar_present = bool(recognized)
                    accepted_short_followup = _accept_v715_short_followup_if_allowed(
                        user_text,
                        state,
                        familiar_present=familiar_present,
                    )
                    directed = is_directed_to_miguel(user_text, state)
                    if active and (
                        directed
                        or accepted_short_followup
                        or _infer_conversation_mode(user_text) != "general"
                        or _short_answer_after_robot_question(user_text, state)
                        or _is_correction_retry_text(user_text)
                        or bool(_creative_fast_allow_topic(user_text))
                    ):
                        if directed:
                            print(f"[V7.14 ADDRESSING] accepted active question={_short_log_text(user_text)}")
                        print(
                            f"[V7.14 CONVERSATION] accepted no-wake turn "
                            f"mode={state.conversation_mode} partner={state.conversation_partner}"
                        )
                        _enqueue_user_turn(
                            user_turn_queue,
                            state,
                            user_text,
                            recognized,
                            authorized=True,
                            authorization_source=_active_authorization_source(state),
                        )
                    elif active:
                        reason = _likely_background_speech_reason(user_text, state)
                        print(f"[V7.14 ADDRESSING] ignored likely background speech={_short_log_text(user_text)} reason={reason}")
                        set_interaction_state(state, _ready_face_state(), _ready_face_text(state))
                    elif _is_owner(recognized) and _is_owner_natural_direct_command(user_text):
                        print("[V7.14 OWNER SESSION] Owner direct command accepted:", user_text)
                        _enqueue_user_turn(
                            user_turn_queue,
                            state,
                            user_text,
                            recognized,
                            authorized=True,
                            authorization_source="owner_session",
                        )
                    elif _is_owner(recognized) and _is_password_session_command(user_text, state):
                        _enqueue_user_turn(
                            user_turn_queue,
                            state,
                            user_text,
                            recognized,
                            authorized=True,
                            authorization_source="owner_session",
                        )
                    elif accepted_short_followup:
                        _enqueue_user_turn(
                            user_turn_queue,
                            state,
                            user_text,
                            recognized,
                            authorized=True,
                            authorization_source="active_conversation",
                        )
                    elif is_barge_in_command(user_text) or _is_global_audio_command(user_text):
                        source = "barge_in" if is_barge_in_command(user_text) else "global_command"
                        _enqueue_user_turn(
                            user_turn_queue,
                            state,
                            user_text,
                            recognized,
                            authorized=True,
                            authorization_source=source,
                        )
                    elif _is_bare_wake_phrase(user_text):
                        _enqueue_user_turn(
                            user_turn_queue,
                            state,
                            user_text,
                            recognized,
                            authorized=True,
                            authorization_source="wake_phrase",
                        )
                    elif _has_v7_5_wake_phrase(user_text):
                        _enqueue_user_turn(
                            user_turn_queue,
                            state,
                            user_text,
                            recognized,
                            authorized=True,
                            authorization_source="wake_phrase",
                            stripped_text=_strip_wake_phrase(user_text),
                        )
                    else:
                        _show_wake_required(state, user_text)
                else:
                    if is_conversation_active(state):
                        set_interaction_state(state, _ready_face_state(), _ready_face_text(state))
                    else:
                        set_interaction_state(state, "idle", "")
                continue

            last_known_person = None
            conversation_active = is_conversation_active(state)
            grace_active = _is_conversation_grace_active(state)
            if face_state.get("face_detected"):
                set_interaction_state(state, "idle", "")
            else:
                set_interaction_state(state, "idle", "")

            _wait_until_listening_allowed(state)
            if state.stop_event.is_set():
                break

            user_text = capture_user_turn_when_ready(state)
            if not user_text:
                set_interaction_state(state, "idle", "")
                continue

            with state.lock:
                shutdown_waiting = bool(state.shutdown_confirmation_pending)
            if shutdown_waiting:
                if _is_shutdown_confirm_text(user_text) or _is_shutdown_request_text(user_text) or _is_shutdown_cancel_text(user_text):
                    _enqueue_user_turn(
                        user_turn_queue,
                        state,
                        user_text,
                        None,
                        authorized=True,
                        authorization_source="global_command",
                    )
                else:
                    set_interaction_state(state, "shutdown_pending", "Confirm shutdown")
                continue

            if _handle_sleep_mode_audio_text(user_turn_queue, state, user_text, None):
                continue

            if is_barge_in_command(user_text):
                _enqueue_user_turn(
                    user_turn_queue,
                    state,
                    user_text,
                    None,
                    authorized=True,
                    authorization_source="barge_in",
                )
                continue

            if conversation_active:
                directed = is_directed_to_miguel(user_text, state)
                accepted_short_followup = _accept_v715_short_followup_if_allowed(
                    user_text,
                    state,
                    familiar_present=False,
                )
                if (
                    directed
                    or accepted_short_followup
                    or _infer_conversation_mode(user_text) != "general"
                    or _short_answer_after_robot_question(user_text, state)
                    or _is_correction_retry_text(user_text)
                    or bool(_creative_fast_allow_topic(user_text))
                ):
                    if directed:
                        print(f"[V7.14 ADDRESSING] accepted active question={_short_log_text(user_text)}")
                    print(
                        f"[V7.14 CONVERSATION] accepted no-wake turn "
                        f"mode={state.conversation_mode} partner={state.conversation_partner}"
                    )
                    _enqueue_user_turn(
                        user_turn_queue,
                        state,
                        user_text,
                        None,
                        authorized=True,
                        authorization_source=_active_authorization_source(state),
                    )
                else:
                    reason = _likely_background_speech_reason(user_text, state)
                    print(f"[V7.14 ADDRESSING] ignored likely background speech={_short_log_text(user_text)} reason={reason}")
                    set_interaction_state(state, _ready_face_state(), _ready_face_text(state))
                continue

            if is_owner_present(state, camera_manager) and _is_owner_natural_direct_command(user_text):
                print("[V7.13 OWNER] Owner direct command accepted:", user_text)
                _enqueue_user_turn(
                    user_turn_queue,
                    state,
                    user_text,
                    None,
                    authorized=True,
                    authorization_source="owner_session",
                )
                continue

            if is_owner_present(state, camera_manager) and _is_password_session_command(user_text, state):
                _enqueue_user_turn(
                    user_turn_queue,
                    state,
                    user_text,
                    None,
                    authorized=True,
                    authorization_source="owner_session",
                )
                continue

            if _is_bare_wake_phrase(user_text):
                _enqueue_user_turn(
                    user_turn_queue,
                    state,
                    user_text,
                    None,
                    authorized=True,
                    authorization_source="wake_phrase",
                )
                continue

            if _has_v7_5_wake_phrase(user_text):
                _enqueue_user_turn(
                    user_turn_queue,
                    state,
                    user_text,
                    None,
                    authorized=True,
                    authorization_source="wake_phrase",
                    stripped_text=_strip_wake_phrase(user_text),
                )
                continue

            if _is_global_without_wake_command(user_text):
                print("[V7.5 IDLE] Global command accepted:", user_text)
                _enqueue_user_turn(
                    user_turn_queue,
                    state,
                    user_text,
                    None,
                    authorized=True,
                    authorization_source="global_command",
                )
                continue

            if grace_active:
                if _is_acceptable_grace_transcript(user_text, state):
                    print("[V7.5 FOLLOWUP] Grace-window transcript accepted:", user_text)
                    _enqueue_user_turn(
                        user_turn_queue,
                        state,
                        user_text,
                        None,
                        authorized=True,
                        authorization_source=_active_authorization_source(state),
                    )
                else:
                    print("[V7.5 AUDIO] Dropped weak grace-window transcript.")
                    set_interaction_state(state, "idle", "")
                continue

            if not _has_v7_5_wake_phrase(user_text):
                _show_wake_required(state, user_text)
                continue

            command_text = _strip_wake_phrase(user_text)
            if command_text:
                _enqueue_user_turn(
                    user_turn_queue,
                    state,
                    command_text,
                    None,
                    authorized=True,
                    authorization_source="wake_phrase",
                    stripped_text=command_text,
                )
                continue

            _wait_until_listening_allowed(state)
            if state.stop_event.is_set():
                break

            next_text = capture_user_turn_when_ready(state)
            if next_text:
                _enqueue_user_turn(
                    user_turn_queue,
                    state,
                    next_text,
                    None,
                    authorized=True,
                    authorization_source="wake_phrase",
                )
            else:
                set_interaction_state(state, "idle", "")

        except Exception as exc:
            if not state.stop_event.is_set():
                print("[V7.5 AUDIO] error:", exc)
                time.sleep(0.25)

    print("[V7.5 AUDIO] AudioWorker stopped.")


def brain_worker(
    camera_manager: full.CameraManager,
    safety: SafetyGuard,
    user_turn_queue: queue.Queue,
    state: RobotRuntimeState,
):
    print("[V7.5 BRAIN] BrainWorker started.")

    while not state.stop_event.is_set():
        try:
            event = user_turn_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            event_authorized = bool(getattr(event, "authorized", False))
            if event_authorized:
                with state.lock:
                    state.brain_is_processing = True
                    state.turn_processing_active = True
                    state.turn_processing_started_at = time.time()
                print(f"[V7.14 TURN] processing=true text={_short_log_text(getattr(event, 'text', ''))}")
            else:
                with state.lock:
                    state.brain_is_processing = True
            keep_running = handle_queued_turn(event, camera_manager, safety, state)
            if not keep_running:
                state.stop_event.set()
        except Exception as exc:
            print("[V7.5 BRAIN] error:", exc)
            set_interaction_state(state, "error", str(exc)[:48])
            v6.speak("I had a brain error while processing that.")
        finally:
            with state.lock:
                route = state.current_turn_latency.get("reply_context", "error")
                state.brain_is_processing = False
                state.turn_processing_active = False
                state.turn_processing_started_at = 0.0
                state.pending_user_turn_count = max(0, state.pending_user_turn_count - 1)
            print(f"[V7.14 TURN] processing=false route={route}")
            user_turn_queue.task_done()

    print("[V7.5 BRAIN] BrainWorker stopped.")


def handle_queued_turn(
    user_text: str | UserTurnEvent,
    camera_manager: full.CameraManager,
    safety: SafetyGuard,
    state: RobotRuntimeState,
) -> bool:
    event = user_text if isinstance(user_text, UserTurnEvent) else None
    raw_user_text = str(event.text if event else user_text or "").strip()
    user_text = raw_user_text
    if not user_text:
        return True

    # Audio capture may enqueue another transcript while this one is routing.
    # Restore the per-event snapshot so its reply and latency cannot be
    # attributed to the newer transcript.
    if event and event.latency:
        with state.lock:
            state.current_turn_latency = dict(event.latency)
            state.current_turn_latency["log_user_text"] = raw_user_text

    expire_conversation_session_if_needed(state)
    event_authorized = bool(getattr(event, "authorized", False))
    auth_source = str(getattr(event, "authorization_source", "") or "")
    event_partner = _normalize_person_name(getattr(event, "recognized_person", None))
    with state.lock:
        active_partner = (
            _normalize_person_name(state.conversation_partner)
            if state.conversation_active and time.time() <= float(state.conversation_until or 0.0)
            else None
        )
    # This is attribution only; authorization remains bound to the immutable
    # event fields and existing identity gates below.
    partner = event_partner or active_partner or _normalize_person_name(_current_owner_partner(state)) or "unknown_wake_user"
    with state.lock:
        if not _normalize_person_name(state.current_turn_latency.get("log_person")):
            state.current_turn_latency["log_person"] = partner
    if event_authorized:
        print(f"[V7.14 AUTH] accepted source={auth_source} text={_short_log_text(user_text)}")

    sleep_result = _route_sleep_control(user_text, state, partner=partner)
    if sleep_result is not None:
        with state.lock:
            sleep_route = str(state.current_turn_latency.get("reply_context") or "")
        # These reply-producing routes run before the general turn logger.
        if sleep_route in {"sleep", "wake"}:
            _remember_accepted_turn(state, user_text)
            _log_user_turn_event(state, user_text, route_hint=sleep_route, partner=partner)
        with state.lock:
            turn_started_at = state.current_turn_latency.get("turn_started_at") or time.monotonic()
            state.current_turn_latency.setdefault("turn_started_at", turn_started_at)
        _mark_route_done(state, turn_started_at)
        return sleep_result

    had_wake_phrase = _has_v7_5_wake_phrase(user_text)
    if _is_bare_wake_phrase(user_text):
        start_conversation_session(state, mode="general", partner=partner, reason="bare_wake")
        _remember_accepted_turn(state, user_text)
        _log_user_turn_event(state, user_text, route_hint="greeting", partner=partner)
        _set_reply_context(state, "greeting")
        v6.speak("Here.")
        with state.lock:
            turn_started_at = state.current_turn_latency.get("turn_started_at") or time.monotonic()
            state.current_turn_latency.setdefault("turn_started_at", turn_started_at)
        _mark_route_done(state, turn_started_at)
        return True

    if had_wake_phrase:
        stripped = str(getattr(event, "stripped_text", "") or _strip_wake_phrase(user_text)).strip()
        if stripped:
            sleep_result = _route_sleep_control(stripped, state, partner=partner)
            if sleep_result is not None:
                with state.lock:
                    turn_started_at = state.current_turn_latency.get("turn_started_at") or time.monotonic()
                    state.current_turn_latency.setdefault("turn_started_at", turn_started_at)
                _mark_route_done(state, turn_started_at)
                return sleep_result
            mode = _infer_conversation_mode(stripped)
            if mode == "general" and _is_story_continue_text(stripped):
                with state.lock:
                    if state.conversation_mode in {"story", "creative"} or (state.last_topic and "story" in state.last_topic.lower()):
                        mode = state.conversation_mode if state.conversation_mode in {"story", "creative"} else "story"
            timeout = None if mode != "robot_control" else _env_float("MIGUEL_CONVERSATION_TIMEOUT_SECONDS", 120.0)
            start_conversation_session(state, mode=mode, partner=partner, timeout_seconds=timeout, reason="wake_plus_command")
            user_text = stripped
        else:
            start_conversation_session(state, mode="general", partner=partner, reason="bare_wake")
            _set_reply_context(state, "greeting")
            v6.speak("Here.")
            return True
    elif event_authorized:
        mode = _infer_conversation_mode(user_text)
        if auth_source == "owner_session" and not is_conversation_active(state):
            start_conversation_session(state, mode="general", partner=partner, reason="owner_direct_command")
        elif auth_source == "global_command":
            start_conversation_session(state, mode="robot_control", partner=partner, reason="global_command")
        elif auth_source == "barge_in":
            extend_conversation_session(state, reason="barge_in")
        elif auth_source == "password_session":
            start_conversation_session(
                state,
                mode="owner_password",
                partner="owner_password",
                timeout_seconds=_env_float("MIGUEL_PASSWORD_SESSION_TIMEOUT_SECONDS", 600.0),
                reason="password_session_turn",
            )
        elif mode not in {"general", "robot_control"}:
            start_conversation_session(state, mode=mode, partner=partner, reason=auth_source or "authorized_turn")
        else:
            if is_conversation_active(state):
                extend_conversation_session(state, reason=auth_source or "authorized_turn")
            else:
                start_conversation_session(state, mode="general", partner=partner, reason=auth_source or "authorized_turn")
    elif is_conversation_active(state):
        with state.lock:
            mode = state.conversation_mode
            session_partner = state.conversation_partner
        print(f"[V7.14 CONVERSATION] accepted no-wake turn mode={mode} partner={session_partner}")
        extend_conversation_session(state, reason="user_turn")
    elif not _is_global_without_wake_command(user_text):
        _show_wake_required(state, user_text)
        return True

    user_text = _resolve_short_context(user_text, state)
    user_text = _choose_first_direct_command(user_text)
    if "star wars" in normalize_command_text(user_text):
        with state.lock:
            state.last_topic = "Star Wars"
            state.last_topic_until = time.time() + 300.0
    with state.lock:
        turn_started_at = state.current_turn_latency.get("turn_started_at") or time.monotonic()
        state.current_turn_latency.setdefault("turn_started_at", turn_started_at)
    print(f"[V7.5 TRANSCRIPT] {user_text}")

    automatic_expression = _automatic_face_expression(user_text)
    if automatic_expression and not _requested_face_expression(user_text):
        _set_face_expression(state, automatic_expression, "automatic")

    _set_reply_context(state, "teacher")
    if _route_portuguese_times_table_reply(user_text, state):
        _mark_route_done(state, turn_started_at)
        return True
    if _route_teacher_mode_local_reply(user_text, state, partner=partner):
        _remember_accepted_turn(state, user_text)
        _log_user_turn_event(state, user_text, route_hint="teacher", partner=partner)
        _set_response_length_context(state, "detailed")
        _mark_route_done(state, turn_started_at)
        return True

    if _route_story_control_request(user_text, state):
        _set_response_length_context(state, "terse")
        _mark_route_done(state, turn_started_at)
        return True

    # Enrollment is local and security-sensitive. Handle it before cloud story
    # classification so a stale story topic cannot intercept the command.
    if _is_enrollment_request_text(user_text) or _should_route_enrollment_followup(user_text, state):
        _remember_accepted_turn(state, user_text)
        _log_user_turn_event(state, user_text, route_hint="enrollment", partner=partner)
        _set_reply_context(state, "enrollment")
        set_interaction_state(state, "enrolling", user_text[:48])
        if _route_enrollment(user_text, state, camera_manager):
            _set_response_length_context(state, "terse")
            _mark_route_done(state, turn_started_at)
            return True

    _set_reply_context(state, "language_policy")
    if _route_language_policy_local_reply(user_text, state):
        _set_response_length_context(state, "terse")
        _mark_route_done(state, turn_started_at)
        return True

    story_detection = _story_mode_detection(user_text)
    if _is_story_finish_request(user_text) and _current_active_topic(state):
        story_detection = _empty_story_detection()
    _exit_stale_story_mode_for_non_story_turn(user_text, story_detection, state)
    if _route_low_confidence_story_intent(story_detection, state):
        _mark_route_done(state, turn_started_at)
        return True

    story_request_allowed = bool(story_detection.get("detected") and story_detection.get("action") == "generate_story")
    if not story_request_allowed and not _language_policy_allows_turn(user_text, state):
        _mark_route_done(state, turn_started_at)
        force_interaction_state(state, _ready_face_state(), _ready_face_text(state))
        return True

    if _route_story_duration_reality_question(user_text):
        _set_response_length_context(state, "normal")
        _mark_route_done(state, turn_started_at)
        return True

    new_story_requested = _is_new_story_request(user_text)
    if new_story_requested:
        _log_story_mode_detection(story_detection)
        if story_detection.get("has_story_request"):
            _log_story_execution_generate()
        _start_new_story_topic(state, user_text, partner=partner)
        with state.lock:
            if state.response_depth_mode == "normal":
                state.response_depth_mode = "long_story"
            if state.conversation_mode in {"general", "wake_required"}:
                state.conversation_mode = "story"
            state.current_turn_latency["reply_context"] = "story"
            state.current_turn_latency["response_depth_mode"] = state.response_depth_mode
            state.current_turn_latency["long_story_target_minutes"] = state.long_story_target_minutes
        _set_response_length_context(state, "long_story")

    if story_detection.get("story_mode") == "story_continuous":
        hard_stop_reason, hard_stop_reply = _local_safety_hard_stop(user_text)
        if hard_stop_reply:
            print(f"[V7.14 SAFETY ROUTER] local_hard_stop reason={hard_stop_reason}")
            _set_reply_context(state, "safety_refusal")
            _set_response_length_context(state, "terse")
            v6.speak(hard_stop_reply)
            _mark_route_done(state, turn_started_at)
            return True

    if (
        story_detection.get("detected")
        and story_detection.get("action") == "generate_story"
        and story_detection.get("story_mode") == "story_continuous"
    ):
        _remember_accepted_turn(state, user_text)
        _log_user_turn_event(state, user_text, route_hint="story", partner=partner)
    if _route_story_continuous_generation(user_text, state, story_detection):
        _mark_route_done(state, turn_started_at)
        return True

    _remember_accepted_turn(state, user_text)
    if not new_story_requested:
        _update_active_topic_from_text(state, user_text)
    _log_user_turn_event(state, user_text, route_hint="accepted", partner=partner)

    mode = _infer_conversation_mode(user_text)
    with state.lock:
        previous_conversation_mode = state.conversation_mode
    if is_conversation_active(state) and mode not in {"general", "robot_control"}:
        start_conversation_session(state, mode=mode, partner=partner, reason="mode_update")
    elif is_conversation_active(state):
        extend_conversation_session(state, reason="route_start")

    barge_result = _route_barge_in_control(user_text, state)
    if barge_result is not None:
        _mark_route_done(state, turn_started_at)
        return barge_result

    _set_reply_context(state, "owner_password_ack")
    if _route_password_owner_session(user_text, state):
        _set_response_length_context(state, "terse")
        _mark_route_done(state, turn_started_at)
        return True

    if _route_timer_local_reply(user_text, state):
        _set_response_length_context(state, "terse")
        _mark_route_done(state, turn_started_at)
        return True

    if _route_system_volume_local_reply(user_text, state):
        _set_response_length_context(state, "terse")
        _mark_route_done(state, turn_started_at)
        return True

    _set_reply_context(state, "teacher")
    if _route_teacher_mode_local_reply(user_text, state, partner=partner):
        _set_response_length_context(state, "detailed")
        _mark_route_done(state, turn_started_at)
        return True

    camera_intent = classify_camera_intent(user_text)
    if _is_project_role_request(user_text):
        camera_intent = "none"
    with state.lock:
        current_conversation_mode = state.conversation_mode
        saved_response_mode = state.response_length_mode
        response_depth_mode = state.response_depth_mode
    inferred_response_mode = infer_response_length_mode(user_text, current_conversation_mode, camera_intent)
    inferred_conversation_mode = _infer_conversation_mode(user_text, camera_intent)
    normalized_for_depth = normalize_command_text(user_text)
    if (
        camera_intent == "none"
        and (story_detection.get("mode") == "long_story" or _is_explicit_long_story_request(user_text))
        and normalized_for_depth not in LONG_STORY_ACTIVATION_PHRASES
        and not _is_depth_status_question(user_text)
    ):
        _log_story_mode_detection(story_detection)
        _set_response_depth_mode(state, "long_story", "explicit_long_story_request")
        response_depth_mode = "long_story"
        inferred_response_mode = "long_story"
        if inferred_conversation_mode == "general":
            inferred_conversation_mode = "story"
        with state.lock:
            if state.conversation_mode in {"general", "wake_required"}:
                state.conversation_mode = "story"
    if inferred_response_mode == "normal" and saved_response_mode in {"detailed", "story", "long_story"}:
        inferred_response_mode = saved_response_mode
    if (
        response_depth_mode == "long_story"
        and camera_intent == "none"
        and (current_conversation_mode in {"creative", "story"} or inferred_conversation_mode in {"creative", "story"})
    ):
        inferred_response_mode = "long_story"
    elif new_story_requested:
        inferred_conversation_mode = "story"
        inferred_response_mode = "long_story"
        response_depth_mode = "long_story"
    elif response_depth_mode == "long_explanation" and camera_intent == "none":
        inferred_response_mode = "detailed"
    if inferred_response_mode == "terse" and not _route_allows_terse_response(user_text, current_conversation_mode, camera_intent):
        inferred_response_mode = "normal"
    _set_response_length_context(state, inferred_response_mode)
    with state.lock:
        state.current_turn_latency["response_depth_mode"] = state.response_depth_mode
    if camera_intent != "none":
        print(f"[V7.5 CAMERA INTENT] {camera_intent}: {user_text}")
        set_interaction_state(state, "looking", user_text[:48])
    elif _is_enrollment_request_text(user_text):
        set_interaction_state(state, "enrolling", user_text[:48])
    else:
        set_interaction_state(state, "thinking", user_text[:48])

    shutdown_result = _route_shutdown_control(user_text, state)
    if shutdown_result is not None:
        _mark_route_done(state, turn_started_at)
        return shutdown_result

    _set_reply_context(state, "normal")
    if _route_heard_repeat(user_text, state):
        _mark_route_done(state, turn_started_at)
        return True

    if _route_sensor_health_local_reply(user_text, camera_manager, state):
        _set_response_length_context(state, "terse")
        _mark_route_done(state, turn_started_at)
        return True

    if _route_audio_or_reply_health_local_reply(user_text, state):
        _set_response_length_context(state, "terse")
        _mark_route_done(state, turn_started_at)
        return True

    _set_reply_context(state, "repeat")
    if _route_repeat_last_reply(user_text, state):
        _mark_route_done(state, turn_started_at)
        return True

    if _route_identity_camera_intent(user_text, camera_intent, camera_manager, state):
        _set_response_length_context(state, "terse")
        _mark_route_done(state, turn_started_at)
        return True

    if _route_scene_camera_intent(user_text, camera_intent, camera_manager, state):
        _mark_route_done(state, turn_started_at)
        return True

    _set_reply_context(state, "identity")
    if _route_preferred_address_request(user_text, state):
        _set_response_length_context(state, "terse")
        _mark_route_done(state, turn_started_at)
        return True

    _set_reply_context(state, "identity")
    if _route_spoken_identity_claim(user_text, state):
        _set_response_length_context(state, "terse")
        _mark_route_done(state, turn_started_at)
        return True

    _set_reply_context(state, "identity")
    if _route_name_correction_local_reply(user_text, state):
        _set_response_length_context(state, "terse")
        _mark_route_done(state, turn_started_at)
        return True

    _set_reply_context(state, "clarification")
    if _route_last_answer_clarification(user_text, state):
        _mark_route_done(state, turn_started_at)
        return True

    _set_reply_context(state, "depth_mode")
    if _route_depth_status_local_reply(user_text, state):
        _mark_route_done(state, turn_started_at)
        return True

    _set_reply_context(state, "depth_mode")
    if _route_response_depth_mode(user_text, state):
        _mark_route_done(state, turn_started_at)
        return True

    _set_reply_context(state, "voice_command")
    if _route_voice_modes_local_reply(user_text, state):
        _mark_route_done(state, turn_started_at)
        return True

    _set_reply_context(state, "face_expression")
    if _route_face_expression_local_reply(user_text, state):
        _set_response_length_context(state, "terse")
        _mark_route_done(state, turn_started_at)
        return True

    _set_reply_context(state, "capabilities")
    if _route_capabilities_local_reply(user_text, state):
        _mark_route_done(state, turn_started_at)
        return True

    _set_reply_context(state, "memory")
    if _route_conversation_memory_local_reply(user_text, state):
        _mark_route_done(state, turn_started_at)
        return True

    _set_reply_context(state, "creative")
    if _route_correction_retry(user_text, state):
        _mark_route_done(state, turn_started_at)
        return True

    _set_reply_context(state, "normal")
    if _route_creative_story_local_reply(user_text, state):
        _mark_route_done(state, turn_started_at)
        return True

    _set_reply_context(state, "normal")
    if _route_long_story_mode(user_text, state):
        _mark_route_done(state, turn_started_at)
        return True

    _set_reply_context(state, "normal")
    if _route_topic_followup_local_reply(user_text, state):
        _mark_route_done(state, turn_started_at)
        return True

    _set_reply_context(state, "local_ack")
    if _route_fast_local_reply(user_text, state):
        _set_response_length_context(state, "terse" if _route_allows_terse_response(user_text, mode, camera_intent) else inferred_response_mode)
        _mark_route_done(state, turn_started_at)
        return True

    _set_reply_context(state, "normal")
    if _route_project_local_reply(user_text, state):
        _mark_route_done(state, turn_started_at)
        return True

    _set_reply_context(state, "location")
    if _route_location_local_reply(user_text, state):
        _set_response_length_context(state, "normal")
        _mark_route_done(state, turn_started_at)
        return True

    _set_reply_context(state, "normal")
    if _route_fun_local_reply(user_text, state):
        _mark_route_done(state, turn_started_at)
        return True

    _set_reply_context(state, "utility")
    local_command_text = normalize_command_text(user_text) or user_text

    if full.handle_v7_local_utility(local_command_text):
        _set_response_length_context(state, "terse")
        _mark_route_done(state, turn_started_at)
        return True

    _set_reply_context(state, "normal")
    if _route_celsius_conversion(local_command_text, state):
        _mark_route_done(state, turn_started_at)
        return True

    _set_reply_context(state, "enrollment")
    if _route_enrollment(user_text, state, camera_manager):
        _set_response_length_context(state, "terse")
        _mark_route_done(state, turn_started_at)
        return True

    if _route_camera_refresh(user_text, camera_manager, state):
        _set_response_length_context(state, "terse")
        _mark_route_done(state, turn_started_at)
        return True

    if _route_identity_camera_intent(user_text, camera_intent, camera_manager, state):
        _set_response_length_context(state, "terse")
        _mark_route_done(state, turn_started_at)
        return True

    if _route_scene_camera_intent(user_text, camera_intent, camera_manager, state):
        _mark_route_done(state, turn_started_at)
        return True

    _set_reply_context(state, "robot_control")
    if full.is_local_robot_control_request(local_command_text):
        _set_response_length_context(state, "terse")
        face_state = camera_manager.get_face_state(max_age_seconds=2.0)
        keep_running = bool(v6.handle_user_turn_with_cached_state(local_command_text, face_state))
        _update_mode_state_from_text(local_command_text, state)
        if not keep_running:
            state.stop_event.set()
            set_interaction_state(state, "shutdown_pending", "Shutdown")
        _mark_route_done(state, turn_started_at)
        return keep_running

    if _should_drop_filler_transcript(user_text, state, camera_intent):
        print("[V7.5 AUDIO] Dropped filler transcript.")
        set_interaction_state(state, "idle", "")
        _mark_route_done(state, turn_started_at)
        return True

    if _route_asr_ambiguity_clarification(user_text, state):
        _mark_route_done(state, turn_started_at)
        return True

    cloud_prompt_text = _recover_contextual_followup_prompt(user_text, state)
    if new_story_requested:
        story_theme = _clean_story_topic(_extract_story_theme(user_text)) or _extract_long_story_topic_hint(user_text) or "new adventure"
        story_style = _extract_story_style(user_text)
        style_clause = f" Style: {story_style}." if story_style else ""
        response_language = story_detection.get("response_language") or story_detection.get("language") or ""
        language_clause = f" Write the story in {response_language}. " if response_language else ""
        cloud_prompt_text = (
            f"Start a new complete story with this theme: {story_theme}. "
            "If the theme is vague, invent the full premise, characters, problem, obstacle, and ending. "
            f"{language_clause}"
            f"{style_clause} "
            f"User request: {user_text}"
        )
    recovered_context = cloud_prompt_text != user_text
    hard_stop_reason, hard_stop_reply = _local_safety_hard_stop(cloud_prompt_text)
    if hard_stop_reply:
        print(f"[V7.14 SAFETY ROUTER] local_hard_stop reason={hard_stop_reason}")
        _set_reply_context(state, "safety_refusal")
        _set_response_length_context(state, "terse")
        v6.speak(hard_stop_reply)
        _mark_route_done(state, turn_started_at)
        return True
    creative_fast_topic = _creative_fast_allow_topic(user_text) if is_conversation_active(state) else None
    if recovered_context and not creative_fast_topic:
        creative_fast_topic = _topic_log_label(_current_active_topic(state)) or "creative"
    if creative_fast_topic:
        start_conversation_session(
            state,
            mode="creative",
            partner=partner,
            reason="creative_fast_allow",
        )
        with state.lock:
            state.last_topic = creative_fast_topic
            state.last_topic_until = time.time() + 300.0
            if "skeleton" in creative_fast_topic:
                state.session_focus = "skeleton superhero"
        _set_reply_context(state, "creative")
        # Preserve an explicit request for a longer answer. Resetting this to
        # normal here made the speech worker cut creative replies even after
        # the user asked Miguel to give more detail.
        _set_response_length_context(state, inferred_response_mode)
        print(f"[V7.14 CREATIVE] fast_allow topic={creative_fast_topic}")
        if should_run_safety_guard(cloud_prompt_text, route_hint="creative", conversation_mode="creative"):
            start = time.time()
            decision = safety.evaluate_user_text(cloud_prompt_text)
            elapsed = time.time() - start
            if elapsed > 1.0:
                print(
                    f"[V7.5 SAFETY] Safety check took {elapsed:.2f}s. "
                    f"category={decision.category} source={decision.source}"
                )
            if not decision.allowed:
                _set_reply_context(state, "safety_refusal")
                _set_response_length_context(state, "terse")
                v6.speak(decision.safe_reply or "I can't help with that.")
                _mark_route_done(state, turn_started_at)
                return True
    else:
        with state.lock:
            safety_mode = state.conversation_mode
        if should_run_safety_guard(cloud_prompt_text, route_hint="normal", conversation_mode=safety_mode):
            start = time.time()
            decision = safety.evaluate_user_text(cloud_prompt_text)
            elapsed = time.time() - start

            if elapsed > 1.0:
                print(
                    f"[V7.5 SAFETY] Safety check took {elapsed:.2f}s. "
                    f"category={decision.category} source={decision.source}"
                )

            if not decision.allowed:
                _set_reply_context(state, "safety_refusal")
                _set_response_length_context(state, "terse")
                v6.speak(decision.safe_reply or "I can't help with that.")
                _mark_route_done(state, turn_started_at)
                return True

            if getattr(decision, "category", "") in {
                "normal_conversation",
                "ambiguous_clarification",
                "fallback_allow",
                "general_clarification",
            }:
                _set_response_length_context(state, "normal")

    # Keep this redundant scene guard from V7 Full as a last defensive check.
    if is_scene_camera_request(user_text):
        set_interaction_state(state, "looking", user_text[:48])
        _set_reply_context(state, "scene_prelude")
        v6.speak("Looking.")
        _set_reply_context(state, "scene")
        reply = full.build_scene_reply(camera_manager)
        v6.speak(reply)
        try:
            v6.update_conversation_memory(user_text=user_text, assistant_reply=reply)
        except Exception:
            pass
        _mark_route_done(state, turn_started_at)
        return True

    with state.lock:
        cloud_response_mode = state.current_turn_latency.get("response_length_mode", state.response_length_mode)
        cloud_conversation_mode = state.conversation_mode
        cloud_depth_mode = state.response_depth_mode
        cloud_allowed_languages = list(state.allowed_conversation_languages or ["english"])
        cloud_long_story_target_minutes = state.long_story_target_minutes
        cloud_story_style = state.long_story_style
        cloud_recovered_story_context = state.recovered_story_context
    cloud_route = "creative" if creative_fast_topic or cloud_conversation_mode in {"creative", "story"} else "normal"
    if cloud_conversation_mode == "story":
        cloud_route = "story"
    if new_story_requested:
        cloud_route = "story"
        cloud_response_mode = "long_story"
        cloud_depth_mode = "long_story"
        if story_detection.get("response_language"):
            cloud_allowed_languages = [story_detection.get("response_language")]
    _set_reply_context(state, cloud_route)
    active_topic_label = _topic_log_label(_current_active_topic(state))
    cloud_user_text = _with_cloud_reply_instructions(
        cloud_prompt_text,
        cloud_response_mode,
        route_hint=cloud_route,
        conversation_mode=cloud_conversation_mode,
        response_depth_mode=cloud_depth_mode,
        active_topic=active_topic_label,
        allowed_languages=cloud_allowed_languages,
        long_story_target_minutes=cloud_long_story_target_minutes,
        story_style=cloud_story_style,
        recovered_story_context=cloud_recovered_story_context,
    )
    cloud_user_text += _live_conversation_context_for_cloud(state)
    face_state = _neutral_conversation_face_state()
    keep_running = bool(v6.handle_user_turn_with_cached_state(cloud_user_text, face_state))
    _mark_route_done(state, turn_started_at)
    return keep_running


def speech_worker(
    original_speak,
    safety: SafetyGuard,
    reply_queue: queue.Queue,
    state: RobotRuntimeState,
):
    print("[V7.5 SPEECH] SpeechWorker started.")

    while not state.stop_event.is_set() or not reply_queue.empty():
        try:
            event = reply_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        text = str(event.text or "").strip()
        latency = getattr(event, "latency", {}) or {}
        context = getattr(event, "context", "normal") or "normal"
        response_length_mode = str(latency.get("response_length_mode") or "normal")
        response_depth_mode = str(latency.get("response_depth_mode") or "normal")
        route = context
        try:
            if state.stop_event.is_set() and route != "shutdown_confirm":
                print("[V7.5 SHUTDOWN] Dropped non-terminal queued reply.")
                with state.lock:
                    state.pending_reply_count = max(0, state.pending_reply_count - 1)
                continue
            if state.stop_speech_event.is_set() and route != "shutdown_confirm":
                print("[V7.5 BARGE-IN] Skipped queued reply after stop request.")
                state.stop_speech_event.clear()
                with state.lock:
                    state.pending_reply_count = max(0, state.pending_reply_count - 1)
                continue
            if route == "shutdown_confirm":
                state.stop_speech_event.clear()
            if text:
                with state.lock:
                    conversation_mode = state.conversation_mode
                    last_user_text = state.last_user_text
                    state_long_story_target_minutes = state.long_story_target_minutes
                long_story_target_minutes = int(
                    latency.get("long_story_target_minutes") or state_long_story_target_minutes or 0
                )
                if (
                    response_length_mode == "terse"
                    and conversation_mode in {"general", "creative", "story", "project", "owner_password"}
                    and route not in TERSE_ALLOWED_ROUTES
                    and text.strip().lower() not in TERSE_ALLOWED_EXACT_REPLIES
                ):
                    print(
                        f"[V7.14 LENGTH] overriding terse->normal "
                        f"route={route} conversation_mode={conversation_mode}"
                    )
                    response_length_mode = "normal"
                max_words = None
                if (
                    response_depth_mode == "long_story"
                    and _long_story_depth_applies_to_route(route, last_user_text, conversation_mode)
                    and response_length_mode != "terse"
                ):
                    explicit_long_story = _is_explicit_long_story_request(last_user_text)
                    max_words = _long_story_spoken_cap_words(explicit_long_story)
                    print(
                        f"[V7.15 LENGTH] depth=long_story allowed_words={max_words} "
                        f"target_minutes={long_story_target_minutes}"
                    )
                elif response_depth_mode == "long_story":
                    print(f"[V7.15 LENGTH] depth=long_story ignored_for_route={route}")
                    if response_length_mode == "long_story":
                        response_length_mode = "normal"
                elif (
                    response_depth_mode == "long_explanation"
                    and route not in TERSE_ALLOWED_ROUTES
                    and response_length_mode != "terse"
                ):
                    max_words = 180
                    print(f"[V7.15 LENGTH] depth=long_explanation allowed_words={max_words}")
                text = make_robot_reply_concise(
                    text,
                    max_words=max_words,
                    context=context,
                    response_length_mode=response_length_mode,
                    response_depth_mode=response_depth_mode,
                )
                print(f"[V7.14 LENGTH] mode={response_length_mode} route={route} words={_word_len(text)}")
                print(f"[V7.14 LENGTH] mode={response_length_mode} conversation_mode={conversation_mode} words={_word_len(text)}")
                if (
                    response_length_mode == "terse"
                    and conversation_mode in {"general", "story", "creative", "project", "owner_password"}
                    and route not in TERSE_ALLOWED_ROUTES
                    and text.strip().lower() not in TERSE_ALLOWED_EXACT_REPLIES
                ):
                    print("[V7.14 LENGTH WARNING] terse used in conversational mode")
                _warn_if_possible_truncation(text)
                if text in TTS_CACHE_CANDIDATES:
                    print(f"[V7.5 TTS CACHE] candidate={text}")
                decision = safety.evaluate_assistant_reply(text)
                if not decision.allowed:
                    print(
                        "[V7.5 SAFETY] Blocked assistant reply. "
                        f"category={decision.category} source={decision.source}"
                    )
                    text = decision.safe_reply or "I can't help with that."

                if state.stop_speech_event.is_set() and route != "shutdown_confirm":
                    print("[V7.5 BARGE-IN] Skipped reply before speech start.")
                    state.stop_speech_event.clear()
                    with state.lock:
                        state.pending_reply_count = max(0, state.pending_reply_count - 1)
                    continue
                if route == "shutdown_confirm":
                    state.stop_speech_event.clear()

                try:
                    with state.lock:
                        state.is_speaking = True
                        state.last_spoken_text = text
                        state.last_robot_text = text
                    prepare_started = time.monotonic()
                    _log_latency("tts_prepare_started", latency.get("turn_started_at"))
                    prepared = None
                    can_prepare = callable(getattr(v6, "prepare_speech_audio", None)) and callable(getattr(v6, "play_prepared_speech", None))
                    if can_prepare:
                        try:
                            prepared = v6.prepare_speech_audio(text, cache=text in TTS_CACHE_CANDIDATES)
                        except Exception as exc:
                            print("[V7.5 SPEECH] TTS prepare failed; falling back:", exc)
                            can_prepare = False
                    playback_started = time.monotonic()
                    latency["tts_prepare_ms"] = max(0.0, playback_started - prepare_started) * 1000
                    if latency.get("turn_started_at"):
                        latency["playback_start_ms"] = max(
                            0.0,
                            playback_started - float(latency["turn_started_at"]),
                        ) * 1000
                    with state.lock:
                        now = time.time()
                        state.last_speech_started_at = now
                        state.last_speaking_started_at = now
                    _log_latency("speak_started", latency.get("turn_started_at"))
                    set_interaction_state(state, "speaking", text[:48])
                    speak_started = time.monotonic()
                    if can_prepare:
                        v6.play_prepared_speech(prepared)
                    else:
                        original_speak(text)
                    speak_finished = time.monotonic()
                    latency["playback_ms"] = max(0.0, speak_finished - speak_started) * 1000
                    if latency.get("turn_started_at"):
                        latency["total_ms"] = max(
                            0.0,
                            speak_finished - float(latency["turn_started_at"]),
                        ) * 1000
                    _log_latency("speak_finished", latency.get("turn_started_at"))
                    _log_latency_summary(latency, text, speak_started, speak_finished)
                    _update_prompt_state(text, state)
                    _log_assistant_reply_event(state, text, route, latency_override=latency)
                    with state.lock:
                        answer_topic = state.session_focus or state.last_topic or ""
                        last_user = state.last_user_text
                        if "skeleton" in normalize_command_text(last_user) or "skeleton" in normalize_command_text(text):
                            answer_topic = "skeleton superhero"
                        state.last_answer_topic = answer_topic or None
                        state.last_answer_route = route
                        state.last_answer_text_short = _short_log_text(text, 120)
                        state.last_answer_at = time.time()
                finally:
                    with state.lock:
                        state.is_speaking = False
                        state.last_speech_finished_at = time.time()
                        current_mode = state.current_mode
                        shutdown_pending = state.shutdown_pending
                        shutdown_confirmation_pending = state.shutdown_confirmation_pending
                        state.pending_reply_count = max(0, state.pending_reply_count - 1)
                    if route == "shutdown_confirm":
                        state.shutdown_acknowledged_event.set()
                    if current_mode == "sleep":
                        set_interaction_state(state, "sleeping", "")
                    elif shutdown_confirmation_pending:
                        print("[V7.14 SHUTDOWN] pending lock active")
                        set_interaction_state(state, "shutdown_pending", "Confirm shutdown")
                    elif shutdown_pending:
                        set_interaction_state(state, "shutdown_pending", "Shutdown")
                    elif context == "scene_prelude":
                        set_interaction_state(state, "looking", "Looking")
                    else:
                        set_interaction_state(state, "idle", "")
        except Exception as exc:
            message = str(exc)
            aplay_interrupted = "Command '['aplay'" in message and "exit status 1" in message
            if state.stop_event.is_set() or "Interrupted system call" in message or aplay_interrupted:
                print("[V7.5 SPEECH] playback interrupted during shutdown/stop.")
            else:
                print("[V7.5 SPEECH] error:", exc)
                set_interaction_state(state, "error", message[:48])
        finally:
            reply_queue.task_done()

    print("[V7.5 SPEECH] SpeechWorker stopped.")


def _camera_startup_health(camera_manager, wait_timeout: float = 2.5) -> dict:
    started_at = time.monotonic()
    thread = getattr(camera_manager, "thread", None)
    thread_alive = bool(thread and thread.is_alive())
    snapshot = None
    error = ""
    try:
        snapshot = camera_manager.get_latest_frame(require_fresh=True, wait_timeout=wait_timeout)
    except Exception as exc:
        error = str(exc)
    frame_available = snapshot is not None
    captured_at = float(getattr(snapshot, "captured_at", 0.0) or 0.0) if snapshot else 0.0
    return {
        "camera_thread_alive": thread_alive,
        "fresh_frame_available": frame_available,
        "frame_age_ms": round(max(0.0, time.time() - captured_at) * 1000) if captured_at else None,
        "check_ms": round((time.monotonic() - started_at) * 1000),
        "status": "online" if thread_alive and frame_available else "degraded",
        "error": error or None,
    }


def _startup_announcement(camera_health: dict) -> str:
    if camera_health.get("status") == "online":
        return "I am Miguel, Marquinho's robot project. Camera and face recognition are online."
    return (
        "I am Miguel, Marquinho's robot project. Voice is online, "
        "but the camera is not providing fresh frames yet."
    )


def _log_incomplete_turn_on_shutdown(state: RobotRuntimeState, reason: str) -> None:
    with state.lock:
        logged_at = float(state.last_logged_user_turn_at or 0.0)
        completed_at = float(state.last_completed_user_turn_at or 0.0)
        user_text = str(state.last_logged_user_turn_text or state.last_user_text or "")
        route = str(state.current_turn_latency.get("reply_context") or "unknown")
        pending_replies = int(state.pending_reply_count or 0)
        pending_turns = int(state.pending_user_turn_count or 0)
        processing = bool(state.turn_processing_active or state.brain_is_processing)
        speaking = bool(state.is_speaking)
    if not user_text or logged_at <= completed_at:
        return
    if speaking:
        stage = "playback"
    elif pending_replies:
        stage = "reply_queue"
    elif processing or pending_turns:
        stage = "routing"
    else:
        stage = "unknown"
    _append_log_event(
        state,
        "turn_interrupted",
        user_text=user_text,
        route=route,
        reason=reason,
        stage=stage,
        elapsed_ms=round(max(0.0, time.time() - logged_at) * 1000),
        pending_replies=pending_replies,
        pending_turns=pending_turns,
        brain_processing=processing,
        speech_active=speaking,
    )


def _launch_debug_handoff() -> bool:
    script_path = THIS_DIR.parent / "tools" / "miguel_debug_last.sh"
    if not script_path.is_file():
        print(f"[V7.5 DEBUG HANDOFF] script missing: {script_path}")
        return False
    try:
        subprocess.Popen(
            [str(script_path), "--voice"],
            cwd=str(THIS_DIR.parent.parent),
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except Exception as exc:
        print(f"[V7.5 DEBUG HANDOFF] launch failed: {exc}")
        print(f"[V7.5 DEBUG HANDOFF] run manually: {script_path}")
        return False
    print(f"[V7.5 DEBUG HANDOFF] detached launcher started: {script_path}")
    return True


def _install_shutdown_signal_handlers(
    stop_event: threading.Event,
    set_termination_reason,
) -> dict[int, object]:
    if threading.current_thread() is not threading.main_thread():
        return {}
    previous_handlers: dict[int, object] = {}
    handled_signals = [signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled_signals.append(signal.SIGHUP)

    def request_shutdown(signum, _frame) -> None:
        signame = signal.Signals(signum).name.lower()
        print(f"[V7.5] {signame} received; requesting graceful shutdown.")
        set_termination_reason(signame)
        stop_event.set()

    for sig in handled_signals:
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, request_shutdown)
    return previous_handlers


def _restore_signal_handlers(previous_handlers: dict[int, object]) -> None:
    if threading.current_thread() is not threading.main_thread():
        return
    for sig, handler in previous_handlers.items():
        try:
            signal.signal(sig, handler)
        except Exception as exc:
            print(f"[V7.5] signal restore warning {sig}:", exc)


def _initialize_startup_conversation_mode(state: RobotRuntimeState) -> None:
    """Start every robot process in the ordinary conversational personality."""
    with state.lock:
        state.current_mode = "normal"
        state.conversation_mode = "wake_required"
        state.response_length_mode = "normal"
        state.response_depth_mode = "normal"
    try:
        robot_memory.set_personality_mode("normal")
    except Exception as exc:
        print("[V7.5 STARTUP] personality reset warning:", exc)


def run_v7_5_queue():
    print("======================================")
    print(" Miguel - Cloud Brain V7.5 Queue ")
    print("======================================")
    print("")
    print("V7.5 Queue Core:")
    print("  - CameraManager owns OAK queue")
    print("  - FaceWorker keeps fresh identity state")
    print("  - AudioWorker, BrainWorker, SpeechWorker use queues")
    print("  - V7 Full remains fallback")
    print("")

    full.face = full.init_optional_face()
    full.face_happy("Miguel online")

    safety = SafetyGuard()
    user_turn_queue = queue.Queue()
    reply_queue = queue.Queue()
    stop_event = threading.Event()
    state = RobotRuntimeState(stop_event=stop_event)
    _initialize_startup_conversation_mode(state)
    state.user_turn_queue = user_turn_queue
    state.reply_queue = reply_queue
    try:
        log_session = robot_memory.start_conversation_log_session()
        state.conversation_log_session_id = log_session.get("session_id")
        print(f"[V7.15 LOG] session={state.conversation_log_session_id} path={log_session.get('path')}")
    except Exception as exc:
        print("[V7.15 LOG] session start warning:", exc)
    _log_password_env_configured_once(state)
    set_interaction_state(state, "starting", "Miguel online")
    original_speak = install_speech_queue(reply_queue, safety, state)

    camera_manager = None
    threads = []
    termination_reason = "shutdown"

    def set_termination_reason(reason: str) -> None:
        nonlocal termination_reason
        if termination_reason == "shutdown":
            termination_reason = reason

    signal_handlers = _install_shutdown_signal_handlers(stop_event, set_termination_reason)

    try:
        with dai.Pipeline() as pipeline:
            camera_manager = full.create_camera_manager_from_live_pipeline(pipeline)
            camera_manager.identity_tracker = state.identity_tracker
            camera_manager.start()
            camera_health = _camera_startup_health(camera_manager)
            _append_log_event(state, "startup_health", camera=camera_health)
            print(f"[V7.15 STARTUP] camera={camera_health}")

            threads = [
                threading.Thread(
                    target=face_worker,
                    args=(camera_manager, state),
                    daemon=True,
                    name="FaceWorker",
                ),
                threading.Thread(
                    target=speech_worker,
                    args=(original_speak, safety, reply_queue, state),
                    daemon=True,
                    name="SpeechWorker",
                ),
                threading.Thread(
                    target=brain_worker,
                    args=(camera_manager, safety, user_turn_queue, state),
                    daemon=True,
                    name="BrainWorker",
                ),
            ]

            for thread in threads:
                thread.start()

            startup_speech_started_at = time.monotonic()
            _speak_with_enqueue_context(
                _startup_announcement(camera_health),
                {
                    "reply_context": "startup",
                    # Give startup announcements the same TTS/queue/playback
                    # observability as user turns.  The log event is written
                    # after playback, so without this timestamp a normal long
                    # announcement looks like an unexplained startup stall.
                    "turn_started_at": startup_speech_started_at,
                    "route_done_at": startup_speech_started_at,
                    "log_user_text": "",
                    "log_person": "unknown",
                    "log_conversation_mode": state.conversation_mode,
                    "log_topic": "",
                },
            )
            reply_queue.join()
            full.face_happy("Miguel online")

            audio_thread = threading.Thread(
                target=audio_worker,
                args=(camera_manager, user_turn_queue, state),
                daemon=True,
                name="AudioWorker",
            )
            audio_thread.start()
            threads.append(audio_thread)

            while not stop_event.is_set():
                _check_timer_tick(state)
                time.sleep(0.2)

    except KeyboardInterrupt:
        termination_reason = "keyboard_interrupt"
        print("[V7.5] Keyboard interrupt.")
    except Exception as exc:
        termination_reason = "runtime_error"
        _append_log_event(
            state,
            "runtime_error",
            error_type=type(exc).__name__,
            message=str(exc)[:500],
        )
        print("[V7.5] runtime error:", exc)

    finally:
        if termination_reason == "shutdown" and stop_event.is_set():
            termination_reason = "stop_requested"
        set_interaction_state(state, "shutdown_pending", "Stopping")
        stop_event.set()
        set_interaction_state(state, "sleeping", "Sleep")

        # A confirmed shutdown is allowed to finish its short terminal reply.
        # This keeps the confirmation turn complete in speech and in the log.
        with state.lock:
            shutdown_confirmation_queued = (
                state.current_turn_latency.get("reply_context") == "shutdown_confirm"
                and state.pending_reply_count > 0
            )
        if shutdown_confirmation_queued:
            state.shutdown_acknowledged_event.wait(
                timeout=max(0.0, _env_float("MIGUEL_SHUTDOWN_ACK_TIMEOUT_SECONDS", 5.0))
            )

        for thread in threads:
            try:
                if thread.is_alive():
                    thread.join(timeout=1.5)
            except Exception as exc:
                print(f"[V7.5] {thread.name} join warning:", exc)

        # Record the terminal turn state only after workers have had a chance
        # to finish or expose the queue/playback stage where shutdown stopped it.
        _log_incomplete_turn_on_shutdown(state, termination_reason)
        _append_log_event(state, "session_end", reason=termination_reason)
        with state.lock:
            debug_handoff_requested = bool(state.debug_handoff_requested)
        if debug_handoff_requested:
            _launch_debug_handoff()

        if camera_manager:
            camera_manager.stop()

        if full.face:
            try:
                full.face.stop()
            except Exception as exc:
                print("[face] stop warning:", exc)

        _restore_signal_handlers(signal_handlers)
        v6.speak = original_speak
        print("Miguel Cloud Brain V7.5 Queue stopped. Jetson stayed on.")


if __name__ == "__main__":
    run_v7_5_queue()
