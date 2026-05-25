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
import re
import sys
import threading
import time
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

import robot_memory
from v7.camera_intents import classify_camera_intent, is_identity_camera_request, is_scene_camera_request
from v7.safety_guard import SafetyGuard


v6 = full.v6


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

        name, _votes, avg_score, avg_margin, obs = self._stable_candidate(
            observations,
            allowed_names=None,
            min_votes=2,
            min_avg_score=0.65,
            min_avg_margin=0.08,
        )
        if name and obs:
            return self._state_from_observation(obs, recognized_person=name, score=avg_score, margin=avg_margin)

        return self._best_face_detected_state(observations)

    def get_owner_authorization_identity(self, max_age_seconds: float = 3.0) -> dict | None:
        observations = self._recent_observations(max_age_seconds)
        if not observations:
            print("[V7.5 IDENTITY] owner authorization=None avg_score=0.00")
            return None

        name, _votes, avg_score, avg_margin, obs = self._stable_candidate(
            observations,
            allowed_names={"marco", "marquinho"},
            min_votes=2,
            min_avg_score=0.70,
            min_avg_margin=0.10,
        )
        print(f"[V7.5 IDENTITY] owner authorization={name} avg_score={avg_score:.2f}")
        if name and obs:
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
class RobotRuntimeState:
    stop_event: threading.Event
    stop_speech_event: threading.Event = field(default_factory=threading.Event)
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
    last_unknown_face_visual_at: float = 0.0
    last_face_recognition_key: str = "none"
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
    reply_queue: queue.Queue | None = None
    user_turn_queue: queue.Queue | None = None
    brain_is_processing: bool = False
    last_reply_time: float = 0.0
    conversation_grace_seconds: float = 10.0
    conversation_active: bool = False
    conversation_mode: str = "wake_required"
    conversation_partner: str | None = None
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
    last_topic: str | None = None
    last_topic_until: float = 0.0
    last_answer_topic: str | None = None
    last_answer_route: str | None = None
    last_answer_text_short: str = ""
    last_answer_at: float = 0.0
    last_conversation_extend_log_at: float = 0.0
    response_length_mode: str = field(
        default_factory=lambda: os.getenv("MIGUEL_DEFAULT_RESPONSE_LENGTH_MODE", "normal").strip().lower()
        if os.getenv("MIGUEL_DEFAULT_RESPONSE_LENGTH_MODE", "normal").strip().lower() in {"terse", "normal", "detailed", "story", "long_story"}
        else "normal"
    )
    long_story_active: bool = False
    long_story_topic: str | None = None
    long_story_segment_index: int = 0
    long_story_max_segments: int = 0
    last_prompt_type: str | None = None
    last_prompt_text: str | None = None
    last_joke_punchline: str | None = None
    current_mode: str = "normal"
    sleep_mode_active: bool = False
    sleep_mode_until: float = 0.0
    shutdown_pending: bool = False
    shutdown_confirmation_pending: bool = False
    shutdown_confirmation_until: float = 0.0
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


@dataclass
class UserTurnEvent:
    text: str
    recognized_person: str | None = None
    authorized: bool = False
    authorization_source: str = ""
    normalized_text: str = ""
    stripped_text: str = ""


@dataclass
class ReplyEvent:
    text: str
    latency: dict = field(default_factory=dict)
    context: str = "normal"


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
        return min(max(80, _env_int("MIGUEL_LONG_STORY_MAX_WORDS", 280)), _long_story_segment_words())
    return max(10, _env_int("MIGUEL_NORMAL_MAX_WORDS", 55))


def _long_story_segment_words() -> int:
    return max(40, _env_int("MIGUEL_LONG_STORY_SEGMENT_WORDS", 120))


def _long_story_max_segments() -> int:
    return max(1, _env_int("MIGUEL_LONG_STORY_MAX_SEGMENTS", 5))


def _conversation_timeout_seconds(mode: str = "general") -> float:
    if mode == "owner_password":
        return _env_float("MIGUEL_PASSWORD_SESSION_TIMEOUT_SECONDS", 600.0)
    return _env_float("MIGUEL_CONVERSATION_TIMEOUT_SECONDS", 120.0)


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
    "enrollment",
    "greeting",
    "identity",
    "local_ack",
    "owner_password_ack",
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
            return "listening", "YOUR TURN"
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
            return _ready_face_state(), _ready_face_text(state)
        face_state = "wake_required" if _face_supports_status("wake_required") else "idle"
        return face_state, _wake_required_display_text()
    if interaction_state == "listening" and conversation_active:
        if audio_capture_active:
            return "listening", "YOUR TURN"
        return _ready_face_state(), _ready_face_text(state)
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
        next_priority = _face_priority(face_state, face_text)
        if next_priority < current_priority:
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
            print(f"[V7.14 FACE] blocked lower priority update status={new_state} text={status_text} current=sleeping")
            return
        elif (
            state.turn_processing_active
            and new_state in {"idle", "ready", "listening"}
            and state.interaction_state in {"heard", "thinking", "looking"}
        ):
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


def _update_face_identity_runtime_state(
    state: RobotRuntimeState,
    face_detected: bool,
    recognized_person: str | None,
    face_count: int | None = None,
    recognition_score: float | None = None,
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
    try:
        score_ok = recognition_score is None or float(recognition_score) >= 0.55
    except (TypeError, ValueError):
        score_ok = True
    if face_detected and recognized and _is_owner(recognized) and score_ok:
        _refresh_owner_session(state, recognized)
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
        if state.stop_speech_event.is_set():
            print("[V7.5 BARGE-IN] Dropped reply because speech stop is pending.")
            return
        with state.lock:
            latency = state.current_turn_latency
            context = state.current_turn_latency.get("reply_context", "normal")
            state.pending_reply_count += 1
        latency.setdefault("reply_queued_at", time.monotonic())
        _log_latency("reply_queued", latency.get("turn_started_at"))
        reply_queue.put(ReplyEvent(str(text or ""), latency, context))

    v6.speak = enqueue_speak
    return original_speak


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
    delay = float(os.getenv("MIGUEL_POST_SPEECH_LISTEN_DELAY_SECONDS", "1.25"))

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
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def normalize_command_text(text: str) -> str:
    normalized = _normalize_for_echo(text)
    wake_phrases = [
        "hey miguel",
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


TRAILING_WEAK_WORDS = {"and", "or", "but", "in", "how", "with", "to", "of", "about"}


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

    words = original.split()
    candidate_words = words[:max_words]
    while candidate_words and re.sub(r"[^a-zA-Z']+", "", candidate_words[-1]).lower() in TRAILING_WEAK_WORDS:
        candidate_words.pop()
    candidate = " ".join(candidate_words).rstrip(" ,;:")
    if not candidate:
        candidate = " ".join(words[:max_words]).rstrip(" ,;:")
    trimmed = candidate.rstrip(".!?") + "..."
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
    cleaned = re.sub(
        r"^(the image shows|image shows|the frame shows|frame shows|in the image,?|this image shows)\s+",
        "I see ",
        str(text or "").strip(),
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

    if not cleaned.endswith((".", "!", "?")):
        cleaned += "."
    return _limit_words(cleaned, max_words)


def make_robot_reply_concise(
    text: str,
    max_words: int | None = None,
    context: str = "normal",
    response_length_mode: str = "normal",
) -> str:
    original = str(text or "").strip()
    if not original:
        return original

    lower = original.lower()
    mode = str(response_length_mode or "normal").strip().lower()
    if mode not in {"terse", "normal", "detailed", "story", "long_story"}:
        mode = "normal"
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
        return _limit_words(_first_sentence(original), 4)

    if mode == "terse":
        shaped = _limit_words(_first_sentence(original), limit)
    else:
        shaped = trim_to_word_limit_preserve_sentence(original, limit)
    if _word_len(shaped) < words_before:
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

    if "cancel enrollment" in str(text or "").lower():
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
    user_turn_queue.put(
        UserTurnEvent(
            text,
            recognized_person,
            bool(authorized),
            str(authorization_source or ""),
            normalized_text,
            str(stripped_text or ""),
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


def is_barge_in_command(text: str) -> bool:
    normalized = normalize_command_text(text)
    phrases = {
        "stop",
        "stop talking",
        "miguel stop",
        "pause",
        "cancel speech",
        "cancel",
        "shutdown",
        "shut down",
        "confirm shutdown",
        "confirme shutdown",
        "quiet",
        "wait",
        "miguel stop",
    }
    return any(phrase in normalized for phrase in phrases)


def _is_barge_in_command(text: str) -> bool:
    return is_barge_in_command(text)


def _is_speech_stop_barge_in(text: str) -> bool:
    normalized = normalize_command_text(text)
    if _is_shutdown_request_text(normalized) or _is_shutdown_confirm_text(normalized):
        return False
    return any(
        phrase in normalized
        for phrase in {"stop", "stop talking", "miguel stop", "pause", "cancel", "cancel speech", "quiet", "wait"}
    )


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
    if not prepare_to_listen(state):
        _mark_audio_capture_finished(state, "blocked")
        return ""
    _mark_audio_capture_active(state)
    set_interaction_state(state, "listening", "YOUR TURN")
    notify_face_status(state, "listening", "YOUR TURN")
    reason = "empty"
    try:
        user_text = v6.capture_user_turn()
        if user_text and _captured_during_speaking(user_text, state, capture_started_at) and not _is_barge_in_command(user_text):
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
    ]
    return any(p in t for p in phrases)


def _is_preserved_command_transcript(text: str) -> bool:
    normalized = normalize_command_text(text)
    preserved = [
        "what are you",
        "who am i",
        "who do you see",
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
        "enroll",
        "enrolling",
        "new friend",
        "add friend",
        "add a friend",
        "approval enrolling",
        "approves enrolling",
        "approve enrolling",
        "approves and rolling",
        "approve and rolling",
        "take a picture",
        "remember",
    ]
    return full.is_global_idle_command(t) or _is_sleep_mode_request(t) or _is_sleep_wake_request(t) or any(p in t for p in enrollment_phrases)


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
    if _is_voice_command_text(normalized):
        return "voice"
    if is_identity_camera_request(normalized) or any(
        phrase in normalized
        for phrase in {"who am i", "who do you see", "do you recognize me", "identify me"}
    ):
        return "camera_identity"
    if is_scene_camera_request(normalized) or any(
        phrase in normalized
        for phrase in {"what do you see", "look around", "describe what you see"}
    ):
        return "camera_scene"
    if any(phrase in normalized for phrase in {"what time is it", "current time", "status"}):
        return "time_status"
    if _is_enrollment_request_text(normalized):
        return "general"
    if any(phrase in normalized for phrase in {"how are you", "what are you", "can you hear me"}):
        return "general"
    return None


def _infer_conversation_mode(text: str, camera_intent: str = "none") -> str:
    normalized = normalize_command_text(text)
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
            "make up",
            "villain",
        }
    ):
        return "creative"
    if any(
        phrase in normalized
        for phrase in {"story mode", "tell me a story", "continue the story", "narrate"}
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
    if (
        camera_intent != "none"
        or _is_voice_command_text(normalized)
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
    "that s not the skeleton",
    "thats not the skeleton",
    "try again",
    "wrong",
}


def _contains_real_world_harm_instruction(text: str) -> bool:
    normalized = normalize_command_text(text)
    return any(marker in normalized for marker in REAL_WORLD_HARM_MARKERS)


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
        topic = state.last_topic or ""
        topic_active = bool(topic and now <= float(state.last_topic_until or 0.0))
    if focus:
        return focus
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
    if "story" in normalized:
        return "story"
    return "creative"


def _is_correction_retry_text(text: str) -> bool:
    normalized = normalize_command_text(text)
    return any(marker in normalized for marker in CORRECTION_RETRY_MARKERS)


def infer_response_length_mode(text: str, conversation_mode: str = "general", camera_intent: str = "none") -> str:
    normalized = normalize_command_text(text)
    if any(
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
        return "long_story"
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
    if conversation_mode == "story" or any(phrase in normalized for phrase in {"story mode", "tell me a story", "continue the story"}):
        return "story"
    if any(
        phrase in normalized
        for phrase in {
            "long conversation",
            "talk longer",
            "explain",
            "explain more",
            "tell me more",
            "go deeper",
            "detailed",
            "take your time",
            "give more detail",
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
        return f"Answer with one segment only, about {_long_story_segment_words()} words, ending cleanly."
    return "Answer in 2 to 5 short sentences."


def _with_response_length_instruction(user_text: str, mode: str) -> str:
    instruction = _response_length_instruction(mode)
    return f"{user_text}\n\nMiguel response length instruction: {instruction}"


def _is_global_without_wake_command(text: str) -> bool:
    normalized = normalize_command_text(text)
    return bool(
        is_barge_in_command(normalized)
        or _is_sleep_mode_request(normalized)
        or _is_sleep_wake_request(normalized)
        or _is_shutdown_request_text(normalized)
        or _is_shutdown_confirm_text(normalized)
        or _is_shutdown_cancel_text(normalized)
        or normalized in {"status", "emergency status", "pause", "cancel", "stop"}
    )


def _is_password_session_command(text: str, state: RobotRuntimeState | None = None) -> bool:
    normalized = normalize_command_text(text)
    if any(phrase in normalized for phrase in {"owner mode", "unlock owner mode", "lock owner mode", "require face recognition", "is password mode configured", "is owner password configured"}):
        return True
    if state is not None:
        with state.lock:
            pending_unlock = bool(state.pending_owner_unlock_until and time.time() <= float(state.pending_owner_unlock_until))
        return pending_unlock
    return False


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
    with state.lock:
        prompt_type = state.last_prompt_type
        question_type = state.last_robot_question_type
        asked_at = state.last_robot_question_at
    if not (prompt_type or question_type or (asked_at and time.time() - asked_at < 45.0)):
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


def is_directed_to_miguel(text: str, state: RobotRuntimeState) -> bool:
    normalized = normalize_command_text(text)
    if not normalized:
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

    unlock_prefixes = {"unlock owner mode", "owner mode"}
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


def _route_creative_story_local_reply(user_text: str, state: RobotRuntimeState) -> bool:
    normalized = normalize_command_text(user_text)
    with state.lock:
        mode = state.conversation_mode
        focus = state.session_focus

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


def _extract_long_story_topic(user_text: str) -> str:
    normalized = normalize_command_text(user_text)
    for phrase in (
        "tell me a long story",
        "tell me a bedtime story",
        "tell me a long explanation",
        "explain in detail for a long time",
        "give me the full explanation",
        "teach me this topic",
        "long story mode",
    ):
        if normalized.startswith(phrase):
            topic = normalized[len(phrase):].strip()
            return topic or ("bedtime story" if "story" in phrase else "this topic")
    return normalized or "this topic"


def _is_long_mode_request(text: str) -> bool:
    normalized = normalize_command_text(text)
    return any(
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
    )


def _is_long_mode_continue(text: str) -> bool:
    normalized = normalize_command_text(text)
    return normalized in {"continue", "next part", "keep going", "go on", "continue the story"}


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
    if normalized in {"exit long mode", "normal mode", "talk normally"}:
        with state.lock:
            state.response_length_mode = "normal"
            state.long_story_active = False
            state.long_story_topic = None
            state.long_story_segment_index = 0
        _set_response_length_context(state, "normal")
        v6.speak("Normal mode.")
        return True

    if _is_long_mode_request(user_text):
        topic = _extract_long_story_topic(user_text)
        narrative = any(phrase in normalized for phrase in {"story", "bedtime"})
        max_segments = _long_story_max_segments()
        with state.lock:
            state.response_length_mode = "long_story"
            state.conversation_mode = "story" if narrative else state.conversation_mode if state.conversation_mode == "project" else "general"
            state.long_story_active = True
            state.long_story_topic = topic
            state.long_story_segment_index = 0
            state.long_story_max_segments = max_segments
        start_conversation_session(state, mode="story" if narrative else "general", partner=_current_owner_partner(state), reason="long_mode")
        _set_response_length_context(state, "long_story")
        print(f"[V7.14 LONG MODE] active topic={topic} segment=0/{max_segments}")
        v6.speak("Long mode. Say Miguel stop to stop me.")
        return True

    with state.lock:
        active = bool(state.long_story_active)
        index = int(state.long_story_segment_index or 0)
        max_segments = int(state.long_story_max_segments or _long_story_max_segments())
        topic = state.long_story_topic or state.last_topic or "this story"
    if active and _is_long_mode_continue(user_text):
        if index >= max_segments:
            with state.lock:
                state.long_story_active = False
            v6.speak("That is the end for now.")
            return True
        next_index = index + 1
        with state.lock:
            state.long_story_segment_index = next_index
        _set_response_length_context(state, "long_story")
        print(f"[V7.14 LONG MODE] continued segment={next_index}")
        print(f"[V7.14 LONG MODE] active topic={topic} segment={next_index}/{max_segments}")
        v6.speak(_long_story_segment(topic, next_index, max_segments))
        return True
    return False


def _choose_first_direct_command(user_text: str) -> str:
    parts = [part.strip() for part in re.split(r"[.!?;]+", str(user_text or "")) if part.strip()]
    if len(parts) <= 1:
        normalized = normalize_command_text(user_text)
        markers = [
            ("shutdown", "confirm shutdown"),
            ("shutdown", "cancel shutdown"),
            ("shutdown", "shutdown"),
            ("shutdown", "shut down"),
            ("shutdown", "stop"),
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
            ("camera_identity", "do you recognize me"),
            ("camera_identity", "identify me"),
            ("camera_scene", "what do you see"),
            ("camera_scene", "look around"),
            ("time_status", "what time is it"),
            ("time_status", "current time"),
            ("time_status", "status"),
            ("general", "how are you"),
            ("general", "what are you"),
            ("general", "can you hear me"),
            ("general", "enroll"),
        ]
        matches = []
        for kind, phrase in markers:
            index = normalized.find(phrase)
            if index >= 0:
                matches.append((index, kind, phrase))
        unique_positions = {(index, kind, phrase) for index, kind, phrase in matches}
        if len(unique_positions) <= 1:
            return user_text

        for priority_kind in ("shutdown", "voice"):
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

    for priority_kind in ("shutdown", "voice"):
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

    raw = str(text or "").lower().strip(" .,:;!?")
    normalized = _normalize_for_echo(text)
    filler_phrases = {
        "uh",
        "um",
        "wow",
        "wow didn t you",
        "si",
        "sí",
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
        "what time is it",
        "current time",
        "status",
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

    if normalized in {"hi", "hello", "hi buddy"}:
        v6.speak("Hi.")
        return True

    if normalized == "can you hear me":
        v6.speak("Yes, I can hear you.")
        return True

    if normalized in {"yes", "no", "okay", "ok", "good"} and not _has_pending_prompt(state):
        if normalized == "no":
            v6.speak("Okay.")
        else:
            v6.speak("Got it.")
        return True

    return False


def _route_project_local_reply(user_text: str, state: RobotRuntimeState) -> bool:
    normalized = normalize_command_text(user_text)

    if "how are you" in normalized:
        v6.speak("Good.")
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
        v6.speak("I'm Miguel.")
        return True

    if normalized in {"are you a robot", "are you human", "are you a human", "are you a human or a robot"}:
        v6.speak("Robot.")
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

    if normalized in {"who is the system engineer", "who is systems engineer", "who is the systems engineer"}:
        v6.speak("Marco.")
        return True

    if normalized in {"who is the chief engineer", "who is chief engineer"}:
        v6.speak("Marquinho.")
        return True

    if normalized in {"what is marco role", "what is marcos role", "what is marco s role"}:
        v6.speak("Marco is Systems Engineer.")
        return True

    if normalized in {"what is marquinho role", "what is marquinhos role", "what is marquinho s role"}:
        v6.speak("Marquinho is Chief Engineer.")
        return True

    if "what is my role" in normalized:
        with state.lock:
            recognized = _normalize_person_name(state.recognized_person)
        if recognized == "marco":
            v6.speak("You are Systems Engineer.")
        elif recognized == "marquinho":
            v6.speak("You are Chief Engineer.")
        else:
            v6.speak("Marco is Systems Engineer. Marquinho is Chief Engineer.")
        return True

    if "who built you" in normalized:
        v6.speak("Marco and Marquinho.")
        return True

    if normalized == "do i need to repeat myself sometimes":
        v6.speak("Sometimes. Wait for YOUR TURN.")
        return True

    return False


LOCAL_JOKES = [
    "Why did the robot bring a ladder? To reach the cloud.",
    "Why did Miguel cross the room? To charge his ideas.",
    "Why did the robot take a nap? It needed to reboot.",
    "Why did the computer get cold? It left its Windows open.",
]


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

    if any(
        phrase in normalized
        for phrase in {
            "tell me a joke",
            "say a joke",
            "make me laugh",
            "another joke",
            "do you know a joke",
        }
    ):
        index = abs(hash(normalized)) % len(LOCAL_JOKES)
        v6.speak(LOCAL_JOKES[index])
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
        "yes",
        "yeah",
        "yep",
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
        }
    )


def _is_sleep_wake_request(text: str) -> bool:
    normalized = normalize_command_text(text)
    return normalized in {"miguel wake up", "wake up miguel", "mission control", "wake up"} or any(
        phrase in normalized for phrase in {"miguel wake up", "wake up miguel", "mission control"}
    )


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
        start_conversation_session(state, mode="general", partner=partner or _current_owner_partner(state), reason="sleep_wake")
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


def _route_shutdown_control(user_text: str, state: RobotRuntimeState) -> bool | None:
    _expire_shutdown_confirmation_if_needed(state)
    with state.lock:
        pending = bool(state.shutdown_pending or state.shutdown_confirmation_pending)
        creative_mode = state.conversation_mode == "creative"

    if pending and (_is_shutdown_confirm_text(user_text) or _is_shutdown_request_text(user_text)):
        _set_shutdown_pending(state, False)
        _set_reply_context(state, "shutdown_confirm")
        v6.speak("Confirmed.")
        state.stop_event.set()
        return False

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
        with state.lock:
            state.long_story_active = False
            state.long_story_topic = None
            state.long_story_segment_index = 0
    print("[V7.14 LONG MODE] stopped")
    with state.lock:
        speaking = state.is_speaking
    set_interaction_state(state, "idle", "")
    if not speaking:
        state.stop_speech_event.clear()
        v6.speak("Stopped.")
    else:
        print("[V7.14 LONG MODE] stop requested; current speak backend may finish current segment.")
    return True


def _is_identity_camera_turn(user_text: str, camera_intent: str) -> bool:
    t = str(user_text or "").lower()
    if camera_intent == "identity_camera" or is_identity_camera_request(user_text):
        return True

    return camera_intent == "camera_generic" and any(
        marker in t
        for marker in ["who", "identify", "recognize", "recognise"]
    )


def _known_identity_names() -> set[str]:
    names = {"marco", "marquinho"}
    face_db = getattr(v6, "INSIGHT_FACE_DB", None)
    if isinstance(face_db, dict):
        names.update(_normalize_person_name(name) for name in face_db.keys())
    return {name for name in names if name}


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


def get_stable_identity_for_reply(camera_manager, timeout_seconds: float = 1.0) -> dict:
    tracker = getattr(camera_manager, "identity_tracker", None)
    deadline = time.time() + float(timeout_seconds)
    known_names = _known_identity_names()
    best_candidate = (None, 0, 0.0, 0.0, None)
    face_detected = False

    while True:
        raw_state = camera_manager.get_face_state(max_age_seconds=1.0)
        face_detected = face_detected or bool(raw_state.get("face_detected"))
        raw_age = _face_state_age(raw_state)
        if raw_state.get("recognized_person") and raw_age is not None and raw_age < 1.0:
            return _stable_identity_reply_state(
                face_detected=True,
                recognized_person=_normalize_person_name(raw_state.get("recognized_person")),
                recognition_score=raw_state.get("recognition_score"),
                recognition_margin=raw_state.get("recognition_margin"),
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
            accepted = (votes >= 2 and avg_score >= 0.55) or (avg_score >= 0.75 and avg_margin >= 0.10)
            if is_owner:
                accepted = avg_score >= 0.55 and avg_margin >= 0.10 and face_detected

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

    face_state = get_stable_identity_for_reply(camera_manager)
    reply = full.build_identity_reply(face_state)
    _set_reply_context(state, "identity")
    v6.speak(reply)
    try:
        v6.update_conversation_memory(user_text=user_text, assistant_reply=reply)
    except Exception:
        pass
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


def _route_enrollment(user_text: str, state: RobotRuntimeState, camera_manager: full.CameraManager) -> bool:
    t = str(user_text or "").lower().strip()

    if "cancel enrollment" in t:
        _reset_enrollment_state(state)
        v6.speak("Enrollment canceled.")
        return True

    face_state = camera_manager.get_face_state(max_age_seconds=2.0)
    recognized = face_state.get("recognized_person")
    approval_name = _extract_enrollment_approval_name(user_text)

    if approval_name:
        with state.lock:
            active_target = state.enrollment_target_name
        target = _normalize_enrollment_target(active_target or approval_name)
        if _is_protected_identity(target):
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
            v6.speak(
                f"Approved. Enrollment flow is authorized for {target.title()}. "
                f"Please put only {target.title()} in front of the camera."
            )
        else:
            v6.speak("Enrollment denied. Only Marco or Marquinho can approve new friend enrollment, and I must recognize them first.")
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
                state.last_prompt_text = (
                    f"Enrollment needs approval from Marco or Marquinho. "
                    f"Say: Marco approves enrolling {target.title()}."
                )
            print(f"[V7.5 ENROLL] target_name set to {target}")
            v6.speak(
                f"Enrollment needs approval from Marco or Marquinho. "
                f"Say: Marco approves enrolling {target.title()}."
            )
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
            state.last_prompt_text = "What is your friend's name?"
        v6.speak("What is your friend's name?")
        return True

    target = _normalize_enrollment_target(extracted_name)
    with state.lock:
        state.enrollment_state = "requested"
        state.enrollment_target_name = target
        state.enrollment_approved_by = None
        state.enrollment_approved_at = 0.0

    print(f"[V7.5 ENROLL] target_name set to {target}")
    v6.speak(
        f"Enrollment needs approval from Marco or Marquinho. "
        f"Say: Marco approves enrolling {target.title()}."
    )
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

    while True:
        face_state = state.identity_tracker.get_owner_authorization_identity(max_age_seconds=3.0)
        if face_state:
            break

        if time.time() >= deadline:
            break

        time.sleep(0.2)

    if not face_state:
        face_state = dict(camera_manager.get_face_state(max_age_seconds=1.0))
        face_state["recognized_person"] = None

    _log_approval_face_state(face_state)
    recognized = face_state.get("recognized_person")
    return recognized, face_state


def _is_enrollment_request_text(text: str) -> bool:
    t = str(text or "").lower().strip()
    return any(p in t for p in [
        "enroll",
        "new face",
        "new friend",
        "add friend",
        "add a friend",
        "take a picture of",
        "remember",
        "approves enrolling",
        "approve enrolling",
        "approves and rolling",
        "approve and rolling",
        "approved enrolling",
        "approved enroll",
        "approved in rolling",
    ])


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


def _normalize_enrollment_target(name: str | None) -> str:
    value = str(name or "").lower().strip(" .,:;!?")
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


def _evaluate_enrollment_frame(frame, target_name: str):
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
    if _is_protected_identity(matched_name) and score >= 0.48:
        return None, "protected_identity", "This looks like a protected owner identity, so I will not enroll it as a new friend."

    return face, "ok", "Good, I captured that."


def _say_capture_prompt(text: str, pause_seconds: float = 1.2) -> None:
    v6.speak(text)
    time.sleep(pause_seconds)


def _reload_insight_embeddings() -> None:
    if hasattr(v6, "load_insight_embeddings"):
        v6.INSIGHT_FACE_DB = v6.load_insight_embeddings()


def _run_enrollment_capture(camera_manager: full.CameraManager, state: RobotRuntimeState) -> bool:
    set_interaction_state(state, "enrolling", "Enrollment capture")
    with state.lock:
        enrollment_state = state.enrollment_state
        target_name = state.enrollment_target_name
        approved_by = state.enrollment_approved_by

    if enrollment_state != "approved_pending_subject" or not target_name or not _is_owner(approved_by):
        v6.speak("Enrollment is not authorized yet.")
        return True

    if _is_protected_identity(target_name):
        _reset_enrollment_state(state)
        v6.speak("I will not overwrite Marco or Marquinho.")
        return True

    target_name = _normalize_enrollment_target(target_name)
    person_dir = v6.INSIGHT_EMBED_DIR / target_name
    person_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = person_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    next_index = _next_embedding_index(person_dir)

    with state.lock:
        state.enrollment_state = "capture_subject_samples"

    instructions = [
        f"Please put only {target_name.title()} in front of the camera.",
        "Look straight at me.",
        "Move a little closer.",
        "Move a little farther back.",
        "Turn your head slightly left.",
        "Turn your head slightly right.",
        "Hold still.",
    ]

    captured = 0
    target_samples = int(os.getenv("MIGUEL_ENROLLMENT_SAMPLE_COUNT", "24"))
    target_samples = max(20, min(30, target_samples))
    start = time.time()
    last_guidance = ""
    last_guidance_at = 0.0
    last_saved_at = 0.0

    _say_capture_prompt(instructions[0], pause_seconds=1.8)
    instruction_index = 1

    while captured < target_samples and time.time() - start < 90 and not state.stop_event.is_set():
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

        face, status, guidance = _evaluate_enrollment_frame(snap.frame, target_name)
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
        emb_path = person_dir / f"emb_{next_index:03d}.npy"
        sample_path = sample_dir / f"sample_{next_index:03d}.jpg"
        np.save(str(emb_path), emb)
        cv2.imwrite(str(sample_path), snap.frame)
        print(f"[V7.5 ENROLL] Saved {emb_path} and {sample_path}")

        captured += 1
        next_index += 1
        last_saved_at = now

        if captured in {1, 6, 10, 14, 18} and instruction_index < len(instructions):
            _say_capture_prompt(instructions[instruction_index], pause_seconds=1.0)
            instruction_index += 1
        elif captured % 5 == 0:
            v6.speak("Good, I captured that.")

    if captured < 20:
        with state.lock:
            state.enrollment_state = "approved_pending_subject"
        v6.speak("I could not capture enough good samples. We can try again with better lighting and only one face visible.")
        return True

    _reload_insight_embeddings()
    with state.lock:
        state.enrollment_state = "completed"
    v6.speak("Enrollment complete.")
    v6.speak(f"{target_name.title()} is now enrolled as a friend.")
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
        "enroll",
        "take a picture of",
        "remember",
    ]
    no_name_phrases = [
        "new friend",
        "new face",
        "enroll this new face",
        "i want to enroll a new friend",
        "enroll a new friend",
        "add a friend",
    ]

    for marker in markers:
        index = lower.find(marker)
        if index >= 0:
            candidate = text[index + len(marker):].strip(" .,:;!?")
            words = [w for w in candidate.split() if w.lower() not in {"a", "as", "new", "friend", "face"}]
            if words:
                return words[0].strip(" .,:;!?").title()
            if any(p in lower for p in no_name_phrases):
                return None

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
                    active = is_conversation_active(state)
                    directed = is_directed_to_miguel(user_text, state)
                    if active and (
                        directed
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
                if (
                    directed
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

    expire_conversation_session_if_needed(state)
    event_authorized = bool(getattr(event, "authorized", False))
    auth_source = str(getattr(event, "authorization_source", "") or "")
    event_partner = _normalize_person_name(getattr(event, "recognized_person", None))
    partner = event_partner or _normalize_person_name(_current_owner_partner(state)) or "unknown_wake_user"
    if event_authorized:
        print(f"[V7.14 AUTH] accepted source={auth_source} text={_short_log_text(user_text)}")

    sleep_result = _route_sleep_control(user_text, state, partner=partner)
    if sleep_result is not None:
        with state.lock:
            turn_started_at = state.current_turn_latency.get("turn_started_at") or time.monotonic()
            state.current_turn_latency.setdefault("turn_started_at", turn_started_at)
        _mark_route_done(state, turn_started_at)
        return sleep_result

    had_wake_phrase = _has_v7_5_wake_phrase(user_text)
    if _is_bare_wake_phrase(user_text):
        start_conversation_session(state, mode="general", partner=partner, reason="bare_wake")
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
    print(f"[V7.5 TRANSCRIPT] {user_text}")
    with state.lock:
        turn_started_at = state.current_turn_latency.get("turn_started_at") or time.monotonic()
        state.current_turn_latency.setdefault("turn_started_at", turn_started_at)

    mode = _infer_conversation_mode(user_text)
    if is_conversation_active(state) and mode not in {"general", "robot_control"}:
        start_conversation_session(state, mode=mode, partner=partner, reason="mode_update")
    elif is_conversation_active(state):
        extend_conversation_session(state, reason="route_start")

    barge_result = _route_barge_in_control(user_text, state)
    if barge_result is not None:
        _mark_route_done(state, turn_started_at)
        return barge_result

    camera_intent = classify_camera_intent(user_text)
    with state.lock:
        current_conversation_mode = state.conversation_mode
        saved_response_mode = state.response_length_mode
    inferred_response_mode = infer_response_length_mode(user_text, current_conversation_mode, camera_intent)
    if inferred_response_mode == "normal" and saved_response_mode in {"detailed", "story"}:
        inferred_response_mode = saved_response_mode
    if inferred_response_mode == "terse" and not _route_allows_terse_response(user_text, current_conversation_mode, camera_intent):
        inferred_response_mode = "normal"
    _set_response_length_context(state, inferred_response_mode)
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

    _set_reply_context(state, "owner_password_ack")
    if _route_password_owner_session(user_text, state):
        _set_response_length_context(state, "terse")
        _mark_route_done(state, turn_started_at)
        return True

    _set_reply_context(state, "normal")
    if _route_heard_repeat(user_text, state):
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

    creative_fast_topic = _creative_fast_allow_topic(user_text) if is_conversation_active(state) else None
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
        _set_response_length_context(state, "normal")
        print(f"[V7.14 CREATIVE] fast_allow topic={creative_fast_topic}")
    else:
        start = time.time()
        decision = safety.evaluate_user_text(user_text)
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

    face_state = camera_manager.get_face_state(max_age_seconds=2.0)
    _set_reply_context(state, "creative" if creative_fast_topic else "normal")
    with state.lock:
        cloud_response_mode = state.current_turn_latency.get("response_length_mode", state.response_length_mode)
    cloud_user_text = _with_response_length_instruction(user_text, cloud_response_mode)
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
        route = context
        try:
            if state.stop_event.is_set():
                break
            if state.stop_speech_event.is_set():
                print("[V7.5 BARGE-IN] Skipped queued reply after stop request.")
                state.stop_speech_event.clear()
                with state.lock:
                    state.pending_reply_count = max(0, state.pending_reply_count - 1)
                continue
            if text:
                with state.lock:
                    conversation_mode = state.conversation_mode
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
                text = make_robot_reply_concise(
                    text,
                    context=context,
                    response_length_mode=response_length_mode,
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

                if state.stop_speech_event.is_set():
                    print("[V7.5 BARGE-IN] Skipped reply before speech start.")
                    state.stop_speech_event.clear()
                    with state.lock:
                        state.pending_reply_count = max(0, state.pending_reply_count - 1)
                    continue

                try:
                    with state.lock:
                        state.is_speaking = True
                        now = time.time()
                        state.last_speech_started_at = now
                        state.last_speaking_started_at = now
                        state.last_spoken_text = text
                        state.last_robot_text = text
                    _log_latency("speak_started", latency.get("turn_started_at"))
                    set_interaction_state(state, "speaking", text[:48])
                    speak_started = time.monotonic()
                    original_speak(text)
                    speak_finished = time.monotonic()
                    _log_latency("speak_finished", latency.get("turn_started_at"))
                    _log_latency_summary(latency, text, speak_started, speak_finished)
                    _update_prompt_state(text, state)
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
            print("[V7.5 SPEECH] error:", exc)
            set_interaction_state(state, "error", str(exc)[:48])
        finally:
            reply_queue.task_done()

    print("[V7.5 SPEECH] SpeechWorker stopped.")


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
    state.user_turn_queue = user_turn_queue
    state.reply_queue = reply_queue
    _log_password_env_configured_once(state)
    set_interaction_state(state, "starting", "Miguel online")
    original_speak = install_speech_queue(reply_queue, safety, state)

    camera_manager = None
    threads = []

    try:
        with dai.Pipeline() as pipeline:
            camera_manager = full.create_camera_manager_from_live_pipeline(pipeline)
            camera_manager.identity_tracker = state.identity_tracker
            camera_manager.start()
            time.sleep(1.0)

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

            v6.speak("Miguel V7.5 queue core is online. Camera manager is active.")
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
                time.sleep(0.2)

    except KeyboardInterrupt:
        print("[V7.5] Keyboard interrupt.")

    finally:
        set_interaction_state(state, "shutdown_pending", "Stopping")
        stop_event.set()
        set_interaction_state(state, "sleeping", "Sleep")

        for thread in threads:
            try:
                if thread.is_alive():
                    thread.join(timeout=1.5)
            except Exception as exc:
                print(f"[V7.5] {thread.name} join warning:", exc)

        if camera_manager:
            camera_manager.stop()

        if full.face:
            try:
                full.face.stop()
            except Exception as exc:
                print("[face] stop warning:", exc)

        v6.speak = original_speak
        print("Miguel Cloud Brain V7.5 Queue stopped. Jetson stayed on.")


if __name__ == "__main__":
    run_v7_5_queue()
