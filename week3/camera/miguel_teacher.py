"""
Local Miguel Teacher Mode.

This module stays independent from camera, cloud, and TTS plumbing so new
subject teachers can be added without changing the robot routing layer.
"""

from __future__ import annotations

import random
import json
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SUBJECTS = {"math", "english", "history", "science", "programming"}
MATH_SUBTOPICS = {
    "multiplication": "multiplication",
    "multiplikation": "multiplication",
    "multiplicacao": "multiplication",
    "multiplicação": "multiplication",
    "multiply": "multiplication",
    "math table": "multiplication",
    "math tables": "multiplication",
    "table": "multiplication",
    "tables": "multiplication",
    "times table": "multiplication",
    "times tables": "multiplication",
    "times": "multiplication",
    "addition": "addition",
    "add": "addition",
    "subtraction": "subtraction",
    "subtract": "subtraction",
    "division": "division",
    "divide": "division",
    "fractions": "fractions",
    "fraction": "fractions",
}

NUMBER_WORDS = {
    "zero": 0,
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
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}

ANSWER_TYPES = {"number", "text", "multiple_choice"}
NEXT_ACTIONS = {"wait_for_answer", "continue_lesson", "ask_user_choice"}
DIFFICULTIES = {"easy", "normal", "hard"}
KNOWN_USER_IDS = {"marquinho": "Marquinho", "marco": "Marco"}


@dataclass(frozen=True)
class TeacherTopicLevel:
    subject: str
    topic: str
    level_number: int
    level_name: str
    description: str
    min_age: int
    max_age: int | None
    exercise_style: str
    local_supported: bool
    cloud_creative_supported: bool


MULTIPLICATION_LEVELS: dict[int, TeacherTopicLevel] = {
    1: TeacherTopicLevel(
        subject="math",
        topic="multiplication",
        level_number=1,
        level_name="Beginner Tables",
        description="1 by 1 through 10 by 10, with repeated addition.",
        min_age=6,
        max_age=None,
        exercise_style="direct facts and simple robot object problems",
        local_supported=True,
        cloud_creative_supported=False,
    ),
    2: TeacherTopicLevel(
        subject="math",
        topic="multiplication",
        level_number=2,
        level_name="Full Table Fluency",
        description="1 by 1 through 12 by 12, focused on harder facts.",
        min_age=7,
        max_age=None,
        exercise_style="mixed facts and short word problems",
        local_supported=True,
        cloud_creative_supported=True,
    ),
    3: TeacherTopicLevel(
        subject="math",
        topic="multiplication",
        level_number=3,
        level_name="Word Problems",
        description="Find multiplication inside short stories.",
        min_age=8,
        max_age=None,
        exercise_style="robot, engineering, and everyday word problems",
        local_supported=True,
        cloud_creative_supported=True,
    ),
    4: TeacherTopicLevel(
        subject="math",
        topic="multiplication",
        level_number=4,
        level_name="Multi-Step Problems",
        description="Multiplication plus addition or subtraction.",
        min_age=9,
        max_age=None,
        exercise_style="multi-step robot and engineering problems",
        local_supported=True,
        cloud_creative_supported=True,
    ),
    5: TeacherTopicLevel(
        subject="math",
        topic="multiplication",
        level_number=5,
        level_name="Challenge Mission Mode",
        description="Creative adaptive Chief Engineer missions.",
        min_age=9,
        max_age=None,
        exercise_style="robot factory, space rescue, and engineering missions",
        local_supported=True,
        cloud_creative_supported=True,
    ),
}


@dataclass
class CloudTeacherContent:
    speech: str
    question: str | None = None
    expected_answer: int | float | str | None = None
    answer_type: str = "number"
    hint: str | None = None
    explanation: str | None = None
    difficulty: str = "normal"
    next_action: str = "wait_for_answer"
    skill_tags: list[str] = field(default_factory=list)


@dataclass
class TeacherRecommendation:
    spoken_summary: str
    recommended_level: int
    recommended_focus: list[str] = field(default_factory=list)
    practice_plan: str = ""
    encouraging_message: str = ""


class CloudTeacherClient:
    def generate_lesson_intro(self, session: "TeacherSession") -> CloudTeacherContent | None:
        raise NotImplementedError

    def generate_creative_question(self, session: "TeacherSession") -> CloudTeacherContent | None:
        raise NotImplementedError

    def generate_hint(self, session: "TeacherSession") -> CloudTeacherContent | None:
        raise NotImplementedError

    def generate_explanation(self, session: "TeacherSession", result: dict[str, Any]) -> CloudTeacherContent | None:
        raise NotImplementedError

    def generate_adaptive_review(self, session: "TeacherSession") -> CloudTeacherContent | None:
        raise NotImplementedError

    def generate_recommendation(self, session: "TeacherSession", progress: dict[str, Any]) -> TeacherRecommendation | dict[str, Any] | str | None:
        raise NotImplementedError


class OpenAICloudTeacherClient(CloudTeacherClient):
    def __init__(self, responses_client: Any, model: str = "gpt-4o-mini"):
        self.responses_client = responses_client
        self.model = model

    def generate_lesson_intro(self, session: "TeacherSession") -> CloudTeacherContent | None:
        return self._generate("lesson_intro", session)

    def generate_creative_question(self, session: "TeacherSession") -> CloudTeacherContent | None:
        return self._generate("creative_question", session, require_math_answer=True)

    def generate_hint(self, session: "TeacherSession") -> CloudTeacherContent | None:
        return self._generate("hint", session)

    def generate_explanation(self, session: "TeacherSession", result: dict[str, Any]) -> CloudTeacherContent | None:
        return self._generate("explanation", session, result=result)

    def generate_adaptive_review(self, session: "TeacherSession") -> CloudTeacherContent | None:
        return self._generate("adaptive_review", session)

    def generate_recommendation(self, session: "TeacherSession", progress: dict[str, Any]) -> TeacherRecommendation | None:
        payload = {
            "request_type": "recommendation",
            "session": session_public_payload(session),
            "progress": compact_progress_payload(progress),
        }
        instructions = (
            "You are Miguel's teacher recommendation generator. Return JSON only. "
            "Keep the spoken summary short, encouraging, safe, and age-appropriate. "
            "Use English unless the local controller explicitly requests another language. "
            "Never decide correctness or overwrite progress. "
            "Schema: {\"spoken_summary\": string, \"recommended_level\": int, "
            "\"recommended_focus\": [string], \"practice_plan\": string, "
            "\"encouraging_message\": string}."
        )
        response = self.responses_client.create(
            model=self.model,
            instructions=instructions,
            input=json.dumps(payload, ensure_ascii=False),
        )
        raw_text = str(getattr(response, "output_text", "") or "").strip()
        return parse_teacher_recommendation(raw_text, session)

    def _generate(
        self,
        request_type: str,
        session: "TeacherSession",
        result: dict[str, Any] | None = None,
        require_math_answer: bool = False,
    ) -> CloudTeacherContent | None:
        payload = {
            "request_type": request_type,
            "session": session_public_payload(session),
            "last_result": result or {},
            "student_context": {
                "name_hint": "Marquinho",
                "project": "father-son robot project named Miguel",
                "preferred_examples": ["robot building", "engineering", "space", "mission control"],
            },
        }
        instructions = (
            "You are Miguel's structured teacher content generator. Return JSON only. "
            "Keep speech short, spoken-friendly, encouraging, and safe for the student's age. "
            "Use English unless the local controller explicitly requests another language. "
            "For Marquinho, prefer robot-building, engineering, space, and mission-style examples. "
            "Never grade answers or update score. The local controller handles scoring and state. "
            "Schema: {\"speech\": string, \"question\": string|null, "
            "\"expected_answer\": int|float|string|null, "
            "\"answer_type\": \"number|text|multiple_choice\", \"hint\": string|null, "
            "\"explanation\": string|null, \"difficulty\": \"easy|normal|hard\", "
            "\"next_action\": \"wait_for_answer|continue_lesson|ask_user_choice\"}. "
            "If you create a math exercise with next_action wait_for_answer, include expected_answer."
        )
        response = self.responses_client.create(
            model=self.model,
            instructions=instructions,
            input=json.dumps(payload, ensure_ascii=False),
        )
        raw_text = str(getattr(response, "output_text", "") or "").strip()
        return parse_cloud_teacher_content(raw_text, session, require_math_answer=require_math_answer)


@dataclass
class TeacherSession:
    active: bool = False
    user_id: str = "guest"
    display_name: str = "Guest"
    student_age: int | None = None
    subject: str | None = None
    subtopic: str | None = None
    topic: str | None = None
    level_number: int | None = None
    level_name: str | None = None
    difficulty: str = "normal"
    current_question: dict[str, Any] | None = None
    current_question_source: str = "local"
    score_correct: int = 0
    score_total: int = 0
    session_attempts: list[dict[str, Any]] = field(default_factory=list)
    session_weak_facts: dict[str, int] = field(default_factory=dict)
    session_start_time: float = field(default_factory=time.time)
    explicit_language: str | None = None
    lesson_step: str = "idle"
    last_explanation: str | None = None


def default_teacher_progress_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "memory" / "teacher_progress"


class TeacherProgressStore:
    def __init__(self, base_dir: str | Path | None = None):
        self.base_dir = Path(base_dir) if base_dir is not None else default_teacher_progress_dir()

    def load(self, user_id: str | None, display_name: str | None = None) -> dict[str, Any]:
        user_id = normalize_user_id(user_id)
        if not user_id:
            return new_progress_profile("guest", display_name or "Guest")
        path = self._path(user_id)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return ensure_progress_shape(data, user_id, display_name)
            except Exception:
                pass
        return new_progress_profile(user_id, display_name or KNOWN_USER_IDS.get(user_id) or user_id.title())

    def save(self, profile: dict[str, Any]) -> None:
        user_id = normalize_user_id(profile.get("user_id"))
        if not user_id or user_id == "guest":
            return
        self.base_dir.mkdir(parents=True, exist_ok=True)
        profile = ensure_progress_shape(profile, user_id, profile.get("display_name"))
        self._path(user_id).write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")

    def _path(self, user_id: str) -> Path:
        return self.base_dir / f"{normalize_user_id(user_id)}.json"


def normalize_user_id(user_id: str | None) -> str:
    normalized = normalize_text(str(user_id or "")).replace(" ", "_")
    if normalized in {"", "unknown", "unknown_wake_user", "tommy"}:
        return ""
    if normalized in {"marquinho", "marco"}:
        return normalized
    return normalized


def new_progress_profile(user_id: str, display_name: str) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "display_name": display_name,
        "age": None,
        "last_subject": None,
        "last_topic": None,
        "last_level": None,
        "subjects": {},
    }


def ensure_progress_shape(data: dict[str, Any], user_id: str, display_name: str | None = None) -> dict[str, Any]:
    data = dict(data or {})
    data["user_id"] = normalize_user_id(data.get("user_id")) or user_id
    data["display_name"] = str(data.get("display_name") or display_name or KNOWN_USER_IDS.get(user_id) or user_id.title())
    data.setdefault("age", None)
    data.setdefault("last_subject", None)
    data.setdefault("last_topic", None)
    data.setdefault("last_level", None)
    data.setdefault("subjects", {})
    return data


def level_progress_key(subject: str | None, topic: str | None, level_number: int | None) -> tuple[str, str, str]:
    return (subject or "math", topic or "multiplication", str(level_number or 1))


def get_level_progress(profile: dict[str, Any], subject: str | None, topic: str | None, level_number: int | None) -> dict[str, Any]:
    subject_key, topic_key, level_key = level_progress_key(subject, topic, level_number)
    subjects = profile.setdefault("subjects", {})
    topics = subjects.setdefault(subject_key, {})
    levels = topics.setdefault(topic_key, {})
    progress = levels.setdefault(
        level_key,
        {
            "attempted": 0,
            "correct": 0,
            "accuracy": 0.0,
            "mastered": False,
            "weak_facts": {},
            "strong_facts": {},
            "last_practiced": None,
            "recent_results": [],
            "recommended_next_level": int(level_key),
            "recommended_focus": [],
        },
    )
    progress.setdefault("weak_facts", {})
    progress.setdefault("strong_facts", {})
    progress.setdefault("recent_results", [])
    progress.setdefault("recommended_focus", [])
    return progress


class SubjectTeacher:
    subject = "subject"

    def start_lesson(self, session: TeacherSession) -> str:
        raise NotImplementedError

    def generate_question(self, session: TeacherSession) -> dict[str, Any]:
        raise NotImplementedError

    def render_question(self, session: TeacherSession, question: dict[str, Any]) -> str:
        raise NotImplementedError

    def check_answer(self, session: TeacherSession, transcript: str) -> dict[str, Any]:
        raise NotImplementedError

    def explain_answer(self, session: TeacherSession, result: dict[str, Any]) -> str:
        raise NotImplementedError

    def adjust_difficulty(self, session: TeacherSession, result: dict[str, Any]) -> None:
        return None


class MathTeacher(SubjectTeacher):
    subject = "math"

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def start_lesson(self, session: TeacherSession) -> str:
        subtopic = session.subtopic or "multiplication"
        session.subtopic = subtopic
        session.topic = subtopic
        if subtopic == "fractions":
            return (
                "Fractions are coming soon. Right now I can teach multiplication, "
                "addition, subtraction, or division."
            )
        explanation = self._lesson_explanation(session)
        session.last_explanation = explanation
        question = self.generate_question(session)
        session.current_question = question
        session.lesson_step = "practice"
        return f"{explanation} Now try this: {self.render_question(session, question)}"

    def generate_question(self, session: TeacherSession) -> dict[str, Any]:
        subtopic = session.subtopic or "multiplication"
        if subtopic == "addition":
            a, b = self._range_pair(session, easy_max=10, normal_min=5, normal_max=30, hard_min=20, hard_max=75)
            return {"type": "addition", "a": a, "b": b, "expected_answer": a + b}
        if subtopic == "subtraction":
            a, b = self._range_pair(session, easy_max=10, normal_min=5, normal_max=30, hard_min=20, hard_max=75)
            bigger, smaller = max(a, b), min(a, b)
            return {"type": "subtraction", "a": bigger, "b": smaller, "expected_answer": bigger - smaller}
        if subtopic == "division":
            divisor = self.rng.randint(2, 12 if session.difficulty != "hard" else 15)
            answer = self.rng.randint(2, 12 if session.difficulty != "hard" else 15)
            return {"type": "division", "a": divisor * answer, "b": divisor, "expected_answer": answer}
        if subtopic == "multiplication":
            level_question = self._multiplication_level_question(session)
            if level_question:
                return level_question
        a, b = self._multiplication_factors(session)
        return {
            "type": "multiplication",
            "a": a,
            "b": b,
            "expected_answer": a * b,
            "fact_key": multiplication_fact_key(a, b),
            "source": "local",
        }

    def render_question(self, session: TeacherSession, question: dict[str, Any]) -> str:
        if question.get("question"):
            return str(question.get("question")).strip()
        qtype = question.get("type")
        a = question.get("a")
        b = question.get("b")
        if qtype == "addition":
            return f"what is {a} plus {b}?"
        if qtype == "subtraction":
            return f"what is {a} minus {b}?"
        if qtype == "division":
            return f"what is {a} divided by {b}?"
        return f"what is {a} times {b}?"

    def check_answer(self, session: TeacherSession, transcript: str) -> dict[str, Any]:
        question = session.current_question
        if not question:
            question = self.generate_question(session)
            session.current_question = question
            return {
                "answered": False,
                "correct": False,
                "reply": f"Let's start with this: {self.render_question(session, question)}",
            }
        expected = question.get("expected_answer")
        answer_type = str(question.get("answer_type") or "number")
        heard = extract_answer_value(transcript, answer_type)
        if heard is None:
            return {
                "answered": False,
                "correct": False,
                "reply": "I did not hear a number. Please try the same question again: "
                + self.render_question(session, question),
            }
        correct = answers_match(heard, expected, answer_type)
        session.score_total += 1
        if correct:
            session.score_correct += 1
        source = str(question.get("source") or session.current_question_source or "local")
        result = {
            "answered": True,
            "correct": correct,
            "heard": heard,
            "expected": expected,
            "question": question,
            "question_source": source,
        }
        self.adjust_difficulty(session, result)
        next_question = self.generate_question(session)
        session.current_question = next_question
        result["next_question"] = next_question
        result["reply"] = self.explain_answer(session, result)
        return result

    def explain_answer(self, session: TeacherSession, result: dict[str, Any]) -> str:
        question = result.get("question") or {}
        next_question = result.get("next_question") or session.current_question or {}
        if result.get("correct"):
            return "Great job. " + self._score_text(session) + " Next one: " + self.render_question(session, next_question)

        expected = result.get("expected")
        qtype = question.get("type")
        if qtype == "cloud_math":
            reason = str(question.get("explanation") or f"The answer is {expected}.")
            return "Good try. " + reason + " " + self._score_text(session) + " Try this one: " + self.render_question(session, next_question)
        if qtype == "multiplication":
            a = int(question.get("a"))
            b = int(question.get("b"))
            repeated = " plus ".join(str(b) for _ in range(min(a, 5)))
            if a > 5:
                repeated += f" plus {b} more groups"
            reason = f"{a} times {b} means {repeated}, which equals {expected}."
        elif qtype == "addition":
            reason = f"{question.get('a')} plus {question.get('b')} equals {expected}."
        elif qtype == "subtraction":
            reason = f"{question.get('a')} minus {question.get('b')} equals {expected}."
        elif qtype == "multiplication_multi_step":
            a = int(question.get("a"))
            b = int(question.get("b"))
            c = int(question.get("c"))
            reason = f"{a} times {b} is {a * b}, then {a * b} minus {c} equals {expected}."
        else:
            reason = f"{question.get('a')} divided by {question.get('b')} equals {expected}."
        return "Good try. " + reason + " " + self._score_text(session) + " Try this one: " + self.render_question(session, next_question)

    def adjust_difficulty(self, session: TeacherSession, result: dict[str, Any]) -> None:
        return None

    def _lesson_explanation(self, session: TeacherSession) -> str:
        subtopic = session.subtopic or "multiplication"
        if subtopic == "addition":
            return "Great, Chief Engineer. Addition means putting numbers together to find the total."
        if subtopic == "subtraction":
            return "Great, Chief Engineer. Subtraction means taking some away and seeing what is left."
        if subtopic == "division":
            return "Great, Chief Engineer. Division means sharing a number into equal groups."
        return (
            "Great, Chief Engineer. Today we will learn multiplication. "
            "Multiplication is a faster way to add the same number many times. "
            "For example, 3 times 4 means 4 plus 4 plus 4, which equals 12."
        )

    def _multiplication_factors(self, session: TeacherSession) -> tuple[int, int]:
        if session.difficulty == "easy":
            return self.rng.randint(1, 5), self.rng.randint(1, 5)
        if session.difficulty == "hard":
            return self.rng.randint(6, 15), self.rng.randint(6, 15)
        if session.level_number == 1:
            return self.rng.randint(1, 10), self.rng.randint(1, 10)
        if session.level_number == 2:
            harder = [6, 7, 8, 9, 11, 12]
            return self.rng.choice(harder), self.rng.randint(1, 12)
        if session.student_age is not None and session.student_age <= 7:
            return self.rng.randint(1, 5), self.rng.randint(1, 5)
        if session.student_age is not None and session.student_age > 10:
            return self.rng.randint(6, 15), self.rng.randint(6, 15)
        return self.rng.randint(2, 12), self.rng.randint(2, 12)

    def _range_pair(
        self,
        session: TeacherSession,
        easy_max: int,
        normal_min: int,
        normal_max: int,
        hard_min: int,
        hard_max: int,
    ) -> tuple[int, int]:
        if session.difficulty == "easy" or (session.student_age is not None and session.student_age <= 7):
            return self.rng.randint(1, easy_max), self.rng.randint(1, easy_max)
        if session.difficulty == "hard" or (session.student_age is not None and session.student_age > 10):
            return self.rng.randint(hard_min, hard_max), self.rng.randint(hard_min, hard_max)
        return self.rng.randint(normal_min, normal_max), self.rng.randint(normal_min, normal_max)

    def _score_text(self, session: TeacherSession) -> str:
        return f"You have answered {session.score_correct} out of {session.score_total} correctly."

    def _multiplication_level_question(self, session: TeacherSession) -> dict[str, Any] | None:
        level = session.level_number or 1
        a, b = self._multiplication_factors(session)
        expected = a * b
        fact_key = multiplication_fact_key(a, b)
        if level == 3:
            return {
                "type": "multiplication",
                "question": f"Miguel needs {a} packs with {b} bolts in each pack. How many bolts is that?",
                "a": a,
                "b": b,
                "expected_answer": expected,
                "fact_key": fact_key,
                "source": "local",
            }
        if level == 4:
            used = self.rng.randint(1, max(1, min(9, expected - 1)))
            return {
                "type": "multiplication_multi_step",
                "question": f"Miguel has {a} boxes with {b} sensors each. He uses {used} sensors. How many are left?",
                "a": a,
                "b": b,
                "c": used,
                "expected_answer": expected - used,
                "fact_key": fact_key,
                "source": "local",
            }
        if level >= 5:
            return {
                "type": "multiplication",
                "question": f"Chief Engineer mission: {a} rover teams each carry {b} repair cells. How many cells launch?",
                "a": a,
                "b": b,
                "expected_answer": expected,
                "fact_key": fact_key,
                "source": "local",
            }
        return {
            "type": "multiplication",
            "a": a,
            "b": b,
            "expected_answer": expected,
            "fact_key": fact_key,
            "source": "local",
        }


class TeacherModeController:
    def __init__(
        self,
        teachers: dict[str, SubjectTeacher] | None = None,
        cloud_client: CloudTeacherClient | None = None,
        progress_store: TeacherProgressStore | None = None,
    ):
        self.session = TeacherSession()
        self.teachers = teachers or {"math": MathTeacher()}
        self.cloud_client = cloud_client
        self.progress_store = progress_store or TeacherProgressStore()
        self.progress_profile = self.progress_store.load("guest", "Guest")

    def set_user_identity(self, user_id: str | None, display_name: str | None = None) -> None:
        normalized = normalize_user_id(user_id)
        if not normalized:
            return
        display = display_name or KNOWN_USER_IDS.get(normalized) or normalized.title()
        if self.session.user_id == normalized:
            self.session.display_name = display
            return
        self.progress_profile = self.progress_store.load(normalized, display)
        self.session.user_id = normalized
        self.session.display_name = display
        saved_age = self.progress_profile.get("age")
        if isinstance(saved_age, int) and self.session.student_age is None:
            self.session.student_age = saved_age

    def detects_teacher_intent(self, transcript: str) -> bool:
        normalized = normalize_text(transcript)
        if self.session.active:
            return True
        teacher_phrases = (
            "teacher mode",
            "teach me",
            "i want to learn",
            "let's study",
            "lets study",
            "math teacher",
            "study math",
            "learn math",
        )
        return any(phrase in normalized for phrase in teacher_phrases)

    def handle(
        self,
        transcript: str,
        user_id: str | None = None,
        display_name: str | None = None,
        explicit_language: str | None = None,
    ) -> str | None:
        if user_id:
            self.set_user_identity(user_id, display_name)
        if explicit_language:
            self.session.explicit_language = explicit_language
        if not self.detects_teacher_intent(transcript):
            return None

        normalized = normalize_text(transcript)
        if self.session.active:
            command_reply = self._handle_active_command(normalized)
            if command_reply:
                return command_reply

        if not self.session.active:
            user_id = self.session.user_id
            display_name = self.session.display_name
            age = self.session.student_age
            explicit_language = self.session.explicit_language
            self.session = TeacherSession(
                active=True,
                user_id=user_id,
                display_name=display_name,
                student_age=age,
                explicit_language=explicit_language,
            )

        age = extract_age(transcript)
        if age is None and self.session.active and self.session.student_age is None:
            heard_number = extract_number(transcript)
            if heard_number is not None and 1 <= heard_number <= 120:
                age = heard_number
        if age is not None:
            self.session.student_age = age
            if self.session.user_id != "guest":
                self.progress_profile["age"] = age

        subject = extract_subject(transcript)
        subtopic = extract_subtopic(transcript)
        level_number = extract_level_selection(transcript)
        if subject:
            self.session.subject = subject
        elif subtopic:
            self.session.subject = "math"
        if subtopic:
            self.session.subtopic = subtopic
            self.session.topic = subtopic
        if level_number:
            self._select_level(level_number)

        if self.session.user_id != "guest" and self.session.display_name:
            recognized_prefix = f"Teacher mode is on. I recognize you as {self.session.display_name}. "
        else:
            recognized_prefix = "Teacher mode is on. "

        if not self.session.subject and self.session.user_id != "guest":
            self.session.lesson_step = "ask_subject"
            saved = self._saved_progress_prompt()
            if saved:
                return saved
            return recognized_prefix + "What do you want to study: math, English, history, science, or programming?"

        if self.session.student_age is None and not self.session.subject:
            self.session.lesson_step = "ask_age"
            return "Teacher mode is on. How old is the student?"

        if not self.session.subject:
            self.session.lesson_step = "ask_subject"
            return "What subject do you want to study? Right now I am best at math."

        if self.session.subject != "math":
            self.session.subject = "math"
            self.session.lesson_step = "ask_subtopic"
            return (
                "I can teach that later, but right now my first teacher skill is math. "
                "Do you want to practice multiplication, addition, subtraction, or division?"
            )

        if not self.session.subtopic:
            self.session.lesson_step = "ask_subtopic"
            return "Do you want to practice multiplication, addition, subtraction, or division?"

        if self.session.subtopic == "multiplication" and not self.session.level_number:
            self.session.lesson_step = "ask_level"
            saved = self._saved_progress_prompt(topic_known=True)
            if saved:
                return saved
            return self._level_choice_prompt()

        teacher = self.teachers.get(self.session.subject)
        if not teacher:
            return (
                "I can teach that later, but right now my first teacher skill is math. "
                "Do you want to practice multiplication, addition, subtraction, or division?"
            )

        if self.session.lesson_step in {"idle", "ask_age", "ask_subject", "ask_subtopic", "ask_level"}:
            return self._start_lesson(teacher)

        result = teacher.check_answer(self.session, transcript)
        if result.get("answered"):
            self._record_answer_progress(result)
        return self._cloud_enhanced_answer_reply(teacher, result)

    def _handle_active_command(self, normalized: str) -> str | None:
        if any(
            phrase in normalized
            for phrase in {
                "exit teacher mode",
                "stop teaching",
                "teacher mode off",
                "turn off teacher mode",
                "turn off the teacher mode",
                "go to normal mode",
                "return to normal mode",
                "back to normal mode",
            }
        ):
            self.session.active = False
            self.session.lesson_step = "idle"
            return "Teacher mode is off. Great work today."
        if (
            "creative challenge" in normalized
            or "mission challenge" in normalized
            or "space challenge" in normalized
            or "robot challenge" in normalized
            or "chief engineer challenge" in normalized
            or "give me a challenge" in normalized
            or "make it fun" in normalized
            or "harder challenge" in normalized
        ):
            reply = self._creative_question_reply()
            if reply:
                return reply
            return self._local_challenge_reply()
        if "continue my level" in normalized or "continue my last level" in normalized or "my last level" in normalized or normalized == "continue":
            return self._continue_saved_level()
        level_number = extract_level_selection(normalized)
        if level_number:
            self._select_level(level_number)
            teacher = self.teachers.get(self.session.subject or "math")
            if not self.session.subject:
                self.session.subject = "math"
            if not self.session.subtopic:
                self.session.subtopic = "multiplication"
                self.session.topic = "multiplication"
            if teacher:
                return self._start_lesson(teacher)
            return self._level_choice_prompt()
        topic_reply = self._handle_active_topic_request(normalized)
        if topic_reply:
            return topic_reply
        if "what is my score" in normalized or "what's my score" in normalized or "score" == normalized:
            return f"You have answered {self.session.score_correct} out of {self.session.score_total} correctly."
        if "new topic" in normalized:
            self.session.subject = None
            self.session.subtopic = None
            self.session.current_question = None
            self.session.lesson_step = "ask_subject"
            return "What math topic do you want next: multiplication, addition, subtraction, or division?"
        if "review" in normalized:
            review = self._cloud_adaptive_review()
            if review:
                return review
            return f"You have answered {self.session.score_correct} out of {self.session.score_total} correctly. Let's keep practicing."
        if "how can i improve" in normalized or "recommend" in normalized or "practice plan" in normalized:
            return self._recommendation_reply()
        if "make it easier" in normalized:
            self.session.difficulty = "easy"
            return self._difficulty_reply("easier")
        if "make it harder" in normalized:
            self.session.difficulty = "hard"
            return self._difficulty_reply("harder")
        if "explain again" in normalized:
            cloud_reply = self._cloud_explanation()
            if cloud_reply:
                return cloud_reply
            if self.session.last_explanation:
                return self.session.last_explanation + " " + self._repeat_current_question()
            return self._repeat_current_question()
        if "hint" in normalized or "help me" in normalized:
            cloud_hint = self._cloud_hint()
            if cloud_hint:
                return cloud_hint
            return "Try breaking it into smaller parts. " + self._repeat_current_question()
        if "repeat the question" in normalized:
            return self._repeat_current_question()
        if "another one" in normalized:
            teacher = self.teachers.get(self.session.subject or "math")
            if not teacher:
                return None
            self.session.current_question = teacher.generate_question(self.session)
            self.session.lesson_step = "practice"
            return "Here is another one: " + teacher.render_question(self.session, self.session.current_question)
        return None

    def _handle_active_topic_request(self, normalized: str) -> str | None:
        if "teacher mode" in normalized and not extract_subtopic(normalized):
            if self.session.current_question:
                return "Teacher mode is already on. " + self._repeat_current_question()
            return "Teacher mode is already on. Tell me your age, subject, or topic."
        wants_lesson = any(
            phrase in normalized
            for phrase in {
                "teach me",
                "teach multiplication",
                "learn",
                "study",
                "practice",
                "want you to teach",
                "wanted you to teach",
            }
        )
        subtopic = extract_subtopic(normalized)
        subject = extract_subject(normalized)
        if not wants_lesson and not subtopic:
            return None
        age = extract_age(normalized)
        if age is not None:
            self.session.student_age = age
        if subject:
            self.session.subject = subject
        elif subtopic:
            self.session.subject = "math"
        if subtopic:
            self.session.subtopic = subtopic
            self.session.topic = subtopic
        level_number = extract_level_selection(normalized)
        if level_number:
            self._select_level(level_number)
        self.session.current_question = None
        if self.session.student_age is None and not self.session.subject:
            self.session.lesson_step = "ask_age"
            return "Great. How old is the student?"
        if not self.session.subject:
            self.session.lesson_step = "ask_subject"
            return "What subject do you want to study? Right now I am best at math."
        if self.session.subject != "math":
            self.session.subject = "math"
            self.session.lesson_step = "ask_subtopic"
            return (
                "I can teach that later, but right now my first teacher skill is math. "
                "Do you want to practice multiplication, addition, subtraction, or division?"
            )
        if not self.session.subtopic:
            self.session.lesson_step = "ask_subtopic"
            return "Do you want to practice multiplication, addition, subtraction, or division?"
        if self.session.subtopic == "multiplication" and not self.session.level_number:
            self.session.lesson_step = "ask_level"
            return self._level_choice_prompt()
        teacher = self.teachers.get(self.session.subject)
        if not teacher:
            return None
        self.session.lesson_step = "ask_subtopic"
        return self._start_lesson(teacher)

    def _difficulty_reply(self, label: str) -> str:
        teacher = self.teachers.get(self.session.subject or "math")
        if not teacher:
            return "Okay, I changed the difficulty."
        self.session.current_question = teacher.generate_question(self.session)
        self.session.lesson_step = "practice"
        return f"Okay, I will make it {label}. Try this: {teacher.render_question(self.session, self.session.current_question)}"

    def _repeat_current_question(self) -> str:
        teacher = self.teachers.get(self.session.subject or "math")
        if not teacher or not self.session.current_question:
            return "I do not have a question yet. Tell me what you want to practice."
        return "The question is: " + teacher.render_question(self.session, self.session.current_question)

    def _local_challenge_reply(self) -> str:
        teacher = self.teachers.get(self.session.subject or "math")
        if not teacher:
            return "Let's keep practicing locally. Tell me which math topic you want."
        self.session.current_question = teacher.generate_question(self.session)
        self.session.current_question_source = "local"
        self.session.lesson_step = "practice"
        return "Cloud challenge is not available, so here is a local mission: " + teacher.render_question(self.session, self.session.current_question)

    def _start_lesson(self, teacher: SubjectTeacher) -> str:
        if self.session.subtopic == "multiplication" and not self.session.level_number:
            self._select_level(int(self.progress_profile.get("last_level") or 1))
        self._remember_last_selection()
        return teacher.start_lesson(self.session)

    def _creative_question_reply(self) -> str | None:
        content = self._safe_cloud_call("generate_creative_question", require_math_answer=True)
        question = self._question_from_cloud_content(content) if content else None
        if not question:
            return None
        self.session.current_question = question
        self.session.current_question_source = "cloud"
        self.session.lesson_step = "practice"
        return combine_speech_and_question(content.speech, question["question"])

    def _cloud_hint(self) -> str | None:
        content = self._safe_cloud_call("generate_hint")
        if not content:
            return None
        hint = content.hint or content.speech
        if not hint:
            return None
        return short_spoken_text(hint)

    def _cloud_explanation(self) -> str | None:
        content = self._safe_cloud_call("generate_explanation", result={"question": self.session.current_question})
        if not content:
            return None
        explanation = content.explanation or content.speech
        if not explanation:
            return None
        return short_spoken_text(explanation) + " " + self._repeat_current_question()

    def _cloud_adaptive_review(self) -> str | None:
        content = self._safe_cloud_call("generate_adaptive_review")
        if not content:
            return None
        return short_spoken_text(content.speech)

    def _cloud_enhanced_answer_reply(self, teacher: SubjectTeacher, result: dict[str, Any]) -> str:
        local_reply = str(result.get("reply") or "")
        return local_reply

    def _select_level(self, level_number: int) -> None:
        level = MULTIPLICATION_LEVELS.get(int(level_number))
        if not level:
            return
        self.session.subject = level.subject
        self.session.subtopic = level.topic
        self.session.topic = level.topic
        self.session.level_number = level.level_number
        self.session.level_name = level.level_name
        self.session.lesson_step = "ask_level"

    def _level_choice_prompt(self) -> str:
        saved_level = self.progress_profile.get("last_level")
        if saved_level:
            return (
                f"Last time you practiced Multiplication Level {saved_level}. "
                "Say continue my level, or choose Level 1 beginner tables, Level 2 full tables, "
                "Level 3 word problems, Level 4 multi-step, or Level 5 challenge mode."
            )
        return (
            "For multiplication, choose Level 1 beginner tables, Level 2 full tables, "
            "Level 3 word problems, Level 4 multi-step, or Level 5 challenge mode."
        )

    def _saved_progress_prompt(self, topic_known: bool = False) -> str | None:
        if self.session.user_id == "guest":
            return None
        last_subject = self.progress_profile.get("last_subject")
        last_topic = self.progress_profile.get("last_topic")
        last_level = self.progress_profile.get("last_level")
        if not (last_subject and last_topic and last_level):
            return None
        progress = get_level_progress(self.progress_profile, last_subject, last_topic, int(last_level))
        if topic_known and last_topic != self.session.subtopic:
            return None
        prefix = f"Teacher mode is on. I recognize you as {self.session.display_name}. " if self.session.user_id != "guest" else ""
        return (
            f"{prefix}Last time you were practicing {last_topic.title()} Level {last_level}. "
            f"You answered {progress.get('correct', 0)} out of {progress.get('attempted', 0)} correctly. "
            f"Say continue my level, or choose a level."
        )

    def _continue_saved_level(self) -> str:
        last_subject = self.progress_profile.get("last_subject") or "math"
        last_topic = self.progress_profile.get("last_topic") or "multiplication"
        last_level = int(self.progress_profile.get("last_level") or 1)
        self.session.subject = last_subject
        self.session.subtopic = last_topic
        self.session.topic = last_topic
        self._select_level(last_level)
        teacher = self.teachers.get(self.session.subject or "math")
        if not teacher:
            return "I can continue math right now. Which multiplication level do you want?"
        return self._start_lesson(teacher)

    def _remember_last_selection(self) -> None:
        if self.session.user_id == "guest":
            return
        self.progress_profile["last_subject"] = self.session.subject
        self.progress_profile["last_topic"] = self.session.subtopic or self.session.topic
        self.progress_profile["last_level"] = self.session.level_number
        self.progress_store.save(self.progress_profile)

    def _record_answer_progress(self, result: dict[str, Any]) -> None:
        question = result.get("question") or {}
        fact_key = question_fact_key(question)
        attempt = {
            "correct": bool(result.get("correct")),
            "expected": result.get("expected"),
            "heard": result.get("heard"),
            "fact": fact_key,
            "source": result.get("question_source") or question.get("source") or "local",
            "created_at": int(time.time()),
        }
        self.session.session_attempts.append(attempt)
        if fact_key and not result.get("correct"):
            self.session.session_weak_facts[fact_key] = self.session.session_weak_facts.get(fact_key, 0) + 1
        if self.session.user_id == "guest":
            return
        progress = get_level_progress(
            self.progress_profile,
            self.session.subject,
            self.session.subtopic or self.session.topic,
            self.session.level_number,
        )
        progress["attempted"] = int(progress.get("attempted") or 0) + 1
        if result.get("correct"):
            progress["correct"] = int(progress.get("correct") or 0) + 1
            if fact_key:
                strong = progress.setdefault("strong_facts", {})
                strong[fact_key] = int(strong.get(fact_key) or 0) + 1
                weak = progress.setdefault("weak_facts", {})
                if fact_key in weak:
                    weak[fact_key] = max(0, int(weak.get(fact_key) or 0) - 1)
                    if weak[fact_key] == 0:
                        weak.pop(fact_key, None)
        elif fact_key:
            weak = progress.setdefault("weak_facts", {})
            weak[fact_key] = int(weak.get(fact_key) or 0) + 1
        progress["accuracy"] = round(progress["correct"] / progress["attempted"], 4) if progress["attempted"] else 0.0
        progress["last_practiced"] = int(time.time())
        recent = progress.setdefault("recent_results", [])
        recent.append(attempt)
        del recent[:-50]
        mastery = evaluate_mastery(progress)
        progress["mastered"] = mastery
        recommendation = local_progress_recommendation(progress, self.session.level_number or 1)
        progress["recommended_next_level"] = recommendation["recommended_level"]
        progress["recommended_focus"] = recommendation["recommended_focus"]
        self._remember_last_selection()
        self.progress_store.save(self.progress_profile)

    def _recommendation_reply(self) -> str:
        progress = get_level_progress(
            self.progress_profile,
            self.session.subject,
            self.session.subtopic or self.session.topic,
            self.session.level_number,
        )
        cloud_reply = self._cloud_recommendation(progress)
        if cloud_reply:
            return cloud_reply
        rec = local_progress_recommendation(progress, self.session.level_number or int(self.progress_profile.get("last_level") or 1))
        focus = ", ".join(rec["recommended_focus"][:3]) if rec["recommended_focus"] else "today's facts"
        return f"{rec['spoken_summary']} Practice focus: {focus}. {rec['practice_plan']}"

    def _cloud_recommendation(self, progress: dict[str, Any]) -> str | None:
        if not self.cloud_client or not hasattr(self.cloud_client, "generate_recommendation"):
            return None
        try:
            content = self.cloud_client.generate_recommendation(self.session, compact_progress_payload(progress))
        except Exception:
            return None
        rec = parse_teacher_recommendation(content, self.session)
        if not rec:
            return None
        local_rec = local_progress_recommendation(progress, self.session.level_number or 1)
        recommended_level = rec.recommended_level if rec.recommended_level in MULTIPLICATION_LEVELS else local_rec["recommended_level"]
        return short_spoken_text(
            f"{rec.spoken_summary} I recommend Level {recommended_level}. {rec.practice_plan} {rec.encouraging_message}",
            max_chars=260,
        )

    def _safe_cloud_call(
        self,
        method_name: str,
        result: dict[str, Any] | None = None,
        require_math_answer: bool = False,
    ) -> CloudTeacherContent | None:
        if not self.cloud_client:
            return None
        try:
            method = getattr(self.cloud_client, method_name)
            if method_name == "generate_explanation":
                content = method(self.session, result or {})
            else:
                content = method(self.session)
        except Exception:
            return None
        if isinstance(content, (str, dict)):
            content = parse_cloud_teacher_content(content, self.session, require_math_answer=require_math_answer)
        if not validate_cloud_teacher_content(content, self.session, require_math_answer=require_math_answer):
            return None
        return content

    def _question_from_cloud_content(self, content: CloudTeacherContent | None) -> dict[str, Any] | None:
        if not content or not content.question:
            return None
        if self.session.subject == "math" and content.expected_answer is None:
            return None
        return {
            "type": "cloud_math" if self.session.subject == "math" else "cloud",
            "question": short_spoken_text(content.question),
            "expected_answer": content.expected_answer,
            "answer_type": content.answer_type,
            "source": "cloud",
            "cloud_difficulty": content.difficulty,
            "explanation": content.explanation,
            "hint": content.hint,
            "skill_tags": list(content.skill_tags or []),
        }


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or "").lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_cloud_teacher_content(
    raw: str | dict[str, Any] | CloudTeacherContent,
    session: TeacherSession,
    require_math_answer: bool = False,
) -> CloudTeacherContent | None:
    if isinstance(raw, CloudTeacherContent):
        return raw if validate_cloud_teacher_content(raw, session, require_math_answer=require_math_answer) else None
    try:
        if isinstance(raw, str):
            text = raw.strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
            data = json.loads(text)
        else:
            data = dict(raw or {})
    except Exception:
        return None
    content = CloudTeacherContent(
        speech=short_spoken_text(str(data.get("speech") or data.get("speech_intro") or "")),
        question=short_spoken_text(str(data.get("question"))) if data.get("question") is not None else None,
        expected_answer=data.get("expected_answer"),
        answer_type=str(data.get("answer_type") or "number"),
        hint=short_spoken_text(str(data.get("hint"))) if data.get("hint") is not None else None,
        explanation=short_spoken_text(str(data.get("explanation"))) if data.get("explanation") is not None else None,
        difficulty=str(data.get("difficulty") or "normal"),
        next_action=str(data.get("next_action") or "wait_for_answer"),
        skill_tags=[str(item) for item in (data.get("skill_tags") or []) if str(item).strip()],
    )
    if not validate_cloud_teacher_content(content, session, require_math_answer=require_math_answer):
        return None
    return content


def validate_cloud_teacher_content(
    content: CloudTeacherContent | None,
    session: TeacherSession,
    require_math_answer: bool = False,
) -> bool:
    if not isinstance(content, CloudTeacherContent):
        return False
    if not content.speech:
        return False
    if content.answer_type not in ANSWER_TYPES:
        return False
    if content.next_action not in NEXT_ACTIONS:
        return False
    if content.difficulty not in DIFFICULTIES:
        return False
    if (
        (session.explicit_language or "").lower() not in {"pt", "pt-br", "portuguese"}
        and any(
            contains_teacher_language_mismatch(value)
            for value in (content.speech, content.question, content.hint, content.explanation)
        )
    ):
        return False
    if session.subject == "math" and (require_math_answer or content.question) and content.next_action == "wait_for_answer":
        if content.expected_answer is None:
            return False
        if content.answer_type == "number" and coerce_number(content.expected_answer) is None:
            return False
    return True


def session_public_payload(session: TeacherSession) -> dict[str, Any]:
    current_question = dict(session.current_question or {})
    if current_question:
        current_question.pop("expected_answer", None)
    return {
        "active": session.active,
        "student_age": session.student_age,
        "subject": session.subject,
        "subtopic": session.subtopic,
        "topic": session.topic,
        "level_number": session.level_number,
        "level_name": session.level_name,
        "difficulty": session.difficulty,
        "score_correct": session.score_correct,
        "score_total": session.score_total,
        "lesson_step": session.lesson_step,
        "current_question": current_question,
    }


def short_spoken_text(text: str, max_chars: int = 260) -> str:
    cleaned = re.sub(r"[\r\n*#`]+", " ", str(text or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    clipped = cleaned[:max_chars].rsplit(" ", 1)[0].strip()
    return clipped or cleaned[:max_chars].strip()


def contains_teacher_language_mismatch(text: str | None) -> bool:
    raw = str(text or "")
    ascii_text = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    normalized = normalize_text(ascii_text)
    if not normalized:
        return False
    portuguese_markers = {
        "oi",
        "missao",
        "missão",
        "aquecimento",
        "resposta",
        "multiplicacao",
        "multiplicação",
        "proxima",
        "próxima",
        "tente",
        "quanto",
    }
    return any(re.search(rf"\b{re.escape(marker)}\b", normalized) for marker in portuguese_markers)


def parse_teacher_recommendation(raw: str | dict[str, Any] | TeacherRecommendation | None, session: TeacherSession) -> TeacherRecommendation | None:
    if isinstance(raw, TeacherRecommendation):
        return raw if validate_teacher_recommendation(raw, session) else None
    try:
        if isinstance(raw, str):
            text = raw.strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
            data = json.loads(text)
        else:
            data = dict(raw or {})
    except Exception:
        return None
    rec = TeacherRecommendation(
        spoken_summary=short_spoken_text(str(data.get("spoken_summary") or "")),
        recommended_level=int(data.get("recommended_level") or session.level_number or 1),
        recommended_focus=[short_spoken_text(str(item), 60) for item in (data.get("recommended_focus") or []) if str(item).strip()],
        practice_plan=short_spoken_text(str(data.get("practice_plan") or "")),
        encouraging_message=short_spoken_text(str(data.get("encouraging_message") or "")),
    )
    return rec if validate_teacher_recommendation(rec, session) else None


def validate_teacher_recommendation(rec: TeacherRecommendation | None, session: TeacherSession) -> bool:
    if not isinstance(rec, TeacherRecommendation):
        return False
    if not rec.spoken_summary:
        return False
    if rec.recommended_level not in MULTIPLICATION_LEVELS:
        return False
    if (session.explicit_language or "").lower() not in {"pt", "pt-br", "portuguese"}:
        if any(contains_teacher_language_mismatch(value) for value in [rec.spoken_summary, rec.practice_plan, rec.encouraging_message, *rec.recommended_focus]):
            return False
    return True


def compact_progress_payload(progress: dict[str, Any]) -> dict[str, Any]:
    weak = dict(progress.get("weak_facts") or {})
    return {
        "attempted": int(progress.get("attempted") or 0),
        "correct": int(progress.get("correct") or 0),
        "accuracy": float(progress.get("accuracy") or 0.0),
        "mastered": bool(progress.get("mastered")),
        "weak_facts": sorted(weak, key=lambda key: weak.get(key, 0), reverse=True)[:5],
        "recent_results": list(progress.get("recent_results") or [])[-10:],
    }


def multiplication_fact_key(a: Any, b: Any) -> str:
    try:
        first = int(a)
        second = int(b)
    except Exception:
        return ""
    low, high = sorted((first, second))
    return f"{low}x{high}"


def question_fact_key(question: dict[str, Any]) -> str:
    if question.get("fact_key"):
        return str(question.get("fact_key"))
    if question.get("a") is not None and question.get("b") is not None and str(question.get("type", "")).startswith("multiplication"):
        return multiplication_fact_key(question.get("a"), question.get("b"))
    return ""


def evaluate_mastery(progress: dict[str, Any]) -> bool:
    attempted = int(progress.get("attempted") or 0)
    correct = int(progress.get("correct") or 0)
    if attempted < 10:
        return False
    accuracy = correct / attempted if attempted else 0.0
    recent = list(progress.get("recent_results") or [])[-10:]
    if len(recent) < 10:
        return False
    recent_accuracy = sum(1 for item in recent if item.get("correct")) / len(recent)
    return accuracy >= 0.85 and recent_accuracy >= 0.80


def local_progress_recommendation(progress: dict[str, Any], current_level: int) -> dict[str, Any]:
    attempted = int(progress.get("attempted") or 0)
    correct = int(progress.get("correct") or 0)
    accuracy = correct / attempted if attempted else 0.0
    weak = dict(progress.get("weak_facts") or {})
    focus = sorted(weak, key=lambda key: weak.get(key, 0), reverse=True)[:3]
    if progress.get("mastered"):
        next_level = min(5, int(current_level or 1) + 1)
        return {
            "spoken_summary": "You are ready for the next mission level.",
            "recommended_level": next_level,
            "recommended_focus": focus,
            "practice_plan": "Try the next level when you want, or do one quick review first.",
        }
    if attempted and accuracy < 0.60:
        return {
            "spoken_summary": "Let's strengthen the basics before moving up.",
            "recommended_level": max(1, int(current_level or 1) - 1),
            "recommended_focus": focus,
            "practice_plan": "Practice a few easier facts slowly and say the groups out loud.",
        }
    if attempted and accuracy < 0.85:
        return {
            "spoken_summary": "You are building fluency. Stay on this level for now.",
            "recommended_level": int(current_level or 1),
            "recommended_focus": focus,
            "practice_plan": "Do another short round and focus on the facts that felt tricky.",
        }
    return {
        "spoken_summary": "You are doing well. Keep practicing this level.",
        "recommended_level": int(current_level or 1),
        "recommended_focus": focus,
        "practice_plan": "Answer ten questions with careful thinking.",
    }


def combine_speech_and_question(speech: str, question: str) -> str:
    speech = short_spoken_text(speech)
    question = short_spoken_text(question)
    if not speech:
        return question
    if not question:
        return speech
    return f"{speech} Now try this: {question}"


def extract_answer_value(text: str, answer_type: str) -> int | float | str | None:
    if answer_type == "number":
        return extract_number(text)
    cleaned = normalize_text(text)
    return cleaned or None


def answers_match(heard: int | float | str | None, expected: int | float | str | None, answer_type: str) -> bool:
    if heard is None or expected is None:
        return False
    if answer_type == "number":
        heard_number = coerce_number(heard)
        expected_number = coerce_number(expected)
        if heard_number is None or expected_number is None:
            return False
        return abs(heard_number - expected_number) < 0.0001
    return normalize_text(str(heard)) == normalize_text(str(expected))


def coerce_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return float(stripped)
        except ValueError:
            extracted = extract_number(stripped)
            return float(extracted) if extracted is not None else None
    return None


def extract_age(text: str) -> int | None:
    normalized = normalize_text(text)
    patterns = [
        r"\b(?:for\s+(?:a\s+)?)?([a-z0-9 -]{1,40})\s+years?\s+old\b",
        r"\b(?:i am|i'm|im|marquinho is|he is|she is)\s+([a-z0-9 -]+?)(?:\s+years?\s+old|\s+year\s+old)?(?:\b|$)",
        r"\bage\s+([a-z0-9 -]+?)(?:\b|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        candidate = match.group(1).strip()
        value = extract_number(candidate)
        if value is not None and 1 <= value <= 120:
            return value
        candidate = re.split(r"\b(?:teach|learn|study|and|math|multiplication|addition|subtraction|division)\b", candidate)[0].strip()
        value = extract_number(candidate)
        if value is not None and 1 <= value <= 120:
            return value
    return None


def extract_subject(text: str) -> str | None:
    normalized = normalize_text(text)
    for subject in SUBJECTS:
        if re.search(rf"\b{re.escape(subject)}\b", normalized):
            return subject
    return None


def extract_subtopic(text: str) -> str | None:
    normalized = normalize_text(text)
    for phrase, subtopic in MATH_SUBTOPICS.items():
        if re.search(rf"\b{re.escape(phrase)}\b", normalized):
            return subtopic
    return None


def extract_level_selection(text: str) -> int | None:
    normalized = normalize_text(text)
    match = re.search(r"\blevel\s+([1-5])\b", normalized)
    if match:
        return int(match.group(1))
    if re.search(r"\bstart\s+level\s+([1-5])\b", normalized):
        return int(re.search(r"\bstart\s+level\s+([1-5])\b", normalized).group(1))
    phrase_map = {
        1: {"beginner", "beginner tables", "basic tables"},
        2: {"full tables", "full table", "full table fluency"},
        3: {"word problems", "story problems"},
        4: {"multi step", "multi-step", "multi step problems", "multistep"},
        5: {"challenge mode", "mission mode"},
    }
    for level, phrases in phrase_map.items():
        if any(phrase in normalized for phrase in phrases):
            return level
    return None


def extract_number(text: str) -> int | None:
    normalized = normalize_text(text).replace("-", " ")
    digit_match = re.search(r"\b\d+\b", normalized)
    if digit_match:
        return int(digit_match.group(0))
    total = 0
    current = 0
    found = False
    for token in normalized.split():
        if token not in NUMBER_WORDS:
            if found:
                break
            continue
        found = True
        value = NUMBER_WORDS[token]
        if value == 100:
            current = max(1, current) * 100
        else:
            current += value
    total += current
    return total if found else None
