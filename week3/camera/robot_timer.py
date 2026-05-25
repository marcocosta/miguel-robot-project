"""Small non-blocking local timer helper for Miguel."""

from __future__ import annotations

import re
import time


_active_timer: dict | None = None

_NUMBER_WORDS = {
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
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
}


def parse_timer_command(text: str) -> dict | None:
    normalized = _normalize(text)
    if not normalized:
        return None

    if normalized in {"cancel timer", "stop timer", "cancel the timer", "stop the timer", "turn off the timer"}:
        return {"intent": "cancel_timer"}

    if normalized in {
        "how is the timer",
        "how much time is left",
        "how much time left",
        "is the timer still running",
        "timer",
        "timer status",
        "what about the timer",
        "what is the timer status",
        "how long is left",
        "how long left",
    }:
        return {"intent": "timer_status"}

    start_text = _strip_start_politeness(normalized)
    if not _looks_like_timer_start(start_text):
        return None

    number_pattern = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty|forty|fifty|sixty)"
    patterns = [
        rf"\b(?:set|start|put)\s+(?:a\s+)?timer\s+(?:for\s+)?({number_pattern})\s*(second|seconds|minute|minutes)\b",
        rf"\btimer\s+for\s+({number_pattern})\s*(second|seconds|minute|minutes)\b",
        rf"\b(?:set|start|put)\s+(?:a\s+)?({number_pattern})\s*(second|seconds|minute|minutes)\s+timer\b",
    ]
    parsed = None
    for pattern in patterns:
        match = re.search(pattern, start_text)
        if match:
            parsed = (_parse_number(match.group(1)), match.group(2))
            break
    if parsed is None:
        return None

    amount, unit = parsed
    if amount is None:
        return None
    seconds = amount * 60 if unit.startswith("minute") else amount
    if seconds <= 0:
        return None
    return {"intent": "start_timer", "seconds": seconds}


def start_timer(seconds: int) -> dict:
    global _active_timer
    duration = max(1, int(seconds))
    now = time.monotonic()
    _active_timer = {
        "duration_seconds": duration,
        "started_at": now,
        "ends_at": now + duration,
    }
    return {"ok": True, "active": True, "seconds": duration, "remaining_seconds": duration}


def cancel_timer() -> dict:
    global _active_timer
    was_active = _active_timer is not None
    _active_timer = None
    return {"ok": True, "active": False, "canceled": was_active}


def get_timer_status() -> dict:
    if _active_timer is None:
        return {"ok": True, "active": False, "remaining_seconds": 0}
    remaining = max(0, int(round(float(_active_timer["ends_at"]) - time.monotonic())))
    return {
        "ok": True,
        "active": remaining > 0,
        "remaining_seconds": remaining,
        "duration_seconds": int(_active_timer["duration_seconds"]),
    }


def timer_tick() -> dict | None:
    global _active_timer
    if _active_timer is None:
        return None
    now = time.monotonic()
    if now < float(_active_timer["ends_at"]):
        return None
    expired = dict(_active_timer)
    _active_timer = None
    return {
        "intent": "timer_expired",
        "duration_seconds": int(expired["duration_seconds"]),
        "active": False,
    }


def _normalize(text: str) -> str:
    lowered = str(text or "").lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def _strip_start_politeness(text: str) -> str:
    cleaned = text
    prefixes = [
        "here we go",
        "can you please",
        "could you please",
        "would you please",
        "can you",
        "could you",
        "would you",
        "please",
    ]
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if cleaned == prefix:
                return ""
            if cleaned.startswith(prefix + " "):
                cleaned = cleaned[len(prefix):].strip()
                changed = True
                break
    return cleaned


def _looks_like_timer_start(text: str) -> bool:
    return "timer" in text and any(marker in text for marker in {"set", "start", "put", "timer for"})


def _parse_number(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    return _NUMBER_WORDS.get(value)


if __name__ == "__main__":
    examples = [
        "can you put a timer for ten seconds?",
        "how is the timer?",
        "timer",
        "cancel the timer",
        "set a timer for one minute",
    ]
    for example in examples:
        print(example, "->", parse_timer_command(example))
