# ~/robot-project/week3/face/face_state.py
# Last updated: 20260522

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FaceMode(str, Enum):
    IDLE = "idle"
    WAKE_REQUIRED = "wake_required"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    HAPPY = "happy"
    CONFUSED = "confused"
    CONFIRM = "confirm"
    SLEEPING = "sleeping"
    ERROR = "error"


@dataclass(frozen=True)
class FaceEvent:
    mode: FaceMode
    text: Optional[str] = None
    mouth_level: float = 0.0
