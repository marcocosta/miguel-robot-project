"""Typed events and state owned by Miguel's conversation layer."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class InteractionState(str, Enum):
    IDLE = "IDLE"
    ENGAGED = "ENGAGED"
    LISTENING = "LISTENING"
    END_CANDIDATE = "END_CANDIDATE"
    PREPARING = "PREPARING"
    SPEAKING = "SPEAKING"


class FloorOwner(str, Enum):
    NONE = "NONE"
    HUMAN = "HUMAN"
    MIGUEL = "MIGUEL"


class Addressee(str, Enum):
    MIGUEL = "MIGUEL"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class VadEvent:
    timestamp: float
    active: bool
    source: str = "audio_rms"
    doa: Optional[float] = None


@dataclass(frozen=True)
class AsrPartialEvent:
    timestamp: float
    text: str
    language: str = "en"


@dataclass(frozen=True)
class AsrFinalEvent:
    timestamp: float
    text: str
    language: str = "en"


@dataclass(frozen=True)
class WakeEvent:
    timestamp: float
    phrase: str
    person: Optional[str] = None


@dataclass(frozen=True)
class RobotSpeechStartedEvent:
    timestamp: float
    text: str


@dataclass(frozen=True)
class RobotSpeechEndedEvent:
    timestamp: float
    text: str
    asked_question: bool = False
    expected_reply_person: Optional[str] = None
