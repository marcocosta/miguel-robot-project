# ~/robot-project/week3/face/face_controller.py
# Last updated: 20260522

from __future__ import annotations

import queue
from typing import Optional

from face_state import FaceEvent, FaceMode
from face_thread import FaceThread


class FaceController:
    supported_statuses = {
        "idle",
        "wake_required",
        "listening",
        "heard",
        "thinking",
        "looking",
        "speaking",
        "sleeping",
        "shutdown_pending",
        "shutdown",
        "confirm",
        "error",
    }
    SUPPORTED_STATUSES = supported_statuses

    def __init__(
        self,
        width: int = 1024,
        height: int = 600,
        fullscreen: bool = True,
        fps: int = 30,
    ) -> None:
        self._events: "queue.Queue[FaceEvent]" = queue.Queue()
        self._face = FaceThread(
            event_queue=self._events,
            width=width,
            height=height,
            fullscreen=fullscreen,
            fps=fps,
        )

    def start(self) -> None:
        self._face.start()

    def stop(self) -> None:
        self._face.stop()
        self._face.join(timeout=2)

    def set_mode(
        self,
        mode: FaceMode,
        text: Optional[str] = None,
        mouth_level: float = 0.0,
    ) -> None:
        self._events.put(
            FaceEvent(
                mode=mode,
                text=text,
                mouth_level=mouth_level,
            )
        )

    def idle(self) -> None:
        if self._face.state.mode == FaceMode.WAKE_REQUIRED:
            return
        self.set_mode(FaceMode.IDLE)

    def listening(self) -> None:
        self.set_mode(FaceMode.LISTENING, "Listening")

    def thinking(self) -> None:
        self.set_mode(FaceMode.THINKING, "Thinking")

    def speaking(self, text: str = "Speaking") -> None:
        self.set_mode(FaceMode.SPEAKING, text)

    def happy(self, text: str = "Happy") -> None:
        if self._face.state.mode == FaceMode.WAKE_REQUIRED:
            return
        self.set_mode(FaceMode.HAPPY, text)

    def confused(self, text: str = "Confused") -> None:
        self.set_mode(FaceMode.CONFUSED, text)

    def sleeping(self) -> None:
        self.set_mode(FaceMode.SLEEPING)

    def error(self, text: str = "Error") -> None:
        self.set_mode(FaceMode.ERROR, text)

    def status(self, state: str, text: Optional[str] = None) -> None:
        state_name = str(state or "idle").strip().lower()
        status_text = str(text or "").strip()
        text_key = status_text.lower()
        if state_name == "idle" and text_key in {"say hey miguel", 'say "hey miguel"', "say “hey miguel”"}:
            state_name = "wake_required"
        mode = {
            "wake_required": FaceMode.WAKE_REQUIRED,
            "listening": FaceMode.LISTENING,
            "heard": FaceMode.THINKING,
            "thinking": FaceMode.THINKING,
            "looking": FaceMode.THINKING,
            "speaking": FaceMode.SPEAKING,
            "idle": FaceMode.IDLE,
            "sleeping": FaceMode.SLEEPING,
            "shutdown_pending": FaceMode.CONFIRM,
            "shutdown": FaceMode.CONFIRM,
            "confirm": FaceMode.CONFIRM,
            "error": FaceMode.ERROR,
        }.get(state_name, FaceMode.IDLE)

        if state_name == "wake_required":
            status_text = 'Say "Hey Miguel"'
        elif state_name == "listening" and status_text.lower() == "your turn":
            status_text = "YOUR TURN"
        elif not status_text:
            status_text = state_name.replace("_", " ").upper()

        self.set_mode(mode, status_text)
