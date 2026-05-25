import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

from .safety_guard import SafetyGuard


@dataclass
class UserTurn:
    text: str
    created_at: float
    speaker: Optional[str] = None


@dataclass
class RobotReply:
    text: str
    created_at: float
    source: str = "brain"


class V7Orchestrator:
    """
    V7 queue architecture.

    This class is intentionally small for the first migration:
    - Audio worker pushes UserTurn.
    - Brain worker consumes UserTurn.
    - Speech worker consumes RobotReply.
    - CameraManager is passed in and owned separately.
    """

    def __init__(self, camera_manager, speak_fn, brain_fn):
        self.camera_manager = camera_manager
        self.speak_fn = speak_fn
        self.brain_fn = brain_fn

        self.safety = SafetyGuard()

        self.user_turns = queue.Queue()
        self.robot_replies = queue.Queue()

        self.stop_event = threading.Event()
        self.brain_thread = None
        self.speech_thread = None

    def start(self):
        self.stop_event.clear()

        self.brain_thread = threading.Thread(target=self._brain_loop, daemon=True)
        self.speech_thread = threading.Thread(target=self._speech_loop, daemon=True)

        self.brain_thread.start()
        self.speech_thread.start()

        print("[V7] Orchestrator started.")

    def stop(self):
        self.stop_event.set()
        print("[V7] Orchestrator stopping.")

    def submit_user_text(self, text, speaker=None):
        self.user_turns.put(UserTurn(text=text, speaker=speaker, created_at=time.time()))

    def _brain_loop(self):
        while not self.stop_event.is_set():
            try:
                turn = self.user_turns.get(timeout=0.2)
            except queue.Empty:
                continue

            decision = self.safety.evaluate_user_text(turn.text)
            if not decision.allowed:
                self.robot_replies.put(RobotReply(
                    text=decision.safe_reply or "I can’t help with that.",
                    created_at=time.time(),
                    source=f"safety:{decision.category}",
                ))
                continue

            try:
                reply_text = self.brain_fn(turn, self.camera_manager)
            except Exception as e:
                print("[V7] brain error:", e)
                reply_text = "I had a brain error while processing that."

            out_decision = self.safety.evaluate_assistant_reply(reply_text)
            if not out_decision.allowed:
                reply_text = out_decision.safe_reply or "I can’t help with that."

            self.robot_replies.put(RobotReply(
                text=reply_text,
                created_at=time.time(),
                source="brain",
            ))

    def _speech_loop(self):
        while not self.stop_event.is_set():
            try:
                reply = self.robot_replies.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self.speak_fn(reply.text)
            except Exception as e:
                print("[V7] speech error:", e)
