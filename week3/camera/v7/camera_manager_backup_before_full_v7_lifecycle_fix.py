import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2


@dataclass
class FrameSnapshot:
    frame: Any
    captured_at: float
    source: str = "camera_manager"


class LatestFrameMessage:
    """
    Small adapter so legacy V6 functions that expect msg.getCvFrame()
    can consume CameraManager's latest frame without reading the OAK queue.
    """

    def __init__(self, frame, captured_at: float):
        self._frame = frame
        self.captured_at = captured_at

    def getCvFrame(self):
        return self._frame.copy()


class CameraManager:
    """
    Single owner for OAK camera frames.

    Rule:
    - Only this class reads from the DepthAI camera queue.
    - Other functions get latest-frame copies through this class.
    - It also mimics tryGet/get/tryGetAll so older V6 functions can use it
      without directly touching the OAK queue.
    """

    def __init__(self, camera_queue, max_frame_age_seconds=0.75):
        self.camera_queue = camera_queue
        self.max_frame_age_seconds = max_frame_age_seconds

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

        self.latest_frame = None
        self.latest_frame_at = 0.0

        self.latest_face_state = {
            "face_detected": False,
            "face_count": 0,
            "recognized_person": None,
            "source": "camera_manager_initial",
            "updated_at": 0.0,
        }

    def start(self):
        if self.thread and self.thread.is_alive():
            return

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print("[V7 CAMERA] CameraManager started.")

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        print("[V7 CAMERA] CameraManager stopped.")

    def _run(self):
        while not self.stop_event.is_set():
            try:
                msg = self._get_newest_raw_msg()

                if msg is None:
                    time.sleep(0.02)
                    continue

                frame = msg.getCvFrame()

                with self.lock:
                    self.latest_frame = frame.copy()
                    self.latest_frame_at = time.time()

            except Exception as e:
                print("[V7 CAMERA] camera loop error:", e)
                time.sleep(0.1)

    def _get_newest_raw_msg(self):
        latest = None

        try:
            if hasattr(self.camera_queue, "tryGetAll"):
                msgs = self.camera_queue.tryGetAll()
                if msgs:
                    latest = msgs[-1]

            if latest is None and hasattr(self.camera_queue, "tryGet"):
                while True:
                    msg = self.camera_queue.tryGet()
                    if msg is None:
                        break
                    latest = msg

            if latest is None and hasattr(self.camera_queue, "get"):
                latest = self.camera_queue.get()

            return latest

        except Exception as e:
            print("[V7 CAMERA] get newest frame error:", e)
            return None

    def get_latest_frame(self, require_fresh=True, wait_timeout=1.2) -> Optional[FrameSnapshot]:
        deadline = time.time() + wait_timeout

        while time.time() < deadline:
            with self.lock:
                if self.latest_frame is not None:
                    age = time.time() - self.latest_frame_at

                    if not require_fresh or age <= self.max_frame_age_seconds:
                        return FrameSnapshot(
                            frame=self.latest_frame.copy(),
                            captured_at=self.latest_frame_at,
                        )

            time.sleep(0.03)

        return None

    def get_latest_message(self, require_fresh=True, wait_timeout=1.2) -> Optional[LatestFrameMessage]:
        snap = self.get_latest_frame(require_fresh=require_fresh, wait_timeout=wait_timeout)
        if snap is None:
            return None
        return LatestFrameMessage(snap.frame, snap.captured_at)

    # Queue-like compatibility API for legacy V6 helpers.
    def get(self):
        msg = self.get_latest_message(require_fresh=True, wait_timeout=1.2)
        if msg is None:
            raise RuntimeError("No fresh camera frame available from CameraManager.")
        return msg

    def tryGet(self):
        return self.get_latest_message(require_fresh=False, wait_timeout=0.02)

    def tryGetAll(self):
        msg = self.get_latest_message(require_fresh=False, wait_timeout=0.02)
        return [msg] if msg is not None else []

    def save_latest_frame(self, prefix="v7_scene") -> Optional[Path]:
        snap = self.get_latest_frame(require_fresh=True, wait_timeout=1.5)

        if snap is None:
            return None

        out_dir = Path.home() / "robot-project/week3/debug_snapshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{prefix}_{int(time.time())}.jpg"

        cv2.imwrite(str(path), snap.frame)
        return path

    def update_face_state(self, face_state: dict):
        with self.lock:
            face_state = dict(face_state or {})
            face_state["updated_at"] = time.time()
            face_state["source"] = "camera_manager_face_worker"
            self.latest_face_state = face_state

    def get_face_state(self, max_age_seconds=1.5) -> dict:
        with self.lock:
            state = dict(self.latest_face_state)

        age = time.time() - float(state.get("updated_at", 0.0))

        if age > max_age_seconds:
            return {
                "face_detected": False,
                "face_count": 0,
                "face_position": "none",
                "recognized_person": None,
                "recognition_score": None,
                "recognition_margin": None,
                "recognition_votes": {},
                "recognition_scores": {},
                "recognizer": "camera_manager_face_state_stale",
                "source": "camera_manager_face_state_stale",
                "age": age,
            }

        return state
