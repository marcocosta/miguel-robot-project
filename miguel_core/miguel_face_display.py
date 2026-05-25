"""Display-face bridge state file for future Waveshare/pygame integration."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class MiguelFaceDisplay:
    """Writes a compact face state JSON document for a separate display app."""

    VALID_STATES = {
        "idle",
        "listening",
        "thinking",
        "speaking",
        "happy",
        "confused",
        "sleeping",
        "error",
        "navigating",
        "mission",
    }

    def __init__(self, state_file: str | Path = "data/miguel_face_state.json") -> None:
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.state_file.exists():
            self.update_face("idle")

    def update_face(
        self,
        state: str,
        emotion: str | None = None,
        is_speaking: bool = False,
        look_x: float = 0.0,
        look_y: float = 0.0,
        message: str | None = None,
    ) -> dict:
        normalized_state = state if state in self.VALID_STATES else "error"
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": normalized_state,
            "emotion": emotion,
            "is_speaking": bool(is_speaking),
            "look_x": self._clamp_look(look_x),
            "look_y": self._clamp_look(look_y),
            "message": message,
            "simulated": True,
        }
        self._write_json_atomic(payload)
        print(f"[MIGUEL_FACE] update_face state={normalized_state}")
        return payload

    def get_face_state(self) -> dict:
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return self.update_face("error", message="face state unavailable")

    def _write_json_atomic(self, payload: dict) -> None:
        temp_path = self.state_file.with_suffix(self.state_file.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_path.replace(self.state_file)

    @staticmethod
    def _clamp_look(value: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        return max(-1.0, min(1.0, numeric))
