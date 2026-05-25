"""Mission lifecycle controller for Miguel Core Lab."""

from __future__ import annotations

from datetime import datetime, timezone


class MiguelMissionController:
    """Track high-level mission state and recent step outcomes."""

    STATES = {"idle", "active", "paused", "completed", "failed", "stopped", "emergency_stop"}

    def __init__(self) -> None:
        self.current_mission: str | None = None
        self.state = "idle"
        self.started_at: str | None = None
        self.updated_at = self._utc_now()
        self.step_count = 0
        self.last_reason: str | None = None
        self.recent_steps: list[dict] = []

    def start_mission(self, mission: str) -> dict:
        self.current_mission = mission
        self.state = "active"
        self.started_at = self._utc_now()
        self.updated_at = self.started_at
        self.step_count = 0
        self.last_reason = None
        self.recent_steps = []
        print(f"[MIGUEL_MISSION] start mission={mission}")
        return self.get_status()

    def pause_mission(self, reason: str | None = None) -> dict:
        self.state = "paused"
        return self._mark(reason, "pause")

    def resume_mission(self) -> dict:
        self.state = "active"
        return self._mark(None, "resume")

    def stop_mission(self, reason: str | None = None) -> dict:
        self.state = "stopped"
        return self._mark(reason, "stop")

    def emergency_stop(self, reason: str | None = None) -> dict:
        self.state = "emergency_stop"
        return self._mark(reason or "emergency_stop", "emergency_stop")

    def complete_mission(self, summary: str | None = None) -> dict:
        self.state = "completed"
        return self._mark(summary, "complete")

    def fail_mission(self, reason: str) -> dict:
        self.state = "failed"
        return self._mark(reason, "fail")

    def get_status(self) -> dict:
        return {
            "current_mission": self.current_mission,
            "state": self.state,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "step_count": self.step_count,
            "last_reason": self.last_reason,
            "recent_steps": list(self.recent_steps),
        }

    def record_step(self, step_result: dict) -> dict:
        self.step_count += 1
        self.updated_at = self._utc_now()
        self.recent_steps.append(dict(step_result))
        self.recent_steps = self.recent_steps[-20:]
        print(f"[MIGUEL_MISSION] record_step count={self.step_count}")
        return self.get_status()

    def _mark(self, reason: str | None, action: str) -> dict:
        self.last_reason = reason
        self.updated_at = self._utc_now()
        print(f"[MIGUEL_MISSION] {action} reason={reason or 'none'}")
        return self.get_status()

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()
