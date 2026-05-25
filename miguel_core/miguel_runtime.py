"""Standalone Miguel Core Lab runtime composition."""

from __future__ import annotations

from pathlib import Path

from .miguel_face_display import MiguelFaceDisplay
from .miguel_hiwonder_bridge import MiguelHiWonderBridge
from .miguel_learning import MiguelLearning
from .miguel_mission import MiguelMissionController
from .miguel_motion import MiguelMotion
from .miguel_navigation import MiguelNavigationDecider
from .miguel_personality import MiguelPersonality
from .miguel_robot_bus import MiguelRobotBus
from .miguel_safety import MiguelSafety


class MiguelRuntime:
    """Composes sandbox-only Miguel Core services."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.robot_bus = MiguelRobotBus(self.data_dir)
        self.safety = MiguelSafety()
        self.mission_controller = MiguelMissionController()
        self.hiwonder = MiguelHiWonderBridge(self.robot_bus, self.data_dir, self.safety)
        self.navigation = MiguelNavigationDecider()
        self.learning = MiguelLearning(self.data_dir)
        self.personality = MiguelPersonality()
        self.face = MiguelFaceDisplay(self.data_dir / "miguel_face_state.json")
        self.motion = MiguelMotion()
        self.started = False

    def start(self) -> dict:
        self.started = True
        self.face.update_face("idle", emotion="ready", message="Miguel Core Lab ready")
        print("[MIGUEL_RUNTIME] start")
        return {"status": "started", "simulated": True}

    def shutdown(self) -> dict:
        self.hiwonder.stop("runtime shutdown")
        self.face.update_face("sleeping", emotion="calm", message="Miguel Core Lab stopped")
        self.started = False
        print("[MIGUEL_RUNTIME] shutdown")
        return {"status": "shutdown", "simulated": True}

    def set_face_state(self, *args, **kwargs) -> dict:
        return self.face.update_face(*args, **kwargs)

    def record_interaction(self, *args, **kwargs) -> dict:
        return self.learning.record_interaction(*args, **kwargs)

    def run_hiwonder_mission_step(self, mission: str) -> dict:
        mission_status = self.mission_controller.get_status()
        if mission_status["state"] == "idle" or mission_status["current_mission"] != mission:
            mission_status = self.mission_controller.start_mission(mission)

        self.face.update_face("navigating", emotion="focused", message=mission)
        telemetry = self.hiwonder.request_telemetry()
        decision = self.navigation.decide_next_action(mission, telemetry)
        command_result = self._execute_hiwonder_decision(decision)
        self.face.update_face("mission", emotion="focused", message=decision.get("action"))

        step_result = {
            "mission": mission,
            "decision": decision,
            "command_result": command_result,
        }
        mission_status = self.mission_controller.record_step(step_result)
        mission_status = self._update_mission_after_command(decision, command_result)
        print(f"[MIGUEL_RUNTIME] run_hiwonder_mission_step mission={mission}")
        return {
            "mission_status": mission_status,
            "telemetry": telemetry,
            "decision": decision,
            "command_result": command_result,
        }

    def _execute_hiwonder_decision(self, decision: dict) -> dict:
        action = decision.get("action", "stop")
        params = dict(decision.get("params") or {})
        if action == "stop":
            return self.hiwonder.stop(params.get("reason") or decision.get("reason"))
        if action == "move_forward":
            return self.hiwonder.move_forward(**params)
        if action == "move_backward":
            return self.hiwonder.move_backward(**params)
        if action == "turn_left":
            return self.hiwonder.turn_left(**params)
        if action == "turn_right":
            return self.hiwonder.turn_right(**params)
        if action == "scan_area":
            return self.hiwonder.scan_area(**params)
        return self.hiwonder.stop(f"unsupported action: {action}")

    def _update_mission_after_command(self, decision: dict, command_result: dict) -> dict:
        validation = command_result.get("safety_validation") or command_result.get("payload", {}).get("safety", {})
        blocked = bool(validation.get("blocked"))
        action = decision.get("action")
        if action == "stop":
            reason = str(decision.get("reason") or validation.get("reason") or "")
        else:
            reason = str(validation.get("reason") or decision.get("reason") or "")

        if reason == "emergency_stop":
            return self.mission_controller.emergency_stop(reason)
        if blocked:
            return self.mission_controller.fail_mission(reason or "safety_blocked")
        if reason.startswith("battery") or reason == "battery_below_minimum":
            return self.mission_controller.fail_mission(reason)
        if action == "stop":
            return self.mission_controller.stop_mission(reason or "stop")
        return self.mission_controller.get_status()
