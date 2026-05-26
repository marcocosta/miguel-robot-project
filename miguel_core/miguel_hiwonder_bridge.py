"""Simulated high-level HiWonder autonomous car bridge."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .miguel_hiwonder_adapter_base import MiguelHiWonderAdapterBase
from .miguel_hiwonder_dry_run_adapter import MiguelHiWonderDryRunAdapter
from .miguel_robot_bus import MiguelRobotBus
from .miguel_safety import MiguelSafety


class MiguelHiWonderBridge:
    """Hardware-safe adapter for future HiWonder car integration."""

    TARGET = "hiwonder_car"
    SAFETY_STOP_IF_OBSTACLE_CM = 35

    def __init__(
        self,
        robot_bus: MiguelRobotBus,
        data_dir: str | Path = "data",
        safety: MiguelSafety | None = None,
        adapter: MiguelHiWonderAdapterBase | None = None,
    ) -> None:
        self.robot_bus = robot_bus
        self.safety = safety or MiguelSafety()
        self.adapter = adapter or MiguelHiWonderDryRunAdapter()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_path = self.data_dir / "miguel_hiwonder_telemetry.json"

    def arm(self) -> dict:
        print("[MIGUEL_HIWONDER] arm")
        return self.adapter.arm()

    def disarm(self) -> dict:
        print("[MIGUEL_HIWONDER] disarm")
        return self.adapter.disarm()

    def stop(self, reason: str | None = None) -> dict:
        print(f"[MIGUEL_HIWONDER] stop reason={reason or 'none'}")
        validation = self.safety.validate_command(
            "stop",
            {"reason": reason or "requested"},
            telemetry=self.robot_bus.get_latest_telemetry(self.TARGET),
        )
        params = {"reason": reason or "requested"}
        safety_payload = self._safety_payload(validation)
        result = self.robot_bus.send_command(
            self.TARGET,
            "stop",
            params,
            safety_payload,
        )
        result["safety_validation"] = validation
        result["adapter_result"] = self.adapter.stop(params["reason"])
        return result

    def move_forward(self, speed: str = "slow", duration_sec: float = 1.0) -> dict:
        return self._movement_command("move_forward", speed, duration_sec)

    def move_backward(self, speed: str = "slow", duration_sec: float = 1.0) -> dict:
        return self._movement_command("move_backward", speed, duration_sec)

    def turn_left(self, speed: str = "slow", duration_sec: float = 1.0) -> dict:
        return self._movement_command("turn_left", speed, duration_sec)

    def turn_right(self, speed: str = "slow", duration_sec: float = 1.0) -> dict:
        return self._movement_command("turn_right", speed, duration_sec)

    def scan_area(self, duration_sec: float = 3.0) -> dict:
        validation = self.safety.validate_command(
            "scan_area",
            {"duration_sec": duration_sec},
            telemetry=self.robot_bus.get_latest_telemetry(self.TARGET),
        )
        params = validation["adjusted_params"]
        print(f"[MIGUEL_HIWONDER] scan_area duration_sec={params['duration_sec']}")
        safety_payload = self._safety_payload(validation)
        result = self.robot_bus.send_command(
            self.TARGET,
            "scan_area",
            params,
            safety_payload,
        )
        result["safety_validation"] = validation
        result["adapter_result"] = self.adapter.send_command("scan_area", params, safety_payload)
        return result

    def request_telemetry(self) -> dict:
        telemetry = self.adapter.request_telemetry()
        telemetry = self._normalize_telemetry(telemetry)
        self._write_json_atomic(self.telemetry_path, telemetry)
        self.robot_bus.record_telemetry(self.TARGET, telemetry)
        print("[MIGUEL_HIWONDER] request_telemetry")
        return telemetry

    def update_telemetry(self, telemetry: dict) -> dict:
        telemetry_record = self._normalize_telemetry(self.adapter.update_telemetry(telemetry))
        self._write_json_atomic(self.telemetry_path, telemetry_record)
        event = self.robot_bus.record_telemetry(self.TARGET, telemetry_record)
        print("[MIGUEL_HIWONDER] update_telemetry")
        return event

    def _movement_command(self, command: str, speed: str, duration_sec: float) -> dict:
        telemetry = self.robot_bus.get_latest_telemetry(self.TARGET)
        validation = self.safety.validate_command(
            command,
            {"speed": speed, "duration_sec": duration_sec},
            telemetry=telemetry,
        )
        if validation["blocked"]:
            print(f"[MIGUEL_HIWONDER] blocked command={command} reason={validation['reason']}")
            stop_params = {
                "reason": f"safety blocked {command}: {validation['reason']}",
                "requested_command": command,
            }
            safety_payload = self._safety_payload(validation)
            result = self.robot_bus.send_command(
                self.TARGET,
                "stop",
                stop_params,
                safety_payload,
            )
            result["safety_validation"] = validation
            result["adapter_result"] = self.adapter.stop(stop_params["reason"])
            return result

        params = validation["adjusted_params"]
        print(f"[MIGUEL_HIWONDER] {command} speed={params['speed']} duration_sec={params['duration_sec']}")
        safety_payload = self._safety_payload(validation)
        result = self.robot_bus.send_command(
            self.TARGET,
            command,
            params,
            safety_payload,
        )
        result["safety_validation"] = validation
        result["adapter_result"] = self.adapter.set_velocity(command, params["speed"], params["duration_sec"])
        return result

    def _safety_payload(self, validation: dict) -> dict:
        payload = dict(validation)
        payload.update(
            {
                "stop_if_obstacle_cm": self.SAFETY_STOP_IF_OBSTACLE_CM,
                "hardware_safe": True,
            }
        )
        return payload

    def _normalize_telemetry(self, telemetry: dict) -> dict:
        telemetry_record = dict(telemetry or {})
        telemetry_record.setdefault("target", self.TARGET)
        telemetry_record.setdefault("simulated", True)
        telemetry_record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        return telemetry_record

    def _read_saved_telemetry(self) -> dict | None:
        if not self.telemetry_path.exists():
            return None
        try:
            return json.loads(self.telemetry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _simulated_safe_idle_telemetry() -> dict:
        return {
            "battery_percent": 87,
            "emergency_stop": False,
            "front_clearance_cm": 120,
            "left_clearance_cm": 95,
            "nearest_obstacle_cm": 95,
            "person_detected": False,
            "person_direction": None,
            "right_clearance_cm": 110,
            "state": "idle",
            "simulated": True,
        }

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_path.replace(path)
