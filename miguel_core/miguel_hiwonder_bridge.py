"""Simulated high-level HiWonder autonomous car bridge."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .miguel_robot_bus import MiguelRobotBus


class MiguelHiWonderBridge:
    """Hardware-safe adapter for future HiWonder car integration."""

    TARGET = "hiwonder_car"
    SAFETY_STOP_IF_OBSTACLE_CM = 35

    def __init__(self, robot_bus: MiguelRobotBus, data_dir: str | Path = "data") -> None:
        self.robot_bus = robot_bus
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_path = self.data_dir / "miguel_hiwonder_telemetry.json"

    def stop(self, reason: str | None = None) -> dict:
        print(f"[MIGUEL_HIWONDER] stop reason={reason or 'none'}")
        return self.robot_bus.send_command(
            self.TARGET,
            "stop",
            {"reason": reason or "requested"},
            {"hardware_safe": True},
        )

    def move_forward(self, speed: str = "slow", duration_sec: float = 1.0) -> dict:
        return self._movement_command("move_forward", speed, duration_sec)

    def move_backward(self, speed: str = "slow", duration_sec: float = 1.0) -> dict:
        return self._movement_command("move_backward", speed, duration_sec)

    def turn_left(self, speed: str = "slow", duration_sec: float = 1.0) -> dict:
        return self._movement_command("turn_left", speed, duration_sec)

    def turn_right(self, speed: str = "slow", duration_sec: float = 1.0) -> dict:
        return self._movement_command("turn_right", speed, duration_sec)

    def scan_area(self, duration_sec: float = 3.0) -> dict:
        print(f"[MIGUEL_HIWONDER] scan_area duration_sec={duration_sec}")
        return self.robot_bus.send_command(
            self.TARGET,
            "scan_area",
            {"duration_sec": max(0.0, float(duration_sec))},
            {"hardware_safe": True},
        )

    def request_telemetry(self) -> dict:
        telemetry = self.robot_bus.get_latest_telemetry(self.TARGET)
        if telemetry is None:
            telemetry = self._read_saved_telemetry()
        if telemetry is None:
            telemetry = self._simulated_safe_idle_telemetry()
            self.update_telemetry(telemetry)
        print("[MIGUEL_HIWONDER] request_telemetry")
        return telemetry

    def update_telemetry(self, telemetry: dict) -> dict:
        telemetry_record = dict(telemetry)
        telemetry_record.setdefault("target", self.TARGET)
        telemetry_record.setdefault("simulated", True)
        telemetry_record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self._write_json_atomic(self.telemetry_path, telemetry_record)
        event = self.robot_bus.record_telemetry(self.TARGET, telemetry_record)
        print("[MIGUEL_HIWONDER] update_telemetry")
        return event

    def _movement_command(self, command: str, speed: str, duration_sec: float) -> dict:
        safe_duration = max(0.0, float(duration_sec))
        print(f"[MIGUEL_HIWONDER] {command} speed={speed} duration_sec={safe_duration}")
        return self.robot_bus.send_command(
            self.TARGET,
            command,
            {"speed": speed, "duration_sec": safe_duration},
            {
                "stop_if_obstacle_cm": self.SAFETY_STOP_IF_OBSTACLE_CM,
                "hardware_safe": True,
            },
        )

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
