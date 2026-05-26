"""Hardware-safe dry-run HiWonder adapter for Miguel Core Lab."""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import count

from .miguel_hiwonder_adapter_base import MiguelHiWonderAdapterBase


class MiguelHiWonderDryRunAdapter(MiguelHiWonderAdapterBase):
    """Default adapter that never sends commands to real hardware."""

    TARGET = "hiwonder_car"
    MOVEMENT_COMMANDS = {"move_forward", "move_backward", "turn_left", "turn_right"}
    MAX_LINEAR_X = 0.12
    MAX_LINEAR_Y = 0.0
    MAX_ANGULAR_Z = 0.45
    MAX_DURATION_SEC = 2.0
    SPEED_SCALE = {"slow": 1.0}

    def __init__(self) -> None:
        self._latest_telemetry: dict | None = None
        self.armed = False
        self.command_log: list[dict] = []
        self._ids = count(1)

    def get_name(self) -> str:
        return "dry_run"

    def arm(self) -> dict:
        self.armed = True
        return self._record("arm", ok=True)

    def disarm(self) -> dict:
        self.armed = False
        return self._record("disarm", ok=True)

    def stop(self, reason: str | None = None) -> dict:
        return self._record(
            "stop",
            ok=True,
            params={"reason": reason or "requested"},
            twist=self._zero_twist(),
        )

    def drive_twist(
        self,
        linear_x: float,
        linear_y: float,
        angular_z: float,
        duration_sec: float,
    ) -> dict:
        if not self.armed:
            return self._record(
                "drive_twist",
                ok=False,
                blocked=True,
                reason="adapter_disarmed",
                params={"duration_sec": duration_sec},
                twist=self._zero_twist(),
            )

        duration = self._cap_number(duration_sec, 0.0, self.MAX_DURATION_SEC)
        twist = {
            "linear_x": self._cap_number(linear_x, -self.MAX_LINEAR_X, self.MAX_LINEAR_X),
            "linear_y": self._cap_number(linear_y, -self.MAX_LINEAR_Y, self.MAX_LINEAR_Y),
            "angular_z": self._cap_number(angular_z, -self.MAX_ANGULAR_Z, self.MAX_ANGULAR_Z),
        }
        result = self._record(
            "drive_twist",
            ok=True,
            params={"duration_sec": duration},
            twist=twist,
        )
        result["followup_stop"] = self.stop("timed movement complete")
        return result

    def set_velocity(self, command: str, speed: str = "slow", duration_sec: float = 1.0) -> dict:
        normalized = str(command or "").strip()
        if normalized == "stop":
            return self.stop("velocity stop")
        if normalized not in self.MOVEMENT_COMMANDS:
            return self._record(
                normalized or "unknown",
                ok=False,
                blocked=True,
                reason="unknown_movement_command",
                params={"speed": speed, "duration_sec": duration_sec},
            )

        normalized_speed = str(speed or "slow")
        if normalized_speed not in self.SPEED_SCALE:
            normalized_speed = "slow"
        scale = self.SPEED_SCALE[normalized_speed]
        linear = self.MAX_LINEAR_X * scale
        angular = self.MAX_ANGULAR_Z * scale
        twist_by_command = {
            "move_forward": (linear, 0.0, 0.0),
            "move_backward": (-linear, 0.0, 0.0),
            "turn_left": (0.0, 0.0, angular),
            "turn_right": (0.0, 0.0, -angular),
        }
        linear_x, linear_y, angular_z = twist_by_command[normalized]
        result = self.drive_twist(linear_x, linear_y, angular_z, duration_sec)
        result["command"] = normalized
        result["params"]["speed"] = normalized_speed
        return result

    def beep(self, freq: int, duration_sec: float) -> dict:
        duration = self._cap_number(duration_sec, 0.0, 1.0)
        return self._record("beep", ok=True, params={"freq": int(freq), "duration_sec": duration})

    def set_led(
        self,
        led_id: int,
        on_time: float,
        off_time: float,
        repeat: int = 1,
    ) -> dict:
        return self._record(
            "set_led",
            ok=True,
            params={
                "led_id": int(led_id),
                "on_time": self._cap_number(on_time, 0.0, 10.0),
                "off_time": self._cap_number(off_time, 0.0, 10.0),
                "repeat": max(0, int(repeat)),
            },
        )

    def send_command(
        self,
        command: str,
        params: dict | None = None,
        safety: dict | None = None,
    ) -> dict:
        params = params or {}
        if command in self.MOVEMENT_COMMANDS:
            result = self.set_velocity(
                command,
                str(params.get("speed") or "slow"),
                float(params.get("duration_sec") or 0.0),
            )
        elif command == "stop":
            result = self.stop(str(params.get("reason") or "requested"))
        else:
            result = self._record(command, ok=True, params=params)
        result["safety"] = safety or {}
        print(f"[MIGUEL_HIWONDER_DRY_RUN] command={command}")
        return result

    def request_telemetry(self) -> dict:
        if self._latest_telemetry is None:
            self._latest_telemetry = self._simulated_safe_idle_telemetry()
        telemetry = dict(self._latest_telemetry)
        telemetry.setdefault("timestamp", self._utc_now())
        print("[MIGUEL_HIWONDER_DRY_RUN] request_telemetry")
        return telemetry

    def update_telemetry(self, telemetry: dict) -> dict:
        telemetry_record = dict(telemetry or {})
        telemetry_record.setdefault("target", self.TARGET)
        telemetry_record.setdefault("simulated", True)
        telemetry_record.setdefault("timestamp", self._utc_now())
        self._latest_telemetry = telemetry_record
        print("[MIGUEL_HIWONDER_DRY_RUN] update_telemetry")
        return dict(telemetry_record)

    def _simulated_safe_idle_telemetry(self) -> dict:
        return self.update_telemetry(
            {
                "battery_percent": 87,
                "emergency_stop": False,
                "front_clearance_cm": 120,
                "left_clearance_cm": 95,
                "nearest_obstacle_cm": 95,
                "person_detected": False,
                "person_direction": None,
                "right_clearance_cm": 110,
                "state": "idle",
            }
        )

    def close(self) -> None:
        self.stop("adapter close")
        self.armed = False

    def _record(
        self,
        command: str,
        ok: bool,
        params: dict | None = None,
        twist: dict | None = None,
        blocked: bool = False,
        reason: str | None = None,
    ) -> dict:
        result = {
            "id": next(self._ids),
            "ok": ok,
            "blocked": blocked,
            "reason": reason or ("ok" if ok else "blocked"),
            "dry_run": True,
            "adapter": self.get_name(),
            "armed": self.armed,
            "command": command,
            "params": params or {},
            "twist": twist,
            "timestamp": self._utc_now(),
        }
        self.command_log.append(result)
        return result

    @staticmethod
    def _zero_twist() -> dict:
        return {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}

    @staticmethod
    def _cap_number(value: object, minimum: float, maximum: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        return max(minimum, min(maximum, numeric))

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()
