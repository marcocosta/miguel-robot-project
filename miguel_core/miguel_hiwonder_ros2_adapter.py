"""Guarded ROS2 HiWonder adapter skeleton for Miguel.

This module intentionally avoids importing ROS2 packages at import time. The
Stage 3 adapter only publishes through explicitly injected test doubles or
future Miguel-owned ROS2 objects.
"""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import count
from typing import Callable

from .miguel_hiwonder_adapter_base import MiguelHiWonderAdapterBase


class MiguelHiWonderRos2Adapter(MiguelHiWonderAdapterBase):
    """Translate Miguel intents into guarded ROS2 Twist-like payloads."""

    TARGET = "hiwonder_car"
    DEFAULT_TOPIC = "/controller/cmd_vel"
    MOVEMENT_COMMANDS = {"move_forward", "move_backward", "turn_left", "turn_right"}
    MAX_LINEAR_X = 0.12
    MAX_LINEAR_Y = 0.0
    MAX_ANGULAR_Z = 0.45
    MAX_DURATION_SEC = 2.0
    SPEED_SCALE = {"slow": 1.0}

    def __init__(
        self,
        node: object | None = None,
        publisher: object | None = None,
        topic: str = DEFAULT_TOPIC,
        twist_factory: Callable[..., object] | None = None,
        clock: object | None = None,
        allow_real_ros2: bool = False,
    ) -> None:
        self.node = node
        self.publisher = publisher
        self.topic = topic
        self.twist_factory = twist_factory
        self.clock = clock
        self.allow_real_ros2 = allow_real_ros2
        self.armed = False
        self.command_log: list[dict] = []
        self._latest_telemetry: dict | None = None
        self._ids = count(1)

        # Stage 3 deliberately does not create ROS2 nodes/publishers. Future
        # hardware smoke-test stages can add a guarded implementation here.
        self.available = publisher is not None

    def get_name(self) -> str:
        return "ros2"

    def is_real_hardware(self) -> bool:
        return self.available and not self._uses_dict_payloads()

    def arm(self) -> dict:
        if not self.available:
            return self._unavailable("arm")
        self.armed = True
        return self._record("arm", ok=True)

    def disarm(self) -> dict:
        stop_result = self.stop("adapter disarm")
        self.armed = False
        result = self._record(
            "disarm",
            ok=stop_result["ok"],
            blocked=stop_result.get("blocked", False),
            error=stop_result.get("error", False),
            reason=stop_result["reason"],
            params={"reason": "adapter disarm"},
            payload=stop_result.get("payload"),
        )
        result["stop_result"] = stop_result
        return result

    def stop(self, reason: str | None = None) -> dict:
        payload = self._twist_payload(0.0, 0.0, 0.0, 0.0)
        if not self.available:
            return self._unavailable("stop", params={"reason": reason or "requested"}, payload=payload)

        publish_result = self._publish(payload)
        if not publish_result["ok"]:
            return self._record(
                "stop",
                ok=False,
                blocked=True,
                error=True,
                reason="publisher_error",
                params={
                    "reason": reason or "requested",
                    "error": publish_result["error"],
                },
                payload=payload,
            )
        return self._record(
            "stop",
            ok=True,
            params={"reason": reason or "requested"},
            payload=payload,
        )

    def drive_twist(
        self,
        linear_x: float,
        linear_y: float,
        angular_z: float,
        duration_sec: float,
    ) -> dict:
        if not self.available:
            return self._unavailable("drive_twist", params={"duration_sec": duration_sec})
        if not self.armed:
            return self._record(
                "drive_twist",
                ok=False,
                blocked=True,
                reason="adapter_disarmed",
                params={"duration_sec": duration_sec},
            )

        duration = self._cap_number(duration_sec, 0.0, self.MAX_DURATION_SEC)
        payload = self._twist_payload(
            self._cap_number(linear_x, -self.MAX_LINEAR_X, self.MAX_LINEAR_X),
            self._cap_number(linear_y, -self.MAX_LINEAR_Y, self.MAX_LINEAR_Y),
            self._cap_number(angular_z, -self.MAX_ANGULAR_Z, self.MAX_ANGULAR_Z),
            duration,
        )
        publish_result = self._publish(payload)
        if not publish_result["ok"]:
            return self._record(
                "drive_twist",
                ok=False,
                blocked=True,
                error=True,
                reason="publisher_error",
                params={
                    "duration_sec": duration,
                    "error": publish_result["error"],
                },
                payload=payload,
            )

        result = self._record("drive_twist", ok=True, params={"duration_sec": duration}, payload=payload)
        result["followup_stop"] = self.stop("timed movement complete")
        if not result["followup_stop"]["ok"]:
            result["ok"] = False
            result["blocked"] = True
            result["error"] = True
            result["reason"] = result["followup_stop"]["reason"]
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
        return self._record(
            "beep",
            ok=False,
            blocked=True,
            reason="unsupported_by_ros2_adapter",
            params={"freq": int(freq), "duration_sec": duration},
        )

    def set_led(
        self,
        led_id: int,
        on_time: float,
        off_time: float,
        repeat: int = 1,
    ) -> dict:
        return self._record(
            "set_led",
            ok=False,
            blocked=True,
            reason="unsupported_by_ros2_adapter",
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
                params.get("duration_sec") or 0.0,
            )
        elif command == "stop":
            result = self.stop(str(params.get("reason") or "requested"))
        else:
            result = self._record(command, ok=False, blocked=True, reason="unknown_command", params=params)
        result["safety"] = safety or {}
        return result

    def request_telemetry(self) -> dict:
        if self._latest_telemetry is None:
            self._latest_telemetry = self.update_telemetry(
                {
                    "battery_percent": None,
                    "emergency_stop": None,
                    "front_clearance_cm": None,
                    "nearest_obstacle_cm": None,
                    "state": "ros2_unavailable" if not self.available else "idle",
                }
            )
        return dict(self._latest_telemetry)

    def update_telemetry(self, telemetry: dict) -> dict:
        telemetry_record = dict(telemetry or {})
        telemetry_record.setdefault("target", self.TARGET)
        telemetry_record.setdefault("simulated", True)
        telemetry_record.setdefault("ros2", True)
        telemetry_record.setdefault("timestamp", self._utc_now())
        self._latest_telemetry = telemetry_record
        return dict(telemetry_record)

    def close(self) -> None:
        self.stop("adapter close")
        self.armed = False

    def _unavailable(
        self,
        command: str,
        params: dict | None = None,
        payload: object | None = None,
    ) -> dict:
        return self._record(
            command,
            ok=False,
            blocked=True,
            reason="adapter_unavailable",
            params=params or {"allow_real_ros2": self.allow_real_ros2},
            payload=payload,
        )

    def _publish(self, payload: object) -> dict:
        try:
            if self.publisher is None:
                raise RuntimeError("publisher unavailable")
            if hasattr(self.publisher, "publish_twist"):
                self.publisher.publish_twist(payload)
            elif hasattr(self.publisher, "publish"):
                self.publisher.publish(payload)
            else:
                raise TypeError("publisher must expose publish(payload) or publish_twist(payload)")
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "error": None}

    def _twist_payload(self, linear_x: float, linear_y: float, angular_z: float, duration_sec: float) -> object:
        if self.twist_factory is not None:
            return self.twist_factory(
                linear_x=linear_x,
                linear_y=linear_y,
                angular_z=angular_z,
                duration_sec=duration_sec,
                topic=self.topic,
            )
        return {
            "topic": self.topic,
            "linear": {"x": linear_x, "y": linear_y, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": angular_z},
            "duration_sec": duration_sec,
            "source": "miguel",
            "adapter": "ros2",
        }

    def _uses_dict_payloads(self) -> bool:
        return self.twist_factory is None

    def _record(
        self,
        command: str,
        ok: bool,
        params: dict | None = None,
        payload: object | None = None,
        blocked: bool = False,
        error: bool = False,
        reason: str | None = None,
    ) -> dict:
        result = {
            "id": next(self._ids),
            "ok": ok,
            "blocked": blocked,
            "error": error,
            "reason": reason or ("ok" if ok else "blocked"),
            "dry_run": False,
            "ros2": True,
            "adapter": self.get_name(),
            "armed": self.armed,
            "available": self.available,
            "command": command,
            "params": params or {},
            "payload": payload,
            "timestamp": self._utc_now(),
        }
        self.command_log.append(result)
        return result

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
