"""Guarded ROS2 HiWonder adapter for Miguel.

This module intentionally avoids importing ROS2 packages at import time. The
adapter publishes through injected test doubles by default, or through a real
ROS2 publisher only when explicitly constructed with allow_real_ros2=True.
"""

from __future__ import annotations

import importlib
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
        readiness_probe: object | None = None,
        require_safe_graph_to_arm: bool = True,
        allow_competing_publishers_override: bool = False,
    ) -> None:
        self.node = node
        self.publisher = publisher
        self.topic = topic
        self.twist_factory = twist_factory
        self.clock = clock
        self.allow_real_ros2 = allow_real_ros2
        self.readiness_probe = readiness_probe
        self.require_safe_graph_to_arm = require_safe_graph_to_arm
        self.allow_competing_publishers_override = allow_competing_publishers_override
        self.armed = False
        self.command_log: list[dict] = []
        self._latest_telemetry: dict | None = None
        self._ids = count(1)
        self._twist_class: object | None = None
        self._owned_node = False

        self.available = publisher is not None
        self.backend = "injected_publisher" if publisher is not None else "unavailable"
        self.real_ros2_enabled = False
        self.hardware_verified = False
        self.unavailable_reason = "adapter_unavailable"
        self.init_error: str | None = None

        if publisher is None and allow_real_ros2:
            self._initialize_real_ros2()

    def get_name(self) -> str:
        return "ros2"

    def is_real_hardware(self) -> bool:
        return self.hardware_verified

    def backend_status(self) -> dict:
        return {
            "adapter": self.get_name(),
            "backend": self.backend,
            "available": self.available,
            "real_ros2_enabled": self.real_ros2_enabled,
            "hardware_verified": self.hardware_verified,
            "allow_real_ros2": self.allow_real_ros2,
            "reason": "ok" if self.available else self.unavailable_reason,
            "init_error": self.init_error,
        }

    def arm(self) -> dict:
        if not self.available:
            return self._unavailable("arm")
        gate_result = self.check_arm_readiness()
        if not gate_result["ok"]:
            return self._record(
                "arm",
                ok=False,
                blocked=True,
                reason=gate_result["reason"],
                params={
                    "readiness_warnings": gate_result.get("warnings", []),
                },
                payload=gate_result.get("readiness_report"),
            )
        self.armed = True
        return self._record(
            "arm",
            ok=True,
            params={"readiness_warnings": gate_result.get("warnings", [])},
            payload=gate_result.get("readiness_report"),
        )

    def check_arm_readiness(self) -> dict:
        return self._validate_arm_readiness()

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
        if not self.available:
            payload = self._plain_twist_payload(0.0, 0.0, 0.0, 0.0)
            return self._unavailable("stop", params={"reason": reason or "requested"}, payload=payload)

        twist_result = self._twist_payload(0.0, 0.0, 0.0, 0.0)
        payload = twist_result.get("payload")
        if not twist_result["ok"]:
            return self._record(
                "stop",
                ok=False,
                blocked=True,
                error=True,
                reason=twist_result["reason"],
                params={
                    "reason": reason or "requested",
                    "error": twist_result["error"],
                },
                payload=payload,
            )

        publish_result = self._publish(payload)
        if not publish_result["ok"]:
            return self._record(
                "stop",
                ok=False,
                blocked=True,
                error=True,
                reason=publish_result["reason"],
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
        twist_result = self._twist_payload(
            self._cap_number(linear_x, -self.MAX_LINEAR_X, self.MAX_LINEAR_X),
            self._cap_number(linear_y, -self.MAX_LINEAR_Y, self.MAX_LINEAR_Y),
            self._cap_number(angular_z, -self.MAX_ANGULAR_Z, self.MAX_ANGULAR_Z),
            duration,
        )
        payload = twist_result.get("payload")
        if not twist_result["ok"]:
            return self._record(
                "drive_twist",
                ok=False,
                blocked=True,
                error=True,
                reason=twist_result["reason"],
                params={
                    "duration_sec": duration,
                    "error": twist_result["error"],
                },
                payload=payload,
            )
        publish_result = self._publish(payload)
        if not publish_result["ok"]:
            return self._record(
                "drive_twist",
                ok=False,
                blocked=True,
                error=True,
                reason=publish_result["reason"],
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

    def close(self) -> dict:
        result = self.stop("adapter close")
        self.armed = False
        if self._owned_node and self.node is not None and hasattr(self.node, "destroy_node"):
            try:
                self.node.destroy_node()
            except Exception as exc:
                result["node_destroy_error"] = str(exc)
            finally:
                self._owned_node = False
        return result

    def _initialize_real_ros2(self) -> None:
        try:
            rclpy = importlib.import_module("rclpy")
            geometry_msgs_msg = importlib.import_module("geometry_msgs.msg")
            twist_class = getattr(geometry_msgs_msg, "Twist")
        except Exception as exc:
            self.unavailable_reason = "ros2_dependency_unavailable"
            self.init_error = str(exc)
            return

        try:
            node = self.node
            if node is None:
                if not hasattr(rclpy, "ok") or not rclpy.ok():
                    rclpy.init(args=None)
                node = rclpy.create_node("miguel_hiwonder_ros2_adapter")
                self._owned_node = True
            publisher = node.create_publisher(twist_class, self.topic, 10)
        except Exception as exc:
            self.unavailable_reason = "ros2_init_error"
            self.init_error = str(exc)
            self._owned_node = False
            return

        self.node = node
        self.publisher = publisher
        self._twist_class = twist_class
        self.available = True
        self.backend = "real_ros2"
        self.real_ros2_enabled = True
        self.hardware_verified = False
        self.unavailable_reason = "ok"
        self.init_error = None

    def _validate_arm_readiness(self) -> dict:
        should_check = self.readiness_probe is not None or (
            self.backend == "real_ros2" and self.require_safe_graph_to_arm
        )
        if not should_check:
            return {"ok": True, "reason": "ok", "warnings": []}

        try:
            if self.readiness_probe is None:
                from .miguel_hiwonder_ros2_graph_probe import MiguelHiWonderRos2GraphProbe

                self.readiness_probe = MiguelHiWonderRos2GraphProbe(node=self.node)
            report = self.readiness_probe.build_readiness_report()
        except Exception as exc:
            return {
                "ok": False,
                "reason": "ros2_readiness_probe_error",
                "readiness_report": {"error": str(exc)},
                "warnings": [],
            }

        warnings = []
        cmd_vel = report.get("graph", {}).get("cmd_vel", {})
        if not cmd_vel and "cmd_vel" in report:
            cmd_vel = report.get("cmd_vel", {})
        direct_motor = report.get("graph", {}).get("direct_motor", {})
        motor_chain = report.get("graph", {}).get("motor_chain", {})

        topic_exists = cmd_vel.get("topic_exists")
        message_type = cmd_vel.get("message_type")
        subscription_count = int(cmd_vel.get("subscription_count") or 0)
        competing_publishers = list(
            report.get("competing_cmd_vel_publishers")
            or cmd_vel.get("competing_publishers")
            or []
        )
        cmd_vel_safe_to_arm = report.get("cmd_vel_safe_to_arm", cmd_vel.get("safe_to_arm"))
        external_direct_motor_publishers = list(
            report.get("external_direct_motor_publishers")
            or direct_motor.get("external_direct_motor_publishers")
            or []
        )
        direct_motor_safe_to_arm = report.get(
            "direct_motor_safe_to_arm",
            direct_motor.get("safe_direct_motor_control"),
        )
        low_level_motor_chain_ok = report.get(
            "low_level_motor_chain_ok",
            motor_chain.get("low_level_motor_chain_ok"),
        )

        if report.get("ok") is False:
            return {
                "ok": False,
                "reason": "ros2_readiness_failed",
                "readiness_report": report,
                "warnings": warnings,
            }
        if topic_exists is not True:
            return {
                "ok": False,
                "reason": "cmd_vel_topic_missing",
                "readiness_report": report,
                "warnings": warnings,
            }
        if message_type != "geometry_msgs/msg/Twist":
            return {
                "ok": False,
                "reason": "cmd_vel_wrong_message_type",
                "readiness_report": report,
                "warnings": warnings,
            }
        if subscription_count < 1:
            return {
                "ok": False,
                "reason": "cmd_vel_no_subscriber",
                "readiness_report": report,
                "warnings": warnings,
            }
        if cmd_vel_safe_to_arm is False and not (
            competing_publishers and self.allow_competing_publishers_override
        ):
            return {
                "ok": False,
                "reason": "ros2_graph_not_safe_to_arm",
                "readiness_report": report,
                "warnings": warnings,
            }
        if competing_publishers and not self.allow_competing_publishers_override:
            return {
                "ok": False,
                "reason": "ros2_graph_not_safe_to_arm",
                "readiness_report": report,
                "warnings": warnings,
            }
        if competing_publishers:
            warnings.append(
                {
                    "reason": "competing_cmd_vel_publishers_override",
                    "publishers": competing_publishers,
                }
            )
        if low_level_motor_chain_ok is False:
            return {
                "ok": False,
                "reason": "low_level_motor_chain_not_ready",
                "readiness_report": report,
                "warnings": warnings,
            }
        if direct_motor_safe_to_arm is False and not (
            external_direct_motor_publishers and self.allow_competing_publishers_override
        ):
            return {
                "ok": False,
                "reason": "direct_motor_graph_not_safe_to_arm",
                "readiness_report": report,
                "warnings": warnings,
            }
        if external_direct_motor_publishers and not self.allow_competing_publishers_override:
            return {
                "ok": False,
                "reason": "direct_motor_graph_not_safe_to_arm",
                "readiness_report": report,
                "warnings": warnings,
            }
        if external_direct_motor_publishers:
            warnings.append(
                {
                    "reason": "external_direct_motor_publishers_override",
                    "publishers": external_direct_motor_publishers,
                }
            )

        for sensor_name in ("battery", "lidar", "odom"):
            readable_key = f"{sensor_name}_readable"
            ok_key = f"{sensor_name}_ok"
            if readable_key in report and report.get(readable_key) is not True:
                return {
                    "ok": False,
                    "reason": f"{sensor_name}_not_readable",
                    "readiness_report": report,
                    "warnings": warnings,
                }
            if ok_key in report and report.get(ok_key) is not True:
                return {
                    "ok": False,
                    "reason": f"{sensor_name}_not_ready",
                    "readiness_report": report,
                    "warnings": warnings,
                }

        return {
            "ok": True,
            "reason": "ok",
            "readiness_report": report,
            "warnings": warnings,
        }

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
            reason=self.unavailable_reason,
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
                return {
                    "ok": False,
                    "reason": "publisher_unavailable",
                    "error": "publisher must expose publish(payload) or publish_twist(payload)",
                }
        except Exception as exc:
            return {"ok": False, "reason": "publisher_error", "error": str(exc)}
        return {"ok": True, "reason": "ok", "error": None}

    def _twist_payload(self, linear_x: float, linear_y: float, angular_z: float, duration_sec: float) -> dict:
        if self.twist_factory is not None:
            try:
                payload = self.twist_factory(
                    linear_x=linear_x,
                    linear_y=linear_y,
                    angular_z=angular_z,
                    duration_sec=duration_sec,
                    topic=self.topic,
                )
            except Exception as exc:
                return {
                    "ok": False,
                    "reason": "twist_factory_error",
                    "error": str(exc),
                    "payload": None,
                }
            return {"ok": True, "reason": "ok", "error": None, "payload": payload}
        if self.real_ros2_enabled and self._twist_class is not None:
            try:
                payload = self._twist_class()
                payload.linear.x = linear_x
                payload.linear.y = linear_y
                payload.linear.z = 0.0
                payload.angular.x = 0.0
                payload.angular.y = 0.0
                payload.angular.z = angular_z
            except Exception as exc:
                return {
                    "ok": False,
                    "reason": "twist_factory_error",
                    "error": str(exc),
                    "payload": None,
                }
            return {"ok": True, "reason": "ok", "error": None, "payload": payload}
        payload = self._plain_twist_payload(linear_x, linear_y, angular_z, duration_sec)
        return {"ok": True, "reason": "ok", "error": None, "payload": payload}

    def _plain_twist_payload(
        self,
        linear_x: float,
        linear_y: float,
        angular_z: float,
        duration_sec: float,
    ) -> dict:
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
            "backend": self.backend,
            "real_ros2_enabled": self.real_ros2_enabled,
            "hardware_verified": self.hardware_verified,
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
