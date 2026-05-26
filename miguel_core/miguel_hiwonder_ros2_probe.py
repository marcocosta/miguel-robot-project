"""Safe ROS2 readiness probe for Miguel HiWonder integration.

The probe never publishes Twist messages and never touches the motion topic. It
only checks imports, reads environment variables, and can create/destroy a probe
node when explicitly asked.
"""

from __future__ import annotations

import importlib
import os
from datetime import datetime, timezone


class MiguelHiWonderRos2Probe:
    """Read-only ROS2 discovery helper for future hardware smoke-test stages."""

    NODE_NAME = "miguel_hiwonder_ros2_probe"

    def __init__(
        self,
        rclpy_module: object | None = None,
        node: object | None = None,
        expected_topic: str = "/controller/cmd_vel",
        expected_msg_type: str = "geometry_msgs/msg/Twist",
    ) -> None:
        self.rclpy_module = rclpy_module
        self.node = node
        self.expected_topic = expected_topic
        self.expected_msg_type = expected_msg_type
        self._owned_node = False

    def check_imports(self) -> dict:
        rclpy_available = self.rclpy_module is not None
        geometry_msgs_available = False
        reason = "ok"
        errors: dict[str, str] = {}

        if self.rclpy_module is None:
            try:
                self.rclpy_module = importlib.import_module("rclpy")
                rclpy_available = True
            except Exception as exc:
                errors["rclpy"] = str(exc)

        try:
            importlib.import_module("geometry_msgs.msg")
            geometry_msgs_available = True
        except Exception as exc:
            errors["geometry_msgs"] = str(exc)

        ok = rclpy_available and geometry_msgs_available
        if not ok:
            reason = "ros2_dependency_unavailable"
        return {
            "ok": ok,
            "rclpy_available": rclpy_available,
            "geometry_msgs_available": geometry_msgs_available,
            "reason": reason,
            "errors": errors,
            "timestamp": self._utc_now(),
        }

    def check_environment(self) -> dict:
        environment = {
            "ROS_DOMAIN_ID": os.environ.get("ROS_DOMAIN_ID"),
            "ROS_LOCALHOST_ONLY": os.environ.get("ROS_LOCALHOST_ONLY"),
            "CYCLONEDDS_URI": os.environ.get("CYCLONEDDS_URI"),
            "RMW_IMPLEMENTATION": os.environ.get("RMW_IMPLEMENTATION"),
        }
        return {
            "ok": True,
            "reason": "environment_read",
            "environment": environment,
            "timestamp": self._utc_now(),
        }

    def create_probe_node(self) -> dict:
        if self.node is not None:
            return {
                "ok": True,
                "node_created": False,
                "node_name": self.NODE_NAME,
                "reason": "using_injected_node",
                "timestamp": self._utc_now(),
            }

        imports = self.check_imports()
        if not imports["rclpy_available"]:
            return {
                "ok": False,
                "node_created": False,
                "node_name": self.NODE_NAME,
                "reason": "ros2_dependency_unavailable",
                "imports": imports,
                "timestamp": self._utc_now(),
            }

        try:
            rclpy = self.rclpy_module
            if not hasattr(rclpy, "ok") or not rclpy.ok():
                rclpy.init(args=None)
            self.node = rclpy.create_node(self.NODE_NAME)
            self._owned_node = True
        except Exception as exc:
            return {
                "ok": False,
                "node_created": False,
                "node_name": self.NODE_NAME,
                "reason": "ros2_probe_error",
                "error": str(exc),
                "timestamp": self._utc_now(),
            }

        return {
            "ok": True,
            "node_created": True,
            "node_name": self.NODE_NAME,
            "reason": "ok",
            "timestamp": self._utc_now(),
        }

    def destroy_probe_node(self) -> dict:
        if self.node is None:
            return {
                "ok": True,
                "node_destroyed": False,
                "reason": "no_node",
                "timestamp": self._utc_now(),
            }
        if not self._owned_node:
            return {
                "ok": True,
                "node_destroyed": False,
                "reason": "injected_node_not_destroyed",
                "timestamp": self._utc_now(),
            }

        try:
            if hasattr(self.node, "destroy_node"):
                self.node.destroy_node()
        except Exception as exc:
            return {
                "ok": False,
                "node_destroyed": False,
                "reason": "ros2_probe_error",
                "error": str(exc),
                "timestamp": self._utc_now(),
            }
        finally:
            self.node = None
            self._owned_node = False

        return {
            "ok": True,
            "node_destroyed": True,
            "reason": "ok",
            "timestamp": self._utc_now(),
        }

    def probe_readiness(self) -> dict:
        imports = self.check_imports()
        environment = self.check_environment()
        node_result = self.create_probe_node() if imports["rclpy_available"] else {
            "ok": False,
            "node_created": False,
            "reason": "ros2_dependency_unavailable",
        }
        ok = imports["ok"] and node_result["ok"]
        return {
            "ok": ok,
            "ros2_available": imports["ok"],
            "node_created": node_result.get("node_created", False),
            "expected_topic": self.expected_topic,
            "expected_msg_type": self.expected_msg_type,
            "can_publish_tested": False,
            "movement_tested": False,
            "hardware_verified": False,
            "imports": imports,
            "environment": environment,
            "node": node_result,
            "reason": "ok" if ok else node_result.get("reason", imports["reason"]),
            "timestamp": self._utc_now(),
        }

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()
