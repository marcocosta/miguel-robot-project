"""Read-only ROS2 graph and sensor probe for Miguel HiWonder Stage 8.

This module is intentionally inert at import time. It does not import ROS2
packages, create nodes, create subscriptions, publish Twist, or arm hardware
unless an explicit instance method is called.
"""

from __future__ import annotations

import importlib
import math
import time
from datetime import datetime, timezone


class MiguelHiWonderRos2GraphProbe:
    """Inspect live HiWonder ROS2 graph state and read one sensor sample."""

    NODE_NAME = "miguel_hiwonder_ros2_graph_probe"
    BATTERY_TOPIC = "/ros_robot_controller/battery"
    LIDAR_TOPIC = "/scan_raw"
    ODOM_TOPIC = "/odom"
    EXPECTED_CMD_VEL_TYPE = "geometry_msgs/msg/Twist"
    MIGUEL_NODE_NAMES = {
        "miguel_hiwonder_ros2_adapter",
        "miguel_hiwonder_ros2_probe",
        "miguel_hiwonder_ros2_graph_probe",
    }
    MIGUEL_NODE_HINTS = ("miguel",)

    def __init__(
        self,
        rclpy_module: object | None = None,
        node: object | None = None,
        expected_cmd_vel_topic: str = "/controller/cmd_vel",
        ignored_publisher_node_names: list[str] | tuple[str, ...] | None = None,
        graph_settle_sec: float = 0.2,
    ) -> None:
        self.rclpy_module = rclpy_module
        self.node = node
        self.expected_cmd_vel_topic = expected_cmd_vel_topic
        self.ignored_publisher_node_names = {
            self._normalize_node_name(name)
            for name in (ignored_publisher_node_names or [])
        }
        self.graph_settle_sec = graph_settle_sec
        self._owned_node = False

    def create_node(self) -> dict:
        if self.node is not None:
            return self._result(ok=True, node_created=False, reason="using_injected_node")
        try:
            rclpy = self._get_rclpy()
            if hasattr(rclpy, "ok") and not rclpy.ok():
                rclpy.init(args=None)
            self.node = rclpy.create_node(self.NODE_NAME)
            self._owned_node = True
            self._settle_graph()
            return self._result(ok=True, node_created=True, node_name=self.NODE_NAME, reason="ok")
        except Exception as exc:
            return self._result(
                ok=False,
                node_created=False,
                reason="ros2_probe_error",
                error=str(exc),
            )

    def destroy_node(self) -> dict:
        if self.node is None:
            return self._result(ok=True, node_destroyed=False, reason="no_node")
        if not self._owned_node:
            return self._result(ok=True, node_destroyed=False, reason="injected_node_not_destroyed")
        try:
            if hasattr(self.node, "destroy_node"):
                self.node.destroy_node()
            return self._result(ok=True, node_destroyed=True, reason="ok")
        except Exception as exc:
            return self._result(
                ok=False,
                node_destroyed=False,
                reason="ros2_probe_error",
                error=str(exc),
            )
        finally:
            self.node = None
            self._owned_node = False

    def list_nodes(self) -> dict:
        node_result = self._ensure_node()
        if not node_result["ok"]:
            return node_result
        try:
            names = []
            if hasattr(self.node, "get_node_names"):
                names = list(self.node.get_node_names())
            elif hasattr(self.node, "node_names"):
                names = list(self.node.node_names)
            return self._result(ok=True, nodes=names, count=len(names), reason="ok")
        except Exception as exc:
            return self._result(ok=False, nodes=[], count=0, reason="ros2_probe_error", error=str(exc))

    def list_topics(self) -> dict:
        node_result = self._ensure_node()
        if not node_result["ok"]:
            return node_result
        try:
            raw_topics = []
            if hasattr(self.node, "get_topic_names_and_types"):
                raw_topics = list(self.node.get_topic_names_and_types())
            elif hasattr(self.node, "topics"):
                raw_topics = list(self.node.topics)

            topics = {}
            for item in raw_topics:
                if isinstance(item, tuple) and len(item) >= 2:
                    name, types = item[0], item[1]
                else:
                    name, types = str(item), []
                topics[str(name)] = list(types or [])
            return self._result(ok=True, topics=topics, count=len(topics), reason="ok")
        except Exception as exc:
            return self._result(ok=False, topics={}, count=0, reason="ros2_probe_error", error=str(exc))

    def inspect_cmd_vel(
        self,
        ignored_publisher_node_names: list[str] | tuple[str, ...] | None = None,
        graph_settle_sec: float | None = None,
    ) -> dict:
        node_result = self._ensure_node()
        if not node_result["ok"]:
            result = self._cmd_vel_base()
            result.update(node_result)
            return result
        try:
            self._settle_graph(graph_settle_sec)
            topics_result = self.list_topics()
            topics = topics_result.get("topics", {}) if topics_result.get("ok") else {}
            topic_types = topics.get(self.expected_cmd_vel_topic, [])
            message_type = topic_types[0] if topic_types else None
            topic_exists = self.expected_cmd_vel_topic in topics
            publisher_nodes = self._topic_endpoint_nodes(self.expected_cmd_vel_topic, "publishers")
            subscriber_nodes = self._topic_endpoint_nodes(self.expected_cmd_vel_topic, "subscribers")
            raw_publisher_count = self._count_publishers(self.expected_cmd_vel_topic)
            raw_subscription_count = self._count_subscribers(self.expected_cmd_vel_topic)
            publisher_count = max(raw_publisher_count, len(publisher_nodes))
            subscription_count = max(raw_subscription_count, len(subscriber_nodes))
            ignored_names = set(self.ignored_publisher_node_names)
            ignored_names.update(
                self._normalize_node_name(name)
                for name in (ignored_publisher_node_names or [])
            )
            competing_publishers = []
            ignored_publishers = []
            for name in publisher_nodes:
                if self._is_ignored_publisher(name, ignored_names):
                    ignored_publishers.append(name)
                else:
                    competing_publishers.append(name)
            unknown_count = max(0, publisher_count - len(publisher_nodes))
            if unknown_count:
                competing_publishers.extend(["unknown"] * unknown_count)
            safe_to_arm = len(competing_publishers) == 0
            return self._result(
                ok=True,
                topic=self.expected_cmd_vel_topic,
                topic_exists=topic_exists,
                message_type=message_type,
                publisher_count=publisher_count,
                subscription_count=subscription_count,
                publisher_nodes=publisher_nodes,
                subscriber_nodes=subscriber_nodes,
                ignored_publishers=ignored_publishers,
                competing_publishers=competing_publishers,
                safe_to_arm=safe_to_arm,
                movement_tested=False,
                hardware_verified=False,
                reason="ok",
            )
        except Exception as exc:
            result = self._cmd_vel_base()
            result.update(self._result(ok=False, reason="ros2_probe_error", error=str(exc)))
            return result

    def read_battery_once(self, timeout_sec: float = 2.0) -> dict:
        try:
            msg_module = importlib.import_module("std_msgs.msg")
            msg = self._read_one_message(msg_module.UInt16, self.BATTERY_TOPIC, timeout_sec)
            if not msg["ok"]:
                return self._sensor_error("battery", msg)
            raw = getattr(msg["message"], "data", None)
            estimated = raw / 1000.0 if isinstance(raw, (int, float)) and raw >= 1000 else None
            return self._result(
                ok=True,
                readable=True,
                sensor="battery",
                topic=self.BATTERY_TOPIC,
                raw=raw,
                estimated_voltage_v=estimated,
                reason="ok",
            )
        except Exception as exc:
            return self._result(
                ok=False,
                readable=False,
                sensor="battery",
                topic=self.BATTERY_TOPIC,
                reason="ros2_probe_error",
                error=str(exc),
            )

    def read_lidar_once(self, timeout_sec: float = 2.0) -> dict:
        try:
            msg_module = importlib.import_module("sensor_msgs.msg")
            msg = self._read_one_message(msg_module.LaserScan, self.LIDAR_TOPIC, timeout_sec)
            if not msg["ok"]:
                return self._sensor_error("lidar", msg)
            summary = self._summarize_lidar(msg["message"])
            return self._result(
                ok=True,
                readable=True,
                sensor="lidar",
                topic=self.LIDAR_TOPIC,
                reason="ok",
                **summary,
            )
        except Exception as exc:
            return self._result(
                ok=False,
                readable=False,
                sensor="lidar",
                topic=self.LIDAR_TOPIC,
                reason="ros2_probe_error",
                error=str(exc),
            )

    def read_odom_once(self, timeout_sec: float = 2.0) -> dict:
        try:
            msg_module = importlib.import_module("nav_msgs.msg")
            msg = self._read_one_message(msg_module.Odometry, self.ODOM_TOPIC, timeout_sec)
            if not msg["ok"]:
                return self._sensor_error("odom", msg)
            summary = self._summarize_odom(msg["message"])
            return self._result(
                ok=True,
                readable=True,
                sensor="odom",
                topic=self.ODOM_TOPIC,
                reason="ok",
                **summary,
            )
        except Exception as exc:
            return self._result(
                ok=False,
                readable=False,
                sensor="odom",
                topic=self.ODOM_TOPIC,
                reason="ros2_probe_error",
                error=str(exc),
            )

    def build_readiness_report(self) -> dict:
        nodes = self.list_nodes()
        topics = self.list_topics()
        cmd_vel = self.inspect_cmd_vel()
        battery = self.read_battery_once()
        lidar = self.read_lidar_once()
        odom = self.read_odom_once()
        topic_map = topics.get("topics", {}) if topics.get("ok") else {}
        expected_topics = {
            self.expected_cmd_vel_topic,
            self.BATTERY_TOPIC,
            self.LIDAR_TOPIC,
            self.ODOM_TOPIC,
        }
        car_graph_visible = bool(topic_map) and any(topic in topic_map for topic in expected_topics)
        ros2_available = bool(nodes.get("ok") or topics.get("ok"))
        return self._result(
            ok=True,
            ros2_available=ros2_available,
            car_graph_visible=car_graph_visible,
            cmd_vel_safe_to_arm=cmd_vel.get("safe_to_arm", False),
            competing_cmd_vel_publishers=cmd_vel.get("competing_publishers", []),
            battery_ok=battery.get("ok", False),
            battery_readable=battery.get("readable", False),
            lidar_ok=lidar.get("ok", False),
            lidar_readable=lidar.get("readable", False),
            odom_ok=odom.get("ok", False),
            odom_readable=odom.get("readable", False),
            movement_tested=False,
            hardware_verified=False,
            graph={"nodes": nodes, "topics": topics, "cmd_vel": cmd_vel},
            sensors={"battery": battery, "lidar": lidar, "odom": odom},
            reason="ok",
        )

    def _get_rclpy(self) -> object:
        if self.rclpy_module is None:
            self.rclpy_module = importlib.import_module("rclpy")
        return self.rclpy_module

    def _ensure_node(self) -> dict:
        if self.node is not None:
            return self._result(ok=True, node_created=False, reason="using_existing_node")
        return self.create_node()

    def _read_one_message(self, msg_type: object, topic: str, timeout_sec: float) -> dict:
        node_result = self._ensure_node()
        if not node_result["ok"]:
            return node_result

        received: list[object] = []

        def callback(message: object) -> None:
            received.append(message)

        subscription = None
        try:
            subscription = self.node.create_subscription(msg_type, topic, callback, 10)
            deadline = time.monotonic() + max(0.0, float(timeout_sec))
            rclpy = self._get_rclpy()
            while not received and time.monotonic() <= deadline:
                if hasattr(rclpy, "spin_once"):
                    rclpy.spin_once(self.node, timeout_sec=0.05)
                else:
                    break
            if not received:
                return self._result(ok=False, readable=False, reason="timeout", topic=topic)
            return self._result(ok=True, readable=True, reason="ok", topic=topic, message=received[0])
        except Exception as exc:
            return self._result(ok=False, readable=False, reason="ros2_probe_error", topic=topic, error=str(exc))
        finally:
            try:
                if subscription is not None and hasattr(self.node, "destroy_subscription"):
                    self.node.destroy_subscription(subscription)
            except Exception:
                pass

    def _count_publishers(self, topic: str) -> int:
        if hasattr(self.node, "count_publishers"):
            return int(self.node.count_publishers(topic))
        return len(getattr(self.node, "publisher_nodes_by_topic", {}).get(topic, []))

    def _count_subscribers(self, topic: str) -> int:
        if hasattr(self.node, "count_subscribers"):
            return int(self.node.count_subscribers(topic))
        return len(getattr(self.node, "subscriber_nodes_by_topic", {}).get(topic, []))

    def _topic_endpoint_nodes(self, topic: str, endpoint_kind: str) -> list[str]:
        method_name = (
            "get_publishers_info_by_topic"
            if endpoint_kind == "publishers"
            else "get_subscriptions_info_by_topic"
        )
        try:
            if hasattr(self.node, method_name):
                endpoint_names = [
                    self._endpoint_node_name(info)
                    for info in getattr(self.node, method_name)(topic)
                ]
                return [name for name in endpoint_names if name]
        except Exception:
            return []
        attr_name = "publisher_nodes_by_topic" if endpoint_kind == "publishers" else "subscriber_nodes_by_topic"
        return [str(name) for name in getattr(self.node, attr_name, {}).get(topic, [])]

    @staticmethod
    def _endpoint_node_name(info: object) -> str:
        node_name = getattr(info, "node_name", None)
        node_namespace = getattr(info, "node_namespace", "")
        if node_name is None and isinstance(info, dict):
            node_name = info.get("node_name") or info.get("name")
            node_namespace = info.get("node_namespace") or info.get("namespace") or ""
        if node_name is None:
            return str(info)
        if node_namespace and str(node_namespace) not in {"/", ""}:
            return f"{node_namespace}/{node_name}".replace("//", "/")
        return str(node_name)

    def _settle_graph(self, graph_settle_sec: float | None = None) -> None:
        settle_sec = self.graph_settle_sec if graph_settle_sec is None else graph_settle_sec
        try:
            settle = max(0.0, float(settle_sec or 0.0))
        except (TypeError, ValueError):
            settle = 0.0
        if settle <= 0.0 or self.node is None:
            return
        try:
            rclpy = self._get_rclpy()
        except Exception:
            return
        deadline = time.monotonic() + settle
        while time.monotonic() <= deadline:
            if hasattr(rclpy, "spin_once"):
                try:
                    rclpy.spin_once(self.node, timeout_sec=min(0.05, settle))
                except Exception:
                    return
            else:
                time.sleep(min(0.05, settle))

    def _summarize_lidar(self, scan: object) -> dict:
        ranges = list(getattr(scan, "ranges", []) or [])
        range_min = float(getattr(scan, "range_min", 0.0) or 0.0)
        range_max = float(getattr(scan, "range_max", float("inf")) or float("inf"))
        angle_min = float(getattr(scan, "angle_min", 0.0) or 0.0)
        angle_increment = float(getattr(scan, "angle_increment", 0.0) or 0.0)
        valid = []
        for index, value in enumerate(ranges):
            try:
                distance_m = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(distance_m):
                continue
            if distance_m < range_min or distance_m > range_max:
                continue
            angle = angle_min + index * angle_increment if angle_increment else None
            valid.append((distance_m, angle))

        def nearest_for_angle(target: float) -> float | None:
            if not valid:
                return None
            if angle_increment:
                sector = math.radians(20)
                sector_values = [
                    distance for distance, angle in valid
                    if angle is not None and abs(self._angle_delta(angle, target)) <= sector
                ]
                if sector_values:
                    return min(sector_values)
            return min(distance for distance, _angle in valid)

        nearest = min((distance for distance, _angle in valid), default=None)
        header = getattr(scan, "header", None)
        return {
            "front_clearance_cm": self._meters_to_cm(nearest_for_angle(0.0)),
            "left_clearance_cm": self._meters_to_cm(nearest_for_angle(math.pi / 2.0)),
            "right_clearance_cm": self._meters_to_cm(nearest_for_angle(-math.pi / 2.0)),
            "nearest_obstacle_cm": self._meters_to_cm(nearest),
            "valid_range_count": len(valid),
            "frame_id": getattr(header, "frame_id", None),
        }

    def _summarize_odom(self, odom: object) -> dict:
        pose = getattr(getattr(odom, "pose", None), "pose", None)
        twist = getattr(getattr(odom, "twist", None), "twist", None)
        position = getattr(pose, "position", None)
        orientation = getattr(pose, "orientation", None)
        linear = getattr(twist, "linear", None)
        angular = getattr(twist, "angular", None)
        header = getattr(odom, "header", None)
        return {
            "x": self._number_or_none(getattr(position, "x", None)),
            "y": self._number_or_none(getattr(position, "y", None)),
            "yaw_approx": self._yaw_from_quaternion(orientation),
            "linear_x": self._number_or_none(getattr(linear, "x", None)),
            "linear_y": self._number_or_none(getattr(linear, "y", None)),
            "angular_z": self._number_or_none(getattr(angular, "z", None)),
            "frame_id": getattr(header, "frame_id", None),
            "child_frame_id": getattr(odom, "child_frame_id", None),
        }

    @staticmethod
    def _yaw_from_quaternion(orientation: object) -> float | None:
        if orientation is None:
            return None
        try:
            x = float(getattr(orientation, "x"))
            y = float(getattr(orientation, "y"))
            z = float(getattr(orientation, "z"))
            w = float(getattr(orientation, "w"))
        except (TypeError, ValueError, AttributeError):
            return None
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _angle_delta(angle: float, target: float) -> float:
        return math.atan2(math.sin(angle - target), math.cos(angle - target))

    @staticmethod
    def _meters_to_cm(value: float | None) -> float | None:
        if value is None:
            return None
        return round(value * 100.0, 3)

    @staticmethod
    def _number_or_none(value: object) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def _cmd_vel_base(self) -> dict:
        return {
            "topic": self.expected_cmd_vel_topic,
            "topic_exists": False,
            "message_type": None,
            "publisher_count": 0,
            "subscription_count": 0,
            "publisher_nodes": [],
            "subscriber_nodes": [],
            "ignored_publishers": [],
            "competing_publishers": [],
            "safe_to_arm": False,
            "movement_tested": False,
            "hardware_verified": False,
        }

    def _sensor_error(self, sensor: str, message_result: dict) -> dict:
        return self._result(
            ok=False,
            readable=False,
            sensor=sensor,
            topic=message_result.get("topic"),
            reason=message_result.get("reason", "ros2_probe_error"),
            error=message_result.get("error"),
        )

    def _is_miguel_node(self, name: str) -> bool:
        normalized = self._normalize_node_name(name)
        return normalized in self.MIGUEL_NODE_NAMES or any(
            hint in normalized for hint in self.MIGUEL_NODE_HINTS
        )

    def _is_ignored_publisher(self, name: str, ignored_names: set[str]) -> bool:
        normalized = self._normalize_node_name(name)
        return normalized in ignored_names or self._is_miguel_node(name)

    @staticmethod
    def _normalize_node_name(name: object) -> str:
        text = str(name or "").strip().lower()
        if "/" in text:
            text = text.rstrip("/").split("/")[-1]
        return text

    def _result(self, **fields: object) -> dict:
        fields.setdefault("timestamp", self._utc_now())
        return dict(fields)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()
