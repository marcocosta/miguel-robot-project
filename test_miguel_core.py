"""Smoke test for the standalone Miguel Core Lab sandbox."""

from __future__ import annotations

import os
import importlib
import math
import sys
import types
from pprint import pprint
from tempfile import TemporaryDirectory

from miguel_core import (
    MiguelHiWonderBridge,
    MiguelHiWonderDryRunAdapter,
    MiguelHiWonderFakeRos2Adapter,
    MiguelHiWonderRealProbe,
    MiguelHiWonderRos2Adapter,
    MiguelHiWonderRos2GraphProbe,
    MiguelHiWonderRos2Probe,
    MiguelRobotBus,
    MiguelRuntime,
)

ROS_TEST_MODULE_ROOTS = {
    "rclpy",
    "geometry_msgs",
    "nav_msgs",
    "sensor_msgs",
    "std_msgs",
    "ros_robot_controller_msgs",
}
_MISSING_MODULE = object()


def _is_ros_test_module(name: str) -> bool:
    root = name.split(".", 1)[0]
    return root in ROS_TEST_MODULE_ROOTS


def _snapshot_ros_modules() -> dict[str, object]:
    return {
        name: module
        for name, module in sys.modules.items()
        if _is_ros_test_module(name)
    }


_INITIAL_REAL_ROS_MODULES = {
    name
    for name, module in _snapshot_ros_modules().items()
    if module is not None and not getattr(module, "_miguel_fake_ros2_module", False)
}


class FakeTwistPublisher:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def publish(self, payload: dict) -> None:
        self.payloads.append(dict(payload))


class FakeTwistMethodPublisher:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def publish_twist(self, payload: dict) -> None:
        self.payloads.append(dict(payload))


class FailingTwistPublisher:
    def publish(self, payload: dict) -> None:
        raise RuntimeError("fake publisher failed")


class MissingPublishTwistPublisher:
    pass


class FailsOnSecondPublishPublisher:
    def __init__(self) -> None:
        self.payloads: list[dict] = []
        self.calls = 0

    def publish(self, payload: dict) -> None:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("followup stop failed")
        self.payloads.append(dict(payload))


def failing_twist_factory(**kwargs: object) -> dict:
    raise RuntimeError("twist factory failed")


def clean_readiness_report(**overrides: object) -> dict:
    report = {
        "ok": True,
        "cmd_vel_safe_to_arm": True,
        "competing_cmd_vel_publishers": [],
        "low_level_motor_chain_ok": True,
        "direct_motor_safe_to_arm": True,
        "external_direct_motor_publishers": [],
        "battery_ok": True,
        "battery_readable": True,
        "lidar_ok": True,
        "lidar_readable": True,
        "odom_ok": True,
        "odom_readable": True,
        "selected_odom_topic": "/odom",
        "movement_tested": False,
        "hardware_verified": False,
        "graph": {
            "cmd_vel": {
                "topic": "/controller/cmd_vel",
                "topic_exists": True,
                "message_type": "geometry_msgs/msg/Twist",
                "publisher_count": 0,
                "subscription_count": 1,
                "publisher_nodes": [],
                "subscriber_nodes": ["odom_publisher"],
                "competing_publishers": [],
                "safe_to_arm": True,
            },
            "direct_motor": {
                "topic": "/ros_robot_controller/set_motor",
                "topic_exists": True,
                "message_type": "ros_robot_controller_msgs/msg/MotorsState",
                "publisher_count": 1,
                "subscription_count": 1,
                "publisher_nodes": ["odom_publisher"],
                "subscriber_nodes": ["ros_robot_controller"],
                "required_subscriber_present": True,
                "external_direct_motor_publishers": [],
                "safe_direct_motor_control": True,
            },
            "motor_chain": {
                "cmd_vel_receiver_ok": True,
                "motor_receiver_ok": True,
                "odom_publisher_to_motor_ok": True,
                "low_level_motor_chain_ok": True,
            },
        },
    }
    report.update(overrides)
    return report


def inactive_readiness_report(**overrides: object) -> dict:
    report = clean_readiness_report(
        cmd_vel_safe_to_arm=False,
        competing_cmd_vel_publishers=["lidar_app"],
        graph={
            "cmd_vel": {
                "topic": "/controller/cmd_vel",
                "topic_exists": True,
                "message_type": "geometry_msgs/msg/Twist",
                "publisher_count": 1,
                "subscription_count": 1,
                "publisher_nodes": ["lidar_app"],
                "subscriber_nodes": ["odom_publisher"],
                "competing_publishers": ["lidar_app"],
                "safe_to_arm": False,
            },
            "direct_motor": {
                "topic": "/ros_robot_controller/set_motor",
                "topic_exists": True,
                "message_type": "ros_robot_controller_msgs/msg/MotorsState",
                "publisher_count": 1,
                "subscription_count": 1,
                "publisher_nodes": ["odom_publisher"],
                "subscriber_nodes": ["ros_robot_controller"],
                "required_subscriber_present": True,
                "external_direct_motor_publishers": [],
                "safe_direct_motor_control": True,
            },
            "motor_chain": {
                "cmd_vel_receiver_ok": True,
                "motor_receiver_ok": True,
                "odom_publisher_to_motor_ok": True,
                "low_level_motor_chain_ok": True,
            },
        },
        cmd_vel_quiet={
            "ok": True,
            "observed_message": False,
            "quiet": True,
            "timeout_sec": 5.0,
            "topic": "/controller/cmd_vel",
            "reason": "timeout",
            "movement_tested": False,
            "hardware_verified": False,
        },
        odom_stationary={
            "ok": True,
            "selected_topic": "/odom_raw",
            "stationary": True,
            "first_twist": {
                "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
            "second_twist": {
                "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
            "max_abs_linear": 0.0,
            "max_abs_angular": 0.0,
            "reason": "ok",
        },
        known_external_publishers=["lidar_app"],
        direct_motor_publishers=[],
        inactive_publishers_observed=True,
        direct_motor_blocking=False,
        relaxed_safe_to_arm=True,
    )
    report.update(overrides)
    return report


class FakeReadinessProbe:
    def __init__(self, report: dict) -> None:
        self.report = report
        self.calls = 0

    def build_readiness_report(self) -> dict:
        self.calls += 1
        return dict(self.report)


class FakeInactiveReadinessProbe(FakeReadinessProbe):
    def build_inactive_publisher_readiness_report(self) -> dict:
        self.calls += 1
        return dict(self.report)


class FakeRos2Vector:
    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class FakeRos2Twist:
    def __init__(self) -> None:
        self.linear = FakeRos2Vector()
        self.angular = FakeRos2Vector()


class FakeRos2Publisher:
    def __init__(self) -> None:
        self.messages: list[FakeRos2Twist] = []

    def publish(self, message: FakeRos2Twist) -> None:
        self.messages.append(message)


class FakeRos2Node:
    def __init__(self) -> None:
        self.publishers: list[FakeRos2Publisher] = []
        self.destroyed = False

    def create_publisher(self, message_type: object, topic: str, depth: int) -> FakeRos2Publisher:
        publisher = FakeRos2Publisher()
        publisher.message_type = message_type
        publisher.topic = topic
        publisher.depth = depth
        self.publishers.append(publisher)
        return publisher

    def destroy_node(self) -> None:
        self.destroyed = True


class FailingRos2Node(FakeRos2Node):
    def destroy_node(self) -> None:
        raise RuntimeError("probe destroy failed")


class FakeEndpointInfo:
    def __init__(self, node_name: str, node_namespace: str = "") -> None:
        self.node_name = node_name
        self.node_namespace = node_namespace


class FakeRos2GraphNode:
    def __init__(
        self,
        nodes: list[str] | None = None,
        topics: list[tuple[str, list[str]]] | None = None,
        publisher_nodes_by_topic: dict[str, list[str]] | None = None,
        subscriber_nodes_by_topic: dict[str, list[str]] | None = None,
        messages_by_topic: dict[str, list[object]] | None = None,
    ) -> None:
        self.node_names = nodes or []
        self.topics = topics or []
        self.publisher_nodes_by_topic = publisher_nodes_by_topic or {}
        self.subscriber_nodes_by_topic = subscriber_nodes_by_topic or {}
        self.messages_by_topic = messages_by_topic or {}
        self.subscriptions: list[object] = []
        self.destroyed_subscriptions: list[object] = []
        self.destroyed = False

    def get_node_names(self) -> list[str]:
        return list(self.node_names)

    def get_topic_names_and_types(self) -> list[tuple[str, list[str]]]:
        return list(self.topics)

    def count_publishers(self, topic: str) -> int:
        return len(self.publisher_nodes_by_topic.get(topic, []))

    def count_subscribers(self, topic: str) -> int:
        return len(self.subscriber_nodes_by_topic.get(topic, []))

    def get_publishers_info_by_topic(self, topic: str) -> list[FakeEndpointInfo]:
        return [FakeEndpointInfo(name) for name in self.publisher_nodes_by_topic.get(topic, [])]

    def get_subscriptions_info_by_topic(self, topic: str) -> list[FakeEndpointInfo]:
        return [FakeEndpointInfo(name) for name in self.subscriber_nodes_by_topic.get(topic, [])]

    def create_subscription(
        self,
        message_type: object,
        topic: str,
        callback: object,
        depth: int,
    ) -> dict:
        subscription = {
            "message_type": message_type,
            "topic": topic,
            "callback": callback,
            "depth": depth,
        }
        self.subscriptions.append(subscription)
        return subscription

    def destroy_subscription(self, subscription: object) -> None:
        self.destroyed_subscriptions.append(subscription)
        if subscription in self.subscriptions:
            self.subscriptions.remove(subscription)

    def spin_once(self) -> None:
        for subscription in list(self.subscriptions):
            topic = subscription["topic"]
            if self.messages_by_topic.get(topic):
                subscription["callback"](self.messages_by_topic[topic].pop(0))

    def destroy_node(self) -> None:
        self.destroyed = True


class FakeUInt16:
    def __init__(self, data: int = 0) -> None:
        self.data = data


class FakeHeader:
    def __init__(self, frame_id: str = "") -> None:
        self.frame_id = frame_id


class FakeLaserScan:
    def __init__(
        self,
        ranges: list[float],
        angle_min: float,
        angle_increment: float,
        range_min: float = 0.0,
        range_max: float = 10.0,
        frame_id: str = "laser",
    ) -> None:
        self.ranges = ranges
        self.angle_min = angle_min
        self.angle_increment = angle_increment
        self.range_min = range_min
        self.range_max = range_max
        self.header = FakeHeader(frame_id)


class FakeVector3:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


class FakeQuaternion:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.w = w


class FakeOdometry:
    def __init__(
        self,
        linear: tuple[float, float, float] = (0.12, 0.01, 0.0),
        angular: tuple[float, float, float] = (0.0, 0.0, -0.2),
    ) -> None:
        self.header = FakeHeader("odom")
        self.child_frame_id = "base_link"
        self.pose = types.SimpleNamespace(
            pose=types.SimpleNamespace(
                position=FakeVector3(1.25, -0.5, 0.0),
                orientation=FakeQuaternion(0.0, 0.0, 0.0, 1.0),
            )
        )
        self.twist = types.SimpleNamespace(
            twist=types.SimpleNamespace(
                linear=FakeVector3(*linear),
                angular=FakeVector3(*angular),
            )
        )


class FakeMotorsState:
    pass


def _make_ros_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module._miguel_fake_ros2_module = True
    if "." not in name:
        module.__path__ = []
    return module


def _make_fake_ros2_modules(node: FakeRos2GraphNode | None = None) -> dict[str, types.ModuleType]:
    rclpy = types.ModuleType("rclpy")
    rclpy._miguel_fake_ros2_module = True
    rclpy.__path__ = []
    rclpy.init_calls = 0
    rclpy.created_nodes = []

    def ok() -> bool:
        return node is not None

    def init(args: object = None) -> None:
        rclpy.init_calls += 1

    def create_node(name: str) -> FakeRos2Node | FakeRos2GraphNode:
        if node is not None:
            node.name = name
            rclpy.created_nodes.append(node)
            return node
        created = FakeRos2Node()
        created.name = name
        rclpy.created_nodes.append(created)
        return created

    def spin_once(spin_node: FakeRos2GraphNode, timeout_sec: float = 0.0) -> None:
        spin_node.spin_once()

    rclpy.ok = ok
    rclpy.init = init
    rclpy.create_node = create_node
    rclpy.spin_once = spin_once

    geometry_msgs = _make_ros_module("geometry_msgs")
    geometry_msgs_msg = _make_ros_module("geometry_msgs.msg")
    geometry_msgs_msg.Twist = FakeRos2Twist
    geometry_msgs.msg = geometry_msgs_msg

    std_msgs = _make_ros_module("std_msgs")
    std_msgs_msg = _make_ros_module("std_msgs.msg")
    std_msgs_msg.UInt16 = FakeUInt16
    std_msgs.msg = std_msgs_msg

    sensor_msgs = _make_ros_module("sensor_msgs")
    sensor_msgs_msg = _make_ros_module("sensor_msgs.msg")
    sensor_msgs_msg.LaserScan = FakeLaserScan
    sensor_msgs.msg = sensor_msgs_msg

    nav_msgs = _make_ros_module("nav_msgs")
    nav_msgs_msg = _make_ros_module("nav_msgs.msg")
    nav_msgs_msg.Odometry = FakeOdometry
    nav_msgs.msg = nav_msgs_msg

    controller_msgs = _make_ros_module("ros_robot_controller_msgs")
    controller_msgs_msg = _make_ros_module("ros_robot_controller_msgs.msg")
    controller_msgs_msg.MotorsState = FakeMotorsState
    controller_msgs.msg = controller_msgs_msg

    return {
        "rclpy": rclpy,
        "geometry_msgs": geometry_msgs,
        "geometry_msgs.msg": geometry_msgs_msg,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msgs_msg,
        "sensor_msgs": sensor_msgs,
        "sensor_msgs.msg": sensor_msgs_msg,
        "nav_msgs": nav_msgs,
        "nav_msgs.msg": nav_msgs_msg,
        "ros_robot_controller_msgs": controller_msgs,
        "ros_robot_controller_msgs.msg": controller_msgs_msg,
    }


def _install_ros2_import_loader(fake_modules: dict[str, types.ModuleType] | None = None) -> dict[str, object]:
    saved = {
        "__import_module__": importlib.import_module,
        "__ros_modules__": _snapshot_ros_modules(),
    }
    fake_modules = fake_modules or {}
    for name in list(sys.modules):
        if _is_ros_test_module(name):
            sys.modules.pop(name, None)
    for name, module in fake_modules.items():
        sys.modules[name] = module

    def test_ros_import(name: str, *args: object, **kwargs: object) -> object:
        if _is_ros_test_module(name):
            if name in fake_modules:
                return fake_modules[name]
            raise ImportError(f"{name} unavailable for test")
        return saved["__import_module__"](name, *args, **kwargs)

    importlib.import_module = test_ros_import
    return saved


def _install_fake_ros2_modules() -> tuple[dict[str, object], types.ModuleType]:
    fake_modules = _make_fake_ros2_modules()
    saved = _install_ros2_import_loader(fake_modules)
    rclpy = fake_modules["rclpy"]
    sys.modules["rclpy"] = rclpy
    return saved, rclpy


def _install_failing_ros2_create_node_module() -> tuple[dict[str, object], types.ModuleType]:
    saved, rclpy = _install_fake_ros2_modules()

    def create_node(name: str) -> FakeRos2Node:
        raise RuntimeError("probe node create failed")

    rclpy.create_node = create_node
    return saved, rclpy


def _install_fake_graph_ros2_modules(node: FakeRos2GraphNode) -> tuple[dict[str, object], types.ModuleType]:
    fake_modules = _make_fake_ros2_modules(node)
    saved = _install_ros2_import_loader(fake_modules)
    rclpy = fake_modules["rclpy"]
    return saved, rclpy


def _install_stage11_ros_sensitive_import_guard() -> dict[str, object]:
    fake_modules = _make_fake_ros2_modules()
    saved = {
        "__import_module__": importlib.import_module,
        "__ros_modules__": _snapshot_ros_modules(),
    }
    for name in list(sys.modules):
        if _is_ros_test_module(name):
            sys.modules.pop(name, None)
    for name, module in fake_modules.items():
        if name != "rclpy":
            sys.modules[name] = module

    def guarded_import_module(name: str, *args: object, **kwargs: object) -> object:
        if name == "rclpy" or name.startswith("rclpy."):
            raise AssertionError(f"Stage 11 test attempted host ROS import: {name}")
        if _is_ros_test_module(name):
            if name in fake_modules and name != "rclpy":
                return fake_modules[name]
            raise AssertionError(f"Stage 11 test attempted non-fake ROS import: {name}")
        return saved["__import_module__"](name, *args, **kwargs)

    graph_probe_module = sys.modules[MiguelHiWonderRos2GraphProbe.__module__]
    adapter_module = sys.modules[MiguelHiWonderRos2Adapter.__module__]
    saved["__graph_import_module__"] = graph_probe_module.importlib.import_module
    saved["__adapter_import_module__"] = adapter_module.importlib.import_module
    graph_probe_module.importlib.import_module = guarded_import_module
    adapter_module.importlib.import_module = guarded_import_module
    return saved


def _restore_modules(saved: dict[str, object]) -> None:
    graph_import_module = saved.get("__graph_import_module__")
    if graph_import_module is not None:
        sys.modules[MiguelHiWonderRos2GraphProbe.__module__].importlib.import_module = graph_import_module
    adapter_import_module = saved.get("__adapter_import_module__")
    if adapter_import_module is not None:
        sys.modules[MiguelHiWonderRos2Adapter.__module__].importlib.import_module = adapter_import_module

    original_import_module = saved.get("__import_module__")
    if original_import_module is not None:
        importlib.import_module = original_import_module

    ros_modules = saved.get("__ros_modules__")
    if isinstance(ros_modules, dict):
        for name in list(sys.modules):
            if _is_ros_test_module(name) and name not in ros_modules:
                sys.modules.pop(name, None)
        for name, module in ros_modules.items():
            sys.modules[name] = module

    for name, module in saved.items():
        if name.startswith("__") or _is_ros_test_module(name):
            continue
        if module is _MISSING_MODULE:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _save_and_remove_modules(*names: str) -> dict[str, object]:
    saved = {name: sys.modules.get(name, _MISSING_MODULE) for name in names}
    if any(_is_ros_test_module(name) for name in names):
        saved["__ros_modules__"] = _snapshot_ros_modules()
        for name in list(sys.modules):
            if _is_ros_test_module(name):
                sys.modules.pop(name, None)
    for name in names:
        sys.modules.pop(name, None)
    return saved


def _assert_no_real_ros_modules_loaded() -> None:
    offenders = []
    for name, module in sys.modules.items():
        if _is_ros_test_module(name) and name not in _INITIAL_REAL_ROS_MODULES and module is not None:
            if not getattr(module, "_miguel_fake_ros2_module", False):
                offenders.append(name)
    assert offenders == [], f"unit tests imported host ROS modules: {offenders[:8]}"


def _destroy_adapter_node_without_publish(adapter: object) -> None:
    node = getattr(adapter, "node", None)
    if node is not None and hasattr(node, "destroy_node"):
        try:
            node.destroy_node()
        except Exception:
            pass
    if hasattr(adapter, "_owned_node"):
        adapter._owned_node = False
    if hasattr(adapter, "node"):
        adapter.node = None


def _runtime() -> MiguelRuntime:
    temp_dir = TemporaryDirectory()
    runtime = MiguelRuntime(temp_dir.name)
    runtime._test_temp_dir = temp_dir
    runtime.start()
    return runtime


def _set_telemetry(runtime: MiguelRuntime, **overrides: object) -> None:
    telemetry = {
        "battery_percent": 82,
        "emergency_stop": False,
        "front_clearance_cm": 120,
        "left_clearance_cm": 50,
        "nearest_obstacle_cm": 60,
        "person_detected": False,
        "person_direction": None,
        "right_clearance_cm": 90,
        "state": "idle",
        "simulated": True,
    }
    telemetry.update(overrides)
    runtime.hiwonder.update_telemetry(telemetry)


def test_explore_room_turn_right_allowed() -> dict:
    runtime = _runtime()
    _set_telemetry(runtime, front_clearance_cm=35, nearest_obstacle_cm=35)

    result = runtime.run_hiwonder_mission_step("explore_room")
    assert result["decision"]["action"] == "turn_right"
    assert result["command_result"]["payload"]["command"] == "turn_right"
    assert result["command_result"]["safety_validation"]["ok"] is True
    assert result["command_result"]["adapter_result"]["dry_run"] is True
    assert result["mission_status"]["state"] == "active"
    runtime.shutdown()
    return result


def test_unsafe_move_forward_blocked() -> dict:
    runtime = _runtime()
    _set_telemetry(runtime, front_clearance_cm=30, nearest_obstacle_cm=60)

    result = runtime.hiwonder.move_forward()
    assert result["payload"]["command"] == "stop"
    assert result["safety_validation"]["blocked"] is True
    assert result["safety_validation"]["reason"] == "front_clearance_too_low"
    runtime.shutdown()
    return result


def test_fast_speed_downgraded() -> dict:
    runtime = _runtime()
    _set_telemetry(runtime)

    result = runtime.hiwonder.turn_right(speed="fast", duration_sec=1.0)
    assert result["payload"]["command"] == "turn_right"
    assert result["payload"]["params"]["speed"] == "slow"
    assert "speed_downgraded_from_fast_to_slow" in result["safety_validation"]["warnings"]
    runtime.shutdown()
    return result


def test_long_duration_capped() -> dict:
    runtime = _runtime()
    _set_telemetry(runtime)

    result = runtime.hiwonder.move_forward(speed="slow", duration_sec=9.0)
    assert result["payload"]["command"] == "move_forward"
    assert result["payload"]["params"]["duration_sec"] == 2.0
    assert "duration_capped_to_2.0_sec" in result["safety_validation"]["warnings"]
    runtime.shutdown()
    return result


def test_emergency_stop_telemetry_stops_mission() -> dict:
    runtime = _runtime()
    _set_telemetry(runtime, emergency_stop=True)

    result = runtime.run_hiwonder_mission_step("explore_room")
    assert result["decision"]["action"] == "stop"
    assert result["command_result"]["payload"]["command"] == "stop"
    assert result["mission_status"]["state"] == "emergency_stop"
    runtime.shutdown()
    return result


def test_mission_controller_records_steps() -> dict:
    runtime = _runtime()
    _set_telemetry(runtime)

    result = runtime.run_hiwonder_mission_step("explore_room")
    status = result["mission_status"]
    assert status["step_count"] == 1
    assert len(status["recent_steps"]) == 1
    runtime.shutdown()
    return result


def test_default_bridge_uses_dry_run_adapter() -> dict:
    runtime = _runtime()
    assert isinstance(runtime.hiwonder.adapter, MiguelHiWonderDryRunAdapter)
    assert runtime.hiwonder.adapter.get_name() == "dry_run"
    telemetry = runtime.hiwonder.request_telemetry()
    assert telemetry["simulated"] is True
    assert telemetry["target"] == "hiwonder_car"
    runtime.shutdown()
    return telemetry


def test_probe_runs_without_crashing() -> dict:
    if os.environ.get("MIGUEL_RUN_REAL_PROBE") != "1":
        return {"skipped": True, "reason": "set MIGUEL_RUN_REAL_PROBE=1 to run host probe"}
    result = MiguelHiWonderRealProbe().probe()
    assert result["ok"] is True
    assert "likely_interfaces" in result
    return result


def test_stage1_adapter_initializes_disarmed_without_commands() -> dict:
    adapter = MiguelHiWonderDryRunAdapter()
    assert adapter.get_name() == "dry_run"
    assert adapter.armed is False
    assert adapter.command_log == []
    return {"armed": adapter.armed, "commands": len(adapter.command_log)}


def test_stage1_runtime_start_emits_no_hardware_commands() -> dict:
    runtime = _runtime()
    adapter = runtime.hiwonder.adapter
    assert isinstance(adapter, MiguelHiWonderDryRunAdapter)
    assert adapter.command_log == []
    runtime.shutdown()
    return {"commands_before_shutdown": 0}


def test_stage1_movement_while_disarmed_blocked() -> dict:
    adapter = MiguelHiWonderDryRunAdapter()
    result = adapter.set_velocity("move_forward", "slow", 1.0)
    assert result["blocked"] is True
    assert result["reason"] == "adapter_disarmed"
    assert len(adapter.command_log) == 1
    return result


def test_stage1_arm_then_move_forward_records_twist_and_stop() -> dict:
    adapter = MiguelHiWonderDryRunAdapter()
    adapter.arm()
    result = adapter.set_velocity("move_forward", "slow", 1.0)
    assert result["ok"] is True
    assert result["blocked"] is False
    assert result["twist"]["linear_x"] > 0
    assert abs(result["twist"]["linear_x"]) <= adapter.MAX_LINEAR_X
    assert result["followup_stop"]["command"] == "stop"
    assert adapter.command_log[-2]["command"] == "move_forward"
    assert adapter.command_log[-1]["command"] == "stop"
    return result


def test_stage1_stale_telemetry_blocks_movement() -> dict:
    runtime = _runtime()
    runtime.hiwonder.arm()
    result = runtime.hiwonder.move_forward()
    assert result["payload"]["command"] == "stop"
    assert result["safety_validation"]["blocked"] is True
    assert result["safety_validation"]["reason"] == "telemetry_missing_or_stale"
    assert result["adapter_result"]["command"] == "stop"
    runtime.shutdown()
    return result


def test_stage1_emergency_stop_records_stop() -> dict:
    runtime = _runtime()
    runtime.hiwonder.arm()
    _set_telemetry(runtime, emergency_stop=True)
    result = runtime.hiwonder.move_forward()
    assert result["payload"]["command"] == "stop"
    assert result["safety_validation"]["reason"] == "emergency_stop"
    assert runtime.hiwonder.adapter.command_log[-1]["command"] == "stop"
    runtime.shutdown()
    return result


def test_stage1_adapter_duration_cap() -> dict:
    adapter = MiguelHiWonderDryRunAdapter()
    adapter.arm()
    result = adapter.set_velocity("move_forward", "slow", 99.0)
    assert result["params"]["duration_sec"] == adapter.MAX_DURATION_SEC
    return result


def test_stage1_adapter_speed_cap() -> dict:
    adapter = MiguelHiWonderDryRunAdapter()
    adapter.arm()
    result = adapter.set_velocity("move_forward", "fast", 1.0)
    assert result["params"]["speed"] == "slow"
    assert abs(result["twist"]["linear_x"]) <= adapter.MAX_LINEAR_X
    return result


def test_stage1_mission_disarmed_records_blocked_not_accepted() -> dict:
    runtime = _runtime()
    _set_telemetry(runtime)
    result = runtime.run_hiwonder_mission_step("explore_room")
    command_result = result["command_result"]
    assert command_result["payload"]["command"] == "move_forward"
    assert command_result["payload"]["status"] == "blocked"
    assert command_result["payload"]["adapter_blocked"] is True
    assert command_result["adapter_result"]["reason"] == "adapter_disarmed"
    runtime.shutdown()
    return result


def test_stage1_move_backward_twist_sign() -> dict:
    adapter = MiguelHiWonderDryRunAdapter()
    adapter.arm()
    result = adapter.set_velocity("move_backward", "slow", 1.0)
    assert result["twist"]["linear_x"] < 0
    assert result["twist"]["angular_z"] == 0.0
    return result


def test_stage1_turn_left_twist_sign() -> dict:
    adapter = MiguelHiWonderDryRunAdapter()
    adapter.arm()
    result = adapter.set_velocity("turn_left", "slow", 1.0)
    assert result["twist"]["linear_x"] == 0.0
    assert result["twist"]["angular_z"] > 0
    return result


def test_stage1_turn_right_twist_sign() -> dict:
    adapter = MiguelHiWonderDryRunAdapter()
    adapter.arm()
    result = adapter.set_velocity("turn_right", "slow", 1.0)
    assert result["twist"]["linear_x"] == 0.0
    assert result["twist"]["angular_z"] < 0
    return result


def test_stage1_negative_duration_clamps_to_zero() -> dict:
    adapter = MiguelHiWonderDryRunAdapter()
    adapter.arm()
    result = adapter.set_velocity("move_forward", "slow", -2.0)
    assert result["params"]["duration_sec"] == 0.0
    return result


def test_stage1_nonnumeric_duration_clamps_to_zero() -> dict:
    adapter = MiguelHiWonderDryRunAdapter()
    adapter.arm()
    result = adapter.set_velocity("move_forward", "slow", "not-a-number")
    assert result["params"]["duration_sec"] == 0.0
    return result


def test_stage1_close_disarms_and_records_stop() -> dict:
    adapter = MiguelHiWonderDryRunAdapter()
    adapter.arm()
    adapter.close()
    assert adapter.armed is False
    assert adapter.command_log[-1]["command"] == "stop"
    assert adapter.command_log[-1]["params"]["reason"] == "adapter close"
    return adapter.command_log[-1]


def test_stage1_low_battery_blocks_movement() -> dict:
    runtime = _runtime()
    runtime.hiwonder.arm()
    _set_telemetry(runtime, battery_percent=10)
    result = runtime.hiwonder.move_forward()
    assert result["payload"]["command"] == "stop"
    assert result["safety_validation"]["reason"] == "battery_below_minimum"
    runtime.shutdown()
    return result


def test_stage1_nearest_obstacle_blocks_movement() -> dict:
    runtime = _runtime()
    runtime.hiwonder.arm()
    _set_telemetry(runtime, nearest_obstacle_cm=20)
    result = runtime.hiwonder.turn_left()
    assert result["payload"]["command"] == "stop"
    assert result["safety_validation"]["reason"] == "nearest_obstacle_too_close"
    runtime.shutdown()
    return result


def test_stage2_bridge_still_defaults_to_dry_run() -> dict:
    runtime = _runtime()
    assert isinstance(runtime.hiwonder.adapter, MiguelHiWonderDryRunAdapter)
    runtime.shutdown()
    return {"adapter": runtime.hiwonder.adapter.get_name()}


def test_stage2_fake_ros2_adapter_starts_disarmed() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderFakeRos2Adapter(publisher)
    assert adapter.armed is False
    assert adapter.get_name() == "fake_ros2"
    return {"armed": adapter.armed}


def test_stage2_fake_ros2_adapter_emits_no_payload_on_construction() -> dict:
    publisher = FakeTwistPublisher()
    MiguelHiWonderFakeRos2Adapter(publisher)
    assert publisher.payloads == []
    return {"payloads": len(publisher.payloads)}


def test_stage2_fake_ros2_movement_while_disarmed_blocks_without_publish() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderFakeRos2Adapter(publisher)
    result = adapter.set_velocity("move_forward", "slow", 1.0)
    assert result["blocked"] is True
    assert result["reason"] == "adapter_disarmed"
    assert publisher.payloads == []
    return result


def test_stage2_fake_ros2_move_forward_publishes_twist_and_stop() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderFakeRos2Adapter(publisher)
    adapter.arm()
    result = adapter.set_velocity("move_forward", "slow", 1.0)
    assert result["ok"] is True
    assert len(publisher.payloads) == 2
    move_payload, stop_payload = publisher.payloads
    assert move_payload["topic"] == "/controller/cmd_vel"
    assert move_payload["source"] == "miguel"
    assert move_payload["fake_ros2"] is True
    assert 0 < move_payload["linear"]["x"] <= adapter.MAX_LINEAR_X
    assert move_payload["angular"]["z"] == 0.0
    assert stop_payload["linear"]["x"] == 0.0
    assert stop_payload["angular"]["z"] == 0.0
    assert result["followup_stop"]["command"] == "stop"
    return result


def test_stage2_fake_ros2_turn_left_and_right_signs() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderFakeRos2Adapter(publisher)
    adapter.arm()
    left = adapter.set_velocity("turn_left", "slow", 1.0)
    right = adapter.set_velocity("turn_right", "slow", 1.0)
    assert left["payload"]["angular"]["z"] > 0
    assert right["payload"]["angular"]["z"] < 0
    return {"left": left, "right": right}


def test_stage2_fake_ros2_stop_while_disarmed_publishes_zero_twist() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderFakeRos2Adapter(publisher)
    result = adapter.stop("manual stop")
    assert result["ok"] is True
    assert len(publisher.payloads) == 1
    payload = publisher.payloads[0]
    assert payload["linear"] == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert payload["angular"] == {"x": 0.0, "y": 0.0, "z": 0.0}
    return result


def test_stage2_fake_ros2_close_publishes_stop_and_disarms() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderFakeRos2Adapter(publisher)
    adapter.arm()
    adapter.close()
    assert adapter.armed is False
    assert publisher.payloads[-1]["linear"]["x"] == 0.0
    assert adapter.command_log[-1]["command"] == "stop"
    return adapter.command_log[-1]


def test_stage2_fake_ros2_duration_cap() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderFakeRos2Adapter(publisher)
    adapter.arm()
    result = adapter.set_velocity("move_forward", "slow", 99.0)
    assert result["payload"]["duration_sec"] == adapter.MAX_DURATION_SEC
    return result


def test_stage2_fake_ros2_speed_cap() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderFakeRos2Adapter(publisher)
    adapter.arm()
    result = adapter.set_velocity("move_forward", "fast", 1.0)
    assert result["params"]["speed"] == "slow"
    assert abs(result["payload"]["linear"]["x"]) <= adapter.MAX_LINEAR_X
    return result


def test_stage21_fake_ros2_publish_twist_method_supported() -> dict:
    publisher = FakeTwistMethodPublisher()
    adapter = MiguelHiWonderFakeRos2Adapter(publisher)
    result = adapter.stop("publish_twist smoke")
    assert result["ok"] is True
    assert len(publisher.payloads) == 1
    assert publisher.payloads[0]["topic"] == "/controller/cmd_vel"
    return result


def test_stage21_bridge_accepts_injected_fake_ros2_adapter() -> dict:
    temp_dir = TemporaryDirectory()
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderFakeRos2Adapter(publisher)
    bus = MiguelRobotBus(temp_dir.name)
    bridge = MiguelHiWonderBridge(bus, temp_dir.name, adapter=adapter)
    bridge.arm()
    bridge.update_telemetry(
        {
            "battery_percent": 82,
            "emergency_stop": False,
            "front_clearance_cm": 120,
            "left_clearance_cm": 50,
            "nearest_obstacle_cm": 60,
            "person_detected": False,
            "person_direction": None,
            "right_clearance_cm": 90,
            "state": "idle",
            "simulated": True,
        }
    )

    result = bridge.move_forward()
    assert result["payload"]["status"] == "accepted"
    assert result["payload"]["adapter_blocked"] is False
    assert result["adapter_result"]["fake_ros2"] is True
    assert result["adapter_result"]["command"] == "move_forward"
    assert result["adapter_result"]["payload"]["linear"]["x"] > 0
    assert len(publisher.payloads) == 2
    assert publisher.payloads[0]["linear"]["x"] > 0
    assert publisher.payloads[1]["linear"]["x"] == 0.0
    temp_dir.cleanup()
    return result


def test_stage21_fake_ros2_publisher_failure_is_reported() -> dict:
    adapter = MiguelHiWonderFakeRos2Adapter(FailingTwistPublisher())
    adapter.arm()
    result = adapter.set_velocity("move_forward", "slow", 1.0)
    assert result["ok"] is False
    assert result["blocked"] is True
    assert result["error"] is True
    assert result["reason"] == "publisher_error"
    return result


def test_stage21_fake_ros2_move_backward_sign() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderFakeRos2Adapter(publisher)
    adapter.arm()
    result = adapter.set_velocity("move_backward", "slow", 1.0)
    assert result["payload"]["linear"]["x"] < 0
    assert publisher.payloads[0]["linear"]["x"] < 0
    assert publisher.payloads[1]["linear"]["x"] == 0.0
    return result


def test_stage21_fake_ros2_negative_duration_clamps_to_zero() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderFakeRos2Adapter(publisher)
    adapter.arm()
    result = adapter.set_velocity("move_forward", "slow", -3.0)
    assert result["ok"] is True
    assert result["payload"]["duration_sec"] == 0.0
    return result


def test_stage21_fake_ros2_nonnumeric_duration_clamps_to_zero() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderFakeRos2Adapter(publisher)
    adapter.arm()
    result = adapter.set_velocity("move_forward", "slow", "not-a-duration")
    assert result["ok"] is True
    assert result["payload"]["duration_sec"] == 0.0
    return result


def test_stage21_fake_ros2_disarm_publishes_stop_and_disarms() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderFakeRos2Adapter(publisher)
    adapter.arm()
    result = adapter.disarm()
    assert adapter.armed is False
    assert result["ok"] is True
    assert result["stop_result"]["command"] == "stop"
    assert publisher.payloads[-1]["linear"]["x"] == 0.0
    assert publisher.payloads[-1]["angular"]["z"] == 0.0
    return result


def test_stage21_fake_ros2_disarm_while_disarmed_is_safe_stop() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderFakeRos2Adapter(publisher)
    result = adapter.disarm()
    assert adapter.armed is False
    assert result["ok"] is True
    assert len(publisher.payloads) == 1
    assert publisher.payloads[0]["linear"]["x"] == 0.0
    return result


def test_stage21_fake_ros2_unknown_command_blocks_without_publish() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderFakeRos2Adapter(publisher)
    adapter.arm()
    result = adapter.set_velocity("spin", "slow", 1.0)
    assert result["ok"] is False
    assert result["blocked"] is True
    assert result["reason"] == "unknown_movement_command"
    assert publisher.payloads == []
    return result


def test_stage3_bridge_still_defaults_to_dry_run() -> dict:
    runtime = _runtime()
    assert isinstance(runtime.hiwonder.adapter, MiguelHiWonderDryRunAdapter)
    runtime.shutdown()
    return {"adapter": runtime.hiwonder.adapter.get_name()}


def test_stage3_ros2_module_import_does_not_import_ros_packages() -> dict:
    saved = _save_and_remove_modules(
        "miguel_core.miguel_hiwonder_ros2_adapter",
        "rclpy",
        "geometry_msgs",
        "geometry_msgs.msg",
    )
    try:
        importlib.import_module("miguel_core.miguel_hiwonder_ros2_adapter")
        assert "rclpy" not in sys.modules
        assert "geometry_msgs" not in sys.modules
        return {"rclpy_imported": False, "geometry_msgs_imported": False}
    finally:
        _restore_modules(saved)


def test_stage3_ros2_without_publisher_is_unavailable() -> dict:
    adapter = MiguelHiWonderRos2Adapter()
    assert adapter.armed is False
    result = adapter.set_velocity("move_forward", "slow", 1.0)
    assert result["ok"] is False
    assert result["blocked"] is True
    assert result["reason"] == "adapter_unavailable"
    assert adapter.command_log[-1]["command"] == "move_forward"
    return result


def test_stage3_ros2_adapter_starts_disarmed() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderRos2Adapter(publisher=publisher)
    assert adapter.armed is False
    assert adapter.get_name() == "ros2"
    assert publisher.payloads == []
    return {"armed": adapter.armed, "payloads": len(publisher.payloads)}


def test_stage3_ros2_disarmed_movement_blocks_without_publish() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderRos2Adapter(publisher=publisher)
    result = adapter.set_velocity("move_forward", "slow", 1.0)
    assert result["blocked"] is True
    assert result["reason"] == "adapter_disarmed"
    assert publisher.payloads == []
    return result


def test_stage3_ros2_stop_while_disarmed_publishes_zero_twist() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderRos2Adapter(publisher=publisher)
    result = adapter.stop("manual stop")
    assert result["ok"] is True
    assert len(publisher.payloads) == 1
    assert publisher.payloads[0]["linear"] == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert publisher.payloads[0]["angular"] == {"x": 0.0, "y": 0.0, "z": 0.0}
    return result


def test_stage3_ros2_arm_move_forward_publishes_twist_and_stop() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderRos2Adapter(publisher=publisher)
    adapter.arm()
    result = adapter.set_velocity("move_forward", "slow", 1.0)
    assert result["ok"] is True
    assert len(publisher.payloads) == 2
    assert 0 < publisher.payloads[0]["linear"]["x"] <= adapter.MAX_LINEAR_X
    assert publisher.payloads[0]["adapter"] == "ros2"
    assert publisher.payloads[1]["linear"]["x"] == 0.0
    assert publisher.payloads[1]["angular"]["z"] == 0.0
    return result


def test_stage3_ros2_move_backward_sign() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderRos2Adapter(publisher=publisher)
    adapter.arm()
    result = adapter.set_velocity("move_backward", "slow", 1.0)
    assert result["payload"]["linear"]["x"] < 0
    assert publisher.payloads[1]["linear"]["x"] == 0.0
    return result


def test_stage3_ros2_turn_left_sign() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderRos2Adapter(publisher=publisher)
    adapter.arm()
    result = adapter.set_velocity("turn_left", "slow", 1.0)
    assert result["payload"]["angular"]["z"] > 0
    return result


def test_stage3_ros2_turn_right_sign() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderRos2Adapter(publisher=publisher)
    adapter.arm()
    result = adapter.set_velocity("turn_right", "slow", 1.0)
    assert result["payload"]["angular"]["z"] < 0
    return result


def test_stage3_ros2_disarm_publishes_stop_and_disarms() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderRos2Adapter(publisher=publisher)
    adapter.arm()
    result = adapter.disarm()
    assert adapter.armed is False
    assert result["ok"] is True
    assert publisher.payloads[-1]["linear"]["x"] == 0.0
    assert result["stop_result"]["command"] == "stop"
    return result


def test_stage3_ros2_close_publishes_stop_and_disarms() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderRos2Adapter(publisher=publisher)
    adapter.arm()
    adapter.close()
    assert adapter.armed is False
    assert publisher.payloads[-1]["linear"]["x"] == 0.0
    assert adapter.command_log[-1]["command"] == "stop"
    return adapter.command_log[-1]


def test_stage3_ros2_publisher_failure_returns_error() -> dict:
    adapter = MiguelHiWonderRos2Adapter(publisher=FailingTwistPublisher())
    adapter.arm()
    result = adapter.set_velocity("move_forward", "slow", 1.0)
    assert result["ok"] is False
    assert result["blocked"] is True
    assert result["error"] is True
    assert result["reason"] == "publisher_error"
    return result


def test_stage3_ros2_negative_duration_clamps_to_zero() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderRos2Adapter(publisher=publisher)
    adapter.arm()
    result = adapter.set_velocity("move_forward", "slow", -1.0)
    assert result["payload"]["duration_sec"] == 0.0
    return result


def test_stage3_ros2_nonnumeric_duration_clamps_to_zero() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderRos2Adapter(publisher=publisher)
    adapter.arm()
    result = adapter.set_velocity("move_forward", "slow", "nope")
    assert result["payload"]["duration_sec"] == 0.0
    return result


def test_stage3_bridge_accepts_injected_ros2_adapter() -> dict:
    temp_dir = TemporaryDirectory()
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderRos2Adapter(publisher=publisher)
    bus = MiguelRobotBus(temp_dir.name)
    bridge = MiguelHiWonderBridge(bus, temp_dir.name, adapter=adapter)
    bridge.arm()
    bridge.update_telemetry(
        {
            "battery_percent": 82,
            "emergency_stop": False,
            "front_clearance_cm": 120,
            "left_clearance_cm": 50,
            "nearest_obstacle_cm": 60,
            "person_detected": False,
            "person_direction": None,
            "right_clearance_cm": 90,
            "state": "idle",
            "simulated": True,
        }
    )

    result = bridge.move_forward()
    assert result["payload"]["status"] == "accepted"
    assert result["payload"]["adapter_blocked"] is False
    assert result["adapter_result"]["adapter"] == "ros2"
    assert result["adapter_result"]["command"] == "move_forward"
    assert len(publisher.payloads) == 2
    assert publisher.payloads[0]["linear"]["x"] > 0
    assert publisher.payloads[1]["linear"]["x"] == 0.0
    temp_dir.cleanup()
    return result


def test_stage31_ros2_injected_publisher_is_not_real_hardware() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderRos2Adapter(
        publisher=publisher,
        twist_factory=lambda **kwargs: dict(kwargs),
    )
    status = adapter.backend_status()
    assert adapter.is_real_hardware() is False
    assert status["adapter"] == "ros2"
    assert status["backend"] == "injected_publisher"
    assert status["real_ros2_enabled"] is False
    assert status["hardware_verified"] is False
    return status


def test_stage31_ros2_real_backend_requires_explicit_opt_in() -> dict:
    import_calls: list[str] = []
    saved = _install_ros2_import_loader()
    blocked_import_module = importlib.import_module

    def missing_ros_import(name: str, *args: object, **kwargs: object) -> object:
        import_calls.append(name)
        return blocked_import_module(name, *args, **kwargs)

    importlib.import_module = missing_ros_import
    try:
        default_adapter = MiguelHiWonderRos2Adapter()
        default_status = default_adapter.backend_status()
        assert default_status["available"] is False
        assert default_status["backend"] == "unavailable"
        assert default_status["real_ros2_enabled"] is False
        assert default_status["reason"] == "adapter_unavailable"
        assert import_calls == []

        real_adapter = MiguelHiWonderRos2Adapter(allow_real_ros2=True)
        real_status = real_adapter.backend_status()
        result = real_adapter.set_velocity("move_forward", "slow", 1.0)
    finally:
        _restore_modules(saved)

    assert "rclpy" in import_calls
    assert real_status["available"] is False
    assert real_status["backend"] == "unavailable"
    assert real_status["real_ros2_enabled"] is False
    assert real_status["hardware_verified"] is False
    assert real_status["reason"] == "ros2_dependency_unavailable"
    assert result["ok"] is False
    assert result["blocked"] is True
    assert result["reason"] == "ros2_dependency_unavailable"
    return {"default": default_status, "real": real_status, "result": result}


def test_stage31_ros2_twist_factory_failure_on_movement_is_structured() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderRos2Adapter(publisher=publisher, twist_factory=failing_twist_factory)
    adapter.arm()
    result = adapter.set_velocity("move_forward", "slow", 1.0)
    assert result["ok"] is False
    assert result["blocked"] is True
    assert result["error"] is True
    assert result["reason"] == "twist_factory_error"
    assert publisher.payloads == []
    return result


def test_stage31_ros2_twist_factory_failure_on_stop_is_structured() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderRos2Adapter(publisher=publisher, twist_factory=failing_twist_factory)
    result = adapter.stop("factory failure")
    assert result["ok"] is False
    assert result["blocked"] is True
    assert result["error"] is True
    assert result["reason"] == "twist_factory_error"
    assert publisher.payloads == []
    return result


def test_stage31_ros2_twist_factory_failure_on_disarm_is_structured() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderRos2Adapter(publisher=publisher, twist_factory=failing_twist_factory)
    adapter.arm()
    result = adapter.disarm()
    assert adapter.armed is False
    assert result["ok"] is False
    assert result["blocked"] is True
    assert result["error"] is True
    assert result["reason"] == "twist_factory_error"
    assert publisher.payloads == []
    return result


def test_stage31_ros2_twist_factory_failure_on_close_is_structured() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderRos2Adapter(publisher=publisher, twist_factory=failing_twist_factory)
    adapter.arm()
    result = adapter.close()
    assert adapter.armed is False
    assert result["ok"] is False
    assert result["blocked"] is True
    assert result["error"] is True
    assert result["reason"] == "twist_factory_error"
    assert publisher.payloads == []
    return result


def test_stage31_ros2_unknown_command_blocks_without_publish() -> dict:
    publisher = FakeTwistPublisher()
    adapter = MiguelHiWonderRos2Adapter(publisher=publisher)
    adapter.arm()
    result = adapter.set_velocity("spin", "slow", 1.0)
    assert result["ok"] is False
    assert result["blocked"] is True
    assert result["reason"] == "unknown_movement_command"
    assert publisher.payloads == []
    return result


def test_stage31_ros2_missing_publish_method_is_structured() -> dict:
    adapter = MiguelHiWonderRos2Adapter(publisher=MissingPublishTwistPublisher())
    result = adapter.stop("missing publisher method")
    assert result["ok"] is False
    assert result["blocked"] is True
    assert result["error"] is True
    assert result["reason"] == "publisher_unavailable"
    return result


def test_stage31_ros2_followup_stop_failure_preserves_movement_result() -> dict:
    publisher = FailsOnSecondPublishPublisher()
    adapter = MiguelHiWonderRos2Adapter(publisher=publisher)
    adapter.arm()
    result = adapter.set_velocity("move_forward", "slow", 1.0)
    assert result["ok"] is False
    assert result["blocked"] is True
    assert result["error"] is True
    assert result["reason"] == "publisher_error"
    assert result["payload"]["linear"]["x"] > 0
    assert result["followup_stop"]["command"] == "stop"
    assert result["followup_stop"]["reason"] == "publisher_error"
    assert len(publisher.payloads) == 1
    return result


def test_stage31_ros2_no_publisher_disarm_is_unavailable() -> dict:
    adapter = MiguelHiWonderRos2Adapter()
    adapter.armed = True
    result = adapter.disarm()
    assert adapter.armed is False
    assert result["ok"] is False
    assert result["blocked"] is True
    assert result["reason"] == "adapter_unavailable"
    assert result["stop_result"]["command"] == "stop"
    return result


def test_stage31_ros2_no_publisher_close_is_unavailable() -> dict:
    adapter = MiguelHiWonderRos2Adapter()
    adapter.armed = True
    result = adapter.close()
    assert adapter.armed is False
    assert result["ok"] is False
    assert result["blocked"] is True
    assert result["reason"] == "adapter_unavailable"
    return result


def test_stage4_allow_real_ros2_missing_dependencies_is_structured() -> dict:
    saved = _install_ros2_import_loader()
    try:
        adapter = MiguelHiWonderRos2Adapter(
            allow_real_ros2=True,
            require_safe_graph_to_arm=False,
        )
    finally:
        _restore_modules(saved)

    status = adapter.backend_status()
    assert status["available"] is False
    assert status["backend"] == "unavailable"
    assert status["real_ros2_enabled"] is False
    assert status["reason"] == "ros2_dependency_unavailable"
    result = adapter.stop("missing deps")
    assert result["reason"] == "ros2_dependency_unavailable"
    return result


def test_stage4_fake_real_ros2_backend_constructs_without_publish() -> dict:
    saved, rclpy = _install_fake_ros2_modules()
    try:
        adapter = MiguelHiWonderRos2Adapter(
            allow_real_ros2=True,
            require_safe_graph_to_arm=False,
        )
        status = adapter.backend_status()
        assert status["available"] is True
        assert status["backend"] == "real_ros2"
        assert status["real_ros2_enabled"] is True
        assert status["hardware_verified"] is False
        assert len(rclpy.created_nodes) == 1
        assert rclpy.created_nodes[0].name == "miguel_hiwonder_ros2_adapter"
        assert len(rclpy.created_nodes[0].publishers) == 1
        assert rclpy.created_nodes[0].publishers[0].messages == []
        return status
    finally:
        _restore_modules(saved)


def test_stage4_fake_real_ros2_stop_publishes_zero_twist() -> dict:
    saved, rclpy = _install_fake_ros2_modules()
    try:
        adapter = MiguelHiWonderRos2Adapter(allow_real_ros2=True)
        result = adapter.stop("manual stop")
        publisher = rclpy.created_nodes[0].publishers[0]
        assert result["ok"] is True
        assert len(publisher.messages) == 1
        message = publisher.messages[0]
        assert isinstance(message, FakeRos2Twist)
        assert message.linear.x == 0.0
        assert message.linear.y == 0.0
        assert message.angular.z == 0.0
        return result
    finally:
        _restore_modules(saved)


def test_stage4_fake_real_ros2_arm_move_forward_publishes_twist_and_stop() -> dict:
    saved, rclpy = _install_fake_ros2_modules()
    try:
        adapter = MiguelHiWonderRos2Adapter(
            allow_real_ros2=True,
            require_safe_graph_to_arm=False,
        )
        adapter.arm()
        result = adapter.set_velocity("move_forward", "slow", 1.0)
        publisher = rclpy.created_nodes[0].publishers[0]
        assert result["ok"] is True
        assert len(publisher.messages) == 2
        move_message, stop_message = publisher.messages
        assert 0 < move_message.linear.x <= adapter.MAX_LINEAR_X
        assert move_message.linear.y == 0.0
        assert move_message.angular.z == 0.0
        assert stop_message.linear.x == 0.0
        assert stop_message.linear.y == 0.0
        assert stop_message.angular.z == 0.0
        return result
    finally:
        _restore_modules(saved)


def test_stage4_fake_real_ros2_close_destroys_owned_node() -> dict:
    saved, rclpy = _install_fake_ros2_modules()
    try:
        adapter = MiguelHiWonderRos2Adapter(
            allow_real_ros2=True,
            require_safe_graph_to_arm=False,
        )
        node = rclpy.created_nodes[0]
        adapter.arm()
        result = adapter.close()
        assert adapter.armed is False
        assert node.destroyed is True
        assert node.publishers[0].messages[-1].linear.x == 0.0
        return result
    finally:
        _restore_modules(saved)


def test_stage5_ros2_probe_module_import_does_not_import_ros_packages() -> dict:
    saved = _save_and_remove_modules(
        "miguel_core.miguel_hiwonder_ros2_probe",
        "rclpy",
        "geometry_msgs",
        "geometry_msgs.msg",
    )
    try:
        importlib.import_module("miguel_core.miguel_hiwonder_ros2_probe")
        assert "rclpy" not in sys.modules
        assert "geometry_msgs" not in sys.modules
        return {"rclpy_imported": False, "geometry_msgs_imported": False}
    finally:
        _restore_modules(saved)


def test_stage5_ros2_probe_constructor_is_inert() -> dict:
    saved = _save_and_remove_modules("rclpy", "geometry_msgs", "geometry_msgs.msg")
    try:
        probe = MiguelHiWonderRos2Probe()
        assert probe.node is None
        assert probe.rclpy_module is None
        assert "rclpy" not in sys.modules
        assert "geometry_msgs" not in sys.modules
        return {"node": probe.node, "rclpy_module": probe.rclpy_module}
    finally:
        _restore_modules(saved)


def test_stage5_ros2_probe_check_imports_handles_missing_modules() -> dict:
    saved = _install_ros2_import_loader()
    try:
        result = MiguelHiWonderRos2Probe().check_imports()
    finally:
        _restore_modules(saved)

    assert result["ok"] is False
    assert result["rclpy_available"] is False
    assert result["geometry_msgs_available"] is False
    assert result["reason"] == "ros2_dependency_unavailable"
    return result


def test_stage5_ros2_probe_check_environment_reads_expected_vars() -> dict:
    old_values = {
        name: os.environ.get(name)
        for name in ("ROS_DOMAIN_ID", "ROS_LOCALHOST_ONLY", "CYCLONEDDS_URI", "RMW_IMPLEMENTATION")
    }
    try:
        os.environ["ROS_DOMAIN_ID"] = "0"
        os.environ["ROS_LOCALHOST_ONLY"] = "0"
        os.environ["CYCLONEDDS_URI"] = "file:///etc/cyclonedds/config.xml"
        os.environ["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
        result = MiguelHiWonderRos2Probe().check_environment()
    finally:
        for name, value in old_values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    assert result["ok"] is True
    assert result["environment"]["ROS_DOMAIN_ID"] == "0"
    assert result["environment"]["CYCLONEDDS_URI"] == "file:///etc/cyclonedds/config.xml"
    return result


def test_stage5_ros2_probe_create_node_with_fake_rclpy() -> dict:
    saved, rclpy = _install_fake_ros2_modules()
    probe = MiguelHiWonderRos2Probe()
    try:
        result = probe.create_probe_node()
        assert result["ok"] is True
        assert result["node_created"] is True
        assert len(rclpy.created_nodes) == 1
        assert rclpy.created_nodes[0].name == "miguel_hiwonder_ros2_probe"
        assert rclpy.created_nodes[0].publishers == []
        return result
    finally:
        probe.destroy_probe_node()
        _restore_modules(saved)


def test_stage5_ros2_probe_destroy_owned_fake_node() -> dict:
    saved, rclpy = _install_fake_ros2_modules()
    probe = MiguelHiWonderRos2Probe()
    try:
        probe.create_probe_node()
        node = rclpy.created_nodes[0]
        result = probe.destroy_probe_node()
        assert result["ok"] is True
        assert result["node_destroyed"] is True
        assert node.destroyed is True
        assert probe.node is None
        return result
    finally:
        _restore_modules(saved)


def test_stage5_ros2_probe_readiness_never_tests_movement() -> dict:
    saved, rclpy = _install_fake_ros2_modules()
    probe = MiguelHiWonderRos2Probe()
    try:
        result = probe.probe_readiness()
        assert result["ok"] is True
        assert result["ros2_available"] is True
        assert result["node_created"] is True
        assert result["expected_topic"] == "/controller/cmd_vel"
        assert result["expected_msg_type"] == "geometry_msgs/msg/Twist"
        assert result["can_publish_tested"] is False
        assert result["movement_tested"] is False
        assert result["hardware_verified"] is False
        assert rclpy.created_nodes[0].publishers == []
        return result
    finally:
        probe.destroy_probe_node()
        _restore_modules(saved)


def test_stage5_ros2_probe_node_creation_failure_is_structured() -> dict:
    saved, _rclpy = _install_failing_ros2_create_node_module()
    try:
        result = MiguelHiWonderRos2Probe().create_probe_node()
        assert result["ok"] is False
        assert result["node_created"] is False
        assert result["reason"] == "ros2_probe_error"
        assert "probe node create failed" in result["error"]
        return result
    finally:
        _restore_modules(saved)


def test_stage5_ros2_probe_destroy_failure_is_structured() -> dict:
    probe = MiguelHiWonderRos2Probe(node=FailingRos2Node())
    probe._owned_node = True
    result = probe.destroy_probe_node()
    assert result["ok"] is False
    assert result["node_destroyed"] is False
    assert result["reason"] == "ros2_probe_error"
    assert "probe destroy failed" in result["error"]
    assert probe.node is None
    return result


def test_stage9_ros2_arm_refuses_competing_publishers() -> dict:
    saved, rclpy = _install_fake_ros2_modules()
    try:
        report = clean_readiness_report(
            cmd_vel_safe_to_arm=False,
            competing_cmd_vel_publishers=["lidar_app"],
            graph={
                "cmd_vel": {
                    "topic_exists": True,
                    "message_type": "geometry_msgs/msg/Twist",
                    "subscription_count": 1,
                    "competing_publishers": ["lidar_app"],
                    "safe_to_arm": False,
                }
            },
        )
        probe = FakeReadinessProbe(report)
        adapter = MiguelHiWonderRos2Adapter(allow_real_ros2=True, readiness_probe=probe)
        result = adapter.arm()
        assert result["ok"] is False
        assert result["blocked"] is True
        assert result["armed"] is False
        assert adapter.armed is False
        assert result["reason"] == "ros2_graph_not_safe_to_arm"
        assert result["payload"]["competing_cmd_vel_publishers"] == ["lidar_app"]
        assert rclpy.created_nodes[0].publishers[0].messages == []
        return result
    finally:
        _restore_modules(saved)


def test_stage9_ros2_arm_refuses_missing_cmd_vel() -> dict:
    saved, _rclpy = _install_fake_ros2_modules()
    try:
        report = clean_readiness_report(
            graph={
                "cmd_vel": {
                    "topic_exists": False,
                    "message_type": None,
                    "subscription_count": 0,
                    "competing_publishers": [],
                }
            }
        )
        adapter = MiguelHiWonderRos2Adapter(
            allow_real_ros2=True,
            readiness_probe=FakeReadinessProbe(report),
        )
        result = adapter.arm()
        assert result["ok"] is False
        assert result["reason"] == "cmd_vel_topic_missing"
        assert adapter.armed is False
        return result
    finally:
        _restore_modules(saved)


def test_stage9_ros2_arm_refuses_no_cmd_vel_subscriber() -> dict:
    saved, _rclpy = _install_fake_ros2_modules()
    try:
        report = clean_readiness_report(
            graph={
                "cmd_vel": {
                    "topic_exists": True,
                    "message_type": "geometry_msgs/msg/Twist",
                    "subscription_count": 0,
                    "competing_publishers": [],
                }
            }
        )
        adapter = MiguelHiWonderRos2Adapter(
            allow_real_ros2=True,
            readiness_probe=FakeReadinessProbe(report),
        )
        result = adapter.arm()
        assert result["ok"] is False
        assert result["reason"] == "cmd_vel_no_subscriber"
        assert adapter.armed is False
        return result
    finally:
        _restore_modules(saved)


def test_stage9_ros2_arm_refuses_wrong_cmd_vel_type() -> dict:
    saved, _rclpy = _install_fake_ros2_modules()
    try:
        report = clean_readiness_report(
            graph={
                "cmd_vel": {
                    "topic_exists": True,
                    "message_type": "std_msgs/msg/String",
                    "subscription_count": 1,
                    "competing_publishers": [],
                }
            }
        )
        adapter = MiguelHiWonderRos2Adapter(
            allow_real_ros2=True,
            readiness_probe=FakeReadinessProbe(report),
        )
        result = adapter.arm()
        assert result["ok"] is False
        assert result["reason"] == "cmd_vel_wrong_message_type"
        assert adapter.armed is False
        return result
    finally:
        _restore_modules(saved)


def test_stage9_ros2_arm_succeeds_with_clean_readiness() -> dict:
    saved, rclpy = _install_fake_ros2_modules()
    try:
        adapter = MiguelHiWonderRos2Adapter(
            allow_real_ros2=True,
            readiness_probe=FakeReadinessProbe(clean_readiness_report()),
        )
        result = adapter.arm()
        assert result["ok"] is True
        assert result["armed"] is True
        assert adapter.armed is True
        assert result["payload"]["cmd_vel_safe_to_arm"] is True
        assert rclpy.created_nodes[0].publishers[0].messages == []
        return result
    finally:
        _restore_modules(saved)


def test_stage9_ros2_arm_override_allows_competing_publishers_with_warning() -> dict:
    saved, rclpy = _install_fake_ros2_modules()
    try:
        report = clean_readiness_report(
            cmd_vel_safe_to_arm=False,
            competing_cmd_vel_publishers=["lidar_app"],
            graph={
                "cmd_vel": {
                    "topic_exists": True,
                    "message_type": "geometry_msgs/msg/Twist",
                    "subscription_count": 1,
                    "competing_publishers": ["lidar_app"],
                    "safe_to_arm": False,
                }
            },
        )
        adapter = MiguelHiWonderRos2Adapter(
            allow_real_ros2=True,
            readiness_probe=FakeReadinessProbe(report),
            allow_competing_publishers_override=True,
        )
        result = adapter.arm()
        assert result["ok"] is True
        assert adapter.armed is True
        assert result["params"]["readiness_warnings"][0]["reason"] == "competing_cmd_vel_publishers_override"
        assert rclpy.created_nodes[0].publishers[0].messages == []
        return result
    finally:
        _restore_modules(saved)


def test_stage9_ros2_movement_blocked_after_failed_arm() -> dict:
    saved, rclpy = _install_fake_ros2_modules()
    try:
        report = clean_readiness_report(
            cmd_vel_safe_to_arm=False,
            competing_cmd_vel_publishers=["joystick_control"],
            graph={
                "cmd_vel": {
                    "topic_exists": True,
                    "message_type": "geometry_msgs/msg/Twist",
                    "subscription_count": 1,
                    "competing_publishers": ["joystick_control"],
                    "safe_to_arm": False,
                }
            },
        )
        adapter = MiguelHiWonderRos2Adapter(
            allow_real_ros2=True,
            readiness_probe=FakeReadinessProbe(report),
        )
        arm_result = adapter.arm()
        move_result = adapter.set_velocity("move_forward", "slow", 1.0)
        assert arm_result["ok"] is False
        assert move_result["ok"] is False
        assert move_result["blocked"] is True
        assert move_result["reason"] == "adapter_disarmed"
        assert rclpy.created_nodes[0].publishers[0].messages == []
        return {"arm": arm_result, "move": move_result}
    finally:
        _restore_modules(saved)


def test_stage9_ros2_arm_refuses_unreadable_sensor() -> dict:
    saved, _rclpy = _install_fake_ros2_modules()
    try:
        report = clean_readiness_report(lidar_readable=False, lidar_ok=False)
        adapter = MiguelHiWonderRos2Adapter(
            allow_real_ros2=True,
            readiness_probe=FakeReadinessProbe(report),
        )
        result = adapter.arm()
        assert result["ok"] is False
        assert result["reason"] == "lidar_not_readable"
        assert adapter.armed is False
        return result
    finally:
        _restore_modules(saved)


def test_stage91_cmd_vel_ignores_explicit_miguel_publishers() -> dict:
    node = FakeRos2GraphNode(
        topics=[("/controller/cmd_vel", ["geometry_msgs/msg/Twist"])],
        publisher_nodes_by_topic={
            "/controller/cmd_vel": [
                "miguel_hiwonder_ros2_adapter",
                "miguel_hiwonder_ros2_probe",
                "miguel_hiwonder_ros2_graph_probe",
            ]
        },
        subscriber_nodes_by_topic={"/controller/cmd_vel": ["odom_publisher"]},
    )
    result = MiguelHiWonderRos2GraphProbe(node=node, graph_settle_sec=0.0).inspect_cmd_vel()
    assert result["publisher_count"] == 3
    assert result["subscription_count"] == 1
    assert result["ignored_publishers"] == [
        "miguel_hiwonder_ros2_adapter",
        "miguel_hiwonder_ros2_probe",
        "miguel_hiwonder_ros2_graph_probe",
    ]
    assert result["competing_publishers"] == []
    assert result["safe_to_arm"] is True
    return result


def test_stage91_unknown_publisher_remains_competing() -> dict:
    node = FakeRos2GraphNode(
        topics=[("/controller/cmd_vel", ["geometry_msgs/msg/Twist"])],
        subscriber_nodes_by_topic={"/controller/cmd_vel": ["odom_publisher"]},
    )
    node.count_publishers = lambda topic: 1
    result = MiguelHiWonderRos2GraphProbe(node=node, graph_settle_sec=0.0).inspect_cmd_vel()
    assert result["publisher_count"] == 1
    assert result["competing_publishers"] == ["unknown"]
    assert result["safe_to_arm"] is False
    return result


def test_stage91_odom_subscriber_endpoint_counts_as_subscription() -> dict:
    node = FakeRos2GraphNode(
        topics=[("/controller/cmd_vel", ["geometry_msgs/msg/Twist"])],
        publisher_nodes_by_topic={"/controller/cmd_vel": ["miguel_hiwonder_ros2_adapter"]},
        subscriber_nodes_by_topic={"/controller/cmd_vel": ["odom_publisher"]},
    )
    node.count_subscribers = lambda topic: 0
    result = MiguelHiWonderRos2GraphProbe(node=node, graph_settle_sec=0.0).inspect_cmd_vel()
    assert result["subscriber_nodes"] == ["odom_publisher"]
    assert result["subscription_count"] == 1
    assert result["safe_to_arm"] is True
    return result


def test_stage91_cmd_vel_no_subscribers_is_not_arm_ready() -> dict:
    node = FakeRos2GraphNode(
        topics=[("/controller/cmd_vel", ["geometry_msgs/msg/Twist"])],
        publisher_nodes_by_topic={"/controller/cmd_vel": ["miguel_hiwonder_ros2_adapter"]},
        subscriber_nodes_by_topic={"/controller/cmd_vel": []},
    )
    result = MiguelHiWonderRos2GraphProbe(node=node, graph_settle_sec=0.0).inspect_cmd_vel()
    assert result["subscription_count"] == 0
    assert result["safe_to_arm"] is True
    adapter = MiguelHiWonderRos2Adapter(
        publisher=FakeTwistPublisher(),
        readiness_probe=FakeReadinessProbe(
            clean_readiness_report(
                graph={
                    "cmd_vel": {
                        "topic_exists": True,
                        "message_type": "geometry_msgs/msg/Twist",
                        "subscription_count": 0,
                        "competing_publishers": [],
                        "safe_to_arm": True,
                    }
                }
            )
        ),
    )
    arm = adapter.arm()
    assert arm["ok"] is False
    assert arm["reason"] == "cmd_vel_no_subscriber"
    return {"inspect": result, "arm": arm}


def test_stage91_arm_succeeds_with_only_miguel_publisher_and_odom_subscriber() -> dict:
    saved, rclpy = _install_fake_ros2_modules()
    try:
        report = clean_readiness_report(
            graph={
                "cmd_vel": {
                    "topic_exists": True,
                    "message_type": "geometry_msgs/msg/Twist",
                    "subscription_count": 1,
                    "publisher_nodes": ["miguel_hiwonder_ros2_adapter"],
                    "subscriber_nodes": ["odom_publisher"],
                    "competing_publishers": [],
                    "ignored_publishers": ["miguel_hiwonder_ros2_adapter"],
                    "safe_to_arm": True,
                }
            }
        )
        adapter = MiguelHiWonderRos2Adapter(
            allow_real_ros2=True,
            readiness_probe=FakeReadinessProbe(report),
        )
        readiness = adapter.check_arm_readiness()
        result = adapter.arm()
        assert readiness["ok"] is True
        assert result["ok"] is True
        assert adapter.armed is True
        assert rclpy.created_nodes[0].publishers[0].messages == []
        return result
    finally:
        _restore_modules(saved)


def test_stage91_arm_fails_with_external_publisher_and_subscriber() -> dict:
    saved, rclpy = _install_fake_ros2_modules()
    try:
        report = clean_readiness_report(
            cmd_vel_safe_to_arm=False,
            competing_cmd_vel_publishers=["unknown"],
            graph={
                "cmd_vel": {
                    "topic_exists": True,
                    "message_type": "geometry_msgs/msg/Twist",
                    "subscription_count": 1,
                    "publisher_nodes": ["unknown"],
                    "subscriber_nodes": ["odom_publisher"],
                    "competing_publishers": ["unknown"],
                    "safe_to_arm": False,
                }
            },
        )
        adapter = MiguelHiWonderRos2Adapter(
            allow_real_ros2=True,
            readiness_probe=FakeReadinessProbe(report),
        )
        result = adapter.arm()
        assert result["ok"] is False
        assert result["reason"] == "ros2_graph_not_safe_to_arm"
        assert adapter.armed is False
        assert rclpy.created_nodes[0].publishers[0].messages == []
        return result
    finally:
        _restore_modules(saved)


def test_stage91_arm_override_allows_external_publisher_with_subscriber() -> dict:
    saved, rclpy = _install_fake_ros2_modules()
    try:
        report = clean_readiness_report(
            cmd_vel_safe_to_arm=False,
            competing_cmd_vel_publishers=["unknown"],
            graph={
                "cmd_vel": {
                    "topic_exists": True,
                    "message_type": "geometry_msgs/msg/Twist",
                    "subscription_count": 1,
                    "publisher_nodes": ["unknown"],
                    "subscriber_nodes": ["odom_publisher"],
                    "competing_publishers": ["unknown"],
                    "safe_to_arm": False,
                }
            },
        )
        adapter = MiguelHiWonderRos2Adapter(
            allow_real_ros2=True,
            readiness_probe=FakeReadinessProbe(report),
            allow_competing_publishers_override=True,
        )
        result = adapter.arm()
        assert result["ok"] is True
        assert adapter.armed is True
        assert result["params"]["readiness_warnings"][0]["publishers"] == ["unknown"]
        assert rclpy.created_nodes[0].publishers[0].messages == []
        return result
    finally:
        _restore_modules(saved)


def test_stage10_odom_fallback_uses_odom_raw() -> dict:
    node = FakeRos2GraphNode(messages_by_topic={"/odom_raw": [FakeOdometry()]})
    saved, _rclpy = _install_fake_graph_ros2_modules(node)
    try:
        result = MiguelHiWonderRos2GraphProbe().read_odom_once(timeout_sec=0.01)
    finally:
        _restore_modules(saved)
    assert result["ok"] is True
    assert result["topic"] == "/odom_raw"
    assert result["selected_topic"] == "/odom_raw"
    assert result["x"] == 1.25
    return result


def test_stage10_set_motor_inspection_detects_required_subscriber() -> dict:
    node = FakeRos2GraphNode(
        topics=[
            (
                "/ros_robot_controller/set_motor",
                ["ros_robot_controller_msgs/msg/MotorsState"],
            )
        ],
        publisher_nodes_by_topic={"/ros_robot_controller/set_motor": ["odom_publisher"]},
        subscriber_nodes_by_topic={"/ros_robot_controller/set_motor": ["ros_robot_controller"]},
    )
    result = MiguelHiWonderRos2GraphProbe(node=node, graph_settle_sec=0.0).inspect_direct_motor()
    assert result["topic_exists"] is True
    assert result["message_type"] == "ros_robot_controller_msgs/msg/MotorsState"
    assert result["required_subscriber_present"] is True
    assert result["safe_direct_motor_control"] is True
    return result


def test_stage10_set_motor_inspection_detects_external_direct_publisher() -> dict:
    node = FakeRos2GraphNode(
        topics=[
            (
                "/ros_robot_controller/set_motor",
                ["ros_robot_controller_msgs/msg/MotorsState"],
            )
        ],
        publisher_nodes_by_topic={
            "/ros_robot_controller/set_motor": ["odom_publisher", "hand_gesture"]
        },
        subscriber_nodes_by_topic={"/ros_robot_controller/set_motor": ["ros_robot_controller"]},
    )
    result = MiguelHiWonderRos2GraphProbe(node=node, graph_settle_sec=0.0).inspect_direct_motor()
    assert result["external_direct_motor_publishers"] == ["hand_gesture"]
    assert result["safe_direct_motor_control"] is False
    assert result["movement_tested"] is False
    assert result["hardware_verified"] is False
    return result


def test_stage10_low_level_motor_chain_ok_when_required_endpoints_exist() -> dict:
    node = FakeRos2GraphNode(
        topics=[
            ("/controller/cmd_vel", ["geometry_msgs/msg/Twist"]),
            (
                "/ros_robot_controller/set_motor",
                ["ros_robot_controller_msgs/msg/MotorsState"],
            ),
        ],
        publisher_nodes_by_topic={
            "/controller/cmd_vel": ["miguel_hiwonder_ros2_adapter"],
            "/ros_robot_controller/set_motor": ["odom_publisher"],
        },
        subscriber_nodes_by_topic={
            "/controller/cmd_vel": ["odom_publisher"],
            "/ros_robot_controller/set_motor": ["ros_robot_controller"],
        },
    )
    probe = MiguelHiWonderRos2GraphProbe(node=node, graph_settle_sec=0.0)
    chain = probe._build_motor_chain_readiness(probe.inspect_cmd_vel(), probe.inspect_direct_motor())
    assert chain["cmd_vel_receiver_ok"] is True
    assert chain["motor_receiver_ok"] is True
    assert chain["odom_publisher_to_motor_ok"] is True
    assert chain["low_level_motor_chain_ok"] is True
    return chain


def test_stage10_low_level_motor_chain_fails_when_endpoint_missing() -> dict:
    node = FakeRos2GraphNode(
        topics=[
            ("/controller/cmd_vel", ["geometry_msgs/msg/Twist"]),
            (
                "/ros_robot_controller/set_motor",
                ["ros_robot_controller_msgs/msg/MotorsState"],
            ),
        ],
        publisher_nodes_by_topic={
            "/controller/cmd_vel": ["miguel_hiwonder_ros2_adapter"],
            "/ros_robot_controller/set_motor": [],
        },
        subscriber_nodes_by_topic={
            "/controller/cmd_vel": ["odom_publisher"],
            "/ros_robot_controller/set_motor": ["ros_robot_controller"],
        },
    )
    probe = MiguelHiWonderRos2GraphProbe(node=node, graph_settle_sec=0.0)
    chain = probe._build_motor_chain_readiness(probe.inspect_cmd_vel(), probe.inspect_direct_motor())
    assert chain["cmd_vel_receiver_ok"] is True
    assert chain["motor_receiver_ok"] is True
    assert chain["odom_publisher_to_motor_ok"] is False
    assert chain["low_level_motor_chain_ok"] is False
    return chain


def test_stage10_ros2_arm_refuses_external_direct_motor_publisher() -> dict:
    saved, rclpy = _install_fake_ros2_modules()
    try:
        report = clean_readiness_report(
            direct_motor_safe_to_arm=False,
            external_direct_motor_publishers=["hand_gesture"],
        )
        adapter = MiguelHiWonderRos2Adapter(
            allow_real_ros2=True,
            readiness_probe=FakeReadinessProbe(report),
        )
        result = adapter.arm()
        assert result["ok"] is False
        assert result["reason"] == "direct_motor_graph_not_safe_to_arm"
        assert adapter.armed is False
        assert rclpy.created_nodes[0].publishers[0].messages == []
        return result
    finally:
        _restore_modules(saved)


def test_stage10_ros2_arm_refuses_incomplete_low_level_motor_chain() -> dict:
    saved, rclpy = _install_fake_ros2_modules()
    try:
        report = clean_readiness_report(low_level_motor_chain_ok=False)
        adapter = MiguelHiWonderRos2Adapter(
            allow_real_ros2=True,
            readiness_probe=FakeReadinessProbe(report),
        )
        result = adapter.arm()
        assert result["ok"] is False
        assert result["reason"] == "low_level_motor_chain_not_ready"
        assert adapter.armed is False
        assert rclpy.created_nodes[0].publishers[0].messages == []
        return result
    finally:
        _restore_modules(saved)


def test_stage10_ros2_arm_succeeds_with_complete_clean_motor_chain() -> dict:
    saved, rclpy = _install_fake_ros2_modules()
    try:
        adapter = MiguelHiWonderRos2Adapter(
            allow_real_ros2=True,
            readiness_probe=FakeReadinessProbe(clean_readiness_report()),
        )
        result = adapter.arm()
        assert result["ok"] is True
        assert adapter.armed is True
        assert result["payload"]["low_level_motor_chain_ok"] is True
        assert result["payload"]["external_direct_motor_publishers"] == []
        assert rclpy.created_nodes[0].publishers[0].messages == []
        return result
    finally:
        _restore_modules(saved)


def test_stage10_movement_blocked_after_direct_motor_arm_failure() -> dict:
    saved, rclpy = _install_fake_ros2_modules()
    try:
        report = clean_readiness_report(
            direct_motor_safe_to_arm=False,
            external_direct_motor_publishers=["hand_gesture"],
        )
        adapter = MiguelHiWonderRos2Adapter(
            allow_real_ros2=True,
            readiness_probe=FakeReadinessProbe(report),
        )
        arm_result = adapter.arm()
        move_result = adapter.set_velocity("move_forward", "slow", 1.0)
        assert arm_result["ok"] is False
        assert move_result["ok"] is False
        assert move_result["reason"] == "adapter_disarmed"
        assert rclpy.created_nodes[0].publishers[0].messages == []
        return {"arm": arm_result, "move": move_result}
    finally:
        _restore_modules(saved)


def test_stage11_cmd_vel_quiet_returns_true_on_timeout() -> dict:
    node = FakeRos2GraphNode()
    saved = _install_stage11_ros_sensitive_import_guard()
    probe = MiguelHiWonderRos2GraphProbe(node=node, graph_settle_sec=0.0)
    try:
        result = probe.observe_cmd_vel_quiet_once(timeout_sec=0.01)
    finally:
        probe.destroy_node()
        node.destroy_node()
        _restore_modules(saved)
    assert result["ok"] is True
    assert result["observed_message"] is False
    assert result["quiet"] is True
    assert result["topic"] == "/controller/cmd_vel"
    assert result["movement_tested"] is False
    assert result["hardware_verified"] is False
    return result


def test_stage11_cmd_vel_quiet_returns_false_when_message_observed() -> dict:
    node = FakeRos2GraphNode(messages_by_topic={"/controller/cmd_vel": [FakeRos2Twist()]})
    saved = _install_stage11_ros_sensitive_import_guard()
    probe = MiguelHiWonderRos2GraphProbe(node=node, graph_settle_sec=0.0)
    try:
        result = probe.observe_cmd_vel_quiet_once(timeout_sec=0.01)
    finally:
        probe.destroy_node()
        node.destroy_node()
        _restore_modules(saved)
    assert result["ok"] is True
    assert result["observed_message"] is True
    assert result["quiet"] is False
    assert result["reason"] == "message_observed"
    return result


def test_stage11_odom_stationary_true_for_zero_twist_samples() -> dict:
    zero = (0.0, 0.0, 0.0)
    node = FakeRos2GraphNode(messages_by_topic={"/odom": [FakeOdometry(zero, zero), FakeOdometry(zero, zero)]})
    saved = _install_stage11_ros_sensitive_import_guard()
    probe = MiguelHiWonderRos2GraphProbe(node=node, graph_settle_sec=0.0)
    try:
        result = probe.observe_odom_stationary(timeout_sec=0.01, gap_sec=0.0)
    finally:
        probe.destroy_node()
        node.destroy_node()
        _restore_modules(saved)
    assert result["ok"] is True
    assert result["selected_topic"] == "/odom"
    assert result["stationary"] is True
    assert result["max_abs_linear"] == 0.0
    assert result["max_abs_angular"] == 0.0
    return result


def test_stage11_odom_stationary_false_for_nonzero_second_sample() -> dict:
    zero = (0.0, 0.0, 0.0)
    node = FakeRos2GraphNode(
        messages_by_topic={
            "/odom": [
                FakeOdometry(zero, zero),
                FakeOdometry((0.05, 0.0, 0.0), zero),
            ]
        }
    )
    saved = _install_stage11_ros_sensitive_import_guard()
    probe = MiguelHiWonderRos2GraphProbe(node=node, graph_settle_sec=0.0)
    try:
        result = probe.observe_odom_stationary(timeout_sec=0.01, gap_sec=0.0)
    finally:
        probe.destroy_node()
        node.destroy_node()
        _restore_modules(saved)
    assert result["ok"] is True
    assert result["stationary"] is False
    assert result["max_abs_linear"] == 0.05
    assert result["reason"] == "odom_twist_nonzero"
    return result


def test_stage11_inactive_readiness_reports_hand_gesture_direct_motor_blocking() -> dict:
    zero = (0.0, 0.0, 0.0)
    node = FakeRos2GraphNode(
        nodes=["/controller", "/odom_publisher"],
        topics=[
            ("/controller/cmd_vel", ["geometry_msgs/msg/Twist"]),
            ("/ros_robot_controller/set_motor", ["ros_robot_controller_msgs/msg/MotorsState"]),
            ("/ros_robot_controller/battery", ["std_msgs/msg/UInt16"]),
            ("/scan_raw", ["sensor_msgs/msg/LaserScan"]),
            ("/odom_raw", ["nav_msgs/msg/Odometry"]),
        ],
        publisher_nodes_by_topic={
            "/controller/cmd_vel": ["lidar_app"],
            "/ros_robot_controller/set_motor": ["odom_publisher", "hand_gesture"],
        },
        subscriber_nodes_by_topic={
            "/controller/cmd_vel": ["odom_publisher"],
            "/ros_robot_controller/set_motor": ["ros_robot_controller"],
        },
        messages_by_topic={
            "/ros_robot_controller/battery": [FakeUInt16(7424)],
            "/scan_raw": [FakeLaserScan([1.0], angle_min=0.0, angle_increment=0.1)],
            "/odom_raw": [FakeOdometry(zero, zero), FakeOdometry(zero, zero), FakeOdometry(zero, zero)],
        },
    )
    saved = _install_stage11_ros_sensitive_import_guard()
    probe = MiguelHiWonderRos2GraphProbe(node=node, graph_settle_sec=0.0)
    try:
        result = probe.build_inactive_publisher_readiness_report(
            cmd_vel_quiet_timeout_sec=0.01,
            odom_timeout_sec=0.01,
            odom_gap_sec=0.0,
        )
    finally:
        probe.destroy_node()
        node.destroy_node()
        _restore_modules(saved)
    assert result["cmd_vel_quiet"]["quiet"] is True
    assert result["odom_stationary"]["stationary"] is True
    assert result["inactive_publishers_observed"] is True
    assert result["direct_motor_publishers"] == ["hand_gesture"]
    assert result["direct_motor_blocking"] is True
    assert result["relaxed_safe_to_arm"] is False
    return result


def test_stage11_adapter_blocks_external_cmd_vel_by_default() -> dict:
    saved = _install_stage11_ros_sensitive_import_guard()
    publisher = FakeTwistPublisher()
    adapter = None
    try:
        adapter = MiguelHiWonderRos2Adapter(
            publisher=publisher,
            readiness_probe=FakeInactiveReadinessProbe(inactive_readiness_report()),
        )
        result = adapter.arm()
        assert result["ok"] is False
        assert result["reason"] == "ros2_graph_not_safe_to_arm"
        assert adapter.armed is False
        assert publisher.payloads == []
        return result
    finally:
        if adapter is not None:
            _destroy_adapter_node_without_publish(adapter)
        _restore_modules(saved)


def test_stage11_adapter_inactive_external_cmd_vel_opt_in_can_pass_without_direct_publishers() -> dict:
    saved = _install_stage11_ros_sensitive_import_guard()
    publisher = FakeTwistPublisher()
    adapter = None
    try:
        adapter = MiguelHiWonderRos2Adapter(
            publisher=publisher,
            readiness_probe=FakeInactiveReadinessProbe(inactive_readiness_report()),
            allow_inactive_external_cmd_vel_publishers=True,
        )
        result = adapter.arm()
        assert result["ok"] is True
        assert adapter.armed is True
        assert result["params"]["readiness_warnings"][0]["reason"] == "inactive_external_cmd_vel_publishers_observed"
        assert publisher.payloads == []
        return result
    finally:
        if adapter is not None:
            _destroy_adapter_node_without_publish(adapter)
        _restore_modules(saved)


def test_stage11_adapter_blocks_hand_gesture_direct_motor_without_direct_override() -> dict:
    saved = _install_stage11_ros_sensitive_import_guard()
    publisher = FakeTwistPublisher()
    adapter = None
    try:
        report = inactive_readiness_report(
            direct_motor_safe_to_arm=False,
            external_direct_motor_publishers=["hand_gesture"],
            direct_motor_publishers=["hand_gesture"],
            direct_motor_blocking=True,
            relaxed_safe_to_arm=False,
        )
        adapter = MiguelHiWonderRos2Adapter(
            publisher=publisher,
            readiness_probe=FakeInactiveReadinessProbe(report),
            allow_inactive_external_cmd_vel_publishers=True,
        )
        result = adapter.arm()
        assert result["ok"] is False
        assert result["reason"] == "direct_motor_graph_not_safe_to_arm"
        assert adapter.armed is False
        assert publisher.payloads == []
        return result
    finally:
        if adapter is not None:
            _destroy_adapter_node_without_publish(adapter)
        _restore_modules(saved)


def test_stage11_adapter_allows_hand_gesture_only_with_direct_override() -> dict:
    saved = _install_stage11_ros_sensitive_import_guard()
    publisher = FakeTwistPublisher()
    adapter = None
    try:
        report = inactive_readiness_report(
            direct_motor_safe_to_arm=False,
            external_direct_motor_publishers=["hand_gesture"],
            direct_motor_publishers=["hand_gesture"],
            direct_motor_blocking=True,
            relaxed_safe_to_arm=False,
        )
        adapter = MiguelHiWonderRos2Adapter(
            publisher=publisher,
            readiness_probe=FakeInactiveReadinessProbe(report),
            allow_inactive_external_cmd_vel_publishers=True,
            allow_external_direct_motor_publishers_override=True,
        )
        result = adapter.arm()
        assert result["ok"] is True
        assert adapter.armed is True
        assert publisher.payloads == []
        return result
    finally:
        if adapter is not None:
            _destroy_adapter_node_without_publish(adapter)
        _restore_modules(saved)


def test_stage11_movement_blocked_after_inactive_arm_failure() -> dict:
    saved = _install_stage11_ros_sensitive_import_guard()
    publisher = FakeTwistPublisher()
    adapter = None
    try:
        report = inactive_readiness_report(
            cmd_vel_quiet={
                "ok": True,
                "observed_message": True,
                "quiet": False,
                "topic": "/controller/cmd_vel",
                "reason": "message_observed",
            },
            inactive_publishers_observed=False,
            relaxed_safe_to_arm=False,
        )
        adapter = MiguelHiWonderRos2Adapter(
            publisher=publisher,
            readiness_probe=FakeInactiveReadinessProbe(report),
            allow_inactive_external_cmd_vel_publishers=True,
        )
        arm_result = adapter.arm()
        move_result = adapter.set_velocity("move_forward", "slow", 1.0)
        assert arm_result["ok"] is False
        assert arm_result["reason"] == "inactive_publisher_readiness_failed"
        assert move_result["ok"] is False
        assert move_result["reason"] == "adapter_disarmed"
        assert publisher.payloads == []
        return {"arm": arm_result, "move": move_result}
    finally:
        if adapter is not None:
            _destroy_adapter_node_without_publish(adapter)
        _restore_modules(saved)


def test_stage8_graph_probe_module_import_is_inert() -> dict:
    saved = _save_and_remove_modules(
        "miguel_core.miguel_hiwonder_ros2_graph_probe",
        "rclpy",
        "std_msgs",
        "std_msgs.msg",
        "sensor_msgs",
        "sensor_msgs.msg",
        "nav_msgs",
        "nav_msgs.msg",
    )
    try:
        importlib.import_module("miguel_core.miguel_hiwonder_ros2_graph_probe")
        assert "rclpy" not in sys.modules
        assert "std_msgs" not in sys.modules
        assert "sensor_msgs" not in sys.modules
        assert "nav_msgs" not in sys.modules
        return {"rclpy_imported": False, "message_packages_imported": False}
    finally:
        _restore_modules(saved)


def test_stage8_graph_probe_constructor_is_inert() -> dict:
    saved = _save_and_remove_modules(
        "rclpy",
        "std_msgs",
        "std_msgs.msg",
        "sensor_msgs",
        "sensor_msgs.msg",
        "nav_msgs",
        "nav_msgs.msg",
    )
    try:
        probe = MiguelHiWonderRos2GraphProbe()
        assert probe.node is None
        assert probe.rclpy_module is None
        assert "rclpy" not in sys.modules
        assert "std_msgs" not in sys.modules
        assert "sensor_msgs" not in sys.modules
        assert "nav_msgs" not in sys.modules
        return {"node": probe.node, "rclpy_module": probe.rclpy_module}
    finally:
        _restore_modules(saved)


def test_stage8_fake_node_list_and_topics_work() -> dict:
    node = FakeRos2GraphNode(
        nodes=["/controller", "/odom_publisher"],
        topics=[
            ("/controller/cmd_vel", ["geometry_msgs/msg/Twist"]),
            ("/scan_raw", ["sensor_msgs/msg/LaserScan"]),
        ],
    )
    probe = MiguelHiWonderRos2GraphProbe(node=node)
    nodes = probe.list_nodes()
    topics = probe.list_topics()
    assert nodes["ok"] is True
    assert "/controller" in nodes["nodes"]
    assert topics["ok"] is True
    assert topics["topics"]["/controller/cmd_vel"] == ["geometry_msgs/msg/Twist"]
    return {"nodes": nodes, "topics": topics}


def test_stage8_cmd_vel_without_competing_publishers_is_safe() -> dict:
    node = FakeRos2GraphNode(
        topics=[("/controller/cmd_vel", ["geometry_msgs/msg/Twist"])],
        publisher_nodes_by_topic={"/controller/cmd_vel": ["miguel_hiwonder_ros2_adapter"]},
        subscriber_nodes_by_topic={"/controller/cmd_vel": ["odom_publisher"]},
    )
    result = MiguelHiWonderRos2GraphProbe(node=node).inspect_cmd_vel()
    assert result["topic_exists"] is True
    assert result["message_type"] == "geometry_msgs/msg/Twist"
    assert result["publisher_count"] == 1
    assert result["subscription_count"] == 1
    assert result["competing_publishers"] == []
    assert result["safe_to_arm"] is True
    assert result["movement_tested"] is False
    assert result["hardware_verified"] is False
    return result


def test_stage8_cmd_vel_with_competing_publishers_is_unsafe() -> dict:
    node = FakeRos2GraphNode(
        topics=[("/controller/cmd_vel", ["geometry_msgs/msg/Twist"])],
        publisher_nodes_by_topic={
            "/controller/cmd_vel": [
                "rosbridge_websocket",
                "lidar_app",
                "miguel_hiwonder_ros2_adapter",
            ]
        },
        subscriber_nodes_by_topic={"/controller/cmd_vel": ["odom_publisher"]},
    )
    result = MiguelHiWonderRos2GraphProbe(node=node).inspect_cmd_vel()
    assert result["safe_to_arm"] is False
    assert result["competing_publishers"] == ["rosbridge_websocket", "lidar_app"]
    assert result["publisher_count"] == 3
    return result


def test_stage8_battery_raw_maps_to_voltage() -> dict:
    node = FakeRos2GraphNode(
        messages_by_topic={
            "/ros_robot_controller/battery": [FakeUInt16(7424)],
        }
    )
    saved, _rclpy = _install_fake_graph_ros2_modules(node)
    try:
        result = MiguelHiWonderRos2GraphProbe().read_battery_once(timeout_sec=0.1)
    finally:
        _restore_modules(saved)
    assert result["ok"] is True
    assert result["readable"] is True
    assert result["raw"] == 7424
    assert result["estimated_voltage_v"] == 7.424
    return result


def test_stage8_lidar_summary_handles_nan_inf() -> dict:
    ranges = [
        1.0,
        float("nan"),
        float("inf"),
        0.8,
        0.5,
        0.3,
        2.0,
    ]
    scan = FakeLaserScan(
        ranges=ranges,
        angle_min=-math.pi / 2,
        angle_increment=math.pi / 6,
        range_min=0.05,
        range_max=5.0,
        frame_id="laser_frame",
    )
    node = FakeRos2GraphNode(messages_by_topic={"/scan_raw": [scan]})
    saved, _rclpy = _install_fake_graph_ros2_modules(node)
    try:
        result = MiguelHiWonderRos2GraphProbe().read_lidar_once(timeout_sec=0.1)
    finally:
        _restore_modules(saved)
    assert result["ok"] is True
    assert result["valid_range_count"] == 5
    assert result["nearest_obstacle_cm"] == 30.0
    assert result["front_clearance_cm"] == 80.0
    assert result["left_clearance_cm"] == 200.0
    assert result["right_clearance_cm"] == 100.0
    assert result["frame_id"] == "laser_frame"
    return result


def test_stage8_odom_summary_handles_minimal_fake_odometry() -> dict:
    node = FakeRos2GraphNode(messages_by_topic={"/odom": [FakeOdometry()]})
    saved, _rclpy = _install_fake_graph_ros2_modules(node)
    try:
        result = MiguelHiWonderRos2GraphProbe().read_odom_once(timeout_sec=0.1)
    finally:
        _restore_modules(saved)
    assert result["ok"] is True
    assert result["x"] == 1.25
    assert result["y"] == -0.5
    assert result["yaw_approx"] == 0.0
    assert result["linear_x"] == 0.12
    assert result["linear_y"] == 0.01
    assert result["angular_z"] == -0.2
    assert result["frame_id"] == "odom"
    assert result["child_frame_id"] == "base_link"
    return result


def test_stage8_sensor_failure_returns_structured_error() -> dict:
    node = FakeRos2GraphNode()
    saved, _rclpy = _install_fake_graph_ros2_modules(node)
    try:
        result = MiguelHiWonderRos2GraphProbe().read_battery_once(timeout_sec=0.0)
    finally:
        _restore_modules(saved)
    assert result["ok"] is False
    assert result["readable"] is False
    assert result["sensor"] == "battery"
    assert result["reason"] == "timeout"
    return result


def test_stage8_readiness_report_combines_graph_and_sensors() -> dict:
    node = FakeRos2GraphNode(
        nodes=["/controller", "/odom_publisher"],
        topics=[
            ("/controller/cmd_vel", ["geometry_msgs/msg/Twist"]),
            ("/ros_robot_controller/battery", ["std_msgs/msg/UInt16"]),
            ("/scan_raw", ["sensor_msgs/msg/LaserScan"]),
            ("/odom", ["nav_msgs/msg/Odometry"]),
        ],
        publisher_nodes_by_topic={"/controller/cmd_vel": ["lidar_app"]},
        subscriber_nodes_by_topic={"/controller/cmd_vel": ["odom_publisher"]},
        messages_by_topic={
            "/ros_robot_controller/battery": [FakeUInt16(7424)],
            "/scan_raw": [
                FakeLaserScan([1.0, 0.75, 1.5], angle_min=-0.1, angle_increment=0.1)
            ],
            "/odom": [FakeOdometry()],
        },
    )
    saved, _rclpy = _install_fake_graph_ros2_modules(node)
    try:
        result = MiguelHiWonderRos2GraphProbe().build_readiness_report()
    finally:
        _restore_modules(saved)
    assert result["ros2_available"] is True
    assert result["car_graph_visible"] is True
    assert result["cmd_vel_safe_to_arm"] is False
    assert result["competing_cmd_vel_publishers"] == ["lidar_app"]
    assert result["battery_ok"] is True
    assert result["battery_readable"] is True
    assert result["lidar_ok"] is True
    assert result["lidar_readable"] is True
    assert result["odom_ok"] is True
    assert result["odom_readable"] is True
    assert result["movement_tested"] is False
    assert result["hardware_verified"] is False
    return result


def main() -> None:
    result = test_explore_room_turn_right_allowed()
    test_unsafe_move_forward_blocked()
    test_fast_speed_downgraded()
    test_long_duration_capped()
    test_emergency_stop_telemetry_stops_mission()
    test_mission_controller_records_steps()
    test_default_bridge_uses_dry_run_adapter()
    test_probe_runs_without_crashing()
    test_stage1_adapter_initializes_disarmed_without_commands()
    test_stage1_runtime_start_emits_no_hardware_commands()
    test_stage1_movement_while_disarmed_blocked()
    test_stage1_arm_then_move_forward_records_twist_and_stop()
    test_stage1_stale_telemetry_blocks_movement()
    test_stage1_emergency_stop_records_stop()
    test_stage1_adapter_duration_cap()
    test_stage1_adapter_speed_cap()
    test_stage1_mission_disarmed_records_blocked_not_accepted()
    test_stage1_move_backward_twist_sign()
    test_stage1_turn_left_twist_sign()
    test_stage1_turn_right_twist_sign()
    test_stage1_negative_duration_clamps_to_zero()
    test_stage1_nonnumeric_duration_clamps_to_zero()
    test_stage1_close_disarms_and_records_stop()
    test_stage1_low_battery_blocks_movement()
    test_stage1_nearest_obstacle_blocks_movement()
    test_stage2_bridge_still_defaults_to_dry_run()
    test_stage2_fake_ros2_adapter_starts_disarmed()
    test_stage2_fake_ros2_adapter_emits_no_payload_on_construction()
    test_stage2_fake_ros2_movement_while_disarmed_blocks_without_publish()
    test_stage2_fake_ros2_move_forward_publishes_twist_and_stop()
    test_stage2_fake_ros2_turn_left_and_right_signs()
    test_stage2_fake_ros2_stop_while_disarmed_publishes_zero_twist()
    test_stage2_fake_ros2_close_publishes_stop_and_disarms()
    test_stage2_fake_ros2_duration_cap()
    test_stage2_fake_ros2_speed_cap()
    test_stage21_fake_ros2_publish_twist_method_supported()
    test_stage21_bridge_accepts_injected_fake_ros2_adapter()
    test_stage21_fake_ros2_publisher_failure_is_reported()
    test_stage21_fake_ros2_move_backward_sign()
    test_stage21_fake_ros2_negative_duration_clamps_to_zero()
    test_stage21_fake_ros2_nonnumeric_duration_clamps_to_zero()
    test_stage21_fake_ros2_disarm_publishes_stop_and_disarms()
    test_stage21_fake_ros2_disarm_while_disarmed_is_safe_stop()
    test_stage21_fake_ros2_unknown_command_blocks_without_publish()
    test_stage3_bridge_still_defaults_to_dry_run()
    test_stage3_ros2_module_import_does_not_import_ros_packages()
    test_stage3_ros2_without_publisher_is_unavailable()
    test_stage3_ros2_adapter_starts_disarmed()
    test_stage3_ros2_disarmed_movement_blocks_without_publish()
    test_stage3_ros2_stop_while_disarmed_publishes_zero_twist()
    test_stage3_ros2_arm_move_forward_publishes_twist_and_stop()
    test_stage3_ros2_move_backward_sign()
    test_stage3_ros2_turn_left_sign()
    test_stage3_ros2_turn_right_sign()
    test_stage3_ros2_disarm_publishes_stop_and_disarms()
    test_stage3_ros2_close_publishes_stop_and_disarms()
    test_stage3_ros2_publisher_failure_returns_error()
    test_stage3_ros2_negative_duration_clamps_to_zero()
    test_stage3_ros2_nonnumeric_duration_clamps_to_zero()
    test_stage3_bridge_accepts_injected_ros2_adapter()
    test_stage31_ros2_injected_publisher_is_not_real_hardware()
    test_stage31_ros2_real_backend_requires_explicit_opt_in()
    test_stage31_ros2_twist_factory_failure_on_movement_is_structured()
    test_stage31_ros2_twist_factory_failure_on_stop_is_structured()
    test_stage31_ros2_twist_factory_failure_on_disarm_is_structured()
    test_stage31_ros2_twist_factory_failure_on_close_is_structured()
    test_stage31_ros2_unknown_command_blocks_without_publish()
    test_stage31_ros2_missing_publish_method_is_structured()
    test_stage31_ros2_followup_stop_failure_preserves_movement_result()
    test_stage31_ros2_no_publisher_disarm_is_unavailable()
    test_stage31_ros2_no_publisher_close_is_unavailable()
    test_stage4_allow_real_ros2_missing_dependencies_is_structured()
    test_stage4_fake_real_ros2_backend_constructs_without_publish()
    test_stage4_fake_real_ros2_stop_publishes_zero_twist()
    test_stage4_fake_real_ros2_arm_move_forward_publishes_twist_and_stop()
    test_stage4_fake_real_ros2_close_destroys_owned_node()
    test_stage5_ros2_probe_module_import_does_not_import_ros_packages()
    test_stage5_ros2_probe_constructor_is_inert()
    test_stage5_ros2_probe_check_imports_handles_missing_modules()
    test_stage5_ros2_probe_check_environment_reads_expected_vars()
    test_stage5_ros2_probe_create_node_with_fake_rclpy()
    test_stage5_ros2_probe_destroy_owned_fake_node()
    test_stage5_ros2_probe_readiness_never_tests_movement()
    test_stage5_ros2_probe_node_creation_failure_is_structured()
    test_stage5_ros2_probe_destroy_failure_is_structured()
    test_stage9_ros2_arm_refuses_competing_publishers()
    test_stage9_ros2_arm_refuses_missing_cmd_vel()
    test_stage9_ros2_arm_refuses_no_cmd_vel_subscriber()
    test_stage9_ros2_arm_refuses_wrong_cmd_vel_type()
    test_stage9_ros2_arm_succeeds_with_clean_readiness()
    test_stage9_ros2_arm_override_allows_competing_publishers_with_warning()
    test_stage9_ros2_movement_blocked_after_failed_arm()
    test_stage9_ros2_arm_refuses_unreadable_sensor()
    test_stage91_cmd_vel_ignores_explicit_miguel_publishers()
    test_stage91_unknown_publisher_remains_competing()
    test_stage91_odom_subscriber_endpoint_counts_as_subscription()
    test_stage91_cmd_vel_no_subscribers_is_not_arm_ready()
    test_stage91_arm_succeeds_with_only_miguel_publisher_and_odom_subscriber()
    test_stage91_arm_fails_with_external_publisher_and_subscriber()
    test_stage91_arm_override_allows_external_publisher_with_subscriber()
    test_stage10_odom_fallback_uses_odom_raw()
    test_stage10_set_motor_inspection_detects_required_subscriber()
    test_stage10_set_motor_inspection_detects_external_direct_publisher()
    test_stage10_low_level_motor_chain_ok_when_required_endpoints_exist()
    test_stage10_low_level_motor_chain_fails_when_endpoint_missing()
    test_stage10_ros2_arm_refuses_external_direct_motor_publisher()
    test_stage10_ros2_arm_refuses_incomplete_low_level_motor_chain()
    test_stage10_ros2_arm_succeeds_with_complete_clean_motor_chain()
    test_stage10_movement_blocked_after_direct_motor_arm_failure()
    test_stage11_cmd_vel_quiet_returns_true_on_timeout()
    test_stage11_cmd_vel_quiet_returns_false_when_message_observed()
    test_stage11_odom_stationary_true_for_zero_twist_samples()
    test_stage11_odom_stationary_false_for_nonzero_second_sample()
    test_stage11_inactive_readiness_reports_hand_gesture_direct_motor_blocking()
    test_stage11_adapter_blocks_external_cmd_vel_by_default()
    test_stage11_adapter_inactive_external_cmd_vel_opt_in_can_pass_without_direct_publishers()
    test_stage11_adapter_blocks_hand_gesture_direct_motor_without_direct_override()
    test_stage11_adapter_allows_hand_gesture_only_with_direct_override()
    test_stage11_movement_blocked_after_inactive_arm_failure()
    test_stage8_graph_probe_module_import_is_inert()
    test_stage8_graph_probe_constructor_is_inert()
    test_stage8_fake_node_list_and_topics_work()
    test_stage8_cmd_vel_without_competing_publishers_is_safe()
    test_stage8_cmd_vel_with_competing_publishers_is_unsafe()
    test_stage8_battery_raw_maps_to_voltage()
    test_stage8_lidar_summary_handles_nan_inf()
    test_stage8_odom_summary_handles_minimal_fake_odometry()
    test_stage8_sensor_failure_returns_structured_error()
    test_stage8_readiness_report_combines_graph_and_sensors()
    _assert_no_real_ros_modules_loaded()

    print("\nTelemetry:")
    pprint(result["telemetry"])
    print("\nDecision:")
    pprint(result["decision"])
    print("\nCommand result:")
    pprint(result["command_result"])
    print("\nMission status:")
    pprint(result["mission_status"])
    print("\nMiguel Core Lab v0.3 smoke test passed.")


if __name__ == "__main__":
    main()
