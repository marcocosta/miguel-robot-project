"""Smoke test for the standalone Miguel Core Lab sandbox."""

from __future__ import annotations

import os
import importlib
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
    MiguelHiWonderRos2Probe,
    MiguelRobotBus,
    MiguelRuntime,
)


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


def _install_fake_ros2_modules() -> tuple[dict[str, object], types.ModuleType]:
    saved = {name: sys.modules.get(name) for name in ("rclpy", "geometry_msgs", "geometry_msgs.msg")}
    rclpy = types.ModuleType("rclpy")
    rclpy.init_calls = 0
    rclpy.created_nodes = []

    def ok() -> bool:
        return False

    def init(args: object = None) -> None:
        rclpy.init_calls += 1

    def create_node(name: str) -> FakeRos2Node:
        node = FakeRos2Node()
        node.name = name
        rclpy.created_nodes.append(node)
        return node

    rclpy.ok = ok
    rclpy.init = init
    rclpy.create_node = create_node

    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msgs_msg.Twist = FakeRos2Twist
    geometry_msgs.msg = geometry_msgs_msg

    sys.modules["rclpy"] = rclpy
    sys.modules["geometry_msgs"] = geometry_msgs
    sys.modules["geometry_msgs.msg"] = geometry_msgs_msg
    return saved, rclpy


def _install_failing_ros2_create_node_module() -> tuple[dict[str, object], types.ModuleType]:
    saved, rclpy = _install_fake_ros2_modules()

    def create_node(name: str) -> FakeRos2Node:
        raise RuntimeError("probe node create failed")

    rclpy.create_node = create_node
    return saved, rclpy


def _restore_modules(saved: dict[str, object]) -> None:
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


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
    sys.modules.pop("miguel_core.miguel_hiwonder_ros2_adapter", None)
    sys.modules.pop("rclpy", None)
    sys.modules.pop("geometry_msgs", None)
    importlib.import_module("miguel_core.miguel_hiwonder_ros2_adapter")
    assert "rclpy" not in sys.modules
    assert "geometry_msgs" not in sys.modules
    return {"rclpy_imported": False, "geometry_msgs_imported": False}


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


def test_stage31_ros2_allow_real_without_publisher_is_not_implemented() -> dict:
    adapter = MiguelHiWonderRos2Adapter(allow_real_ros2=True)
    status = adapter.backend_status()
    result = adapter.set_velocity("move_forward", "slow", 1.0)
    assert status["available"] is False
    assert status["real_ros2_enabled"] is False
    assert status["hardware_verified"] is False
    assert result["ok"] is False
    assert result["blocked"] is True
    assert result["reason"] in {"ros2_dependency_unavailable", "ros2_init_error"}
    return result


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
    import miguel_core.miguel_hiwonder_ros2_adapter as ros2_module

    original_import_module = ros2_module.importlib.import_module

    def missing_ros_import(name: str, *args: object, **kwargs: object) -> object:
        if name in {"rclpy", "geometry_msgs.msg"}:
            raise ImportError(f"{name} unavailable for test")
        return original_import_module(name, *args, **kwargs)

    ros2_module.importlib.import_module = missing_ros_import
    try:
        adapter = MiguelHiWonderRos2Adapter(allow_real_ros2=True)
    finally:
        ros2_module.importlib.import_module = original_import_module

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
        adapter = MiguelHiWonderRos2Adapter(allow_real_ros2=True)
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
        adapter = MiguelHiWonderRos2Adapter(allow_real_ros2=True)
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
        adapter = MiguelHiWonderRos2Adapter(allow_real_ros2=True)
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
    sys.modules.pop("miguel_core.miguel_hiwonder_ros2_probe", None)
    sys.modules.pop("rclpy", None)
    sys.modules.pop("geometry_msgs", None)
    importlib.import_module("miguel_core.miguel_hiwonder_ros2_probe")
    assert "rclpy" not in sys.modules
    assert "geometry_msgs" not in sys.modules
    return {"rclpy_imported": False, "geometry_msgs_imported": False}


def test_stage5_ros2_probe_constructor_is_inert() -> dict:
    sys.modules.pop("rclpy", None)
    sys.modules.pop("geometry_msgs", None)
    probe = MiguelHiWonderRos2Probe()
    assert probe.node is None
    assert probe.rclpy_module is None
    assert "rclpy" not in sys.modules
    assert "geometry_msgs" not in sys.modules
    return {"node": probe.node, "rclpy_module": probe.rclpy_module}


def test_stage5_ros2_probe_check_imports_handles_missing_modules() -> dict:
    import miguel_core.miguel_hiwonder_ros2_probe as probe_module

    original_import_module = probe_module.importlib.import_module

    def missing_ros_import(name: str, *args: object, **kwargs: object) -> object:
        if name in {"rclpy", "geometry_msgs.msg"}:
            raise ImportError(f"{name} unavailable for test")
        return original_import_module(name, *args, **kwargs)

    probe_module.importlib.import_module = missing_ros_import
    try:
        result = MiguelHiWonderRos2Probe().check_imports()
    finally:
        probe_module.importlib.import_module = original_import_module

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
    try:
        probe = MiguelHiWonderRos2Probe()
        result = probe.create_probe_node()
        assert result["ok"] is True
        assert result["node_created"] is True
        assert len(rclpy.created_nodes) == 1
        assert rclpy.created_nodes[0].name == "miguel_hiwonder_ros2_probe"
        assert rclpy.created_nodes[0].publishers == []
        return result
    finally:
        _restore_modules(saved)


def test_stage5_ros2_probe_destroy_owned_fake_node() -> dict:
    saved, rclpy = _install_fake_ros2_modules()
    try:
        probe = MiguelHiWonderRos2Probe()
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
    try:
        probe = MiguelHiWonderRos2Probe()
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
    test_stage31_ros2_allow_real_without_publisher_is_not_implemented()
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
