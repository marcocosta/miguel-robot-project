"""Smoke test for the standalone Miguel Core Lab sandbox."""

from __future__ import annotations

from pprint import pprint
from tempfile import TemporaryDirectory

from miguel_core import MiguelHiWonderDryRunAdapter, MiguelHiWonderRealProbe, MiguelRuntime


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
