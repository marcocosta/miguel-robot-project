"""Smoke test for the standalone Miguel Core Lab sandbox."""

from __future__ import annotations

from pprint import pprint

from miguel_core import MiguelRuntime


def main() -> None:
    runtime = MiguelRuntime()
    runtime.start()

    runtime.personality.set_personality("mission_control")
    runtime.set_face_state("listening", emotion="attentive", look_x=0.2, look_y=0.0)
    runtime.set_face_state("thinking", emotion="focused", message="planning")

    runtime.hiwonder.update_telemetry(
        {
            "battery_percent": 82,
            "emergency_stop": False,
            "front_clearance_cm": 35,
            "left_clearance_cm": 50,
            "nearest_obstacle_cm": 35,
            "person_detected": False,
            "person_direction": None,
            "right_clearance_cm": 90,
            "state": "idle",
            "simulated": True,
        }
    )

    result = runtime.run_hiwonder_mission_step("explore_room")
    print("\nTelemetry:")
    pprint(result["telemetry"])
    print("\nDecision:")
    pprint(result["decision"])
    print("\nCommand result:")
    pprint(result["command_result"])

    interaction = runtime.record_interaction(
        "Remember that I like mission mode.",
        "Got it. I will keep mission mode as a learning candidate.",
        {"source": "test_miguel_core"},
    )
    print("\nLearning interaction:")
    pprint(interaction)

    print("\nMotion stub:")
    pprint(runtime.motion.scan_room())
    pprint(runtime.motion.track_face(1.4, -1.2))

    runtime.shutdown()


if __name__ == "__main__":
    main()
