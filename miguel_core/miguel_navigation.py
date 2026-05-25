"""Mission-level navigation decisions for Miguel Core Lab."""

from __future__ import annotations


class MiguelNavigationDecider:
    """Turn telemetry into high-level simulated car actions."""

    def decide_next_action(self, mission: str, telemetry: dict) -> dict:
        base_stop = self._base_safety_stop(telemetry)
        if base_stop is not None:
            print(f"[MIGUEL_NAVIGATION] safety_stop reason={base_stop['reason']}")
            return base_stop

        normalized_mission = (mission or "").strip().lower()
        if normalized_mission == "explore_room":
            return self.decide_explore_room(telemetry)
        if normalized_mission == "follow_person":
            return self.decide_follow_person(telemetry)
        if normalized_mission == "return_home":
            return self.decide_return_home(telemetry)

        print(f"[MIGUEL_NAVIGATION] unknown_mission mission={mission}")
        return {
            "mission": mission,
            "action": "stop",
            "params": {"reason": f"unknown mission: {mission}"},
            "reason": "unknown_mission",
        }

    def decide_explore_room(self, telemetry: dict) -> dict:
        front_clearance = self._number(telemetry.get("front_clearance_cm"), default=999.0)
        left_clearance = self._number(telemetry.get("left_clearance_cm"), default=0.0)
        right_clearance = self._number(telemetry.get("right_clearance_cm"), default=0.0)

        if front_clearance < 45:
            action = "turn_right" if right_clearance >= left_clearance else "turn_left"
            print(f"[MIGUEL_NAVIGATION] explore_room action={action}")
            return {
                "mission": "explore_room",
                "action": action,
                "params": {"speed": "slow", "duration_sec": 1.0},
                "reason": "front_blocked_turn_toward_clearer_side",
            }

        print("[MIGUEL_NAVIGATION] explore_room action=move_forward")
        return {
            "mission": "explore_room",
            "action": "move_forward",
            "params": {"speed": "slow", "duration_sec": 1.0},
            "reason": "front_clear",
        }

    def decide_follow_person(self, telemetry: dict) -> dict:
        if not bool(telemetry.get("person_detected", False)):
            print("[MIGUEL_NAVIGATION] follow_person action=scan_area")
            return {
                "mission": "follow_person",
                "action": "scan_area",
                "params": {"duration_sec": 3.0},
                "reason": "person_not_detected",
            }

        direction = str(telemetry.get("person_direction") or "").lower()
        if direction == "front":
            action = "move_forward"
        elif direction == "left":
            action = "turn_left"
        elif direction == "right":
            action = "turn_right"
        else:
            action = "scan_area"

        print(f"[MIGUEL_NAVIGATION] follow_person action={action}")
        params = {"duration_sec": 1.0}
        if action != "scan_area":
            params["speed"] = "slow"
        return {
            "mission": "follow_person",
            "action": action,
            "params": params,
            "reason": f"person_direction_{direction or 'unknown'}",
        }

    def decide_return_home(self, telemetry: dict) -> dict:
        print("[MIGUEL_NAVIGATION] return_home action=stop")
        return {
            "mission": "return_home",
            "action": "stop",
            "params": {"reason": "return_home placeholder"},
            "reason": "return_home_placeholder",
        }

    def _base_safety_stop(self, telemetry: dict) -> dict | None:
        if bool(telemetry.get("emergency_stop", False)):
            return self._stop_decision("emergency_stop")
        battery = self._number(telemetry.get("battery_percent"), default=100.0)
        if battery < 15:
            return self._stop_decision("battery_below_15_percent")
        nearest = self._number(telemetry.get("nearest_obstacle_cm"), default=999.0)
        if nearest < 30:
            return self._stop_decision("nearest_obstacle_under_30_cm")
        return None

    @staticmethod
    def _stop_decision(reason: str) -> dict:
        return {
            "mission": "safety",
            "action": "stop",
            "params": {"reason": reason},
            "reason": reason,
        }

    @staticmethod
    def _number(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
