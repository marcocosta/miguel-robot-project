"""Command safety validation for Miguel Core Lab."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class MiguelSafety:
    """Validate simulated mobile robot commands before bridge execution."""

    MOVEMENT_COMMANDS = {"move_forward", "move_backward", "turn_left", "turn_right"}

    def __init__(
        self,
        allowed_commands: list[str] | None = None,
        max_forward_duration_sec: float = 2.0,
        max_backward_duration_sec: float = 1.5,
        max_turn_duration_sec: float = 1.5,
        max_scan_duration_sec: float = 5.0,
        allowed_speeds: list[str] | None = None,
        min_front_clearance_cm: float = 45,
        min_nearest_obstacle_cm: float = 30,
        min_battery_percent: float = 15,
        max_telemetry_age_sec: float = 5.0,
    ) -> None:
        self.allowed_commands = set(
            allowed_commands
            or [
                "stop",
                "move_forward",
                "move_backward",
                "turn_left",
                "turn_right",
                "scan_area",
                "request_telemetry",
            ]
        )
        self.max_forward_duration_sec = float(max_forward_duration_sec)
        self.max_backward_duration_sec = float(max_backward_duration_sec)
        self.max_turn_duration_sec = float(max_turn_duration_sec)
        self.max_scan_duration_sec = float(max_scan_duration_sec)
        self.allowed_speeds = list(allowed_speeds or ["slow"])
        self.min_front_clearance_cm = float(min_front_clearance_cm)
        self.min_nearest_obstacle_cm = float(min_nearest_obstacle_cm)
        self.min_battery_percent = float(min_battery_percent)
        self.max_telemetry_age_sec = float(max_telemetry_age_sec)

    def validate_command(
        self,
        command: str,
        params: dict | None = None,
        safety: dict | None = None,
        telemetry: dict | None = None,
    ) -> dict:
        adjusted_params = dict(params or {})
        warnings: list[str] = []
        normalized_command = str(command or "").strip()

        if normalized_command not in self.allowed_commands:
            return self._result(False, normalized_command, "unknown_command", adjusted_params, warnings)

        if normalized_command in {"stop", "request_telemetry"}:
            return self._result(True, normalized_command, "always_allowed", adjusted_params, warnings)

        if normalized_command == "scan_area":
            self._cap_duration(
                adjusted_params,
                self.max_scan_duration_sec,
                warnings,
            )
            return self._result(True, normalized_command, "scan_allowed", adjusted_params, warnings)

        if normalized_command in self.MOVEMENT_COMMANDS:
            if not self.is_telemetry_fresh(telemetry):
                return self._result(False, normalized_command, "telemetry_missing_or_stale", adjusted_params, warnings)

            if bool((telemetry or {}).get("emergency_stop", False)):
                return self._result(False, normalized_command, "emergency_stop", adjusted_params, warnings)

            battery = self._number((telemetry or {}).get("battery_percent"), default=100.0)
            if battery < self.min_battery_percent:
                return self._result(False, normalized_command, "battery_below_minimum", adjusted_params, warnings)

            nearest = self._number((telemetry or {}).get("nearest_obstacle_cm"), default=999.0)
            if nearest < self.min_nearest_obstacle_cm:
                return self._result(False, normalized_command, "nearest_obstacle_too_close", adjusted_params, warnings)

            if normalized_command == "move_forward":
                front = self._number((telemetry or {}).get("front_clearance_cm"), default=999.0)
                if front < self.min_front_clearance_cm:
                    return self._result(False, normalized_command, "front_clearance_too_low", adjusted_params, warnings)

            speed = str(adjusted_params.get("speed") or "slow")
            if speed not in self.allowed_speeds:
                adjusted_params["speed"] = "slow"
                warnings.append(f"speed_downgraded_from_{speed}_to_slow")
            else:
                adjusted_params["speed"] = speed

            self._cap_duration(adjusted_params, self._max_duration_for(normalized_command), warnings)
            return self._result(True, normalized_command, "movement_allowed", adjusted_params, warnings)

        return self._result(False, normalized_command, "unsupported_command", adjusted_params, warnings)

    def is_telemetry_fresh(self, telemetry: dict | None) -> bool:
        if not telemetry:
            return False

        raw_timestamp = telemetry.get("timestamp", telemetry.get("updated_at"))
        if raw_timestamp is None:
            return bool(telemetry.get("simulated") and telemetry.get("target") == "hiwonder_car")

        timestamp = self._parse_timestamp(raw_timestamp)
        if timestamp is None:
            return False

        age_sec = (datetime.now(timezone.utc) - timestamp).total_seconds()
        return age_sec <= self.max_telemetry_age_sec

    def summarize_limits(self) -> dict:
        return {
            "allowed_commands": sorted(self.allowed_commands),
            "allowed_speeds": list(self.allowed_speeds),
            "max_forward_duration_sec": self.max_forward_duration_sec,
            "max_backward_duration_sec": self.max_backward_duration_sec,
            "max_turn_duration_sec": self.max_turn_duration_sec,
            "max_scan_duration_sec": self.max_scan_duration_sec,
            "min_front_clearance_cm": self.min_front_clearance_cm,
            "min_nearest_obstacle_cm": self.min_nearest_obstacle_cm,
            "min_battery_percent": self.min_battery_percent,
            "max_telemetry_age_sec": self.max_telemetry_age_sec,
        }

    def _cap_duration(self, params: dict, max_duration: float, warnings: list[str]) -> None:
        duration = max(0.0, self._number(params.get("duration_sec"), default=0.0))
        if duration > max_duration:
            params["duration_sec"] = max_duration
            warnings.append(f"duration_capped_to_{max_duration}_sec")
        else:
            params["duration_sec"] = duration

    def _max_duration_for(self, command: str) -> float:
        if command == "move_forward":
            return self.max_forward_duration_sec
        if command == "move_backward":
            return self.max_backward_duration_sec
        return self.max_turn_duration_sec

    def _result(
        self,
        ok: bool,
        command: str,
        reason: str,
        adjusted_params: dict,
        warnings: list[str],
    ) -> dict:
        result = {
            "ok": ok,
            "command": command,
            "reason": reason,
            "adjusted_params": adjusted_params,
            "blocked": not ok,
            "warnings": list(warnings),
        }
        print(f"[MIGUEL_SAFETY] command={command} ok={ok} reason={reason}")
        return result

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime | None:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str):
            try:
                cleaned = value.replace("Z", "+00:00")
                parsed = datetime.fromisoformat(cleaned)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            except ValueError:
                try:
                    return datetime.fromtimestamp(float(value), tz=timezone.utc)
                except ValueError:
                    return None
        return None

    @staticmethod
    def _number(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
