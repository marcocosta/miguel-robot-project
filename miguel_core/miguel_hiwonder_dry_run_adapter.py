"""Hardware-safe dry-run HiWonder adapter for Miguel Core Lab."""

from __future__ import annotations

from datetime import datetime, timezone

from .miguel_hiwonder_adapter_base import MiguelHiWonderAdapterBase


class MiguelHiWonderDryRunAdapter(MiguelHiWonderAdapterBase):
    """Default adapter that never sends commands to real hardware."""

    TARGET = "hiwonder_car"

    def __init__(self) -> None:
        self._latest_telemetry: dict | None = None

    def get_name(self) -> str:
        return "dry_run"

    def send_command(
        self,
        command: str,
        params: dict | None = None,
        safety: dict | None = None,
    ) -> dict:
        result = {
            "ok": True,
            "dry_run": True,
            "adapter": self.get_name(),
            "command": command,
            "params": params or {},
            "safety": safety or {},
        }
        print(f"[MIGUEL_HIWONDER_DRY_RUN] command={command}")
        return result

    def request_telemetry(self) -> dict:
        if self._latest_telemetry is None:
            self._latest_telemetry = self._simulated_safe_idle_telemetry()
        telemetry = dict(self._latest_telemetry)
        telemetry.setdefault("timestamp", self._utc_now())
        print("[MIGUEL_HIWONDER_DRY_RUN] request_telemetry")
        return telemetry

    def update_telemetry(self, telemetry: dict) -> dict:
        telemetry_record = dict(telemetry or {})
        telemetry_record.setdefault("target", self.TARGET)
        telemetry_record.setdefault("simulated", True)
        telemetry_record.setdefault("timestamp", self._utc_now())
        self._latest_telemetry = telemetry_record
        print("[MIGUEL_HIWONDER_DRY_RUN] update_telemetry")
        return dict(telemetry_record)

    def _simulated_safe_idle_telemetry(self) -> dict:
        return self.update_telemetry(
            {
                "battery_percent": 87,
                "emergency_stop": False,
                "front_clearance_cm": 120,
                "left_clearance_cm": 95,
                "nearest_obstacle_cm": 95,
                "person_detected": False,
                "person_direction": None,
                "right_clearance_cm": 110,
                "state": "idle",
            }
        )

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()
