"""Generic simulated command, event, and telemetry bus for Miguel Core Lab."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MiguelRobotBus:
    """Small file-backed bus for simulated robot commands and events."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.event_log_path = self.data_dir / "miguel_events.jsonl"
        self.event_log_path.touch(exist_ok=True)
        self._latest_telemetry: dict[str, dict[str, Any]] = {}

    def send_command(
        self,
        target: str,
        command: str,
        params: dict | None = None,
        safety: dict | None = None,
    ) -> dict:
        safety_payload = safety or {}
        event = self._build_event(
            "command",
            {
                "target": target,
                "command": command,
                "params": params or {},
                "safety": safety_payload,
                "safety_allowed": safety_payload.get("ok"),
                "safety_blocked": safety_payload.get("blocked"),
                "simulated": True,
                "status": "accepted",
            },
        )
        self._append_event(event)
        print(f"[MIGUEL_ROBOT_BUS] command target={target} command={command}")
        return event

    def record_telemetry(self, target: str, telemetry: dict) -> dict:
        telemetry_record = dict(telemetry)
        telemetry_record.setdefault("target", target)
        telemetry_record.setdefault("simulated", True)
        telemetry_record.setdefault("timestamp", self._utc_now())
        self._latest_telemetry[target] = telemetry_record

        event = self._build_event(
            "telemetry",
            {
                "target": target,
                "telemetry": telemetry_record,
            },
        )
        self._append_event(event)
        print(f"[MIGUEL_ROBOT_BUS] telemetry target={target}")
        return event

    def get_latest_telemetry(self, target: str) -> dict | None:
        telemetry = self._latest_telemetry.get(target)
        if telemetry is None:
            return None
        return dict(telemetry)

    def get_event_log(self, limit: int = 20) -> list[dict]:
        if limit <= 0 or not self.event_log_path.exists():
            return []

        lines = self.event_log_path.read_text(encoding="utf-8").splitlines()
        events: list[dict] = []
        for line in lines[-limit:]:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    def _build_event(self, event_type: str, payload: dict) -> dict:
        return {
            "timestamp": self._utc_now(),
            "event_type": event_type,
            "payload": payload,
        }

    def _append_event(self, event: dict) -> None:
        with self.event_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()
