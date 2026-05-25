"""Adapter base for Miguel HiWonder integration experiments."""

from __future__ import annotations


class MiguelHiWonderAdapterBase:
    """Minimal adapter interface for simulated or future real HiWonder backends."""

    def get_name(self) -> str:
        return self.__class__.__name__

    def is_real_hardware(self) -> bool:
        return False

    def send_command(
        self,
        command: str,
        params: dict | None = None,
        safety: dict | None = None,
    ) -> dict:
        raise NotImplementedError

    def request_telemetry(self) -> dict:
        raise NotImplementedError

    def update_telemetry(self, telemetry: dict) -> dict:
        return dict(telemetry or {})

    def close(self) -> None:
        return None
