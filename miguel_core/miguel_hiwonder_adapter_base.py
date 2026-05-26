"""Adapter base for Miguel HiWonder integration experiments."""

from __future__ import annotations


class MiguelHiWonderAdapterBase:
    """Minimal adapter interface for simulated or future real HiWonder backends."""

    def get_name(self) -> str:
        return self.__class__.__name__

    def is_real_hardware(self) -> bool:
        return False

    def arm(self) -> dict:
        raise NotImplementedError

    def disarm(self) -> dict:
        raise NotImplementedError

    def stop(self, reason: str | None = None) -> dict:
        raise NotImplementedError

    def drive_twist(
        self,
        linear_x: float,
        linear_y: float,
        angular_z: float,
        duration_sec: float,
    ) -> dict:
        raise NotImplementedError

    def set_velocity(self, command: str, speed: str = "slow", duration_sec: float = 1.0) -> dict:
        raise NotImplementedError

    def beep(self, freq: int, duration_sec: float) -> dict:
        raise NotImplementedError

    def set_led(
        self,
        led_id: int,
        on_time: float,
        off_time: float,
        repeat: int = 1,
    ) -> dict:
        raise NotImplementedError

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
