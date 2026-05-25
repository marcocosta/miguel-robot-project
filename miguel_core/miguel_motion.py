"""Future body/head motion stubs for Miguel Core Lab."""

from __future__ import annotations


class MiguelMotion:
    """No-op motion bridge that records intent but never controls hardware."""

    def look_left(self) -> dict:
        return self._stub("look_left")

    def look_right(self) -> dict:
        return self._stub("look_right")

    def look_center(self) -> dict:
        return self._stub("look_center")

    def look_up(self) -> dict:
        return self._stub("look_up")

    def look_down(self) -> dict:
        return self._stub("look_down")

    def nod_yes(self) -> dict:
        return self._stub("nod_yes")

    def nod_no(self) -> dict:
        return self._stub("nod_no")

    def scan_room(self) -> dict:
        return self._stub("scan_room")

    def track_face(self, x: float, y: float) -> dict:
        return self._stub("track_face", {"x": self._clamp(x), "y": self._clamp(y)})

    def emergency_stop(self) -> dict:
        return self._stub("emergency_stop", {"priority": "highest"})

    def _stub(self, action: str, params: dict | None = None) -> dict:
        result = {
            "action": action,
            "params": params or {},
            "status": "stubbed",
            "hardware_controlled": False,
        }
        print(f"[MIGUEL_MOTION] {action}")
        return result

    @staticmethod
    def _clamp(value: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        return max(-1.0, min(1.0, numeric))
