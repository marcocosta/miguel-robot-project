"""Optional persistent USB monitor for Seeed's XVF3800 control interface.

Protocol values are a deliberately small subset of Seeed's upstream
``python_control/xvf_host.py``. Importing this module never requires PyUSB.
"""

from dataclasses import asdict, dataclass
import math
import struct
import threading
import time
from typing import Any, Optional


DEFAULT_VID = 0x2886
DEFAULT_PID = 0x001A
CONTROL_SUCCESS = 0
SERVICER_COMMAND_RETRY = 64
COMMANDS = {
    "VERSION": (48, 0, 3, "uint8"),
    "DOA_VALUE": (20, 18, 2, "uint16"),
    "AEC_AZIMUTH_VALUES": (33, 75, 4, "float"),
    "AEC_SPENERGY_VALUES": (33, 80, 4, "float"),
}


@dataclass
class XVFHealth:
    available: bool = False
    vid: Optional[int] = None
    pid: Optional[int] = None
    firmware: Optional[str] = None
    vad_supported: bool = False
    doa_supported: bool = False
    last_read_monotonic: Optional[float] = None
    consecutive_errors: int = 0
    reason: str = "not_probed"
    supported_commands: tuple[str, ...] = ()


class XVF3800Monitor:
    """Thread-safe, persistent PyUSB control connection; no polling thread."""

    def __init__(self, vid: int = DEFAULT_VID, pid: int = DEFAULT_PID, device: Any = None):
        self.vid = vid
        self.pid = pid
        self._device = device
        self._usb_util = None
        self._lock = threading.Lock()
        self._health = XVFHealth(vid=vid, pid=pid)
        self._probe()

    def _probe(self) -> None:
        try:
            if self._device is None:
                import usb.core
                import usb.util
                self._usb_util = usb.util
                find_kwargs = {}
                try:
                    import libusb_package
                    find_kwargs["backend"] = libusb_package.get_libusb1_backend()
                except ImportError:
                    # System libusb remains a supported deployment path.
                    pass
                self._device = usb.core.find(
                    idVendor=self.vid,
                    idProduct=self.pid,
                    **find_kwargs,
                )
                if self._device is None:
                    # Some firmware/device variants can carry another PID.
                    self._device = next(
                        (candidate for candidate in usb.core.find(find_all=True, **find_kwargs)
                         if int(getattr(candidate, "idVendor", 0)) == self.vid),
                        None,
                    )
            if self._device is None:
                self._health.reason = "device_not_found"
                return
            self._health.vid = int(getattr(self._device, "idVendor", self.vid))
            self._health.pid = int(getattr(self._device, "idProduct", self.pid))
            supported = []
            try:
                version = self._read("VERSION")
                self._health.firmware = ".".join(str(value) for value in version)
                supported.append("VERSION")
            except Exception as exc:
                self._health.firmware = None
                self._health.reason = f"version_read_failed: {type(exc).__name__}: {exc}"
            # Probe only documented read-only commands. Current Seeed firmware
            # exposes combined DoA/VAD through DOA_VALUE; older host-control
            # firmware may instead expose azimuth and speech-energy telemetry.
            for command in ("DOA_VALUE", "AEC_AZIMUTH_VALUES", "AEC_SPENERGY_VALUES"):
                try:
                    self._read(command)
                    supported.append(command)
                except Exception:
                    continue
            self._health.supported_commands = tuple(supported)
            self._health.doa_supported = bool(
                {"DOA_VALUE", "AEC_AZIMUTH_VALUES"}.intersection(supported)
            )
            self._health.vad_supported = bool(
                {"DOA_VALUE", "AEC_SPENERGY_VALUES"}.intersection(supported)
            )
            self._health.available = bool(supported)
            if self._health.reason == "not_probed":
                self._health.reason = "ok" if len(supported) > 1 else "control_commands_unavailable"
        except Exception as exc:
            self._health.available = False
            self._health.reason = f"{type(exc).__name__}: {exc}"

    def _read(self, name: str) -> tuple:
        resid, cmdid, count, kind = COMMANDS[name]
        sizes = {"uint8": 1, "uint16": 2, "float": 4}
        length = 1 + count * sizes[kind]
        response = None
        for attempt in range(10):
            response = self._device.ctrl_transfer(0xC0, 0, 0x80 | cmdid, resid, length, timeout=250)
            status = int(response[0])
            if status == CONTROL_SUCCESS:
                break
            if status != SERVICER_COMMAND_RETRY:
                raise OSError(f"XVF status={status} command={name}")
            if attempt < 9:
                time.sleep(0.01)
        else:
            raise TimeoutError(f"XVF command retry exhausted: {name}")
        payload = bytes(response)[1:]
        formats = {"uint8": "B", "uint16": "H", "float": "f"}
        return struct.unpack("<" + formats[kind] * count, payload)

    def available(self) -> bool:
        return self._health.available

    def firmware_version(self) -> Optional[str]:
        return self._health.firmware

    def _combined_doa_vad(self) -> tuple[int, bool]:
        with self._lock:
            try:
                values = self._read("DOA_VALUE")
                self._health.last_read_monotonic = time.monotonic()
                self._health.consecutive_errors = 0
                return int(values[0]), bool(values[1])
            except Exception:
                self._health.consecutive_errors += 1
                raise

    def read_doa(self) -> Optional[int]:
        if not self.available() or not self._health.doa_supported:
            return None
        if "DOA_VALUE" in self._health.supported_commands:
            return self._combined_doa_vad()[0]
        with self._lock:
            values = self._read("AEC_AZIMUTH_VALUES")
            self._health.last_read_monotonic = time.monotonic()
            self._health.consecutive_errors = 0
        # The fourth value is the auto-selected beam, in radians.
        return round(math.degrees(float(values[-1]))) % 360

    def read_vad(self) -> Optional[bool]:
        if not self.available() or not self._health.vad_supported:
            return None
        if "DOA_VALUE" in self._health.supported_commands:
            return self._combined_doa_vad()[1]
        with self._lock:
            values = self._read("AEC_SPENERGY_VALUES")
            self._health.last_read_monotonic = time.monotonic()
            self._health.consecutive_errors = 0
        # Speech energy is a firmware-provided VAD-like signal. A positive
        # selected-beam energy means speech is present; Stage 2 can calibrate a
        # stricter threshold from live diagnostics.
        return bool(values and float(values[-1]) > 0.0)

    def read_doa_vad(self) -> tuple[Optional[int], Optional[bool]]:
        if not self.available():
            return None, None
        if "DOA_VALUE" in self._health.supported_commands:
            doa, vad = self._combined_doa_vad()
            return doa, vad
        return self.read_doa(), self.read_vad()

    def health(self) -> dict:
        return asdict(self._health)

    def close(self) -> None:
        with self._lock:
            if self._device is not None and self._usb_util is not None:
                self._usb_util.dispose_resources(self._device)
            self._device = None

    def restart(self) -> bool:
        self.close()
        self._health = XVFHealth(vid=self.vid, pid=self.pid)
        self._probe()
        return self.available()

    def startup_log(self) -> str:
        health = self.health()
        if health["available"]:
            return (
                "[XVF3800]\n"
                "available=true\n"
                f"vid=0x{health['vid']:04x}\n"
                f"pid=0x{health['pid']:04x}\n"
                f"firmware={health['firmware']}\n"
                f"vad_supported={str(health['vad_supported']).lower()}\n"
                f"doa_supported={str(health['doa_supported']).lower()}\n"
                f"commands={','.join(health['supported_commands']) or 'none'}\n"
                f"reason={health['reason']}"
            )
        return (
            "[XVF3800]\n"
            "available=false\n"
            f"vid=0x{health['vid']:04x}\n"
            f"pid=0x{health['pid']:04x}\n"
            f"firmware={health['firmware']}\n"
            f"vad_supported={str(health['vad_supported']).lower()}\n"
            f"doa_supported={str(health['doa_supported']).lower()}\n"
            f"reason={health['reason']}\n"
            "fallback=existing_audio"
        )


if __name__ == "__main__":
    monitor = XVF3800Monitor()
    try:
        print(monitor.startup_log())
        if monitor.available():
            doa, vad = monitor.read_doa_vad()
            print(f"doa={doa} vad={vad}")
    finally:
        monitor.close()
