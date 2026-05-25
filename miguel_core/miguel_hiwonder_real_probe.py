"""Read-only HiWonder environment discovery for Miguel Core Lab."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path


class MiguelHiWonderRealProbe:
    """Inspect likely local HiWonder interfaces without starting hardware."""

    PATH_HINTS = [
        "~/hiwonder",
        "~/Hiwonder",
        "~/ros2_ws",
        "~/catkin_ws",
        "~/robot-project",
        "/home/marquinho",
        "/opt/ros",
        "/usr/local/lib/python3.10/dist-packages",
        "/usr/lib/python3/dist-packages",
    ]
    MODULE_HINTS = [
        "rospy",
        "rclpy",
        "roslaunch",
        "smbus",
        "serial",
        "cv2",
        "depthai",
        "hiwonder",
        "HiwonderSDK",
        "ArmIK",
        "mecanum",
        "lidar",
        "ydlidar",
    ]
    PROCESS_KEYWORDS = ["ros", "lidar", "ydlidar", "hiwonder", "camera", "depth", "python"]

    def probe(self) -> dict:
        paths = self.probe_paths()
        python_modules = self.probe_python_imports()
        processes = self.probe_processes()
        network_ports = self.probe_network_ports()
        likely_interfaces = self._infer_interfaces(paths, python_modules)
        notes = [
            "Read-only probe only; no hardware commands were sent.",
            "Python modules were checked with importlib.util.find_spec only.",
        ]
        result = {
            "ok": True,
            "paths": paths,
            "python_modules": python_modules,
            "processes": processes,
            "network_ports": network_ports,
            "likely_interfaces": likely_interfaces,
            "notes": notes,
        }
        print("[MIGUEL_HIWONDER_PROBE] probe complete")
        return result

    def probe_paths(self) -> list[dict]:
        results = []
        for raw_path in self.PATH_HINTS:
            path = Path(raw_path).expanduser()
            entry = {
                "path": str(path),
                "exists": path.exists(),
                "is_dir": path.is_dir(),
            }
            if path.exists() and path.is_dir():
                try:
                    entry["children_sample"] = sorted(child.name for child in path.iterdir())[:20]
                except OSError as exc:
                    entry["error"] = str(exc)
            results.append(entry)
        print("[MIGUEL_HIWONDER_PROBE] paths checked")
        return results

    def probe_processes(self) -> list[dict]:
        try:
            completed = subprocess.run(
                ["ps", "aux"],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return [{"ok": False, "error": str(exc)}]

        results = []
        for line in completed.stdout.splitlines()[1:]:
            lowered = line.lower()
            matches = [keyword for keyword in self.PROCESS_KEYWORDS if keyword in lowered]
            if matches:
                results.append({"line": line[:300], "matches": matches})
        print("[MIGUEL_HIWONDER_PROBE] processes checked")
        return results[:50]

    def probe_python_imports(self) -> list[dict]:
        results = []
        for module_name in self.MODULE_HINTS:
            try:
                spec = importlib.util.find_spec(module_name)
                origin = getattr(spec, "origin", None) if spec else None
                results.append({"module": module_name, "found": spec is not None, "origin": origin})
            except (ImportError, ValueError, AttributeError) as exc:
                results.append({"module": module_name, "found": False, "error": str(exc)})
        print("[MIGUEL_HIWONDER_PROBE] python modules checked")
        return results

    def probe_network_ports(self) -> list[dict]:
        command = None
        if shutil.which("ss"):
            command = ["ss", "-ltnup"]
        elif shutil.which("netstat"):
            command = ["netstat", "-ltnup"]
        if command is None:
            return [{"ok": False, "error": "ss/netstat not available"}]

        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return [{"ok": False, "command": command[0], "error": str(exc)}]

        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        print("[MIGUEL_HIWONDER_PROBE] network ports checked")
        return [{"command": " ".join(command), "line": line[:300]} for line in lines[:80]]

    def _infer_interfaces(self, paths: list[dict], python_modules: list[dict]) -> list[str]:
        found_modules = {entry["module"] for entry in python_modules if entry.get("found")}
        existing_paths = {entry["path"] for entry in paths if entry.get("exists")}
        interfaces: list[str] = []

        if "rospy" in found_modules or Path("/opt/ros/noetic").exists():
            interfaces.append("ros1")
        if "rclpy" in found_modules or any(path.endswith("/ros2_ws") for path in existing_paths):
            interfaces.append("ros2")
        if (
            {"hiwonder", "HiwonderSDK", "ArmIK", "mecanum"} & found_modules
            or any("hiwonder" in path.lower() for path in existing_paths)
        ):
            interfaces.append("python_sdk")
        if "serial" in found_modules:
            interfaces.append("serial")
        if "depthai" in found_modules:
            interfaces.append("depthai")
        if not interfaces:
            interfaces.append("unknown")
        return interfaces
