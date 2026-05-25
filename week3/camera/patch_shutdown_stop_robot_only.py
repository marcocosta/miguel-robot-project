from pathlib import Path

memory_path = Path.home() / "robot-project/week3/camera/robot_memory.py"
brain_path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"

memory = memory_path.read_text()
brain = brain_path.read_text()

# -------------------------------------------------------------------
# 1) robot_memory.py:
# Replace shutdown confirmation block so it returns __STOP_ROBOT__,
# not __SHUTDOWN__.
# -------------------------------------------------------------------

memory = memory.replace(
    'memory["robot_mode"] = "shutdown"\n        save_memory(memory)\n        return "__SHUTDOWN__"',
    'memory["robot_mode"] = "normal"\n        save_memory(memory)\n        return "__STOP_ROBOT__"'
)

# If there are repeated variants, replace all.
memory = memory.replace('return "__SHUTDOWN__"', 'return "__STOP_ROBOT__"')

# Make the prompt wording clear.
memory = memory.replace(
    "Shutdown confirmation required. Say confirm shutdown if you want me to power off the Jetson.",
    "Shutdown confirmation required. Say confirm shutdown if you want me to stop the robot program. The Jetson will stay on."
)

memory_path.write_text(memory.replace("\t", "    "))

# -------------------------------------------------------------------
# 2) robot_cloud_brain_v6_threaded.py:
# Replace real Jetson shutdown handler with stop-program-only handler.
# -------------------------------------------------------------------

old_handler = '''    if mode_reply == "__SHUTDOWN__":
        speak("Confirmed. Miguel is shutting down the Jetson now. Mission saved.")
        subprocess.Popen(["sudo", "shutdown", "now"])
        return False

'''

new_handler = '''    if mode_reply in {"__SHUTDOWN__", "__STOP_ROBOT__"}:
        speak("Confirmed. I am stopping the Miguel robot program now. The Jetson will stay on. Mission saved.")
        return False

'''

if old_handler in brain:
    brain = brain.replace(old_handler, new_handler)
else:
    # Fallback: handle any direct sudo shutdown path.
    brain = brain.replace(
        'speak("Confirmed. Miguel is shutting down the Jetson now. Mission saved.")\n        subprocess.Popen(["sudo", "shutdown", "now"])\n        return False',
        'speak("Confirmed. I am stopping the Miguel robot program now. The Jetson will stay on. Mission saved.")\n        return False'
    )

# Remove any remaining direct sudo shutdown call as safety.
brain = brain.replace('subprocess.Popen(["sudo", "shutdown", "now"])', 'print("[SHUTDOWN] Jetson shutdown disabled; stopping robot only.")')

brain_path.write_text(brain)

print("Patched shutdown mode: stop robot program only; Jetson stays on.")
