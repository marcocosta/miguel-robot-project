from pathlib import Path
import re

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"
text = path.read_text()

# Longer natural follow-up before falling back to wake mode.
text = re.sub(r"ACTIVE_FOLLOWUP_SECONDS\s*=\s*\d+", "ACTIVE_FOLLOWUP_SECONDS = 90", text)
text = re.sub(r"CONVERSATION_GRACE_SECONDS\s*=\s*\d+", "CONVERSATION_GRACE_SECONDS = 90", text)

# Stop command should be explicit; "stop right now" should not kill Miguel.
old = '''    if words & {"quit", "stop", "exit", "shutdown"}:
        speak("Miguel is shutting down conversation mode. Mission saved.")
        return False
'''
new = '''    explicit_exit_phrases = {
        "quit robot",
        "exit robot",
        "stop robot program",
        "stop miguel program",
        "end program",
    }

    normalized_text = user_text.lower().strip()

    if normalized_text in explicit_exit_phrases:
        speak("Miguel is shutting down conversation mode. Mission saved.")
        return False
'''
if old in text:
    text = text.replace(old, new)
else:
    print("Warning: explicit stop block not found.")

# Reduce vision work/logs while sleep mode is active.
old = '''    while not STOP_EVENT.is_set() and pipeline.isRunning():
        if AUDIO_CAPTURE_ACTIVE.is_set():
            time.sleep(0.2)
            continue
'''
new = '''    while not STOP_EVENT.is_set() and pipeline.isRunning():
        try:
            if get_robot_mode() == "sleep":
                time.sleep(5.0)
                continue
        except Exception:
            pass

        if AUDIO_CAPTURE_ACTIVE.is_set():
            time.sleep(0.2)
            continue
'''
if old in text:
    text = text.replace(old, new)
else:
    print("Warning: vision worker sleep insertion not found.")

# Reduce repeated sleep terminal spam by increasing wake timeout.
text = text.replace(
    "wake_detected = listen_for_wake(timeout_seconds=3)",
    "wake_detected = listen_for_wake(timeout_seconds=6)"
)

# Strong spoken-output instructions.
old = '''- Do not claim you can move unless local_state says motion hardware exists.
'''
new = '''- Do not claim you can move unless local_state says motion hardware exists.
- Output plain spoken text only. Do not use Markdown, bullets, asterisks, code formatting, tables, or emojis.
- Keep spoken replies short: usually 1 to 3 sentences unless the user asks for detail.
'''
if old in text:
    text = text.replace(old, new)
else:
    print("Warning: speech rules insertion not found.")

path.write_text(text)
print("Patched runtime behavior.")
