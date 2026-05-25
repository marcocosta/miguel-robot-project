from pathlib import Path
import re

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"
text = path.read_text()

# Give child/long-form speech more time.
text = re.sub(r"MAX_TURN_SECONDS\s*=\s*[0-9.]+", "MAX_TURN_SECONDS = 20.0", text)

# Slightly longer silence timeout so Miguel does not cut off pauses.
text = re.sub(r"SILENCE_SECONDS\s*=\s*[0-9.]+", "SILENCE_SECONDS = 1.8", text)

# Keep threshold high enough to avoid random quiet captures.
text = re.sub(r"SPEECH_RMS_THRESHOLD\s*=\s*\d+", "SPEECH_RMS_THRESHOLD = 1200", text)

# Slow vision background cadence a bit to reduce CPU load.
text = re.sub(r"VISION_UPDATE_SECONDS\s*=\s*[0-9.]+", "VISION_UPDATE_SECONDS = 3.0", text)

# Add recording flag after STOP_EVENT if missing.
if "AUDIO_CAPTURE_ACTIVE = threading.Event()" not in text:
    text = text.replace(
        "STOP_EVENT = threading.Event()\n",
        "STOP_EVENT = threading.Event()\nAUDIO_CAPTURE_ACTIVE = threading.Event()\n",
    )

# Pause vision worker while audio capture is active.
old = """    while not STOP_EVENT.is_set() and pipeline.isRunning():
        try:
            state = detect_face_state(camera_queue)
"""
new = """    while not STOP_EVENT.is_set() and pipeline.isRunning():
        if AUDIO_CAPTURE_ACTIVE.is_set():
            time.sleep(0.2)
            continue

        try:
            state = detect_face_state(camera_queue)
"""
text = text.replace(old, new)

# Wrap capture_user_turn with AUDIO_CAPTURE_ACTIVE flag.
old = """def capture_user_turn():
    print("Listening for your turn with OpenAI transcription...")
"""
new = """def capture_user_turn():
    print("Listening for your turn with OpenAI transcription...")
    AUDIO_CAPTURE_ACTIVE.set()
"""
text = text.replace(old, new)

old = """    finally:
        stop_stream(proc)
"""
new = """    finally:
        stop_stream(proc)
        try:
            AUDIO_CAPTURE_ACTIVE.clear()
        except NameError:
            pass
"""
# Replace only first occurrence after capture_user_turn. This may hit the correct one in current file.
text = text.replace(old, new, 1)

# Make should_respond_naturally accept short follow-ups during active conversation.
old = """    if familiar_person and active_followup and len(words) >= 2:
        return True
"""
new = """    if familiar_person and active_followup and len(words) >= 1:
        return True
"""
text = text.replace(old, new)

# Treat direct continuation words as valid follow-ups.
if "CONTINUATION_WORDS" not in text:
    insert_after = """REQUEST_WORDS = [
    "tell", "say", "look", "check", "explain", "calculate",
    "show", "remember", "describe", "give", "help", "find"
]
"""
    continuation = """
CONTINUATION_WORDS = {
    "yes", "no", "yeah", "yep", "nope", "maybe",
    "imagination", "real", "fiction", "continue", "exactly"
}
"""
    text = text.replace(insert_after, insert_after + continuation)

old = """    if text_has_any(text, REQUEST_WORDS):
        return True
"""
new = """    if text_has_any(text, REQUEST_WORDS):
        return True

    if familiar_person and active_followup and words & CONTINUATION_WORDS:
        return True
"""
text = text.replace(old, new)

path.write_text(text)
print(f"Patched V6.2: {path}")
