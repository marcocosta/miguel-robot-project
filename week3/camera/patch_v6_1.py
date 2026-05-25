from pathlib import Path
import re

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"
text = path.read_text()

# ------------------------------------------------------------
# 1) Make wake listening timeout-aware so Miguel can leave wake mode
# if the vision thread recognizes a familiar face.
# ------------------------------------------------------------

old_func_start = text.find("def listen_for_wake():")
if old_func_start == -1:
    raise SystemExit("Could not find listen_for_wake().")

# Find next function after listen_for_wake
next_def = text.find("\ndef capture_user_turn", old_func_start)
if next_def == -1:
    raise SystemExit("Could not find end of listen_for_wake().")

new_listen_for_wake = r'''def listen_for_wake(timeout_seconds=None):
    print(f"\n{ROBOT_NAME} is idle. Say: 'Hey Miguel', 'Hey me go', or 'Mission Control'.")

    proc = open_raw_mic_stream()
    recognizer = KaldiRecognizer(vosk_model, AUDIO_RATE)

    last_reset = time.time()
    start_time = time.time()

    try:
        while True:
            if timeout_seconds is not None and (time.time() - start_time) >= timeout_seconds:
                print("Wake listen timeout reached.")
                return False

            raw = proc.stdout.read(CHUNK_BYTES)
            if not raw:
                continue

            mono_bytes, rms = stereo_raw_to_mono_bytes(raw)
            if not mono_bytes:
                continue

            if recognizer.AcceptWaveform(mono_bytes):
                result = json.loads(recognizer.Result())
                final_text = result.get("text", "").lower().strip()

                if final_text:
                    print(f"Idle final: {final_text}")

                    if text_contains_any(final_text, WAKE_PHRASES):
                        print("Wake phrase detected from final text.")
                        return True

                recognizer = KaldiRecognizer(vosk_model, AUDIO_RATE)
                last_reset = time.time()

            else:
                partial_text = json.loads(recognizer.PartialResult()).get("partial", "").lower().strip()

                if partial_text:
                    print(f"Idle partial: {partial_text}")

                    if text_contains_any(partial_text, WAKE_PHRASES):
                        print("Wake phrase detected from partial text.")
                        return True

            if time.time() - last_reset > 5:
                recognizer = KaldiRecognizer(vosk_model, AUDIO_RATE)
                last_reset = time.time()

    finally:
        stop_stream(proc)
'''

text = text[:old_func_start] + new_listen_for_wake + text[next_def:]


# ------------------------------------------------------------
# 2) Replace the aggressive custom greeting shortcut.
# Old behavior triggers greeting on "who", "see", "hey", etc.
# New behavior only greets on short pure greeting messages.
# ------------------------------------------------------------

helper_marker = "def handle_user_turn_with_cached_state(user_text, cached_local_state):"
helper_idx = text.find(helper_marker)
if helper_idx == -1:
    raise SystemExit("Could not find handle_user_turn_with_cached_state().")

insert_helper = r'''
def is_simple_greeting(user_text):
    text = user_text.lower().strip()
    words = set(re.findall(r"\b\w+\b", text))

    greeting_words = {"hello", "hi", "hey"}
    question_request_words = {
        "what", "who", "when", "where", "why", "how",
        "see", "look", "tell", "check", "calculate",
        "record", "remember", "status", "weather", "time",
    }

    if not (words & greeting_words):
        return False

    if words & question_request_words:
        return False

    # Only short greeting-style turns.
    return len(words) <= 3


'''
text = text[:helper_idx] + insert_helper + text[helper_idx:]

old_shortcut = '''    if local_state.get("recognized_person") in CUSTOM_GREETINGS and words & {"hello", "hi", "hey", "look", "see", "who"}:
        speak(CUSTOM_GREETINGS[local_state["recognized_person"]])
        return True
'''

new_shortcut = '''    if local_state.get("recognized_person") in CUSTOM_GREETINGS and is_simple_greeting(user_text):
        speak(CUSTOM_GREETINGS[local_state["recognized_person"]])
        return True
'''

if old_shortcut not in text:
    print("Warning: old greeting shortcut not found exactly; trying regex replacement.")
    text = re.sub(
        r'    if local_state\.get\("recognized_person"\) in CUSTOM_GREETINGS and words & \{[^}]+\}:\n        speak\(CUSTOM_GREETINGS\[local_state\["recognized_person"\]\]\)\n        return True\n',
        new_shortcut,
        text,
    )
else:
    text = text.replace(old_shortcut, new_shortcut)


# ------------------------------------------------------------
# 3) Refresh cached local state AFTER the user finishes speaking,
# right before brain processing. This prevents using stale unknown state.
# ------------------------------------------------------------

text = text.replace(
    "keep_running = handle_user_turn_with_cached_state(user_text, local_state)",
    "fresh_state = get_cached_local_state()\n                    keep_running = handle_user_turn_with_cached_state(user_text, fresh_state)"
)

# There are two branches; indentation differs in wake branch.
text = text.replace(
    "keep_running = handle_user_turn_with_cached_state(user_text, local_state)",
    "fresh_state = get_cached_local_state()\n                    keep_running = handle_user_turn_with_cached_state(user_text, fresh_state)"
)


# ------------------------------------------------------------
# 4) In wake-required branch, do not block forever. Check every 3 sec.
# If vision recognizes a familiar person while waiting, switch to natural mode.
# ------------------------------------------------------------

old_wake = '''                    print("[IDLE] No familiar person recognized. Wake phrase required.")
                    listen_for_wake()
                    speak("I'm listening now.")
                    print("[READY] Start talking now.")

                    local_state = get_cached_local_state()
                    user_text = capture_user_turn()
'''

new_wake = '''                    print("[IDLE] No familiar person recognized. Wake phrase required.")
                    wake_detected = listen_for_wake(timeout_seconds=3)

                    # If the vision thread recognized someone while we were waiting,
                    # skip wake mode and return to the top of the loop.
                    latest_state = get_cached_local_state()
                    if not wake_detected and latest_state.get("recognized_person") and is_vision_state_fresh(latest_state):
                        print("[IDLE] Familiar person recognized during wake wait. Switching to natural mode.")
                        continue

                    if not wake_detected:
                        continue

                    speak("I'm listening now.")
                    print("[READY] Start talking now.")

                    local_state = get_cached_local_state()
                    user_text = capture_user_turn()
'''

if old_wake not in text:
    print("Warning: wake branch not found exactly. You may need manual patch.")
else:
    text = text.replace(old_wake, new_wake)


path.write_text(text)
print(f"Patched V6.1: {path}")
