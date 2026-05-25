from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"
text = path.read_text()

# 1) Add a longer grace window constant.
if "CONVERSATION_GRACE_SECONDS" not in text:
    text = text.replace(
        "ACTIVE_FOLLOWUP_SECONDS = 30",
        "ACTIVE_FOLLOWUP_SECONDS = 45\nCONVERSATION_GRACE_SECONDS = 45"
    )

# 2) Add helper to build fallback state from last known person.
helper_marker = "def handle_user_turn_with_cached_state(user_text, cached_local_state):"
idx = text.find(helper_marker)
if idx == -1:
    raise SystemExit("Could not find handle_user_turn_with_cached_state.")

helper = r'''
def make_last_known_person_state(person_name):
    return {
        "face_detected": True,
        "face_count": 1,
        "face_position": "center",
        "recognized_person": person_name,
        "recognition_score": None,
        "recognition_margin": None,
        "recognition_votes": {person_name: 1} if person_name else {},
        "recognition_scores": {},
        "recognizer": "conversation_grace_cached_identity",
        "last_vision_update": time.time(),
    }


'''
if "def make_last_known_person_state(" not in text:
    text = text[:idx] + helper + text[idx:]

# 3) Ensure last_known_person exists in run loop.
text = text.replace(
    "last_announced_person = None\n    last_reply_time = 0",
    "last_announced_person = None\n    last_known_person = None\n    last_reply_time = 0"
)

# 4) When a person is recognized, remember them.
text = text.replace(
    "if recognized_person and vision_fresh:",
    "if recognized_person and vision_fresh:\n                    last_known_person = recognized_person"
)

# 5) Add grace mode branch before wake-required branch.
old = '''                else:
                    last_announced_person = None
                    print("[IDLE] No familiar person recognized. Wake phrase required.")
                    wake_detected = listen_for_wake(timeout_seconds=3)
'''

new = '''                elif last_known_person and (time.time() - last_reply_time) < CONVERSATION_GRACE_SECONDS:
                    # Keep natural follow-up mode alive even if vision briefly drops.
                    friendly_person = last_known_person.replace("_", " ")
                    local_state = get_cached_local_state()

                    if not local_state.get("recognized_person"):
                        local_state = make_last_known_person_state(last_known_person)

                    print(f"[FOLLOWUP] Grace mode active for {friendly_person}. Start talking now.")
                    ready_beep()
                    user_text = capture_user_turn()

                    if not user_text:
                        print("[AUDIO] No speech captured during follow-up grace.")
                        continue

                    print(f"[TRANSCRIPT] {user_text}")

                    # In grace mode, answer even short replies like yes/no/imagination.
                    print("[THINKING] Miguel is processing follow-up.")
                    keep_running = handle_user_turn_with_cached_state(user_text, local_state)
                    last_reply_time = time.time()

                else:
                    last_announced_person = None
                    print("[IDLE] No familiar person recognized. Wake phrase required.")
                    wake_detected = listen_for_wake(timeout_seconds=3)
'''

if old not in text:
    raise SystemExit("Could not find wake-required branch to replace.")
text = text.replace(old, new)

path.write_text(text)
print(f"Patched V6.4 follow-up grace mode: {path}")
