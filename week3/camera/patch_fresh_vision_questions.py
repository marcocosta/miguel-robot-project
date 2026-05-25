from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"
text = path.read_text()

helper = r'''
def get_state_for_user_turn(user_text, camera_queue, fallback_state):
    """
    For normal conversation, use cached state.
    For vision questions, force a fresh camera scan so Miguel does not answer from stale memory.
    """
    try:
        if is_vision_question(user_text):
            print("[VISION] Fresh scan requested for vision question.")
            fresh_state = detect_face_state(camera_queue)
            set_cached_local_state(fresh_state)
            return fresh_state
    except Exception as e:
        print("[VISION] Fresh scan failed, using cached state:", e)

    return fallback_state

'''

marker = "def run_v6_threaded_conversation():"
idx = text.find(marker)

if idx == -1:
    raise SystemExit("Could not find run_v6_threaded_conversation().")

if "def get_state_for_user_turn(" not in text:
    text = text[:idx] + helper + "\n" + text[idx:]

# Replace direct handle calls so they refresh vision state when needed.
text = text.replace(
    "keep_running = handle_user_turn_with_cached_state(user_text, local_state)",
    "turn_state = get_state_for_user_turn(user_text, camera_queue, local_state)\n                    keep_running = handle_user_turn_with_cached_state(user_text, turn_state)"
)

text = text.replace(
    "keep_running = handle_user_turn_with_cached_state(user_text, fresh_state)",
    "turn_state = get_state_for_user_turn(user_text, camera_queue, fresh_state)\n                    keep_running = handle_user_turn_with_cached_state(user_text, turn_state)"
)

path.write_text(text)
print("Patched fresh vision scan for vision questions.")
