from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"
text = path.read_text()

helper = r'''
def flush_camera_queue(camera_queue, drain_frames=12):
    """
    Drain old buffered camera frames so fresh vision questions use the live view,
    not stale frames from before the user blocked/moved the camera.
    """
    drained = 0

    try:
        for _ in range(drain_frames):
            msg = None

            if hasattr(camera_queue, "tryGet"):
                msg = camera_queue.tryGet()
            else:
                break

            if msg is None:
                break

            drained += 1

    except Exception as e:
        print("[VISION] Camera queue flush warning:", e)

    if drained:
        print(f"[VISION] Flushed {drained} stale camera frames.")

'''

marker = "def get_state_for_user_turn(user_text, camera_queue, fallback_state):"
idx = text.find(marker)

if idx == -1:
    raise SystemExit("Could not find get_state_for_user_turn().")

if "def flush_camera_queue(" not in text:
    text = text[:idx] + helper + "\n" + text[idx:]

old = '''        if is_vision_question(user_text):
            print("[VISION] Fresh scan requested for vision question.")
            fresh_state = detect_face_state(camera_queue)
            set_cached_local_state(fresh_state)
            return fresh_state
'''

new = '''        if is_vision_question(user_text):
            print("[VISION] Fresh scan requested for vision question.")
            flush_camera_queue(camera_queue, drain_frames=16)
            time.sleep(0.35)
            flush_camera_queue(camera_queue, drain_frames=8)
            fresh_state = detect_face_state(camera_queue)
            fresh_state["fresh_scan_for_question"] = True
            fresh_state["fresh_scan_time"] = time.time()
            set_cached_local_state(fresh_state)
            return fresh_state
'''

if old not in text:
    raise SystemExit("Could not find fresh scan block to replace.")

text = text.replace(old, new)

path.write_text(text)
print("Patched fresh vision queue flushing.")
