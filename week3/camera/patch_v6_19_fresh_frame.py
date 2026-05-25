from pathlib import Path
import re

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"
text = path.read_text()

# ------------------------------------------------------------
# 1) Replace/upgrade flush_camera_queue().
# ------------------------------------------------------------

flush_pattern = r'''def flush_camera_queue\(camera_queue, drain_frames=.*?\):.*?\n(?=def |\n#|\Z)'''

flush_new = r'''def flush_camera_queue(camera_queue, drain_frames=80):
    """
    Aggressively drain old buffered camera frames.

    OAK/DepthAI queues can keep frames from before the user blocked or moved
    the camera. For any explicit vision request, stale frames are worse than
    slower response.
    """
    drained = 0

    try:
        # Best case: DepthAI supports tryGetAll().
        if hasattr(camera_queue, "tryGetAll"):
            for _ in range(3):
                msgs = camera_queue.tryGetAll()
                if not msgs:
                    break
                drained += len(msgs)
            if drained:
                print(f"[VISION] Flushed {drained} stale camera frames via tryGetAll.")
            return drained

        # Fallback: repeatedly tryGet().
        if hasattr(camera_queue, "tryGet"):
            for _ in range(drain_frames):
                msg = camera_queue.tryGet()
                if msg is None:
                    break
                drained += 1

    except Exception as e:
        print("[VISION] Camera queue flush warning:", e)

    if drained:
        print(f"[VISION] Flushed {drained} stale camera frames.")

    return drained

'''

text, n = re.subn(flush_pattern, flush_new, text, count=1, flags=re.DOTALL)
if n == 0:
    print("Warning: flush_camera_queue() not found; inserting helper.")
    marker = "def get_state_for_user_turn"
    idx = text.find(marker)
    if idx == -1:
        raise SystemExit("Could not find insertion point.")
    text = text[:idx] + flush_new + "\n" + text[idx:]


# ------------------------------------------------------------
# 2) Add capture_latest_camera_msg().
# ------------------------------------------------------------

marker = "def capture_fresh_camera_frame"
idx = text.find(marker)
if idx == -1:
    raise SystemExit("Could not find capture_fresh_camera_frame().")

latest_helper = r'''def capture_latest_camera_msg(camera_queue, warmup_seconds=1.0, discard_after_wait=8):
    """
    Return a camera message that is much more likely to represent the live view.

    Method:
    1. Drain old frames.
    2. Wait for new sensor frames.
    3. Discard several post-wait frames.
    4. Use the newest message available.
    """
    flush_camera_queue(camera_queue, drain_frames=120)
    time.sleep(warmup_seconds)

    latest = None

    # Discard some frames generated after the wait.
    for _ in range(discard_after_wait):
        try:
            if hasattr(camera_queue, "get"):
                latest = camera_queue.get()
        except Exception:
            break

    # Now drain anything else available and keep only the newest.
    try:
        if hasattr(camera_queue, "tryGetAll"):
            msgs = camera_queue.tryGetAll()
            if msgs:
                latest = msgs[-1]
        elif hasattr(camera_queue, "tryGet"):
            while True:
                msg = camera_queue.tryGet()
                if msg is None:
                    break
                latest = msg
    except Exception as e:
        print("[VISION] Latest-frame drain warning:", e)

    if latest is None:
        latest = camera_queue.get()

    return latest


'''

if "def capture_latest_camera_msg(" not in text:
    text = text[:idx] + latest_helper + "\n" + text[idx:]


# ------------------------------------------------------------
# 3) Replace capture_fresh_camera_frame() to use latest-frame logic.
# ------------------------------------------------------------

cap_pattern = r'''def capture_fresh_camera_frame\(camera_queue, save_debug=True\):.*?\n(?=def |\n#|\Z)'''

cap_new = r'''def capture_fresh_camera_frame(camera_queue, save_debug=True):
    """
    Capture the latest live frame from the OAK camera for scene understanding.
    This is scene-level vision, not face identity.
    """
    with CAMERA_LOCK:
        msg = capture_latest_camera_msg(
            camera_queue,
            warmup_seconds=1.0,
            discard_after_wait=10,
        )
        frame = msg.getCvFrame()

    out_path = None

    if save_debug:
        out_dir = Path.home() / "robot-project/week3/debug_snapshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"scene_describe_{int(time.time())}.jpg"

        try:
            cv2.imwrite(str(out_path), frame)
            print(f"[SCENE] Saved fresh scene frame: {out_path}")
        except Exception as e:
            print("[SCENE] Could not save scene frame:", e)

    return frame, out_path

'''

text, n = re.subn(cap_pattern, cap_new, text, count=1, flags=re.DOTALL)
if n == 0:
    raise SystemExit("Could not replace capture_fresh_camera_frame().")


# ------------------------------------------------------------
# 4) Make face fresh scan also more aggressive.
# ------------------------------------------------------------

old = '''            with CAMERA_LOCK:
                flush_camera_queue(camera_queue, drain_frames=24)
                time.sleep(0.45)
                flush_camera_queue(camera_queue, drain_frames=12)
                fresh_state = detect_face_state(camera_queue)
'''

new = '''            with CAMERA_LOCK:
                flush_camera_queue(camera_queue, drain_frames=120)
                time.sleep(0.8)
                flush_camera_queue(camera_queue, drain_frames=40)
                fresh_state = detect_face_state(camera_queue)
'''

if old in text:
    text = text.replace(old, new)
else:
    print("Warning: face fresh scan block not found exactly. It may already be different.")


# ------------------------------------------------------------
# 5) Add a visible log marker so we know this patch is active.
# ------------------------------------------------------------

text = text.replace(
    'print("[SCENE] Fresh scene description requested.")',
    'print("[SCENE] Fresh scene description requested. Using V6.19 latest-frame capture.")'
)

path.write_text(text)
print("Patched V6.19 fresh-frame camera capture.")
