from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"
text = path.read_text()

# 1) Add camera lock near STATE_LOCK.
if "CAMERA_LOCK = threading.Lock()" not in text:
    text = text.replace(
        "STATE_LOCK = threading.Lock()",
        "STATE_LOCK = threading.Lock()\nCAMERA_LOCK = threading.Lock()"
    )

# 2) Lock camera use inside vision_worker.
old = """        try:
            state = detect_face_state(camera_queue)
            set_cached_local_state(state)
"""

new = """        try:
            with CAMERA_LOCK:
                state = detect_face_state(camera_queue)
            set_cached_local_state(state)
"""

if old in text:
    text = text.replace(old, new)
else:
    print("Warning: vision_worker camera lock block not found exactly.")

# 3) Lock and harden fresh vision scan.
old = """        if is_vision_question(user_text):
            print("[VISION] Fresh scan requested for vision question.")
            flush_camera_queue(camera_queue, drain_frames=16)
            time.sleep(0.35)
            flush_camera_queue(camera_queue, drain_frames=8)
            fresh_state = detect_face_state(camera_queue)
            fresh_state["fresh_scan_for_question"] = True
            fresh_state["fresh_scan_time"] = time.time()
            set_cached_local_state(fresh_state)
            return fresh_state
"""

new = """        if is_vision_question(user_text):
            print("[VISION] Fresh scan requested for vision question.")

            with CAMERA_LOCK:
                flush_camera_queue(camera_queue, drain_frames=24)
                time.sleep(0.45)
                flush_camera_queue(camera_queue, drain_frames=12)
                fresh_state = detect_face_state(camera_queue)

            fresh_state["fresh_scan_for_question"] = True
            fresh_state["fresh_scan_time"] = time.time()

            # Strict camera-truth mode:
            # If there is no confirmed recognized_person, do not let the cloud brain
            # turn raw scores into "best guess Marco." Treat it as unconfirmed.
            if fresh_state.get("face_detected") and not fresh_state.get("recognized_person"):
                fresh_state["unconfirmed_face_like_pattern"] = True
                fresh_state["vision_warning"] = (
                    "Fresh scan found a face-like pattern, but identity was not confirmed. "
                    "Do not claim this is Marco or Marquinho."
                )
                fresh_state["recognition_scores_for_debug"] = fresh_state.get("recognition_scores", {})
                fresh_state["recognition_scores"] = {}
                fresh_state["recognition_votes"] = {}

            set_cached_local_state(fresh_state)
            return fresh_state
"""

if old in text:
    text = text.replace(old, new)
else:
    print("Warning: fresh vision block not found exactly.")

# 4) Add local strict response for vision questions before cloud brain.
marker = """    if local_state.get("face_detected") and local_state.get("recognized_person") is None:
        print("[BRAIN] Unknown face detected. Auto-enrollment remains disabled during tuning.")
"""

replacement = """    if is_vision_question(user_text):
        if not local_state.get("face_detected"):
            reply = "I do not see a confirmed face right now."
            speak(reply)
            update_conversation_memory(assistant_reply=reply)
            return True

        if local_state.get("unconfirmed_face_like_pattern"):
            reply = (
                "I detect a face-like pattern, but I cannot confirm it is a real person. "
                "I will not call it Marco or Marquinho."
            )
            speak(reply)
            update_conversation_memory(assistant_reply=reply)
            return True

        if local_state.get("recognized_person"):
            person = local_state.get("recognized_person").replace("_", " ")
            score = local_state.get("recognition_score")
            if score is not None:
                reply = f"I see one face and I recognize it as {person}, with confidence about {score:.2f}."
            else:
                reply = f"I see one face and I recognize it as {person}."
            speak(reply)
            update_conversation_memory(assistant_reply=reply)
            return True

    if local_state.get("face_detected") and local_state.get("recognized_person") is None:
        print("[BRAIN] Unknown face detected. Auto-enrollment remains disabled during tuning.")
"""

if marker in text:
    text = text.replace(marker, replacement)
else:
    print("Warning: strict vision response insertion marker not found.")

path.write_text(text)
print("Patched V6.11 camera truth mode.")
