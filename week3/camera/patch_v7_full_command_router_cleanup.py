from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v7_full.py"
text = path.read_text()

# ------------------------------------------------------------
# 1. Add local command helpers before handle_user_turn().
# ------------------------------------------------------------

marker = "def handle_user_turn(user_text: str, camera_manager: CameraManager, safety: SafetyGuard) -> bool:"
idx = text.find(marker)
if idx == -1:
    raise SystemExit("Could not find handle_user_turn().")

helpers = r'''
def is_local_robot_control_request(user_text: str) -> bool:
    """
    Commands about Miguel itself should not go through semantic safety first.
    They should be handled by local robot memory / mode / voice logic.
    """
    t = str(user_text or "").lower().strip()

    phrases = [
        # Voice controls.
        "voice",
        "deep voice",
        "robot voice",
        "robotic voice",
        "natural voice",
        "friendly voice",
        "story voice",
        "storyteller voice",
        "narrator voice",
        "change voice",
        "switch voice",
        "what voices",
        "voice options",
        "which voice",

        # Modes.
        "sleep mode",
        "wake up",
        "shutdown",
        "shut down",
        "confirm shutdown",
        "confirm shut down",
        "mission control",

        # Simple robot utility/status.
        "status",
        "what time",
        "time is it",
        "current time",
    ]

    return any(p in t for p in phrases)


def handle_v7_local_utility(user_text: str) -> bool:
    """
    Very small V7-local utility layer for things V6 cloud brain may not know.
    Returns True if handled.
    """
    t = str(user_text or "").lower().strip()

    if "what time" in t or "time is it" in t or "current time" in t:
        from datetime import datetime
        now = datetime.now().strftime("%I:%M %p").lstrip("0")
        v6.speak(f"The local Jetson time is {now}.")
        return True

    return False


def route_camera_intent_now(user_text: str, camera_intent: str, camera_manager: CameraManager) -> bool:
    """
    Absolute V7 rule:
    Any camera/vision/see/look request uses camera truth directly,
    not semantic safety and not conversation memory.
    """
    t = str(user_text or "").lower()

    if is_identity_camera_request(user_text):
        face_state = camera_manager.get_face_state(max_age_seconds=2.0)
        reply = build_identity_reply(face_state)
        v6.speak(reply)
        try:
            v6.update_conversation_memory(user_text=user_text, assistant_reply=reply)
        except Exception:
            pass
        return True

    if is_scene_camera_request(user_text) or camera_intent == "camera_generic":
        # If generic wording includes "who", treat it as identity.
        if "who" in t or "identify" in t or "recognize" in t or "recognise" in t:
            face_state = camera_manager.get_face_state(max_age_seconds=2.0)
            reply = build_identity_reply(face_state)
        else:
            reply = build_scene_reply(camera_manager)

        v6.speak(reply)
        try:
            v6.update_conversation_memory(user_text=user_text, assistant_reply=reply)
        except Exception:
            pass
        return True

    return False

'''

if "def is_local_robot_control_request(" not in text:
    text = text[:idx] + helpers + "\n" + text[idx:]


# ------------------------------------------------------------
# 2. Insert hard routing at start of handle_user_turn, after camera_intent print.
# ------------------------------------------------------------

old = '''    camera_intent = classify_camera_intent(user_text)
    if camera_intent != "none":
        print(f"[V7 FULL CAMERA INTENT] {camera_intent}: {user_text}")

    # Camera requests are robot-control/vision commands.
'''

new = '''    camera_intent = classify_camera_intent(user_text)
    if camera_intent != "none":
        print(f"[V7 FULL CAMERA INTENT] {camera_intent}: {user_text}")

    # Local utility commands that should never require cloud safety.
    if handle_v7_local_utility(user_text):
        return True

    # Camera requests are robot-control/vision commands.
    # They must use live camera truth immediately.
    if route_camera_intent_now(user_text, camera_intent, camera_manager):
        return True

    # Miguel local controls like voice/sleep/shutdown should be handled by V6 local logic,
    # not semantic safety/cloud brain first.
    if is_local_robot_control_request(user_text):
        face_state = camera_manager.get_face_state(max_age_seconds=2.0)
        return bool(v6.handle_user_turn_with_cached_state(user_text, face_state))

    # Camera requests are robot-control/vision commands.
'''

if old not in text:
    raise SystemExit("Could not find camera_intent block.")

text = text.replace(old, new)


# ------------------------------------------------------------
# 3. Strengthen shutdown cleanup: keep a handle to face_thread and join it.
# ------------------------------------------------------------

text = text.replace(
    "    camera_manager = None\n\n    try:",
    "    camera_manager = None\n    face_thread = None\n\n    try:"
)

text = text.replace(
    '''            face_thread = threading.Thread(
                target=face_worker,
                args=(camera_manager, stop_event),
                daemon=True,
            )
            face_thread.start()
''',
    '''            face_thread = threading.Thread(
                target=face_worker,
                args=(camera_manager, stop_event),
                daemon=True,
            )
            face_thread.start()
'''
)

old_finally = '''    finally:
        stop_event.set()

        if camera_manager:
            camera_manager.stop()

        print("Miguel Cloud Brain V7 Full stopped.")
'''

new_finally = '''    finally:
        stop_event.set()

        # Give the face/ONNX worker a chance to leave before closing DepthAI resources.
        try:
            if face_thread and face_thread.is_alive():
                face_thread.join(timeout=1.5)
        except Exception as e:
            print("[V7 FULL] face thread join warning:", e)

        if camera_manager:
            camera_manager.stop()

        print("Miguel Cloud Brain V7 Full stopped.")
'''

if old_finally not in text:
    raise SystemExit("Could not find finally block.")

text = text.replace(old_finally, new_finally)

path.write_text(text)
print("Patched Full V7 command routing and shutdown cleanup.")
