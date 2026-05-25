from pathlib import Path

intents_path = Path.home() / "robot-project/week3/camera/v7/camera_intents.py"
brain_path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v7_full.py"

intents = intents_path.read_text()
brain = brain_path.read_text()

# -------------------------------------------------------------------
# 1. Improve camera_intents.py for imperfect speech phrases.
# -------------------------------------------------------------------

start = intents.find("def is_identity_camera_request(user_text: str) -> bool:")
if start == -1:
    raise SystemExit("Could not find is_identity_camera_request().")
end = intents.find("\ndef ", start + 1)
if end == -1:
    raise SystemExit("Could not find end of is_identity_camera_request().")

new_identity = r'''def is_identity_camera_request(user_text: str) -> bool:
    """
    Requests that should use local face recognition / identity.
    Includes imperfect ASR variants like "who you see".
    """
    text = normalize_text(user_text)

    phrases = [
        "can you see me",
        "do you see me",
        "who do you see",
        "who you see",
        "who are you seeing",
        "who is there",
        "who is in front",
        "who is in front of you",
        "who is in front of the camera",
        "who am i",
        "do you recognize me",
        "do you recognise me",
        "recognize me",
        "recognise me",
        "identify me",
        "whose face",
        "which person",
        "which face",
        "is it marco",
        "is it marquinho",
    ]

    return any(p in text for p in phrases)

'''

intents = intents[:start] + new_identity + intents[end+1:]

start = intents.find("def is_scene_camera_request(user_text: str) -> bool:")
if start == -1:
    raise SystemExit("Could not find is_scene_camera_request().")
end = intents.find("\ndef ", start + 1)
if end == -1:
    raise SystemExit("Could not find end of is_scene_camera_request().")

new_scene = r'''def is_scene_camera_request(user_text: str) -> bool:
    """
    Requests that should use fresh image scene description.
    Broad "do you see?" questions should use camera, not safety.
    """
    text = normalize_text(user_text)

    # Identity requests take priority.
    if is_identity_camera_request(text):
        return False

    phrases = [
        "describe",
        "describe what you see",
        "describe what you are seeing",
        "describe your camera",
        "describe your camera view",
        "describe the scene",
        "what do you see",
        "what can you see",
        "what are you seeing",
        "do you see",
        "can you see",
        "what is your camera seeing",
        "what's your camera seeing",
        "what is in front of you",
        "what's in front of you",
        "what is in front of the camera",
        "what's in front of the camera",
        "look around",
        "look and tell me",
        "refresh your camera",
        "camera view",
        "your view",
        "what is there",
        "what's there",
        "is the camera blocked",
        "is your vision blocked",
        "black board",
        "blocked camera",
    ]

    return any(p in text for p in phrases)

'''

intents = intents[:start] + new_scene + intents[end+1:]

intents_path.write_text(intents)


# -------------------------------------------------------------------
# 2. Patch Full V7 handle_user_turn so camera_generic still routes to camera.
# -------------------------------------------------------------------

old = '''    # Camera requests are robot-control/vision commands.
    # Do not send ordinary "what do you see?" / "who do you see?" through cloud safety.
    # They route directly to camera truth.
    if is_scene_camera_request(user_text):
        reply = build_scene_reply(camera_manager)
        v6.speak(reply)
        try:
            v6.update_conversation_memory(user_text=user_text, assistant_reply=reply)
        except Exception:
            pass
        return True

    if is_identity_camera_request(user_text):
        face_state = camera_manager.get_face_state(max_age_seconds=2.0)
        reply = build_identity_reply(face_state)
        v6.speak(reply)
        try:
            v6.update_conversation_memory(user_text=user_text, assistant_reply=reply)
        except Exception:
            pass
        return True
'''

new = '''    # Camera requests are robot-control/vision commands.
    # Do not send ordinary "what do you see?" / "who do you see?" / imperfect ASR
    # variants through cloud safety. They route directly to camera truth.
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
        reply = build_scene_reply(camera_manager)
        v6.speak(reply)
        try:
            v6.update_conversation_memory(user_text=user_text, assistant_reply=reply)
        except Exception:
            pass
        return True
'''

if old not in brain:
    raise SystemExit("Could not find Full V7 camera routing block.")

brain = brain.replace(old, new)
brain_path.write_text(brain)

print("Patched V7 Full camera generic routing.")
