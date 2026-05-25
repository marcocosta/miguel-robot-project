from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"
text = path.read_text()

# -------------------------------------------------------------------
# 1. Replace is_scene_description_request with broader routing.
# -------------------------------------------------------------------

start = text.find("def is_scene_description_request(user_text):")
if start == -1:
    raise SystemExit("Could not find is_scene_description_request().")

end = text.find("\ndef ", start + 1)
if end == -1:
    raise SystemExit("Could not find end of is_scene_description_request().")

new_func = r'''def is_scene_description_request(user_text):
    text = user_text.lower().strip()

    phrases = [
        "describe what you see",
        "describe what you are seeing",
        "describe the scene",
        "describe in front of you",
        "describe your camera view",
        "describe your view",
        "describe what your camera is seeing",
        "what is your camera seeing",
        "what your camera is seeing",
        "what do you see in front of you",
        "what can you see in front of you",
        "what is in front of you",
        "what's in front of you",
        "what are you looking at",
        "look around and describe",
        "refresh your camera view and tell me what you see",
        "use the camera and describe",
        "tell me what you see in front of the camera",
        "what do you see now",
        "what can you see now",
    ]

    # These are identity/face-recognition requests, not scene-description requests.
    identity_phrases = [
        "can you see me",
        "do you see me",
        "who do you see",
        "who am i",
        "do you recognize me",
        "recognize me",
        "identify me",
    ]

    if any(p in text for p in identity_phrases):
        return False

    return any(p in text for p in phrases)

'''

text = text[:start] + new_func + text[end:]


# -------------------------------------------------------------------
# 2. Make get_state_for_user_turn avoid face scan for scene description.
# Scene description has its own fresh frame capture.
# -------------------------------------------------------------------

old = '''        if is_vision_question(user_text) or is_camera_related_request(user_text):
            print("[VISION] Fresh scan requested for camera-related question.")
'''

new = '''        if (not is_scene_description_request(user_text)) and (is_vision_question(user_text) or is_camera_related_request(user_text)):
            print("[VISION] Fresh scan requested for camera-related question.")
'''

if old in text:
    text = text.replace(old, new)
else:
    print("Warning: fresh scan camera-related trigger not found exactly.")


# -------------------------------------------------------------------
# 3. Add stronger scene prompt for blocked camera truth.
# -------------------------------------------------------------------

old_prompt = '''                                "You are Miguel, a small father-son robot. "
                                "Describe the camera image in one or two short spoken sentences. "
                                "Be honest and cautious. If the image is blocked, dark, blurry, or unclear, say that. "
                                "Do not identify a person's real identity from the image. "
                                "You may say 'a person' or 'a face' only if visually obvious."
'''

new_prompt = '''                                "You are Miguel, a small father-son robot. "
                                "Describe the latest camera frame in one or two short spoken sentences. "
                                "Mention only what is visibly present: blockage, darkness, objects, lighting, people count, and rough positions. "
                                "If the image is blocked, dark, blurry, or unclear, say that directly. "
                                "Do not identify a person's real identity from the image. "
                                "Only say there is a person or face if it is clearly visible in this exact image."
'''

if old_prompt in text:
    text = text.replace(old_prompt, new_prompt)
else:
    print("Warning: scene prompt block not found exactly.")

path.write_text(text)
print("Patched V6.17 scene routing.")
