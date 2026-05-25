"""
Miguel V7 camera intent routing.

Policy:
- Any camera/vision action request about seeing, looking, camera, vision, recognition,
  or describing the robot view must use fresh camera data.
- Normal conversation may use cached memory.
- Conversation identity must never be treated as camera evidence.
"""


def normalize_text(user_text: str) -> str:
    return str(user_text or "").lower().strip()


def is_identity_camera_request(user_text: str) -> bool:
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

def is_scene_camera_request(user_text: str) -> bool:
    """
    Requests that should use fresh image scene description.
    Broad "do you see?" questions should use camera, not safety.
    """
    text = normalize_text(user_text)

    if _is_descriptive_visual_statement(text):
        return False

    # Identity requests take priority.
    if is_identity_camera_request(text):
        return False

    phrases = [
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

def is_any_camera_request(user_text: str) -> bool:
    """
    Any user request where Miguel must not answer from conversation memory.
    This intentionally requires a camera/vision action pattern. Bare words like
    "face", "person", "people", or "see" can be normal robot-planning talk.
    """
    text = normalize_text(user_text)

    if _is_descriptive_visual_statement(text):
        return False

    phrases = [
        "what do you see",
        "what can you see",
        "what are you seeing",
        "do you see",
        "can you see",
        "look at",
        "look around",
        "look and tell me",
        "describe what you see",
        "describe what you are seeing",
        "who do you see",
        "who you see",
        "who am i",
        "who is in front",
        "recognize me",
        "recognise me",
        "identify me",
        "is the camera blocked",
        "is your vision blocked",
        "detect the camera",
        "blocked camera",
    ]

    return any(p in text for p in phrases)


def _is_descriptive_visual_statement(text: str) -> bool:
    """
    Avoid routing Miguel's own/vision-model descriptions back into the camera.
    These are reports, not commands to use the camera.
    """
    t = normalize_text(text)
    starters = [
        "i see ",
        "image shows ",
        "the image shows ",
        "frame shows ",
        "the frame shows ",
        "camera sees ",
        "the camera sees ",
        "i checked the camera ",
        "i do not see a confirmed face",
        "i checked the camera and recognize",
    ]

    if not any(t.startswith(s) for s in starters):
        return False

    command_markers = [
        "?",
        "can you",
        "do you",
        "what do",
        "who do",
        "look at",
        "describe what",
        "is the camera blocked",
        "recognize me",
        "recognise me",
        "identify me",
        "detect the camera",
    ]

    return not any(m in t for m in command_markers)


def classify_camera_intent(user_text: str) -> str:
    """
    Returns:
      identity_camera
      scene_camera
      camera_generic
      none
    """
    text = normalize_text(user_text)

    if _is_descriptive_visual_statement(text):
        return "none"

    if is_identity_camera_request(text):
        return "identity_camera"

    if is_scene_camera_request(text):
        return "scene_camera"

    if is_any_camera_request(text):
        return "camera_generic"

    return "none"
