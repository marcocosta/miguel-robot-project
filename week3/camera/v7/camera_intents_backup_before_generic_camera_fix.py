"""
Miguel V7 camera intent routing.

Policy:
- Any request about seeing, looking, camera, vision, faces, recognition,
  or describing the robot view must use fresh camera data.
- Normal conversation may use cached memory.
- Conversation identity must never be treated as camera evidence.
"""


def normalize_text(user_text: str) -> str:
    return str(user_text or "").lower().strip()


def is_identity_camera_request(user_text: str) -> bool:
    """
    Requests that should use local face recognition / identity.
    """
    text = normalize_text(user_text)

    phrases = [
        "can you see me",
        "do you see me",
        "who do you see",
        "who am i",
        "who is in front",
        "who is in front of you",
        "who is in front of the camera",
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
    This includes broad 'what do you see' prompts.
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
    """
    text = normalize_text(user_text)

    phrases = [
        "see",
        "seeing",
        "saw",
        "look",
        "looking",
        "camera",
        "vision",
        "view",
        "describe",
        "in front of you",
        "in front of the camera",
        "face",
        "person",
        "people",
        "recognize",
        "recognise",
        "identify",
        "who am i",
        "who do you see",
        "blocked",
        "black board",
        "dark",
    ]

    return any(p in text for p in phrases)


def classify_camera_intent(user_text: str) -> str:
    """
    Returns:
      identity_camera
      scene_camera
      camera_generic
      none
    """
    if is_identity_camera_request(user_text):
        return "identity_camera"

    if is_scene_camera_request(user_text):
        return "scene_camera"

    if is_any_camera_request(user_text):
        return "camera_generic"

    return "none"
