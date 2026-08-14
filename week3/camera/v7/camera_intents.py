"""
Miguel V7 camera intent routing.

Policy:
- Any camera/vision action request about seeing, looking, camera, vision, recognition,
  or describing the robot view must use fresh camera data.
- Normal conversation may use cached memory.
- Conversation identity must never be treated as camera evidence.
"""

import unicodedata


def normalize_text(user_text: str) -> str:
    text = unicodedata.normalize("NFKD", str(user_text or "").lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.strip().split())


def is_identity_camera_request(user_text: str) -> bool:
    """
    Requests that should use local face recognition / identity.
    Includes imperfect ASR variants like "who you see".
    """
    text = normalize_text(user_text)

    # A mention of an earlier recognition event can be part of a privacy or
    # project question, rather than a request to identify the current frame.
    retrospective_markers = (
        "que voce me reconheceu",
        "quando voce me reconheceu",
        "como voce me reconheceu",
        "you recognized me",
        "when you recognized me",
        "how you recognized me",
    )
    explanatory_markers = (
        "essas imagens",
        "as imagens",
        "minhas imagens",
        "my images",
        "these images",
        "where are the images",
        "onde ficam as imagens",
        "o que acontece com",
        "what happens to",
    )
    if any(marker in text for marker in retrospective_markers) and (
        any(marker in text for marker in explanatory_markers) or len(text.split()) >= 14
    ):
        return False

    phrases = [
        "can you see me",
        "do you see me",
        "who do you see",
        "who can you see",
        "who you see",
        "who are you seeing",
        "who is there",
        "who are they",
        "who are those two",
        "who are those two people",
        "who are these two",
        "who are these two people",
        "who are those people",
        "who are these people",
        "who are the people",
        "who are the two people",
        "who are both people",
        "who are both of us",
        "who is this face",
        "who is that face",
        "who is his face",
        "who is her face",
        "who is this child",
        "who is that child",
        "identify this face",
        "whose face is this",
        "who is in front",
        "who is in front of you",
        "who is in front of the camera",
        "person in front of the board",
        "person in front of board",
        "who is in the frame",
        "who are in the frame",
        "who am i",
        "do you recognize me",
        "do you recognise me",
        "do you recognize this person",
        "do you recognise this person",
        "recognize this person",
        "recognise this person",
        "recognize that person",
        "recognise that person",
        "do you recognize us",
        "do you recognise us",
        "recognize us",
        "recognise us",
        "recognize both",
        "recognise both",
        "recognize me",
        "recognise me",
        "identify me",
        "identify us",
        "identify the people",
        "identify those people",
        "identify these people",
        "who else is there",
        "anybody else",
        "see any other face",
        "any other face besides me",
        "other face besides me",
        "anyone behind me",
        "anybody behind me",
        "whose face",
        "which person",
        "which face",
        "is it marco",
        "is it marquinho",
        "can you see marco",
        "can you see marquinho",
        "can you see marquinho in the back",
        "can you see marco in the back",
        "voce consegue me ver",
        "consegue me ver",
        "voce pode me ver",
        "voce me ve",
        "quem voce ve",
        "quem voce esta vendo",
        "quem esta ai",
        "quem esta na sua frente",
        "quem esta na camera",
        "quem esta no quadro",
        "quem sou eu",
        "voce me reconhece",
        "voce reconhece a gente",
        "reconheca meu rosto",
        "identifique meu rosto",
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
        "what you can see",
        "what are you seeing",
        "do you see",
        "can you see",
        "what is your camera seeing",
        "what's your camera seeing",
        "what is in front of you",
        "what's in front of you",
        "what is in front of the camera",
        "what's in front of the camera",
        "what is this object",
        "what's this object",
        "what is that object",
        "what's that object",
        "what is the object",
        "what object is this",
        "describe the object",
        "object in front of you",
        "object that i am holding",
        "object that im holding",
        "object that i'm holding",
        "holding in front of you",
        "look around",
        "look and tell me",
        "check out your camera",
        "check your camera",
        "check your camera again",
        "refresh your camera",
        "flash your camera",
        "camera view",
        "your view",
        "what is there",
        "what's there",
        "what are those people doing",
        "what are these people doing",
        "what are those two people doing",
        "what are these two people doing",
        "what are they doing",
        "what is happening in front of you",
        "what is happening in the frame",
        "is the camera blocked",
        "is your vision blocked",
        "black board",
        "blocked camera",
        "descreva o que voce esta vendo",
        "descreva o que voce ve",
        "descreva sua camera",
        "descreva a cena",
        "o que voce esta vendo",
        "o que voce ve",
        "mostre o que voce esta vendo",
        "mostre o que voce ve",
        "olhe ao redor",
        "olha ao redor",
        "verifique sua camera",
        "atualize sua camera",
        "a camera esta bloqueada",
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
        "what you can see",
        "what are you seeing",
        "do you see",
        "can you see",
        "look at",
        "look around",
        "look and tell me",
        "check out your camera",
        "check your camera",
        "check your camera again",
        "refresh your camera",
        "flash your camera",
        "describe what you see",
        "describe what you are seeing",
        "what is this object",
        "what's this object",
        "what is that object",
        "what's that object",
        "what is the object",
        "what object is this",
        "describe the object",
        "object in front of you",
        "object that i am holding",
        "object that im holding",
        "object that i'm holding",
        "holding in front of you",
        "who do you see",
        "who can you see",
        "who you see",
        "who are they",
        "who are those people",
        "who are those two people",
        "who are these people",
        "who are the people",
        "who is this face",
        "who is that face",
        "who is his face",
        "who is her face",
        "who is this child",
        "who is that child",
        "identify this face",
        "whose face is this",
        "who am i",
        "who is in front",
        "who is in the frame",
        "who else is there",
        "see any other face",
        "any other face besides me",
        "other face besides me",
        "anyone behind me",
        "anybody behind me",
        "recognize me",
        "recognise me",
        "recognize this person",
        "recognise this person",
        "person in front of the board",
        "person in front of board",
        "recognize us",
        "recognise us",
        "identify me",
        "identify the people",
        "is the camera blocked",
        "is your vision blocked",
        "detect the camera",
        "what are those people doing",
        "what are those two people doing",
        "what are they doing",
        "blocked camera",
        "voce consegue me ver",
        "consegue me ver",
        "quem voce ve",
        "quem voce esta vendo",
        "voce me reconhece",
        "descreva o que voce esta vendo",
        "descreva o que voce ve",
        "o que voce esta vendo",
        "o que voce ve",
        "mostre o que voce esta vendo",
        "mostre o que voce ve",
        "olhe ao redor",
        "olha ao redor",
        "verifique sua camera",
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

    # The same retrospective privacy/project question may also contain a
    # broad scene phrase such as "imagens que voce ve".  Suppress all live
    # camera routing for that compound turn, not only identity routing.
    if any(
        marker in text
        for marker in {
            "que voce me reconheceu",
            "quando voce me reconheceu",
            "como voce me reconheceu",
            "you recognized me",
            "when you recognized me",
            "how you recognized me",
        }
    ) and (
        any(
            marker in text
            for marker in {
                "essas imagens",
                "as imagens",
                "minhas imagens",
                "my images",
                "these images",
                "where are the images",
                "onde ficam as imagens",
                "o que acontece com",
                "what happens to",
            }
        )
        or len(text.split()) >= 14
    ):
        return "none"

    if is_identity_camera_request(text):
        return "identity_camera"

    if is_scene_camera_request(text):
        return "scene_camera"

    if is_any_camera_request(text):
        return "camera_generic"

    return "none"
