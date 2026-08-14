from v7.camera_intents import (
    classify_camera_intent,
    is_identity_camera_request,
    is_scene_camera_request,
)


def test_portuguese_identity_camera_requests() -> None:
    phrases = [
        "Miguel, voce consegue me ver?",
        "Miguel, você me reconhece?",
        "Quem voce esta vendo?",
    ]

    for phrase in phrases:
        assert is_identity_camera_request(phrase) is True
        assert classify_camera_intent(phrase) == "identity_camera"


def test_portuguese_scene_camera_requests() -> None:
    phrases = [
        "Miguel, descreva o que voce esta vendo.",
        "Miguel, descreva o que você vê.",
        "Miguel, mostre o que voce esta vendo na sua camera.",
    ]

    for phrase in phrases:
        assert is_scene_camera_request(phrase) is True
        assert classify_camera_intent(phrase) == "scene_camera"


def test_object_followups_route_to_scene_camera() -> None:
    phrases = [
        "what is this object?",
        "flash your camera and describe what is this object that i'm holding in front of you.",
        "describe the object that I am holding in front of you",
    ]

    for phrase in phrases:
        assert is_scene_camera_request(phrase) is True
        assert classify_camera_intent(phrase) == "scene_camera"


def test_log_scene_question_routes_to_scene_camera() -> None:
    phrase = "hey miguel, what you can see?"

    assert is_scene_camera_request(phrase) is True
    assert classify_camera_intent(phrase) == "scene_camera"


def test_log_identity_question_routes_to_identity_camera() -> None:
    phrase = "do you recognize this person in front of the board?"

    assert is_identity_camera_request(phrase) is True
    assert classify_camera_intent(phrase) == "identity_camera"


def test_previous_recognition_discussion_is_not_a_live_identity_request() -> None:
    phrase = (
        "Miguel, eu estou tentando aprender sobre o projeto. Essas imagens que voce ve "
        "da gente aqui, que voce me reconheceu, onde ficam armazenadas?"
    )
    assert classify_camera_intent(phrase) == "none"
