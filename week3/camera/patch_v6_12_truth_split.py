from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"
text = path.read_text()

# -------------------------------------------------------------------
# 1) Fix make_last_known_person_state().
# It must NOT claim face_detected=True or recognized_person=person.
# It is conversation context only, not camera truth.
# -------------------------------------------------------------------

old = '''def make_last_known_person_state(person_name):
    return {
        "face_detected": True,
        "face_count": 1,
        "face_position": "center",
        "recognized_person": person_name,
        "recognition_score": None,
        "recognition_margin": None,
        "recognition_votes": {person_name: 1} if person_name else {},
        "recognition_scores": {},
        "recognizer": "conversation_grace_cached_identity",
        "last_vision_update": time.time(),
    }
'''

new = '''def make_last_known_person_state(person_name):
    return {
        # Conversation identity only.
        # This is NOT camera truth and must not be used to say "I see you."
        "face_detected": False,
        "face_count": 0,
        "face_position": "unknown",
        "recognized_person": None,
        "conversation_person": person_name,
        "recognition_score": None,
        "recognition_margin": None,
        "recognition_votes": {},
        "recognition_scores": {},
        "recognizer": "conversation_grace_cached_identity",
        "camera_truth": "unknown_grace_identity_only",
        "last_vision_update": time.time(),
    }
'''

if old in text:
    text = text.replace(old, new)
else:
    print("Warning: make_last_known_person_state block not found exactly.")


# -------------------------------------------------------------------
# 2) Prevent hysteresis from converting fresh no-face / unconfirmed scans
# back into recognized_person.
# -------------------------------------------------------------------

old = '''        # Identity hysteresis only applies to actual face-detected uncertain frames.
        # Do not use hysteresis for no-face frames.
        elif (
            previous_person
            and not new_state.get("recognized_person")
            and new_state.get("face_detected")
            and previous_age <= IDENTITY_HYSTERESIS_SECONDS
        ):
            new_state = dict(new_state)
            new_state["recognized_person"] = previous_person
            new_state["recognizer"] = "insightface_arcface_hysteresis"
            new_state.setdefault("recognition_votes", {previous_person: 1})
'''

new = '''        # Identity hysteresis only applies to normal live camera frames.
        # Do not apply hysteresis to fresh vision questions or unconfirmed face-like patterns.
        elif (
            previous_person
            and not new_state.get("recognized_person")
            and new_state.get("face_detected")
            and not new_state.get("fresh_scan_for_question")
            and not new_state.get("unconfirmed_face_like_pattern")
            and previous_age <= IDENTITY_HYSTERESIS_SECONDS
        ):
            new_state = dict(new_state)
            new_state["recognized_person"] = previous_person
            new_state["recognizer"] = "insightface_arcface_hysteresis"
            new_state.setdefault("recognition_votes", {previous_person: 1})
'''

if old in text:
    text = text.replace(old, new)
else:
    print("Warning: hysteresis block not found exactly.")


# -------------------------------------------------------------------
# 3) Add helper to detect unreliable visual state.
# -------------------------------------------------------------------

helper = r'''
def is_camera_truth_state(local_state):
    recognizer = local_state.get("recognizer", "")
    if recognizer == "conversation_grace_cached_identity":
        return False
    if local_state.get("camera_truth") == "unknown_grace_identity_only":
        return False
    return True


def spoken_person_name_from_state(local_state):
    return (
        local_state.get("recognized_person")
        or local_state.get("conversation_person")
        or "friend"
    )

'''

marker = "def handle_user_turn_with_cached_state(user_text, cached_local_state):"
idx = text.find(marker)

if idx == -1:
    raise SystemExit("Could not find handle_user_turn_with_cached_state().")

if "def is_camera_truth_state(" not in text:
    text = text[:idx] + helper + "\n" + text[idx:]


# -------------------------------------------------------------------
# 4) In vision questions, never trust conversation_grace_cached_identity.
# If the state is grace-only, answer with no confirmed camera face.
# -------------------------------------------------------------------

old = '''    if is_vision_question(user_text):
        if not local_state.get("face_detected"):
            reply = "I do not see a confirmed face right now."
            speak(reply)
            update_conversation_memory(assistant_reply=reply)
            return True
'''

new = '''    if is_vision_question(user_text):
        if not is_camera_truth_state(local_state):
            reply = "I do not have a confirmed live camera view of a face right now."
            speak(reply)
            update_conversation_memory(assistant_reply=reply)
            return True

        if not local_state.get("face_detected"):
            reply = "I do not see a confirmed face right now."
            speak(reply)
            update_conversation_memory(assistant_reply=reply)
            return True
'''

if old in text:
    text = text.replace(old, new)
else:
    print("Warning: vision question no-face block not found exactly.")


# -------------------------------------------------------------------
# 5) Remove visual claims before sending to cloud when state is grace-only.
# This stops replies like "I see you centered" when no fresh camera truth exists.
# -------------------------------------------------------------------

old = '''    local_skill_reply = maybe_handle_local_skill(user_text, local_state)
'''

new = '''    if not is_camera_truth_state(local_state):
        local_state = dict(local_state)
        local_state["face_detected"] = False
        local_state["face_count"] = 0
        local_state["face_position"] = "unknown"
        local_state["recognized_person"] = None
        local_state["vision_warning"] = "No confirmed live camera truth; do not say you see or recognize anyone."

    local_skill_reply = maybe_handle_local_skill(user_text, local_state)
'''

if old in text:
    text = text.replace(old, new, 1)
else:
    print("Warning: local_skill insertion marker not found.")

path.write_text(text)
print("Patched V6.12: separated conversation identity from camera truth.")
