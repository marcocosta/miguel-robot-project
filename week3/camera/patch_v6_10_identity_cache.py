from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"
text = path.read_text()

# Add session identity globals before handle_user_turn_with_cached_state.
marker = "def handle_user_turn_with_cached_state(user_text, cached_local_state):"
idx = text.find(marker)
if idx == -1:
    raise SystemExit("Could not find handle_user_turn_with_cached_state().")

session_block = r'''
SESSION_IDENTITY_OVERRIDE = {
    "person": None,
    "created_at": 0.0,
    "reason": "",
}

VISION_HARD_STALE_SECONDS = 6.0


def set_session_identity_override(person, reason="manual"):
    person = (person or "").lower().strip().replace(" ", "_")
    if person not in {"marco", "marquinho"}:
        return False

    SESSION_IDENTITY_OVERRIDE["person"] = person
    SESSION_IDENTITY_OVERRIDE["created_at"] = time.time()
    SESSION_IDENTITY_OVERRIDE["reason"] = reason
    print(f"[IDENTITY] Manual session override set to {person}: {reason}")
    return True


def clear_session_identity_override():
    SESSION_IDENTITY_OVERRIDE["person"] = None
    SESSION_IDENTITY_OVERRIDE["created_at"] = 0.0
    SESSION_IDENTITY_OVERRIDE["reason"] = ""


def is_vision_question(user_text):
    text = user_text.lower()
    phrases = [
        "do you see",
        "can you see",
        "who am i",
        "who is this",
        "do you recognize",
        "anyone in front",
        "is anyone there",
        "what do you see",
        "how many faces",
        "ghost",
        "ghosts",
    ]
    return any(p in text for p in phrases)


def apply_identity_override_to_state(local_state):
    person = SESSION_IDENTITY_OVERRIDE.get("person")
    if not person:
        return local_state

    state = dict(local_state)
    state["recognized_person"] = person
    state["manual_identity_override"] = True
    state["manual_identity_reason"] = SESSION_IDENTITY_OVERRIDE.get("reason", "")
    return state


def handle_manual_identity_command(user_text):
    text = user_text.lower()

    if "i am marquinho" in text or "i'm marquinho" in text or "treat me as marquinho" in text:
        set_session_identity_override("marquinho", "user said they are Marquinho")
        return "Got it. For this session I will treat you as Chief Engineer Marquinho, even if the camera is unsure."

    if "i am marco" in text or "i'm marco" in text or "treat me as marco" in text:
        set_session_identity_override("marco", "user said they are Marco")
        return "Got it. For this session I will treat you as Marco."

    if "clear identity override" in text or "forget manual identity" in text:
        clear_session_identity_override()
        return "Manual identity override cleared. I will rely on camera recognition again."

    return None

'''

if "SESSION_IDENTITY_OVERRIDE" not in text:
    text = text[:idx] + session_block + "\n" + text[idx:]

# Replace set_cached_local_state to clear identity when no face and avoid preserving ghost identities.
old = '''def set_cached_local_state(new_state):
    with STATE_LOCK:
        now = time.time()

        previous_person = SHARED_LOCAL_STATE.get("recognized_person")
        previous_update = SHARED_LOCAL_STATE.get("last_vision_update", 0.0)
        previous_age = now - previous_update

        # Identity hysteresis:
        # Do not drop Marco/Marquinho immediately because of one weak/uncertain frame.
        if (
            previous_person
            and not new_state.get("recognized_person")
            and new_state.get("face_detected")
            and previous_age <= IDENTITY_HYSTERESIS_SECONDS
        ):
            new_state = dict(new_state)
            new_state["recognized_person"] = previous_person
            new_state["recognizer"] = "insightface_arcface_hysteresis"
            new_state.setdefault("recognition_votes", {previous_person: 1})

        SHARED_LOCAL_STATE.clear()
        SHARED_LOCAL_STATE.update(new_state)
        SHARED_LOCAL_STATE["last_vision_update"] = now
'''

new = '''def set_cached_local_state(new_state):
    with STATE_LOCK:
        now = time.time()

        previous_person = SHARED_LOCAL_STATE.get("recognized_person")
        previous_update = SHARED_LOCAL_STATE.get("last_vision_update", 0.0)
        previous_age = now - previous_update

        # If the vision worker sees no face, clear camera identity immediately.
        # This prevents "ghost Marco" from persisting after the person leaves.
        if not new_state.get("face_detected"):
            new_state = dict(new_state)
            new_state["recognized_person"] = None
            new_state["recognition_votes"] = {}
            new_state["recognizer"] = "insightface_arcface_no_face_clear"

        # Identity hysteresis only applies to actual face-detected uncertain frames.
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

        SHARED_LOCAL_STATE.clear()
        SHARED_LOCAL_STATE.update(new_state)
        SHARED_LOCAL_STATE["last_vision_update"] = now
'''

if old in text:
    text = text.replace(old, new)
else:
    print("set_cached_local_state exact block not found; skipping replacement.")

# Insert manual identity handling and override application after mode/topic handling.
old = '''    if topic_reply:
        speak(topic_reply)
        update_conversation_memory(assistant_reply=topic_reply)
        return True

    if local_state.get("face_detected") and local_state.get("recognized_person") is None:
'''

new = '''    if topic_reply:
        speak(topic_reply)
        update_conversation_memory(assistant_reply=topic_reply)
        return True

    identity_reply = handle_manual_identity_command(user_text)
    if identity_reply:
        speak(identity_reply)
        update_conversation_memory(assistant_reply=identity_reply)
        return True

    # For vision questions, do not trust conversation-grace cached identity.
    # It may be stale when someone left or switched places.
    if is_vision_question(user_text):
        latest_state = get_cached_local_state()
        age = time.time() - latest_state.get("last_vision_update", 0.0)

        if latest_state.get("recognizer") == "conversation_grace_cached_identity" or age > VISION_HARD_STALE_SECONDS:
            local_state = latest_state
            local_state["recognized_person"] = None
            local_state["vision_warning"] = "camera state stale; need fresh vision update"
        else:
            local_state = latest_state

    local_state = apply_identity_override_to_state(local_state)

    if local_state.get("face_detected") and local_state.get("recognized_person") is None:
'''

if old in text:
    text = text.replace(old, new)
else:
    print("Could not find topic_reply insertion area.")

path.write_text(text)
print("Patched identity cache and manual override.")
