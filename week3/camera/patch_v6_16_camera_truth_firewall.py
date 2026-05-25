from pathlib import Path
import re

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"
text = path.read_text()

# ============================================================
# 1. Add camera-truth firewall helpers.
# ============================================================

marker = "def handle_user_turn_with_cached_state(user_text, cached_local_state):"
idx = text.find(marker)

if idx == -1:
    raise SystemExit("Could not find handle_user_turn_with_cached_state().")

helpers = r'''
def is_camera_memory_only_state(local_state):
    return (
        local_state.get("recognizer") == "conversation_grace_cached_identity"
        or local_state.get("camera_truth") == "unknown_grace_identity_only"
    )


def sanitize_camera_memory_state(local_state):
    """
    Conversation memory is allowed to remember who was speaking,
    but it must never become camera evidence.
    """
    if not is_camera_memory_only_state(local_state):
        return local_state

    state = dict(local_state)
    state["conversation_person"] = state.get("conversation_person") or state.get("recognized_person")
    state["face_detected"] = False
    state["face_count"] = 0
    state["face_position"] = "unknown"
    state["recognized_person"] = None
    state["recognition_score"] = None
    state["recognition_margin"] = None
    state["recognition_votes"] = {}
    state["recognition_scores"] = {}
    state["camera_truth"] = "conversation_memory_only_not_camera"
    state["vision_warning"] = (
        "Conversation identity only. Do not say the camera sees a face, "
        "a person, Marco, or Marquinho."
    )
    return state


def is_camera_related_request(user_text):
    text = user_text.lower()

    phrases = [
        "what do you see",
        "what can you see",
        "who do you see",
        "who am i",
        "can you see me",
        "do you see me",
        "do you see anyone",
        "is anyone there",
        "in front of you",
        "in front of the camera",
        "camera",
        "vision",
        "blocked",
        "black board",
        "box in front",
        "face",
        "recognize me",
        "recognise me",
        "identify me",
        "describe what you see",
        "describe the scene",
    ]

    return any(p in text for p in phrases)


def reply_makes_camera_claim(reply):
    text = reply.lower()

    claim_phrases = [
        "i see",
        "i can see",
        "my camera sees",
        "camera sees",
        "i recognize",
        "i recognise",
        "i detect a face",
        "i see one face",
        "i see a face",
        "face centered",
        "face in the center",
        "in front of me",
        "in front of my camera",
        "looks like marco",
        "looks like marquinho",
    ]

    return any(p in text for p in claim_phrases)


def camera_truth_allows_visual_claim(local_state):
    if is_camera_memory_only_state(local_state):
        return False

    if local_state.get("fresh_scan_for_question") and not local_state.get("face_detected"):
        return False

    if not local_state.get("face_detected"):
        return False

    return True


def safe_reply_after_camera_firewall(reply, local_state):
    """
    Final guard before speech. If the model tries to claim visual/camera truth
    from memory-only state, override it.
    """
    if reply_makes_camera_claim(reply) and not camera_truth_allows_visual_claim(local_state):
        return (
            "I cannot confirm that with my live camera right now. "
            "I may remember who I was talking to, but my camera is not confirming a face."
        )

    return reply

'''

if "def sanitize_camera_memory_state(" not in text:
    text = text[:idx] + helpers + "\n" + text[idx:]


# ============================================================
# 2. Harden make_last_known_person_state if present.
# ============================================================

pattern = r'''def make_last_known_person_state\(person_name\):\n    return \{.*?\n    \}\n'''
replacement = '''def make_last_known_person_state(person_name):
    return {
        # Conversation identity only. This is NOT camera truth.
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
        "vision_warning": "Conversation identity only. Do not say the camera sees anyone.",
        "last_vision_update": time.time(),
    }
'''

text, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
if count == 0:
    print("Warning: make_last_known_person_state() not found or already changed.")


# ============================================================
# 3. Sanitize cached state at the start of handle_user_turn.
# ============================================================

old = '''    local_state = cached_local_state
    print("[BRAIN] Using cached local state:", local_state)
'''

new = '''    local_state = sanitize_camera_memory_state(cached_local_state)
    print("[BRAIN] Using cached local state:", local_state)
'''

if old in text:
    text = text.replace(old, new, 1)
else:
    print("Warning: local_state assignment block not found.")


# ============================================================
# 4. Broaden fresh camera scan trigger.
# get_state_for_user_turn should refresh camera for any camera-related request.
# ============================================================

old = '''        if is_vision_question(user_text):
            print("[VISION] Fresh scan requested for vision question.")
'''

new = '''        if is_vision_question(user_text) or is_camera_related_request(user_text):
            print("[VISION] Fresh scan requested for camera-related question.")
'''

if old in text:
    text = text.replace(old, new)
else:
    print("Warning: get_state_for_user_turn fresh scan trigger not found.")


# ============================================================
# 5. Prevent hysteresis from turning unconfirmed or fresh no-face scans
# back into old identity.
# ============================================================

old = '''        elif (
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

new = '''        elif (
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
    print("Warning: hysteresis block not found or already patched.")


# ============================================================
# 6. Before any cloud reply is spoken, enforce camera truth firewall.
# ============================================================

old = '''    speak(reply)
    update_conversation_memory(assistant_reply=reply)
'''

new = '''    reply = safe_reply_after_camera_firewall(reply, local_state)
    speak(reply)
    update_conversation_memory(assistant_reply=reply)
'''

if old in text:
    text = text.replace(old, new)
else:
    print("Warning: speak(reply) cloud block not found.")


# ============================================================
# 7. Add strong cloud rule.
# ============================================================

rule_anchor = '''- Output plain spoken text only. Do not use Markdown, bullets, asterisks, code formatting, tables, or emojis.
'''
rule_add = '''- Never claim the camera sees a face/person unless local_state.face_detected is true and local_state.recognizer is not conversation_grace_cached_identity.
- If local_state.camera_truth says conversation_memory_only_not_camera, say you remember the conversation but cannot confirm with the camera.
'''

if rule_anchor in text and rule_add not in text:
    text = text.replace(rule_anchor, rule_anchor + rule_add)

path.write_text(text)
print("Patched V6.16 camera truth firewall.")
