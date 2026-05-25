from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"
text = path.read_text()

# -------------------------------------------------------------------
# 1. Add centralized vision intent helpers.
# -------------------------------------------------------------------

marker = "def handle_user_turn_with_cached_state(user_text, cached_local_state):"
idx = text.find(marker)

if idx == -1:
    raise SystemExit("Could not find handle_user_turn_with_cached_state().")

helpers = r'''
def is_any_camera_request(user_text):
    """
    Any request about seeing, looking, camera, vision, faces, recognition,
    or describing the robot's view must use fresh camera data, never memory.
    """
    text = user_text.lower()

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
        "what is there",
        "what's there",
        "around you",
        "face",
        "person",
        "people",
        "recognize",
        "recognise",
        "identify",
        "who am i",
        "who is there",
        "who do you see",
        "can you see me",
        "do you see me",
        "blocked",
        "black board",
        "dark",
    ]

    return any(p in text for p in phrases)


def is_identity_camera_request(user_text):
    """
    Requests that should use local face recognition.
    """
    text = user_text.lower()

    phrases = [
        "can you see me",
        "do you see me",
        "who do you see",
        "who am i",
        "who is in front",
        "do you recognize me",
        "recognize me",
        "recognise me",
        "identify me",
        "whose face",
        "which person",
    ]

    return any(p in text for p in phrases)


def is_scene_camera_request(user_text):
    """
    Requests that should use fresh image scene description.
    This includes broad 'what do you see' prompts.
    """
    text = user_text.lower()

    # Identity requests take priority and are handled separately.
    if is_identity_camera_request(user_text):
        return False

    phrases = [
        "describe",
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
    ]

    return any(p in text for p in phrases)


def build_identity_reply_from_fresh_state(local_state):
    """
    Speak only from fresh face-recognition camera state.
    """
    if local_state.get("recognizer") == "conversation_grace_cached_identity":
        return "I cannot answer that from memory. I need a fresh camera view."

    if not local_state.get("face_detected"):
        return "I checked the camera, and I do not see a confirmed face right now."

    if local_state.get("unconfirmed_face_like_pattern"):
        return (
            "I checked the camera and detected a face-like pattern, "
            "but I cannot confirm it is a real person or identify who it is."
        )

    person = local_state.get("recognized_person")
    if person:
        score = local_state.get("recognition_score")
        if score is not None:
            return f"I checked the camera and recognize one face as {person}, with confidence about {score:.2f}."
        return f"I checked the camera and recognize one face as {person}."

    return "I checked the camera and see a face, but I cannot identify who it is."

'''

if "def is_any_camera_request(" not in text:
    text = text[:idx] + helpers + "\n" + text[idx:]


# -------------------------------------------------------------------
# 2. Make scene request detector delegate to the new centralized helper.
# -------------------------------------------------------------------

start = text.find("def is_scene_description_request(user_text):")
if start != -1:
    end = text.find("\ndef ", start + 1)
    if end == -1:
        raise SystemExit("Could not find end of is_scene_description_request().")

    new_func = '''def is_scene_description_request(user_text):
    return is_scene_camera_request(user_text)

'''
    text = text[:start] + new_func + text[end+1:]


# -------------------------------------------------------------------
# 3. Fresh scan trigger must catch any camera request unless scene description
# has already captured its own frame.
# -------------------------------------------------------------------

old = '''        if (not is_scene_description_request(user_text)) and (is_vision_question(user_text) or is_camera_related_request(user_text)):
            print("[VISION] Fresh scan requested for camera-related question.")
'''

new = '''        if (not is_scene_description_request(user_text)) and is_any_camera_request(user_text):
            print("[VISION] Fresh scan requested for camera-related question.")
'''

if old in text:
    text = text.replace(old, new)
else:
    print("Warning: exact fresh-scan trigger not found. Trying fallback.")
    text = text.replace(
        '''        if is_vision_question(user_text) or is_camera_related_request(user_text):
            print("[VISION] Fresh scan requested for camera-related question.")
''',
        new
    )


# -------------------------------------------------------------------
# 4. Route camera requests before normal brain handling in both transcript paths.
# -------------------------------------------------------------------

def patch_transcript_block(block_label):
    global text
    old = f'''                    empty_followup_count = 0
                    print(f"[TRANSCRIPT] {{user_text}}")

                    print("[THINKING] Miguel is processing{block_label}.")
'''
    new = f'''                    empty_followup_count = 0
                    print(f"[TRANSCRIPT] {{user_text}}")

                    if is_scene_camera_request(user_text):
                        reply = describe_scene_now(camera_queue)
                        speak(reply)
                        update_conversation_memory(user_text=user_text, assistant_reply=reply)
                        last_reply_time = time.time()
                        continue

                    print("[THINKING] Miguel is processing{block_label}.")
'''
    if old in text:
        text = text.replace(old, new)

patch_transcript_block("")
patch_transcript_block(" follow-up")

# Some versions have comments between transcript and thinking; add simpler fallback if needed.
text = text.replace(
'''                    # In grace mode, answer even short replies like yes/no/imagination.
                    print("[THINKING] Miguel is processing follow-up.")
''',
'''                    if is_scene_camera_request(user_text):
                        reply = describe_scene_now(camera_queue)
                        speak(reply)
                        update_conversation_memory(user_text=user_text, assistant_reply=reply)
                        last_reply_time = time.time()
                        continue

                    # In grace mode, answer even short replies like yes/no/imagination.
                    print("[THINKING] Miguel is processing follow-up.")
'''
)


# -------------------------------------------------------------------
# 5. Inside handle_user_turn, identity camera requests get a direct camera-truth reply.
# -------------------------------------------------------------------

anchor = '''    if is_vision_question(user_text):
'''

insert = '''    if is_identity_camera_request(user_text):
        reply = build_identity_reply_from_fresh_state(local_state)
        speak(reply)
        update_conversation_memory(assistant_reply=reply)
        return True

    if is_vision_question(user_text):
'''

if anchor in text and "build_identity_reply_from_fresh_state(local_state)" not in text[text.find("def handle_user_turn_with_cached_state"):]:
    text = text.replace(anchor, insert, 1)
else:
    print("Warning: identity camera insertion may already exist or anchor missing.")


# -------------------------------------------------------------------
# 6. Stronger cloud rule: memory can never answer camera requests.
# -------------------------------------------------------------------

rule_anchor = '''- Never claim the camera sees a face/person unless local_state.face_detected is true and local_state.recognizer is not conversation_grace_cached_identity.
'''
rule_add = '''- For any request about seeing, looking, camera, vision, faces, or describing what is in front of you, do not answer from memory. Require fresh camera data.
'''

if rule_anchor in text and rule_add not in text:
    text = text.replace(rule_anchor, rule_anchor + rule_add)

path.write_text(text)
print("Patched V6.18 camera-first vision routing.")
