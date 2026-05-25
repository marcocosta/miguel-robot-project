from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"
text = path.read_text()

# ------------------------------------------------------------
# 1. Import V7 scene classifier into V6 file.
# ------------------------------------------------------------

if "from v7.camera_intents import is_scene_camera_request as v7_is_scene_camera_request" not in text:
    marker = "from pathlib import Path"
    idx = text.find(marker)
    if idx == -1:
        raise SystemExit("Could not find import marker.")

    insert = '''from pathlib import Path

try:
    from v7.camera_intents import is_scene_camera_request as v7_is_scene_camera_request
except Exception:
    def v7_is_scene_camera_request(user_text):
        return False
'''
    text = text.replace(marker, insert, 1)


# ------------------------------------------------------------
# 2. Force is_scene_description_request() to use V7 classifier.
# ------------------------------------------------------------

start = text.find("def is_scene_description_request(user_text):")
if start != -1:
    end = text.find("\ndef ", start + 1)
    if end == -1:
        raise SystemExit("Could not find end of is_scene_description_request().")

    new_func = '''def is_scene_description_request(user_text):
    return v7_is_scene_camera_request(user_text)

'''
    text = text[:start] + new_func + text[end+1:]
else:
    print("Warning: is_scene_description_request() not found.")


# ------------------------------------------------------------
# 3. Add helper to handle scene command before any brain/cached-state logic.
# ------------------------------------------------------------

helper_marker = "def run_v6_threaded_conversation():"
idx = text.find(helper_marker)
if idx == -1:
    raise SystemExit("Could not find run_v6_threaded_conversation().")

helper = r'''
def maybe_handle_scene_camera_request(user_text, camera_queue):
    """
    Absolute rule:
    If the user asks Miguel to see/look/describe what is in front of the camera,
    use a fresh scene frame. Do not answer from cached face state or memory.
    """
    if not v7_is_scene_camera_request(user_text):
        return False

    print("[V7 ROUTE] Enforcing scene_camera -> fresh scene description.")
    reply = describe_scene_now(camera_queue)
    speak(reply)

    try:
        update_conversation_memory(user_text=user_text, assistant_reply=reply)
    except Exception:
        pass

    return True

'''

if "def maybe_handle_scene_camera_request(" not in text:
    text = text[:idx] + helper + "\n" + text[idx:]


# ------------------------------------------------------------
# 4. Patch all transcript locations before THINKING.
# ------------------------------------------------------------

patterns = [
    '''                    empty_followup_count = 0
                    print(f"[TRANSCRIPT] {user_text}")

                    print("[THINKING] Miguel is processing.")
''',
    '''                    empty_followup_count = 0
                    print(f"[TRANSCRIPT] {user_text}")

                    # In grace mode, answer even short replies like yes/no/imagination.
                    print("[THINKING] Miguel is processing follow-up.")
''',
    '''                    empty_followup_count = 0
                    print(f"[TRANSCRIPT] {user_text}")

                    if is_scene_description_request(user_text):
                        reply = describe_scene_now(camera_queue)
                        speak(reply)
                        update_conversation_memory(user_text=user_text, assistant_reply=reply)
                        last_reply_time = time.time()
                        continue

                    print("[THINKING] Miguel is processing.")
''',
    '''                    empty_followup_count = 0
                    print(f"[TRANSCRIPT] {user_text}")

                    if is_scene_description_request(user_text):
                        reply = describe_scene_now(camera_queue)
                        speak(reply)
                        update_conversation_memory(user_text=user_text, assistant_reply=reply)
                        last_reply_time = time.time()
                        continue

                    # In grace mode, answer even short replies like yes/no/imagination.
                    print("[THINKING] Miguel is processing follow-up.")
''',
]

replacements = [
    '''                    empty_followup_count = 0
                    print(f"[TRANSCRIPT] {user_text}")

                    if maybe_handle_scene_camera_request(user_text, camera_queue):
                        last_reply_time = time.time()
                        continue

                    print("[THINKING] Miguel is processing.")
''',
    '''                    empty_followup_count = 0
                    print(f"[TRANSCRIPT] {user_text}")

                    if maybe_handle_scene_camera_request(user_text, camera_queue):
                        last_reply_time = time.time()
                        continue

                    # In grace mode, answer even short replies like yes/no/imagination.
                    print("[THINKING] Miguel is processing follow-up.")
''',
    '''                    empty_followup_count = 0
                    print(f"[TRANSCRIPT] {user_text}")

                    if maybe_handle_scene_camera_request(user_text, camera_queue):
                        last_reply_time = time.time()
                        continue

                    print("[THINKING] Miguel is processing.")
''',
    '''                    empty_followup_count = 0
                    print(f"[TRANSCRIPT] {user_text}")

                    if maybe_handle_scene_camera_request(user_text, camera_queue):
                        last_reply_time = time.time()
                        continue

                    # In grace mode, answer even short replies like yes/no/imagination.
                    print("[THINKING] Miguel is processing follow-up.")
''',
]

for old, new in zip(patterns, replacements):
    if old in text:
        text = text.replace(old, new)

path.write_text(text)
print("Patched V7 scene routing enforcement into V6 loop.")
