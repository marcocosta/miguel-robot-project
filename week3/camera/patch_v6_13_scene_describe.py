from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"
text = path.read_text()

# ------------------------------------------------------------
# 1) Add imports needed for image encoding.
# ------------------------------------------------------------
if "import base64" not in text:
    text = text.replace("import os\n", "import os\nimport base64\n")

# ------------------------------------------------------------
# 2) Add scene description helpers before run_v6_threaded_conversation().
# ------------------------------------------------------------
marker = "def run_v6_threaded_conversation():"
idx = text.find(marker)
if idx == -1:
    raise SystemExit("Could not find run_v6_threaded_conversation().")

helper = r'''
def is_scene_description_request(user_text):
    text = user_text.lower()

    phrases = [
        "describe what you see",
        "describe what you are seeing",
        "describe the scene",
        "describe in front of you",
        "what is in front of you",
        "what's in front of you",
        "what are you looking at",
        "look around and describe",
        "describe your camera view",
        "describe your view",
    ]

    return any(p in text for p in phrases)


def capture_fresh_camera_frame(camera_queue, save_debug=True):
    """
    Capture one fresh frame from the OAK camera.
    This is scene-level vision, not face identity.
    """
    with CAMERA_LOCK:
        flush_camera_queue(camera_queue, drain_frames=24)
        time.sleep(0.45)
        flush_camera_queue(camera_queue, drain_frames=12)

        msg = camera_queue.get()
        frame = msg.getCvFrame()

    out_path = None

    if save_debug:
        out_dir = Path.home() / "robot-project/week3/debug_snapshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"scene_describe_{int(time.time())}.jpg"

        try:
            cv2.imwrite(str(out_path), frame)
            print(f"[SCENE] Saved fresh scene frame: {out_path}")
        except Exception as e:
            print("[SCENE] Could not save scene frame:", e)

    return frame, out_path


def describe_scene_with_openai(image_path):
    """
    Use cloud vision for natural scene description.
    Keep answer short because Miguel is speaking it aloud.
    """
    if not image_path or not Path(image_path).exists():
        return "I could not capture a fresh camera image to describe."

    try:
        image_bytes = Path(image_path).read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        response = client.responses.create(
            model=os.getenv("MIGUEL_VISION_MODEL", "gpt-4o-mini"),
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are Miguel, a small father-son robot. "
                                "Describe the camera image in one or two short spoken sentences. "
                                "Be honest and cautious. If the image is blocked, dark, blurry, or unclear, say that. "
                                "Do not identify a person's real identity from the image. "
                                "You may say 'a person' or 'a face' only if visually obvious."
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    ],
                }
            ],
        )

        reply = getattr(response, "output_text", None)

        if not reply:
            # Fallback extraction for SDK variations.
            parts = []
            for item in getattr(response, "output", []) or []:
                for content in getattr(item, "content", []) or []:
                    txt = getattr(content, "text", None)
                    if txt:
                        parts.append(txt)
            reply = " ".join(parts).strip()

        return reply or "I looked, but I could not produce a clear scene description."

    except Exception as e:
        print("[SCENE] OpenAI vision description failed:", e)
        return "I captured an image, but my cloud vision description failed."


def describe_scene_now(camera_queue):
    print("[SCENE] Fresh scene description requested.")

    try:
        frame, image_path = capture_fresh_camera_frame(camera_queue, save_debug=True)
    except Exception as e:
        print("[SCENE] Fresh frame capture failed:", e)
        return "I could not capture a fresh camera frame right now."

    return describe_scene_with_openai(image_path)

'''

if "def is_scene_description_request(" not in text:
    text = text[:idx] + helper + "\n" + text[idx:]


# ------------------------------------------------------------
# 3) Route scene description requests before normal brain handling.
# We patch all main-loop handle calls by inserting a scene branch after transcript.
# ------------------------------------------------------------

old = '''                    empty_followup_count = 0
                    print(f"[TRANSCRIPT] {user_text}")

                    print("[THINKING] Miguel is processing.")
'''

new = '''                    empty_followup_count = 0
                    print(f"[TRANSCRIPT] {user_text}")

                    if is_scene_description_request(user_text):
                        reply = describe_scene_now(camera_queue)
                        speak(reply)
                        update_conversation_memory(user_text=user_text, assistant_reply=reply)
                        last_reply_time = time.time()
                        continue

                    print("[THINKING] Miguel is processing.")
'''

if old in text:
    text = text.replace(old, new)
else:
    print("Warning: main THINKING transcript block not found exactly.")


old2 = '''                    empty_followup_count = 0
                    print(f"[TRANSCRIPT] {user_text}")

                    # In grace mode, answer even short replies like yes/no/imagination.
                    print("[THINKING] Miguel is processing follow-up.")
'''

new2 = '''                    empty_followup_count = 0
                    print(f"[TRANSCRIPT] {user_text}")

                    if is_scene_description_request(user_text):
                        reply = describe_scene_now(camera_queue)
                        speak(reply)
                        update_conversation_memory(user_text=user_text, assistant_reply=reply)
                        last_reply_time = time.time()
                        continue

                    # In grace mode, answer even short replies like yes/no/imagination.
                    print("[THINKING] Miguel is processing follow-up.")
'''

if old2 in text:
    text = text.replace(old2, new2)
else:
    print("Warning: follow-up THINKING transcript block not found exactly.")

path.write_text(text)
print("Patched V6.13 scene description skill.")
