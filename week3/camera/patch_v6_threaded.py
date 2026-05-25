from pathlib import Path
import re

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"
text = path.read_text()

# Add threading import if missing.
if "import threading" not in text:
    text = text.replace("import time\n", "import time\nimport threading\n")

# Tune audio threshold higher to avoid weak/background captures.
text = re.sub(
    r"SPEECH_RMS_THRESHOLD\s*=\s*\d+",
    "SPEECH_RMS_THRESHOLD = 900",
    text,
)

# Reduce per-scan InsightFace frames. Vision will run in background now.
text = re.sub(
    r"INSIGHT_SCAN_FRAMES\s*=\s*\d+",
    "INSIGHT_SCAN_FRAMES = 2",
    text,
)

marker = "# ============================================================\n# V4 Natural Conversation Mode"
idx = text.find(marker)

if idx == -1:
    raise SystemExit("Could not find V4 Natural Conversation Mode marker.")

prefix = text[:idx]

v6_block = r'''
# ============================================================
# V6 Threaded Natural Conversation Mode
# Background vision cache + main audio/brain/speaker loop
# ============================================================

import threading

ACTIVE_FOLLOWUP_SECONDS = 30
VISION_UPDATE_SECONDS = 2.0
VISION_STALE_SECONDS = 8.0

QUESTION_WORDS = [
    "what", "who", "when", "where", "why", "how",
    "can you", "could you", "do you", "are you", "is it",
    "will you", "would you"
]

REQUEST_WORDS = [
    "tell", "say", "look", "check", "explain", "calculate",
    "show", "remember", "describe", "give", "help", "find"
]

DIRECT_ADDRESS_WORDS = [
    "miguel", "robot", "mission control"
]

STOP_EVENT = threading.Event()
STATE_LOCK = threading.Lock()

SHARED_LOCAL_STATE = {
    "face_detected": False,
    "face_count": 0,
    "face_position": "none",
    "recognized_person": None,
    "recognition_score": None,
    "recognition_margin": None,
    "recognition_votes": {},
    "recognition_scores": {},
    "recognizer": "insightface_arcface",
    "last_vision_update": 0.0,
}


def text_has_any(text, phrases):
    text = text.lower()
    return any(p in text for p in phrases)


def get_cached_local_state():
    with STATE_LOCK:
        return dict(SHARED_LOCAL_STATE)


def set_cached_local_state(new_state):
    with STATE_LOCK:
        SHARED_LOCAL_STATE.clear()
        SHARED_LOCAL_STATE.update(new_state)
        SHARED_LOCAL_STATE["last_vision_update"] = time.time()


def vision_worker(camera_queue, pipeline):
    print("[VISION] Background vision worker started.")

    while not STOP_EVENT.is_set() and pipeline.isRunning():
        try:
            state = detect_face_state(camera_queue)
            set_cached_local_state(state)

            person = state.get("recognized_person")
            face_detected = state.get("face_detected")
            position = state.get("face_position")

            if person:
                print(f"[VISION] recognized={person} position={position} score={state.get('recognition_score')}")
            elif face_detected:
                print(f"[VISION] face detected position={position}, unknown")
            else:
                print("[VISION] no face")

        except Exception as e:
            print("[VISION] error:", e)

        time.sleep(VISION_UPDATE_SECONDS)

    print("[VISION] Background vision worker stopped.")


def should_respond_naturally(user_text, familiar_person=None, active_followup=False):
    text = user_text.lower().strip()

    if not text:
        return False

    words = set(re.findall(r"\b\w+\b", text))

    if words & {"quit", "stop", "exit", "shutdown"}:
        return True

    if text_has_any(text, DIRECT_ADDRESS_WORDS):
        return True

    if text_has_any(text, QUESTION_WORDS):
        return True

    if text_has_any(text, REQUEST_WORDS):
        return True

    if familiar_person and active_followup and len(words) >= 2:
        return True

    return False


def is_vision_state_fresh(local_state):
    last_update = local_state.get("last_vision_update", 0.0)
    return (time.time() - last_update) <= VISION_STALE_SECONDS


def handle_user_turn_with_cached_state(user_text, cached_local_state):
    words = set(re.findall(r"\b\w+\b", user_text.lower()))

    if not user_text:
        print("[AUDIO] Empty transcript ignored.")
        return True

    if words & {"quit", "stop", "exit", "shutdown"}:
        speak("Miguel is shutting down conversation mode. Mission saved.")
        return False

    local_state = cached_local_state
    print("[BRAIN] Using cached local state:", local_state)

    if local_state.get("face_detected") and local_state.get("recognized_person") is None:
        print("[BRAIN] Unknown face detected. Auto-enrollment remains disabled during tuning.")

    local_skill_reply = maybe_handle_local_skill(user_text, local_state)
    if local_skill_reply:
        speak(local_skill_reply)
        return True

    if local_state.get("recognized_person") in CUSTOM_GREETINGS and words & {"hello", "hi", "hey", "look", "see", "who"}:
        speak(CUSTOM_GREETINGS[local_state["recognized_person"]])
        return True

    try:
        reply = ask_cloud_brain(user_text, local_state)
    except Exception as e:
        print("[BRAIN] OpenAI API error:", e)
        reply = "My cloud brain is not reachable right now, but my local systems are still online."

    speak(reply)
    return True


def run_v6_threaded_conversation():
    print(f"Starting {ROBOT_NAME} Cloud Brain V6 - Threaded Natural Conversation Mode.")
    print_loaded_faces()
    print("Mode:")
    print("  - Background vision recognition runs continuously")
    print("  - Familiar face visible: no wake phrase required")
    print("  - Unknown/no face: wake phrase required")
    print("  - Miguel uses cached vision state for faster replies")
    print("  - Press Ctrl+C to stop")
    print()

    last_announced_person = None
    last_reply_time = 0

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build()
        cam_out = cam.requestOutput((FRAME_W, FRAME_H))
        camera_queue = cam_out.createOutputQueue(maxSize=4, blocking=True)

        pipeline.start()

        vision_thread = threading.Thread(
            target=vision_worker,
            args=(camera_queue, pipeline),
            daemon=True,
        )
        vision_thread.start()

        speak("Miguel threaded conversation mode is online. I will watch and listen more naturally.")

        keep_running = True

        try:
            while keep_running and pipeline.isRunning():
                local_state = get_cached_local_state()
                recognized_person = local_state.get("recognized_person")
                active_followup = (time.time() - last_reply_time) < ACTIVE_FOLLOWUP_SECONDS
                vision_fresh = is_vision_state_fresh(local_state)

                if recognized_person and vision_fresh:
                    friendly_person = recognized_person.replace("_", " ")

                    if last_announced_person != recognized_person:
                        speak(f"I see {friendly_person}. Conversation mode is ready.")
                        last_announced_person = recognized_person

                    print(f"[READY] Familiar person present: {friendly_person}. Start talking now.")
                    user_text = capture_user_turn()

                    if not user_text:
                        print("[AUDIO] No speech captured. Returning to ready state.")
                        continue

                    print(f"[TRANSCRIPT] {user_text}")

                    if not should_respond_naturally(
                        user_text,
                        familiar_person=recognized_person,
                        active_followup=active_followup,
                    ):
                        print(f"[IGNORED] Background speech: {user_text}")
                        continue

                    print("[THINKING] Miguel is processing.")
                    keep_running = handle_user_turn_with_cached_state(user_text, local_state)
                    last_reply_time = time.time()

                else:
                    last_announced_person = None
                    print("[IDLE] No familiar person recognized. Wake phrase required.")
                    listen_for_wake()
                    speak("I'm listening now.")
                    print("[READY] Start talking now.")

                    local_state = get_cached_local_state()
                    user_text = capture_user_turn()

                    if not user_text:
                        print("[AUDIO] No speech captured after wake phrase.")
                        continue

                    print(f"[TRANSCRIPT] {user_text}")
                    print("[THINKING] Miguel is processing.")
                    keep_running = handle_user_turn_with_cached_state(user_text, local_state)
                    last_reply_time = time.time()

        finally:
            STOP_EVENT.set()
            vision_thread.join(timeout=2)

    print("Miguel Cloud Brain V6 stopped.")


if __name__ == "__main__":
    run_v6_threaded_conversation()
'''

path.write_text(prefix + v6_block)
print(f"Created threaded V6: {path}")
