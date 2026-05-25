from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v4.py"
text = path.read_text()

marker = 'print(f"Starting {ROBOT_NAME} Cloud Brain V2.")'
idx = text.find(marker)

if idx == -1:
    marker = 'print(f"Starting {ROBOT_NAME} Cloud Brain V3.")'
    idx = text.find(marker)

if idx == -1:
    raise SystemExit("Could not find main-loop marker. Open the file and check the startup print line.")

prefix = text[:idx]

v4_main = r'''
# ============================================================
# V4 Natural Conversation Mode
# ============================================================

ACTIVE_FOLLOWUP_SECONDS = 30
PRESENCE_SCAN_SECONDS = 3

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


def text_has_any(text, phrases):
    text = text.lower()
    return any(p in text for p in phrases)


def should_respond_naturally(user_text, familiar_person=None, active_followup=False):
    text = user_text.lower().strip()

    if not text:
        return False

    words = set(re.findall(r"\b\w+\b", text))

    # Always respond to shutdown words.
    if words & {"quit", "stop", "exit", "shutdown"}:
        return True

    # Always respond if directly addressed.
    if text_has_any(text, DIRECT_ADDRESS_WORDS):
        return True

    # Respond to obvious questions.
    if text_has_any(text, QUESTION_WORDS):
        return True

    # Respond to obvious requests.
    if text_has_any(text, REQUEST_WORDS):
        return True

    # In follow-up mode, allow shorter replies.
    if familiar_person and active_followup:
        # But still ignore very short non-commands.
        if len(words) >= 2:
            return True

    return False


def summarize_person(local_state):
    person = local_state.get("recognized_person")
    if not person:
        return None
    return person.replace("_", " ")


def run_v4_conversation():
    print(f"Starting {ROBOT_NAME} Cloud Brain V4 - Natural Conversation Mode.")
    print_loaded_faces()
    print("Mode:")
    print("  - Familiar face visible: no wake phrase required")
    print("  - Unknown/no face: wake phrase required")
    print("  - Miguel replies to questions, requests, direct address, and follow-ups")
    print("Wake phrases:", ", ".join(WAKE_PHRASES))
    print("Say Ctrl+C to stop from terminal.")
    print()

    last_known_person = None
    last_reply_time = 0
    announced_person = None

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build()
        cam_out = cam.requestOutput((FRAME_W, FRAME_H))
        queue = cam_out.createOutputQueue(maxSize=4, blocking=True)

        pipeline.start()

        speak("Miguel natural conversation mode is online.")

        keep_running = True

        while keep_running and pipeline.isRunning():
            # Look around first.
            local_state = detect_face_state(queue)
            recognized_person = local_state.get("recognized_person")
            active_followup = (time.time() - last_reply_time) < ACTIVE_FOLLOWUP_SECONDS

            print("Presence state:", local_state)

            if recognized_person:
                friendly_person = recognized_person.replace("_", " ")

                if announced_person != recognized_person:
                    speak(f"I see {friendly_person}. Conversation mode is ready.")
                    announced_person = recognized_person

                last_known_person = recognized_person

                # Familiar person: no wake phrase needed.
                print(f"Familiar person present: {friendly_person}. Listening without wake phrase.")
                user_text = capture_user_turn()

                if not user_text:
                    continue

                if not should_respond_naturally(
                    user_text,
                    familiar_person=recognized_person,
                    active_followup=active_followup,
                ):
                    print(f"Ignored background speech: {user_text}")
                    continue

                keep_running = handle_user_turn(user_text, queue)
                last_reply_time = time.time()

            else:
                announced_person = None

                # Unknown/no familiar face: require wake phrase.
                print("No familiar person recognized. Wake phrase required.")
                listen_for_wake()
                speak("I'm listening.")
                user_text = capture_user_turn()

                if not user_text:
                    continue

                keep_running = handle_user_turn(user_text, queue)
                last_reply_time = time.time()

    print("Miguel Cloud Brain V4 stopped.")


if __name__ == "__main__":
    run_v4_conversation()
'''

path.write_text(prefix + v4_main)
print(f"Patched V4 file: {path}")
