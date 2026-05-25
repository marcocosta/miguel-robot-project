from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"
text = path.read_text()

# Ensure imports from robot_memory.
if "get_voice_mode" not in text:
    text = text.replace(
        "get_personality_mode,",
        "get_personality_mode,\n    get_voice_mode,"
    )

if "handle_enrollment_security_command" not in text:
    text = text.replace(
        "handle_robot_mode_command,",
        "handle_robot_mode_command,\n    handle_enrollment_security_command,\n    enrollment_is_unlocked,\n    get_enrollment_authorizer,"
    )

# Add voice-mode mapping helper before speak().
marker = "def clean_text_for_speech"
idx = text.find(marker)
if idx == -1:
    raise SystemExit("Could not find clean_text_for_speech().")

voice_helper = r'''
def get_tts_config_for_voice_mode():
    """
    Voice mode mapping:
      natural_voice  -> OpenAI TTS cedar/marin-style natural voice
      friendly_voice -> warmer OpenAI voice
      deep_voice     -> deeper OpenAI voice
      story_voice    -> narrator-style OpenAI voice
      robot_voice    -> local espeak fallback
    """
    try:
        mode = get_voice_mode()
    except Exception:
        mode = "robot_voice"

    if mode == "natural_voice":
        return {
            "engine": "openai",
            "voice": "cedar",
            "instructions": "Speak naturally and clearly as Miguel, a friendly small robot. Keep it concise.",
        }

    if mode == "friendly_voice":
        return {
            "engine": "openai",
            "voice": "coral",
            "instructions": "Speak warmly and friendly, like a helpful home robot. Keep it concise.",
        }

    if mode == "deep_voice":
        return {
            "engine": "openai",
            "voice": "onyx",
            "instructions": "Speak with a calm, deeper robot voice. Clear, steady, and concise.",
        }

    if mode == "story_voice":
        return {
            "engine": "openai",
            "voice": "fable",
            "instructions": "Speak like a gentle storyteller, expressive but still concise.",
        }

    return {
        "engine": "espeak",
        "voice": None,
        "instructions": None,
    }


'''

if "def get_tts_config_for_voice_mode():" not in text:
    text = text[:idx] + voice_helper + "\n" + text[idx:]

# Replace get_tts_voice_for_mode if present to use new mapping.
start = text.find("def get_tts_voice_for_mode():")
if start != -1:
    end = text.find("\ndef ", start + 1)
    if end != -1:
        replacement = r'''def get_tts_voice_for_mode():
    cfg = get_tts_config_for_voice_mode()
    return cfg.get("voice"), cfg.get("instructions")

'''
        text = text[:start] + replacement + text[end+1:]

# Harden speak_with_openai_tts to fall back if robot_voice.
old = '''def speak_with_openai_tts(speech_text: str):
    wav_path = Path(tempfile.gettempdir()) / "miguel_speech_openai.wav"
    voice, instructions = get_tts_voice_for_mode()

    if not voice:
        speak_with_espeak(speech_text)
        return

    response = client.audio.speech.create(
        model=MIGUEL_TTS_MODEL,
        voice=voice,
        input=speech_text,
        instructions=instructions,
        response_format="wav",
    )

    response.write_to_file(str(wav_path))
    subprocess.run(["aplay", "-q", "-D", SPEAKER_DEVICE, str(wav_path)], check=True)
'''

new = '''def speak_with_openai_tts(speech_text: str):
    wav_path = Path(tempfile.gettempdir()) / "miguel_speech_openai.wav"
    cfg = get_tts_config_for_voice_mode()

    if cfg.get("engine") != "openai":
        speak_with_espeak(speech_text)
        return

    response = client.audio.speech.create(
        model=MIGUEL_TTS_MODEL,
        voice=cfg.get("voice", "cedar"),
        input=speech_text,
        instructions=cfg.get("instructions"),
        response_format="wav",
    )

    response.write_to_file(str(wav_path))
    subprocess.run(["aplay", "-q", "-D", SPEAKER_DEVICE, str(wav_path)], check=True)
'''

if old in text:
    text = text.replace(old, new)
else:
    print("Warning: speak_with_openai_tts block not found exactly. Voice mode may already be patched.")

# Enrollment command detection helpers.
marker = "def handle_user_turn_with_cached_state(user_text, cached_local_state):"
idx = text.find(marker)
if idx == -1:
    raise SystemExit("Could not find handle_user_turn_with_cached_state().")

enroll_helpers = r'''
def is_face_enrollment_request(user_text):
    text = user_text.lower()
    phrases = [
        "enroll me as",
        "enroll this face as",
        "learn my face as",
        "learn this face as",
        "remember my face as",
        "add me as",
        "add new friend",
    ]
    return any(p in text for p in phrases)


def extract_requested_enrollment_name(user_text):
    text = user_text.lower().strip()
    markers = [
        "enroll me as",
        "enroll this face as",
        "learn my face as",
        "learn this face as",
        "remember my face as",
        "add me as",
    ]

    for marker in markers:
        if marker in text:
            name = text.split(marker, 1)[1].strip(" .!?")
            if name:
                return name.replace(" ", "_")[:32]

    return "new_friend"

'''

if "def is_face_enrollment_request(" not in text:
    text = text[:idx] + enroll_helpers + "\n" + text[idx:]

# Insert enrollment security handling after mode_reply/topic/identity command area.
anchor = '''    identity_reply = handle_manual_identity_command(user_text)
    if identity_reply:
        speak(identity_reply)
        update_conversation_memory(assistant_reply=identity_reply)
        return True

    # For vision questions, do not trust conversation-grace cached identity.
'''

insert = '''    identity_reply = handle_manual_identity_command(user_text)
    if identity_reply:
        speak(identity_reply)
        update_conversation_memory(assistant_reply=identity_reply)
        return True

    # Enrollment security:
    # Only Marco or Marquinho may unlock enrollment, and they must be confirmed by camera truth.
    confirmed_camera_person = None
    if is_camera_truth_state(local_state):
        confirmed_camera_person = local_state.get("recognized_person")

    enrollment_security_reply = handle_enrollment_security_command(user_text, confirmed_camera_person)
    if enrollment_security_reply:
        speak(enrollment_security_reply)
        update_conversation_memory(assistant_reply=enrollment_security_reply)
        return True

    if is_face_enrollment_request(user_text):
        requested_name = extract_requested_enrollment_name(user_text)

        if not enrollment_is_unlocked():
            reply = (
                "Enrollment is locked. Only Marco or Marquinho can unlock new friend enrollment first."
            )
            speak(reply)
            update_conversation_memory(assistant_reply=reply)
            return True

        authorizer = get_enrollment_authorizer()
        reply = (
            f"Enrollment is authorized by {authorizer}. "
            f"I can start learning the new friend name {requested_name}, "
            "but the capture routine is not connected yet. Next step is to capture clear face samples."
        )
        speak(reply)
        update_conversation_memory(assistant_reply=reply)
        return True

    # For vision questions, do not trust conversation-grace cached identity.
'''

if anchor in text:
    text = text.replace(anchor, insert)
else:
    print("Warning: enrollment insertion anchor not found exactly.")

path.write_text(text)
print("Patched robot_cloud_brain_v6_threaded.py with voice mode mapping and owner-gated enrollment.")
