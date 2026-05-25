from pathlib import Path
import re

memory_path = Path.home() / "robot-project/week3/camera/robot_memory.py"
brain_path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"

memory = memory_path.read_text()
brain = brain_path.read_text()

# ============================================================
# 1. Replace handle_voice_mode_command() in robot_memory.py
# ============================================================

start = memory.find("def handle_voice_mode_command(user_text):")
if start == -1:
    marker = "def handle_robot_mode_command(user_text, recognized_person=None):"
    start = memory.find(marker)
    if start == -1:
        raise SystemExit("Could not find voice command or robot mode function.")
    insert_mode = True
else:
    insert_mode = False

if not insert_mode:
    end = memory.find("\ndef ", start + 1)
    if end == -1:
        raise SystemExit("Could not find end of handle_voice_mode_command().")
else:
    end = start

voice_func = r'''
def normalize_voice_mode(mode):
    aliases = {
        "natural_robot": "natural_voice",
        "natural": "natural_voice",
        "normal": "natural_voice",
        "normal_voice": "natural_voice",
        "default_voice": "natural_voice",
        "friendly": "friendly_voice",
        "deep": "deep_voice",
        "story": "story_voice",
        "robot": "robot_voice",
        "robotic_voice": "robot_voice",
        "local_robot": "robot_voice",
    }
    return aliases.get(str(mode or "").strip(), str(mode or "robot_voice").strip())


def get_voice_mode():
    memory = load_memory()
    mode = normalize_voice_mode(memory.get("voice_mode", "robot_voice"))

    if memory.get("voice_mode") != mode:
        memory["voice_mode"] = mode
        save_memory(memory)

    return mode


def set_voice_mode(mode):
    memory = load_memory()
    memory["voice_mode"] = normalize_voice_mode(mode)
    save_memory(memory)


def handle_voice_mode_command(user_text):
    text = user_text.lower().strip()
    memory = load_memory()

    # Normalize old stored values.
    memory["voice_mode"] = normalize_voice_mode(memory.get("voice_mode", "robot_voice"))

    voice_options_reply = (
        "I have five voice modes: robot voice, natural voice, friendly voice, "
        "deep voice, and story voice."
    )

    if any(p in text for p in [
        "what voice options",
        "which voice options",
        "what voices do you have",
        "which voices do you have",
        "voice options",
        "list voices",
        "available voices",
        "which options of voice",
        "what options of voice",
    ]):
        save_memory(memory)
        return voice_options_reply

    voice_map = [
        (
            [
                "robot voice",
                "robotic voice",
                "use robot voice",
                "use robotic voice",
                "go to robot voice",
                "go to robotic voice",
                "switch to robot voice",
                "switch to robotic voice",
                "classic robot voice",
                "local voice",
            ],
            "robot_voice",
            "Robot voice activated.",
        ),
        (
            [
                "natural voice",
                "use natural voice",
                "more natural voice",
                "speak naturally",
                "normal voice",
                "go to normal voice",
                "use normal voice",
                "default voice",
            ],
            "natural_voice",
            "Natural voice activated.",
        ),
        (
            [
                "friendly voice",
                "warmer voice",
                "use friendly voice",
                "go to friendly voice",
            ],
            "friendly_voice",
            "Friendly voice activated.",
        ),
        (
            [
                "deep voice",
                "deeper voice",
                "use deep voice",
                "go to deep voice",
            ],
            "deep_voice",
            "Deep voice activated.",
        ),
        (
            [
                "story voice",
                "narrator voice",
                "use story voice",
                "go to story voice",
            ],
            "story_voice",
            "Story voice activated.",
        ),
        (
            [
                "mission control voice",
                "space robot voice",
            ],
            "robot_voice",
            "Mission Control robot voice activated.",
        ),
    ]

    for phrases, mode, reply in voice_map:
        if any(p in text for p in phrases):
            memory["voice_mode"] = mode
            save_memory(memory)
            return reply

    if any(p in text for p in [
        "what voice are you using",
        "current voice",
        "which voice are you using",
        "which voice are you now",
        "what voice are you now",
        "voice mode",
    ]):
        mode = normalize_voice_mode(memory.get("voice_mode", "robot_voice"))
        memory["voice_mode"] = mode
        save_memory(memory)
        return f"My current voice mode is {mode}."

    save_memory(memory)
    return None

'''

if insert_mode:
    memory = memory[:start] + voice_func + "\n" + memory[start:]
else:
    memory = memory[:start] + voice_func + memory[end:]


# Remove duplicate older get_voice_mode/set_voice_mode definitions if they appear after our new block.
# Keep first occurrence from our block.
def remove_duplicate_defs(text, func_name):
    first = text.find(f"def {func_name}(")
    if first == -1:
        return text
    second = text.find(f"\ndef {func_name}(", first + 1)
    while second != -1:
        end = text.find("\ndef ", second + 1)
        if end == -1:
            text = text[:second]
            break
        text = text[:second] + text[end:]
        second = text.find(f"\ndef {func_name}(", first + 1)
    return text

memory = remove_duplicate_defs(memory, "get_voice_mode")
memory = remove_duplicate_defs(memory, "set_voice_mode")

memory_path.write_text(memory.replace("\t", "    "))


# ============================================================
# 2. Ensure brain TTS mapping supports current voice modes.
# ============================================================

if "def get_tts_config_for_voice_mode():" not in brain:
    marker = "def clean_text_for_speech"
    idx = brain.find(marker)
    if idx == -1:
        raise SystemExit("Could not find clean_text_for_speech in brain file.")

    helper = r'''
def get_tts_config_for_voice_mode():
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
            "instructions": "Speak like a gentle storyteller, expressive but concise.",
        }

    # robot_voice = local espeak, clearly different from OpenAI voices.
    return {
        "engine": "espeak",
        "voice": None,
        "instructions": None,
    }

'''
    brain = brain[:idx] + helper + "\n" + brain[idx:]


# Replace speak_with_openai_tts with config-aware version.
start = brain.find("def speak_with_openai_tts(speech_text: str):")
if start != -1:
    end = brain.find("\ndef ", start + 1)
    if end != -1:
        new_tts = r'''def speak_with_openai_tts(speech_text: str):
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
        brain = brain[:start] + new_tts + brain[end+1:]


# Make sure robot_voice uses espeak even if MIGUEL_TTS_ENGINE=openai.
start = brain.find("def speak(text: str):")
if start != -1:
    end = brain.find("\ndef ", start + 1)
    if end != -1:
        speak_block = brain[start:end]
        if "get_tts_config_for_voice_mode()" not in speak_block:
            new_speak = r'''def speak(text: str):
    print(f"{ROBOT_NAME} says: {text}")

    speech_text = clean_text_for_speech(text)
    if not speech_text:
        return

    try:
        cfg = get_tts_config_for_voice_mode()

        if cfg.get("engine") == "espeak":
            speak_with_espeak(speech_text)
        elif MIGUEL_TTS_ENGINE.lower() == "openai":
            speak_with_openai_tts(speech_text)
        else:
            speak_with_espeak(speech_text)

    except Exception as e:
        print("[TTS] Voice failed; falling back to espeak:", e)
        speak_with_espeak(speech_text)

'''
            brain = brain[:start] + new_speak + brain[end+1:]


brain_path.write_text(brain)

print("Patched voice modes and TTS mapping.")
