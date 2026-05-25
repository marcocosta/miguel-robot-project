from pathlib import Path
import re

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"
text = path.read_text()

# Ensure get_voice_mode import.
if "get_voice_mode" not in text:
    text = text.replace(
        "get_personality_mode,",
        "get_personality_mode,\n    get_voice_mode,"
    )

# Make sure os is available.
if "import os" not in text:
    text = text.replace("import time\n", "import time\nimport os\n")

# Add speech constants after OpenAI client/model constants if possible.
if "MIGUEL_TTS_ENGINE" not in text:
    insert_after = 'OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")'
    if insert_after in text:
        text = text.replace(
            insert_after,
            insert_after + '\nMIGUEL_TTS_ENGINE = os.getenv("MIGUEL_TTS_ENGINE", "openai")\nMIGUEL_TTS_MODEL = os.getenv("MIGUEL_TTS_MODEL", "gpt-4o-mini-tts")\nMIGUEL_TTS_VOICE = os.getenv("MIGUEL_TTS_VOICE", "cedar")'
        )
    else:
        text = text.replace(
            "client = OpenAI()",
            "client = OpenAI()\nMIGUEL_TTS_ENGINE = os.getenv(\"MIGUEL_TTS_ENGINE\", \"openai\")\nMIGUEL_TTS_MODEL = os.getenv(\"MIGUEL_TTS_MODEL\", \"gpt-4o-mini-tts\")\nMIGUEL_TTS_VOICE = os.getenv(\"MIGUEL_TTS_VOICE\", \"cedar\")"
        )

# Remove old clean_text_for_speech if present.
m = re.search(r"\ndef clean_text_for_speech\(.*?\n(?=def |\n#|\Z)", text, flags=re.DOTALL)
if m:
    text = text[:m.start()] + "\n" + text[m.end():]

# Replace speak() function.
start = text.find("def speak(")
if start == -1:
    raise SystemExit("Could not find speak().")

next_def = text.find("\ndef ", start + 1)
if next_def == -1:
    next_def = text.find("\n# ", start + 1)
if next_def == -1:
    raise SystemExit("Could not find end of speak().")

new_speak = r'''
def clean_text_for_speech(text: str) -> str:
    if not text:
        return ""

    text = str(text)

    # Remove Markdown and screen-only formatting.
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = text.replace("`", "")
    text = text.replace("#", "")
    text = text.replace("* ", "")
    text = text.replace("- ", "")
    text = text.replace("•", ". ")

    # Replace visual separators with spoken pauses.
    text = text.replace("—", ", ")
    text = text.replace("–", ", ")
    text = text.replace("|", ", ")

    # Remove markdown links but keep label.
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Avoid reading weird symbols.
    text = re.sub(r"[^\w\s.,?!'\":;()/%-]", "", text)

    # Keep speech concise and clean.
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_tts_voice_for_mode():
    mode = "natural_robot"
    try:
        mode = get_voice_mode()
    except Exception:
        pass

    if mode == "deep_robot":
        return "onyx", "Speak with a calm, deep, friendly robot voice. Keep it natural, not theatrical."
    if mode == "friendly":
        return "coral", "Speak warmly and naturally, like a friendly home robot."
    if mode == "story":
        return "fable", "Speak like a gentle storyteller, expressive but concise."
    if mode == "local_robot":
        return None, None

    # Default natural robot voice.
    return MIGUEL_TTS_VOICE, "Speak as Miguel, a friendly father-son project robot. Natural, clear, concise, slightly robotic but not mechanical."


def speak_with_espeak(speech_text: str):
    wav_path = Path(tempfile.gettempdir()) / "miguel_speech.wav"

    # Slightly slower, less harsh than default espeak.
    subprocess.run(
        ["espeak", "-s", "145", "-p", "42", "-a", "135", "-w", str(wav_path), speech_text],
        check=True,
    )
    subprocess.run(["aplay", "-q", "-D", SPEAKER_DEVICE, str(wav_path)], check=True)


def speak_with_openai_tts(speech_text: str):
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


def speak(text: str):
    print(f"{ROBOT_NAME} says: {text}")

    speech_text = clean_text_for_speech(text)
    if not speech_text:
        return

    try:
        if MIGUEL_TTS_ENGINE.lower() == "openai":
            speak_with_openai_tts(speech_text)
        else:
            speak_with_espeak(speech_text)
    except Exception as e:
        print("[TTS] OpenAI/natural voice failed; falling back to espeak:", e)
        speak_with_espeak(speech_text)

'''

text = text[:start] + new_speak + text[next_def:]

path.write_text(text)
print("Patched OpenAI TTS voice engine.")
