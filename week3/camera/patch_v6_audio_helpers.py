from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"
text = path.read_text()

# Ensure wave import exists.
if "import wave" not in text:
    text = text.replace("import time\n", "import time\nimport wave\n")

helper_block = r'''
def write_mono_wav(wav_path, mono_chunks):
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(AUDIO_RATE)
        wf.writeframes(b"".join(mono_chunks))


def local_vosk_transcribe_wav(wav_path):
    try:
        with wave.open(str(wav_path), "rb") as wf:
            recognizer = KaldiRecognizer(vosk_model, wf.getframerate())
            parts = []

            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    part = result.get("text", "").lower().strip()
                    if part:
                        parts.append(part)

            final = json.loads(recognizer.FinalResult())
            final_text = final.get("text", "").lower().strip()
            if final_text:
                parts.append(final_text)

        return " ".join(parts).strip()

    except Exception as e:
        print("Local Vosk fallback transcription error:", e)
        return ""


def transcribe_audio_openai(wav_path):
    prompt = (
        "This is speech to a small father-son robot named Miguel. "
        "Important names and terms: Miguel, Marquinho, Marco, Mission Control, "
        "Jetson, Jetson Orin Nano, OAK-D Lite, ReSpeaker, Creative speaker, "
        "what do you see, who am I, status, weather, time, calculate."
    )

    try:
        with open(wav_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=OPENAI_TRANSCRIBE_MODEL,
                file=audio_file,
                prompt=prompt,
                response_format="text",
            )

        if isinstance(transcript, str):
            return transcript.lower().strip()

        return str(getattr(transcript, "text", transcript)).lower().strip()

    except Exception as e:
        print("OpenAI transcription error:", e)
        print("Falling back to local Vosk transcription.")
        return local_vosk_transcribe_wav(wav_path)

'''

if "def write_mono_wav(" not in text:
    marker = "def capture_user_turn():"
    idx = text.find(marker)
    if idx == -1:
        raise SystemExit("Could not find capture_user_turn().")
    text = text[:idx] + helper_block + "\n" + text[idx:]
else:
    print("write_mono_wav already exists; no insertion needed.")

path.write_text(text)
print(f"Patched audio helpers into: {path}")
