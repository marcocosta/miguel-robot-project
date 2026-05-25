from pathlib import Path
import re

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v5.py"
text = path.read_text()

# Ensure wave import exists.
if "import wave" not in text:
    text = text.replace("import time\n", "import time\nimport wave\n")

# Add transcription model env setting after OPENAI_MODEL if not present.
if "OPENAI_TRANSCRIBE_MODEL" not in text:
    text = text.replace(
        'OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")',
        'OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")\nOPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")'
    )

new_capture_block = r'''
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
                    text = result.get("text", "").lower().strip()
                    if text:
                        parts.append(text)

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

        # Defensive fallback for SDK response objects.
        return str(getattr(transcript, "text", transcript)).lower().strip()

    except Exception as e:
        print("OpenAI transcription error:", e)
        print("Falling back to local Vosk transcription.")
        return local_vosk_transcribe_wav(wav_path)


def capture_user_turn():
    print("Listening for your turn with OpenAI transcription...")

    proc = open_raw_mic_stream()

    start_time = time.time()
    last_voice_time = start_time
    speech_started = False
    mono_chunks = []
    rms_peak = 0.0

    wav_path = AUDIO_DIR / "miguel_user_turn_openai.wav"

    try:
        while True:
            now = time.time()

            if now - start_time > MAX_TURN_SECONDS:
                print("Max turn time reached.")
                break

            raw = proc.stdout.read(CHUNK_BYTES)
            if not raw:
                continue

            mono_bytes, rms = stereo_raw_to_mono_bytes(raw)

            if not mono_bytes:
                continue

            rms_peak = max(rms_peak, rms)

            # Always keep audio after capture starts.
            if speech_started:
                mono_chunks.append(mono_bytes)

            # Start capture when speech is detected.
            if rms > SPEECH_RMS_THRESHOLD:
                if not speech_started:
                    print(f"Speech started. RMS={rms:.1f}")
                    speech_started = True
                    mono_chunks.append(mono_bytes)

                last_voice_time = time.time()

            if speech_started:
                turn_age = time.time() - start_time
                silence_age = time.time() - last_voice_time

                if turn_age >= MIN_TURN_SECONDS and silence_age >= SILENCE_SECONDS:
                    print("Silence timeout reached.")
                    break

        if not mono_chunks:
            print("No speech captured.")
            return ""

        write_mono_wav(wav_path, mono_chunks)
        print(f"Saved user audio: {wav_path}")
        print(f"Peak RMS: {rms_peak:.1f}")

        user_text = transcribe_audio_openai(wav_path)
        print(f"Final heard by OpenAI transcription: {user_text}")

        return user_text

    finally:
        stop_stream(proc)
'''

# Replace capture_user_turn function and helper area before Face recognition helpers.
pattern = r"def capture_user_turn\(\):.*?(?=\n\n# ============================================================\n# Face recognition helpers)"
new_text, count = re.subn(pattern, new_capture_block, text, flags=re.DOTALL)

if count != 1:
    raise SystemExit(f"Could not replace capture_user_turn cleanly. Replacements made: {count}")

path.write_text(new_text)
print(f"Patched V5 transcription in: {path}")
