import cv2
import depthai as dai
import json
import os
import re
import subprocess
import tempfile
import time
import threading
import wave
import numpy as np
from pathlib import Path
from openai import OpenAI
from vosk import Model, KaldiRecognizer
from robot_skills import maybe_handle_local_skill
from robot_memory import (
    get_robot_mode,
    get_personality_mode,
    get_voice_mode,
    get_memory_context,
    handle_robot_mode_command,
    handle_long_term_topic_command,
    append_turn_to_active_topic,
    should_auto_attach_to_topic,
    get_long_term_topic_context,
)


# ============================================================
# Miguel Robot - Cloud Brain V2
# Local: mic, speaker, camera, face recognition, enrollment
# Cloud: conversation + reasoning
# ============================================================

ROBOT_NAME = "Miguel"

FRAME_W = 640
FRAME_H = 480
FACE_SIZE = 160

# Stable ALSA names. These avoid changing card numbers after reboot.
MIC_DEVICE = "hw:CARD=Array,DEV=0"          # ReSpeaker XVF3800
SPEAKER_DEVICE = "plughw:CARD=Audio,DEV=0" # USB-C/3.5mm adapter / Creative speaker

AUDIO_RATE = 16000
AUDIO_CHANNELS = 2
CHUNK_MS = 250
CHUNK_BYTES = int(AUDIO_RATE * AUDIO_CHANNELS * 2 * (CHUNK_MS / 1000.0))

SPEECH_RMS_THRESHOLD = 1200
SILENCE_SECONDS = 1.8
MIN_TURN_SECONDS = 1.0
MAX_TURN_SECONDS = 20.0

WAKE_PHRASES = [
    "hey miguel",
    "miguel",
    "hey me go",
    "me go",
    "hey michael",
    "michael",
    "mission control",
    "robot",
    "hey robot",
]

END_TURN_PHRASES = [
    "your turn",
    "answer now",
    "go ahead",
    "over",
    "mission complete",
]

YES_WORDS = {"yes", "yeah", "yep", "sure", "ok", "okay", "please"}
NO_WORDS = {"no", "nope", "not", "later"}

BASE = Path.home() / "robot-project/week3"
AUDIO_DIR = BASE / "audio"
FACES_DIR = BASE / "faces"
MODEL_PATH = BASE / "models/vosk-model-small-en-us-0.15"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
FACES_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
MIGUEL_TTS_ENGINE = os.getenv("MIGUEL_TTS_ENGINE", "openai")
MIGUEL_TTS_MODEL = os.getenv("MIGUEL_TTS_MODEL", "gpt-4o-mini-tts")
MIGUEL_TTS_VOICE = os.getenv("MIGUEL_TTS_VOICE", "cedar")

client = OpenAI()

if not MODEL_PATH.exists():
    raise RuntimeError(f"Vosk model not found at: {MODEL_PATH}")

vosk_model = Model(str(MODEL_PATH))

face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    raise RuntimeError(f"Could not load Haar cascade: {face_cascade_path}")

CUSTOM_GREETINGS = {
    "marquinho": "Hello Chief Engineer Marquinho. Miguel recognizes you.",
    "marco": "Hello Marco. Systems Engineer online.",
}

last_unknown_face_prompt_time = 0
UNKNOWN_FACE_PROMPT_COOLDOWN = 60


# ============================================================
# Audio helpers
# ============================================================


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


def open_raw_mic_stream():
    cmd = [
        "arecord",
        "-q",
        "-D", MIC_DEVICE,
        "-f", "S16_LE",
        "-r", str(AUDIO_RATE),
        "-c", str(AUDIO_CHANNELS),
        "-t", "raw",
    ]

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )


def stop_stream(proc):
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=1)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def stereo_raw_to_mono_bytes(raw_bytes: bytes):
    samples = np.frombuffer(raw_bytes, dtype=np.int16)

    if len(samples) < 2:
        return b"", 0.0

    samples = samples[: len(samples) - (len(samples) % 2)]
    stereo = samples.reshape(-1, 2)

    mono = stereo.mean(axis=1).astype(np.int16)

    rms = float(np.sqrt(np.mean(mono.astype(np.float32) ** 2))) if len(mono) else 0.0

    return mono.tobytes(), rms


def text_contains_any(text: str, phrases):
    text = text.lower()
    return any(phrase in text for phrase in phrases)

def listen_for_wake(timeout_seconds=None):
    print(f"\n{ROBOT_NAME} is idle. Say: 'Hey Miguel', 'Hey me go', or 'Mission Control'.")

    proc = open_raw_mic_stream()
    recognizer = KaldiRecognizer(vosk_model, AUDIO_RATE)

    last_reset = time.time()
    start_time = time.time()

    try:
        while True:
            if timeout_seconds is not None and (time.time() - start_time) >= timeout_seconds:
                print("Wake listen timeout reached.")
                return False

            raw = proc.stdout.read(CHUNK_BYTES)
            if not raw:
                continue

            mono_bytes, rms = stereo_raw_to_mono_bytes(raw)
            if not mono_bytes:
                continue

            if recognizer.AcceptWaveform(mono_bytes):
                result = json.loads(recognizer.Result())
                final_text = result.get("text", "").lower().strip()

                if final_text:
                    print(f"Idle final: {final_text}")

                    if text_contains_any(final_text, WAKE_PHRASES):
                        print("Wake phrase detected from final text.")
                        return True

                recognizer = KaldiRecognizer(vosk_model, AUDIO_RATE)
                last_reset = time.time()

            else:
                partial_text = json.loads(recognizer.PartialResult()).get("partial", "").lower().strip()

                if partial_text:
                    print(f"Idle partial: {partial_text}")

                    if text_contains_any(partial_text, WAKE_PHRASES):
                        print("Wake phrase detected from partial text.")
                        return True

            if time.time() - last_reset > 5:
                recognizer = KaldiRecognizer(vosk_model, AUDIO_RATE)
                last_reset = time.time()

    finally:
        stop_stream(proc)
        try:
            AUDIO_CAPTURE_ACTIVE.clear()
        except NameError:
            pass


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


def capture_user_turn():
    print("Listening for your turn with OpenAI transcription...")
    AUDIO_CAPTURE_ACTIVE.set()

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


# ============================================================
# Face recognition helpers
# ============================================================

def preprocess_face(gray_frame, face_box):
    x, y, w, h = face_box

    pad = int(0.15 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(gray_frame.shape[1], x + w + pad)
    y2 = min(gray_frame.shape[0], y + h + pad)

    face = gray_frame[y1:y2, x1:x2]
    face = cv2.resize(face, (FACE_SIZE, FACE_SIZE))
    face = cv2.equalizeHist(face)
    face = cv2.GaussianBlur(face, (3, 3), 0)

    return face


def lbp_image(gray):
    center = gray[1:-1, 1:-1]
    code = np.zeros_like(center, dtype=np.uint8)

    code |= ((gray[:-2, :-2] >= center) << 7).astype(np.uint8)
    code |= ((gray[:-2, 1:-1] >= center) << 6).astype(np.uint8)
    code |= ((gray[:-2, 2:] >= center) << 5).astype(np.uint8)
    code |= ((gray[1:-1, 2:] >= center) << 4).astype(np.uint8)
    code |= ((gray[2:, 2:] >= center) << 3).astype(np.uint8)
    code |= ((gray[2:, 1:-1] >= center) << 2).astype(np.uint8)
    code |= ((gray[2:, :-2] >= center) << 1).astype(np.uint8)
    code |= ((gray[1:-1, :-2] >= center) << 0).astype(np.uint8)

    return code


def lbp_histogram(gray, grid=8):
    lbp = lbp_image(gray)
    h, w = lbp.shape
    cell_h = h // grid
    cell_w = w // grid

    features = []

    for gy in range(grid):
        for gx in range(grid):
            cell = lbp[gy * cell_h:(gy + 1) * cell_h, gx * cell_w:(gx + 1) * cell_w]
            hist, _ = np.histogram(cell, bins=256, range=(0, 256))
            hist = hist.astype(np.float32)
            hist /= hist.sum() + 1e-6
            features.append(hist)

    return np.concatenate(features)


def chi_square_distance(a, b):
    return float(0.5 * np.sum(((a - b) ** 2) / (a + b + 1e-6)))


def load_face_templates():
    templates = {}

    for person_dir in FACES_DIR.iterdir():
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        person_features = []

        for img_path in sorted(person_dir.glob("face_*.png")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (FACE_SIZE, FACE_SIZE))
            img = cv2.equalizeHist(img)
            img = cv2.GaussianBlur(img, (3, 3), 0)

            person_features.append(lbp_histogram(img))

        if person_features:
            templates[person_name] = person_features

    return templates


FACE_TEMPLATES = load_face_templates()


def print_loaded_faces():
    print("Loaded face templates:")
    if not FACE_TEMPLATES:
        print("  none")
    for name, templates in FACE_TEMPLATES.items():
        print(f"  {name}: {len(templates)} samples")


def recognize_face(face_img):
    if not FACE_TEMPLATES:
        return None, 999.0, {}

    feature = lbp_histogram(face_img)
    person_scores = {}

    best_name = None
    best_distance = 999.0

    for person_name, templates in FACE_TEMPLATES.items():
        distances = sorted(chi_square_distance(feature, t) for t in templates)
        best_few = distances[:min(5, len(distances))]
        avg_distance = sum(best_few) / len(best_few)
        person_scores[person_name] = avg_distance

        if avg_distance < best_distance:
            best_distance = avg_distance
            best_name = person_name

    sorted_scores = sorted(person_scores.items(), key=lambda item: item[1])

    if len(sorted_scores) >= 2:
        margin = sorted_scores[1][1] - sorted_scores[0][1]
    else:
        margin = 999.0

    print(f"Recognition scores: {person_scores}")
    print(f"Recognition result candidate: {best_name}, distance={best_distance:.3f}, margin={margin:.3f}")

    # Conservative: avoid wrong confident greetings.
    if best_distance <= 32.0 and margin >= 5.0:
        return best_name, best_distance, person_scores

    return None, best_distance, person_scores

def detect_face_state(queue, frames_to_scan=60):
    recognition_votes = {}
    recognition_distances = {}
    face_positions = []
    face_count_seen = 0

    for _ in range(frames_to_scan):
        frame = queue.get().getCvFrame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(45, 45),
        )

        if len(faces) == 0:
            continue

        face_count_seen = max(face_count_seen, len(faces))

        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        cx = x + w // 2

        if cx < FRAME_W * 0.35:
            position = "left"
        elif cx > FRAME_W * 0.65:
            position = "right"
        else:
            position = "center"

        face_positions.append(position)

        face_img = preprocess_face(gray, faces[0])
        name, distance, scores = recognize_face(face_img)

        if name:
            recognition_votes[name] = recognition_votes.get(name, 0) + 1
            recognition_distances.setdefault(name, []).append(distance)

    if face_count_seen == 0:
        return {
            "face_detected": False,
            "face_count": 0,
            "face_position": "none",
            "recognized_person": None,
            "recognition_distance": None,
            "recognition_votes": {},
        }

    if face_positions:
        face_position = max(set(face_positions), key=face_positions.count)
    else:
        face_position = "center"

    if not recognition_votes:
        return {
            "face_detected": True,
            "face_count": face_count_seen,
            "face_position": face_position,
            "recognized_person": None,
            "recognition_distance": None,
            "recognition_votes": {},
        }

    sorted_votes = sorted(recognition_votes.items(), key=lambda item: item[1], reverse=True)
    best_name, best_votes = sorted_votes[0]
    second_votes = sorted_votes[1][1] if len(sorted_votes) > 1 else 0

    avg_distance = sum(recognition_distances[best_name]) / len(recognition_distances[best_name])

    print(f"Recognition votes: {recognition_votes}")
    print(
        f"Best recognition: {best_name}, "
        f"votes={best_votes}, second_votes={second_votes}, "
        f"avg_distance={avg_distance:.3f}"
    )

    # Conservative decision:
    # Miguel only recognizes someone if several frames agree.
    # This avoids wrong greetings when Marco/Marquinho scores are close.
    if best_votes >= 5 and best_votes >= second_votes + 3:
        recognized_person = best_name
    else:
        recognized_person = None

    return {
        "face_detected": True,
        "face_count": face_count_seen,
        "face_position": face_position,
        "recognized_person": recognized_person,
        "recognition_distance": avg_distance,
        "recognition_votes": recognition_votes,
    }


def sanitize_person_name(text):
    text = text.lower().strip()

    blocked_names = {
        "hey_me",
        "me_go",
        "hey_me_go",
        "miguel",
        "hey_miguel",
        "michael",
        "hey_michael",
        "robot",
        "hey_robot",
        "mission_control",
    }

    raw_name = text.replace(" ", "_")
    if raw_name in blocked_names:
        return None

    text = re.sub(r"[^a-z0-9 ]", "", text)
    words = [w for w in text.split() if w not in {"my", "name", "is", "i", "am", "im"}]

    if not words:
        return None

    person_name = "_".join(words[:2])

    if person_name in blocked_names:
        return None

    return person_name

def capture_face_samples(queue, person_name, sample_count=15):
    person_dir = FACES_DIR / person_name
    person_dir.mkdir(parents=True, exist_ok=True)

    existing = list(person_dir.glob("face_*.png"))
    next_index = len(existing) + 1
    saved = 0
    last_save = 0

    speak(f"Okay {person_name.replace('_', ' ')}. Please stay in front of the camera while I learn your face.")

    start = time.time()

    while saved < sample_count and time.time() - start < 45:
        frame = queue.get().getCvFrame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 60),
        )

        if len(faces) == 0:
            continue

        now = time.time()
        if now - last_save < 0.35:
            continue

        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        face_img = preprocess_face(gray, faces[0])

        out_path = person_dir / f"face_{next_index:03d}.png"
        cv2.imwrite(str(out_path), face_img)

        print(f"Saved face sample: {out_path}")

        next_index += 1
        saved += 1
        last_save = now

    return saved

def ask_to_enroll_unknown_face(queue):
    global FACE_TEMPLATES
    global last_unknown_face_prompt_time

    now = time.time()
    if now - last_unknown_face_prompt_time < UNKNOWN_FACE_PROMPT_COOLDOWN:
        return False

    last_unknown_face_prompt_time = now

    speak("I see a face I do not recognize. Do you want to become my friend? Please say yes or no.")

    answer = capture_user_turn()
    answer_words = set(re.findall(r"\b\w+\b", answer.lower()))

    if answer_words & NO_WORDS:
        speak("Okay. I will not learn this face now.")
        return False

    if not (answer_words & YES_WORDS):
        speak("I am not sure if that was yes. I will not learn this face yet.")
        return False

    speak("Great. Please say your name clearly.")
    name_text = capture_user_turn()
    person_name = sanitize_person_name(name_text)

    if not person_name:
        speak("I could not understand the name clearly. I will not save this face yet.")
        return False

    friendly_name = person_name.replace("_", " ")

    speak(f"I heard the name {friendly_name}. Please confirm. Say yes to confirm, or no to cancel.")
    confirm_text = capture_user_turn()
    confirm_words = set(re.findall(r"\b\w+\b", confirm_text.lower()))

    if confirm_words & NO_WORDS:
        speak("Okay. I cancelled the new friend enrollment.")
        return False

    if not (confirm_words & YES_WORDS):
        speak("I did not hear a clear confirmation. I cancelled the enrollment.")
        return False

    speak(f"Confirmed. {friendly_name}, please stay in front of the camera while I learn your face.")

    saved = capture_face_samples(queue, person_name, sample_count=15)

    if saved >= 5:
        FACE_TEMPLATES = load_face_templates()
        speak(f"Thank you {friendly_name}. I learned your face.")
        print_loaded_faces()
        return True

    speak("I could not capture enough good pictures. We can try again later.")
    return False



# ============================================================
# Cloud brain
# ============================================================

def ask_cloud_brain(user_text, local_state):
    system_instructions = f"""
You are {ROBOT_NAME}, the high-level conversation brain for a small father-son robot project.

Identity:
- Your name is Miguel.
- Chief Engineer: Marquinho.
- Systems Engineer: Marco.

Personality:
- Friendly, curious, and mission-control inspired.
- Speak naturally to a child and a parent building you together.
- Keep replies short: 1 or 2 sentences.

Rules:
- Use local_state when relevant.
- Use conversation_memory to understand short follow-ups like ok, yes, no, repeat that, or I did not understand.
- Use miguel_memory for saved user preferences, profile notes, and selected personality mode.
- Use long_term_topic_context to resume or continue topics the user wants to discuss over days.
- If there is an active long-term topic, connect short follow-ups to that topic unless the user clearly changes topic.
- If personality_mode is creative, be more imaginative and propose ideas.
- If personality_mode is teacher, explain slowly with simple examples.
- If personality_mode is engineer, be more technical and precise.
- If personality_mode is quiet, keep the answer very short.
- If personality_mode is mission_control, use a friendly father-son robot project tone.
- Stay on the current topic unless the user clearly changes topics.
- Do not randomly return to robot testing, face enrollment, or voice testing unless the user asks about the robot.
- Speak in English unless the user clearly asks for Portuguese or uses a full Portuguese sentence.
- Do not invent things you cannot sense.
- If asked what you see, only use local_state.
- If recognized_person is present, you may greet that person by name, but do not turn every answer into a greeting.
- Do not claim you can move unless local_state says motion hardware exists.
- Output plain spoken text only. Do not use Markdown, bullets, asterisks, code formatting, tables, or emojis.
- Keep spoken replies short: usually 1 to 3 sentences unless the user asks for detail.
- Keep spoken replies short: usually 1 to 3 sentences unless the user asks for detail.
- Do not use Markdown, bullets, asterisks, tables, or emojis in spoken replies.
"""

    payload = {
        "user_text": user_text,
        "local_state": local_state,
        "conversation_memory": CONVERSATION_MEMORY,
        "miguel_memory": get_memory_context(local_state.get("recognized_person")),
        "long_term_topic_context": get_long_term_topic_context(),
        "personality_mode": get_personality_mode(),
        "robot_capabilities": [
            "hear speech through ReSpeaker",
            "speak through Creative speaker",
            "detect faces with OAK-D Lite",
            "recognize enrolled faces locally",
            "ask unknown people if they want to be enrolled as friends",
            "hold short conversations using a cloud brain",
        ],
    }

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=system_instructions,
        input=json.dumps(payload),
    )

    return response.output_text.strip()


def handle_user_turn(user_text, queue):
    words = set(re.findall(r"\b\w+\b", user_text.lower()))

    if not user_text:
        speak("I did not understand. Please try again.")
        return True

    explicit_exit_phrases = {
        "quit robot",
        "exit robot",
        "stop robot program",
        "stop miguel program",
        "end program",
    }

    normalized_text = user_text.lower().strip()

    if normalized_text in explicit_exit_phrases:
        speak("Miguel is shutting down conversation mode. Mission saved.")
        return False

    local_state = detect_face_state(queue)
    print("Local state:", local_state)
    topic = infer_simple_topic(user_text)
    update_conversation_memory(user_text=user_text, topic=topic)

    short_ack_reply = handle_short_acknowledgement(user_text)
    if short_ack_reply:
        speak(short_ack_reply)
        update_conversation_memory(assistant_reply=short_ack_reply)
        return True

    local_skill_reply = maybe_handle_local_skill(user_text, local_state)
    if local_skill_reply:
        speak(local_skill_reply)
        update_conversation_memory(assistant_reply=local_skill_reply)
        if should_auto_attach_to_topic(user_text):
            append_turn_to_active_topic(user_text, local_skill_reply, local_state.get("recognized_person"))
        return True

    if local_state["face_detected"] and local_state["recognized_person"] is None:
    	print("Unknown face detected, but auto-enrollment is disabled during recognition tuning.")

    if local_state["recognized_person"] in CUSTOM_GREETINGS and words & {"hello", "hi", "hey", "look", "see", "who"}:
        speak(CUSTOM_GREETINGS[local_state["recognized_person"]])
        return True

    try:
        reply = ask_cloud_brain(user_text, local_state)
    except Exception as e:
        print("OpenAI API error:", e)
        reply = "My cloud brain is not reachable right now, but my local systems are still online."

    speak(reply)
    update_conversation_memory(assistant_reply=reply)
    if should_auto_attach_to_topic(user_text):
        append_turn_to_active_topic(user_text, reply, local_state.get("recognized_person"))
    return True


# ============================================================
# Main
# ============================================================



# ============================================================
# InsightFace / ArcFace Offline Recognition Override
# This overrides the older LBP detect_face_state implementation.
# ============================================================

from insightface.app import FaceAnalysis

INSIGHT_EMBED_DIR = BASE / "face_embeddings"
INSIGHT_SIMILARITY_THRESHOLD = 0.50
INSIGHT_MARGIN_THRESHOLD = 0.08
INSIGHT_MIN_ACCEPTED_VOTES = 2
INSIGHT_SCAN_FRAMES = 2

print("Loading Miguel InsightFace recognizer...")
insight_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
insight_app.prepare(ctx_id=-1, det_size=(640, 640))


def insight_normalize_embedding(embedding):
    emb = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb
    return emb / norm


def load_insight_embeddings():
    db = {}

    if not INSIGHT_EMBED_DIR.exists():
        print(f"InsightFace embedding folder not found: {INSIGHT_EMBED_DIR}")
        return db

    for person_dir in INSIGHT_EMBED_DIR.iterdir():
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        embeddings = []

        for emb_path in sorted(person_dir.glob("emb_*.npy")):
            try:
                emb = np.load(str(emb_path))
                embeddings.append(insight_normalize_embedding(emb))
            except Exception as e:
                print(f"Could not load embedding {emb_path}: {e}")

        if embeddings:
            db[person_name] = embeddings

    return db


INSIGHT_FACE_DB = load_insight_embeddings()

print("Loaded InsightFace embeddings:")
if not INSIGHT_FACE_DB:
    print("  none")
else:
    for name, embs in INSIGHT_FACE_DB.items():
        print(f"  {name}: {len(embs)} samples")


def recognize_insight_embedding(embedding):
    if not INSIGHT_FACE_DB:
        return None, 0.0, 0.0, {}

    emb = insight_normalize_embedding(embedding)
    scores = {}

    for person_name, known_embeddings in INSIGHT_FACE_DB.items():
        sims = [float(np.dot(emb, known)) for known in known_embeddings]
        sims = sorted(sims, reverse=True)
        top = sims[:min(5, len(sims))]
        scores[person_name] = sum(top) / len(top)

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_name, best_score = sorted_scores[0]
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
    margin = best_score - second_score

    print(
        f"InsightFace scores: {scores} | "
        f"candidate={best_name} score={best_score:.3f} margin={margin:.3f}"
    )

    # Normal confident match.
    if best_score >= INSIGHT_SIMILARITY_THRESHOLD and margin >= INSIGHT_MARGIN_THRESHOLD:
        return best_name, best_score, margin, scores

    # Low-score but very strong separation.
    # Useful for Marquinho when lighting/angle lowers absolute score but Marco is far away.
    if best_score >= 0.45 and margin >= 0.25:
        return best_name, best_score, margin, scores

    return None, best_score, margin, scores


def detect_face_state(queue, frames_to_scan=INSIGHT_SCAN_FRAMES):
    recognition_votes = {}
    recognition_scores = {}
    face_positions = []
    face_count_seen = 0
    best_overall_score = 0.0
    best_overall_margin = 0.0
    best_overall_scores = {}

    for _ in range(frames_to_scan):
        frame = queue.get().getCvFrame()
        faces = insight_app.get(frame)

        if not faces:
            continue

        face_count_seen = max(face_count_seen, len(faces))

        # Use largest face for identity and position.
        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )

        face = faces[0]
        x1, y1, x2, y2 = face.bbox.astype(int)
        cx = int((x1 + x2) / 2)

        if cx < FRAME_W * 0.35:
            position = "left"
        elif cx > FRAME_W * 0.65:
            position = "right"
        else:
            position = "center"

        face_positions.append(position)

        name, score, margin, scores = recognize_insight_embedding(face.embedding)

        if score > best_overall_score:
            best_overall_score = score
            best_overall_margin = margin
            best_overall_scores = scores

        if name:
            recognition_votes[name] = recognition_votes.get(name, 0) + 1
            recognition_scores.setdefault(name, []).append(score)

    if face_count_seen == 0:
        return {
            "face_detected": False,
            "face_count": 0,
            "face_position": "none",
            "recognized_person": None,
            "recognition_score": None,
            "recognition_margin": None,
            "recognition_votes": {},
            "recognition_scores": {},
            "recognizer": "insightface_arcface",
        }

    face_position = max(set(face_positions), key=face_positions.count) if face_positions else "center"

    if not recognition_votes:
        return {
            "face_detected": True,
            "face_count": face_count_seen,
            "face_position": face_position,
            "recognized_person": None,
            "recognition_score": best_overall_score,
            "recognition_margin": best_overall_margin,
            "recognition_votes": {},
            "recognition_scores": best_overall_scores,
            "recognizer": "insightface_arcface",
        }

    sorted_votes = sorted(recognition_votes.items(), key=lambda item: item[1], reverse=True)
    best_name, best_votes = sorted_votes[0]
    second_votes = sorted_votes[1][1] if len(sorted_votes) > 1 else 0

    avg_score = sum(recognition_scores[best_name]) / len(recognition_scores[best_name])

    print(
        f"InsightFace votes: {recognition_votes} | "
        f"best={best_name} votes={best_votes} second_votes={second_votes} avg_score={avg_score:.3f}"
    )

    # Conservative vote rule:
    # Accept if we have repeated accepted frames and winner is clearly ahead.
    if best_votes >= INSIGHT_MIN_ACCEPTED_VOTES and best_votes >= second_votes + 1:
        recognized_person = best_name
    else:
        recognized_person = None

    return {
        "face_detected": True,
        "face_count": face_count_seen,
        "face_position": face_position,
        "recognized_person": recognized_person,
        "recognition_score": avg_score,
        "recognition_margin": best_overall_margin,
        "recognition_votes": recognition_votes,
        "recognition_scores": best_overall_scores,
        "recognizer": "insightface_arcface",
    }



# ============================================================
# V6 Threaded Natural Conversation Mode
# Background vision cache + main audio/brain/speaker loop
# ============================================================

import threading

ACTIVE_FOLLOWUP_SECONDS = 90
CONVERSATION_GRACE_SECONDS = 90
VISION_UPDATE_SECONDS = 3.0
VISION_STALE_SECONDS = 8.0
IDENTITY_HYSTERESIS_SECONDS = 12.0

QUESTION_WORDS = [
    "what", "who", "when", "where", "why", "how",
    "can you", "could you", "do you", "are you", "is it",
    "will you", "would you"
]

REQUEST_WORDS = [
    "tell", "say", "look", "check", "explain", "calculate",
    "show", "remember", "describe", "give", "help", "find"
]

CONTINUATION_WORDS = {
    "yes", "no", "yeah", "yep", "nope", "maybe",
    "imagination", "real", "fiction", "continue", "exactly"
}

DIRECT_ADDRESS_WORDS = [
    "miguel", "robot", "mission control"
]

STOP_EVENT = threading.Event()
AUDIO_CAPTURE_ACTIVE = threading.Event()
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
        now = time.time()

        previous_person = SHARED_LOCAL_STATE.get("recognized_person")
        previous_update = SHARED_LOCAL_STATE.get("last_vision_update", 0.0)
        previous_age = now - previous_update

        # If the vision worker sees no face, clear camera identity immediately.
        # This prevents "ghost Marco" from persisting after the person leaves.
        if not new_state.get("face_detected"):
            new_state = dict(new_state)
            new_state["recognized_person"] = None
            new_state["recognition_votes"] = {}
            new_state["recognizer"] = "insightface_arcface_no_face_clear"

        # Identity hysteresis only applies to actual face-detected uncertain frames.
        # Do not use hysteresis for no-face frames.
        elif (
            previous_person
            and not new_state.get("recognized_person")
            and new_state.get("face_detected")
            and previous_age <= IDENTITY_HYSTERESIS_SECONDS
        ):
            new_state = dict(new_state)
            new_state["recognized_person"] = previous_person
            new_state["recognizer"] = "insightface_arcface_hysteresis"
            new_state.setdefault("recognition_votes", {previous_person: 1})

        SHARED_LOCAL_STATE.clear()
        SHARED_LOCAL_STATE.update(new_state)
        SHARED_LOCAL_STATE["last_vision_update"] = now


def vision_worker(camera_queue, pipeline):
    print("[VISION] Background vision worker started.")

    while not STOP_EVENT.is_set() and pipeline.isRunning():
        try:
            if get_robot_mode() == "sleep":
                time.sleep(5.0)
                continue
        except Exception:
            pass

        if AUDIO_CAPTURE_ACTIVE.is_set():
            time.sleep(0.2)
            continue

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

    if familiar_person and active_followup and words & CONTINUATION_WORDS:
        return True

    # Natural conversation mode:
    # If Miguel knows who is speaking, a normal sentence is probably intentional.
    if familiar_person and len(words) >= 3:
        return True

    if familiar_person and active_followup and len(words) >= 1:
        return True

    return False


def is_vision_state_fresh(local_state):
    last_update = local_state.get("last_vision_update", 0.0)
    return (time.time() - last_update) <= VISION_STALE_SECONDS



def is_simple_greeting(user_text):
    text = user_text.lower().strip()
    words = set(re.findall(r"\b\w+\b", text))

    greeting_words = {"hello", "hi", "hey"}
    question_request_words = {
        "what", "who", "when", "where", "why", "how",
        "see", "look", "tell", "check", "calculate",
        "record", "remember", "status", "weather", "time",
    }

    if not (words & greeting_words):
        return False

    if words & question_request_words:
        return False

    # Only short greeting-style turns.
    return len(words) <= 3



def make_last_known_person_state(person_name):
    return {
        "face_detected": True,
        "face_count": 1,
        "face_position": "center",
        "recognized_person": person_name,
        "recognition_score": None,
        "recognition_margin": None,
        "recognition_votes": {person_name: 1} if person_name else {},
        "recognition_scores": {},
        "recognizer": "conversation_grace_cached_identity",
        "last_vision_update": time.time(),
    }



CONVERSATION_MEMORY = {
    "last_user_text": "",
    "last_assistant_reply": "",
    "last_topic": "",
}


def update_conversation_memory(user_text=None, assistant_reply=None, topic=None):
    if user_text is not None:
        CONVERSATION_MEMORY["last_user_text"] = user_text
    if assistant_reply is not None:
        CONVERSATION_MEMORY["last_assistant_reply"] = assistant_reply
    if topic is not None:
        CONVERSATION_MEMORY["last_topic"] = topic


def infer_simple_topic(text):
    t = text.lower()
    if "life" in t or "extreme" in t or "extremophile" in t:
        return "life in extreme environments"
    if "gravity" in t:
        return "gravity"
    if "big bang" in t:
        return "Big Bang theory"
    if "creativ" in t:
        return "creativity"
    if "robot" in t or "face recognition" in t or "voice" in t:
        return "Miguel robot project"
    return CONVERSATION_MEMORY.get("last_topic", "")


def is_short_acknowledgement(user_text):
    t = user_text.lower().strip()
    return t in {"ok", "okay", "yes", "yeah", "yep", "no", "nope", "não", "sim"}


def handle_short_acknowledgement(user_text):
    t = user_text.lower().strip()
    last_reply = CONVERSATION_MEMORY.get("last_assistant_reply", "")
    last_topic = CONVERSATION_MEMORY.get("last_topic", "")

    if t in {"ok", "okay"}:
        if last_topic:
            return f"Okay. We can keep going about {last_topic}, or you can ask me the next question."
        return "Okay. I'm ready for your next question."

    if t in {"no", "nope", "não"}:
        return "Okay, no problem. Tell me what you want instead."

    if t in {"yes", "yeah", "yep", "sim"}:
        if last_reply:
            return "Yes — continuing from what I just said. What part should I explain next?"
        return "Yes. Tell me what you want to do next."

    return None



SESSION_IDENTITY_OVERRIDE = {
    "person": None,
    "created_at": 0.0,
    "reason": "",
}

VISION_HARD_STALE_SECONDS = 6.0


def set_session_identity_override(person, reason="manual"):
    person = (person or "").lower().strip().replace(" ", "_")
    if person not in {"marco", "marquinho"}:
        return False

    SESSION_IDENTITY_OVERRIDE["person"] = person
    SESSION_IDENTITY_OVERRIDE["created_at"] = time.time()
    SESSION_IDENTITY_OVERRIDE["reason"] = reason
    print(f"[IDENTITY] Manual session override set to {person}: {reason}")
    return True


def clear_session_identity_override():
    SESSION_IDENTITY_OVERRIDE["person"] = None
    SESSION_IDENTITY_OVERRIDE["created_at"] = 0.0
    SESSION_IDENTITY_OVERRIDE["reason"] = ""


def is_vision_question(user_text):
    text = user_text.lower()
    phrases = [
        "do you see",
        "can you see",
        "who am i",
        "who is this",
        "do you recognize",
        "anyone in front",
        "is anyone there",
        "what do you see",
        "how many faces",
        "ghost",
        "ghosts",
    ]
    return any(p in text for p in phrases)


def apply_identity_override_to_state(local_state):
    person = SESSION_IDENTITY_OVERRIDE.get("person")
    if not person:
        return local_state

    state = dict(local_state)
    state["recognized_person"] = person
    state["manual_identity_override"] = True
    state["manual_identity_reason"] = SESSION_IDENTITY_OVERRIDE.get("reason", "")
    return state


def handle_manual_identity_command(user_text):
    text = user_text.lower()

    if "i am marquinho" in text or "i'm marquinho" in text or "treat me as marquinho" in text:
        set_session_identity_override("marquinho", "user said they are Marquinho")
        return "Got it. For this session I will treat you as Chief Engineer Marquinho, even if the camera is unsure."

    if "i am marco" in text or "i'm marco" in text or "treat me as marco" in text:
        set_session_identity_override("marco", "user said they are Marco")
        return "Got it. For this session I will treat you as Marco."

    if "clear identity override" in text or "forget manual identity" in text:
        clear_session_identity_override()
        return "Manual identity override cleared. I will rely on camera recognition again."

    return None


def handle_user_turn_with_cached_state(user_text, cached_local_state):
    words = set(re.findall(r"\b\w+\b", user_text.lower()))

    if not user_text:
        print("[AUDIO] Empty transcript ignored.")
        return True

    explicit_exit_phrases = {
        "quit robot",
        "exit robot",
        "stop robot program",
        "stop miguel program",
        "end program",
    }

    normalized_text = user_text.lower().strip()

    if normalized_text in explicit_exit_phrases:
        speak("Miguel is shutting down conversation mode. Mission saved.")
        return False

    local_state = cached_local_state
    print("[BRAIN] Using cached local state:", local_state)

    mode_reply = handle_robot_mode_command(user_text, local_state.get("recognized_person"))
    if mode_reply == "__SILENT__":
        print("[MODE] Sleep mode: silently ignored user text.")
        return True

    if mode_reply == "__SHUTDOWN__":
        speak("Confirmed. Miguel is shutting down the Jetson now. Mission saved.")
        subprocess.Popen(["sudo", "shutdown", "now"])
        return False

    if mode_reply:
        speak(mode_reply)
        return True

    topic_reply = handle_long_term_topic_command(
        user_text,
        local_state.get("recognized_person"),
        CONVERSATION_MEMORY,
    )
    if topic_reply:
        speak(topic_reply)
        update_conversation_memory(assistant_reply=topic_reply)
        return True

    identity_reply = handle_manual_identity_command(user_text)
    if identity_reply:
        speak(identity_reply)
        update_conversation_memory(assistant_reply=identity_reply)
        return True

    # For vision questions, do not trust conversation-grace cached identity.
    # It may be stale when someone left or switched places.
    if is_vision_question(user_text):
        latest_state = get_cached_local_state()
        age = time.time() - latest_state.get("last_vision_update", 0.0)

        if latest_state.get("recognizer") == "conversation_grace_cached_identity" or age > VISION_HARD_STALE_SECONDS:
            local_state = latest_state
            local_state["recognized_person"] = None
            local_state["vision_warning"] = "camera state stale; need fresh vision update"
        else:
            local_state = latest_state

    local_state = apply_identity_override_to_state(local_state)

    if local_state.get("face_detected") and local_state.get("recognized_person") is None:
        print("[BRAIN] Unknown face detected. Auto-enrollment remains disabled during tuning.")

    topic = infer_simple_topic(user_text)
    update_conversation_memory(user_text=user_text, topic=topic)

    short_ack_reply = handle_short_acknowledgement(user_text)
    if short_ack_reply:
        speak(short_ack_reply)
        update_conversation_memory(assistant_reply=short_ack_reply)
        return True

    local_skill_reply = maybe_handle_local_skill(user_text, local_state)
    if local_skill_reply:
        speak(local_skill_reply)
        update_conversation_memory(assistant_reply=local_skill_reply)
        if should_auto_attach_to_topic(user_text):
            append_turn_to_active_topic(user_text, local_skill_reply, local_state.get("recognized_person"))
        return True

    if local_state.get("recognized_person") in CUSTOM_GREETINGS and is_simple_greeting(user_text):
        greeting = CUSTOM_GREETINGS[local_state["recognized_person"]]
        speak(greeting)
        update_conversation_memory(assistant_reply=greeting)
        return True

    try:
        reply = ask_cloud_brain(user_text, local_state)
    except Exception as e:
        print("[BRAIN] OpenAI API error:", e)
        reply = "My cloud brain is not reachable right now, but my local systems are still online."

    speak(reply)
    update_conversation_memory(assistant_reply=reply)
    if should_auto_attach_to_topic(user_text):
        append_turn_to_active_topic(user_text, reply, local_state.get("recognized_person"))
    return True



def ready_beep():
    """Short audible cue: Miguel is ready for the user to speak."""
    try:
        beep_path = Path(tempfile.gettempdir()) / "miguel_ready_beep.wav"

        # Generate a short beep using Python only.
        import wave as _wave
        import math as _math
        import struct as _struct

        sample_rate = 22050
        duration = 0.16
        frequency = 880
        volume = 0.22

        n_samples = int(sample_rate * duration)

        with _wave.open(str(beep_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)

            for i in range(n_samples):
                t = i / sample_rate
                # Tiny fade in/out to avoid click.
                fade = min(i / 800, (n_samples - i) / 800, 1.0)
                value = int(32767 * volume * fade * _math.sin(2 * _math.pi * frequency * t))
                wf.writeframes(_struct.pack("<h", value))

        subprocess.run(["aplay", "-q", "-D", SPEAKER_DEVICE, str(beep_path)], check=False)

    except Exception as e:
        print("[BEEP] Could not play ready beep:", e)



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
    last_known_person = None
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

                if get_robot_mode() == "sleep":
                    print("[SLEEP] Miguel is quiet. Wake phrase required.")
                    wake_detected = listen_for_wake(timeout_seconds=6)

                    if wake_detected:
                        # Route the wake phrase through mode command handling.
                        mode_reply = handle_robot_mode_command("Miguel wake up", recognized_person)
                        if mode_reply and mode_reply != "__SILENT__":
                            speak(mode_reply)
                        last_reply_time = time.time()

                    continue

                if recognized_person and vision_fresh:
                    last_known_person = recognized_person
                    friendly_person = recognized_person.replace("_", " ")

                    if last_announced_person != recognized_person:
                        speak(f"I see {friendly_person}. Conversation mode is ready.")
                        last_announced_person = recognized_person

                    print(f"[READY] Familiar person present: {friendly_person}. Start talking now.")
                    ready_beep()
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
                    fresh_state = get_cached_local_state()
                    keep_running = handle_user_turn_with_cached_state(user_text, fresh_state)
                    last_reply_time = time.time()

                elif last_known_person and (time.time() - last_reply_time) < CONVERSATION_GRACE_SECONDS:
                    # Keep natural follow-up mode alive even if vision briefly drops.
                    friendly_person = last_known_person.replace("_", " ")
                    local_state = get_cached_local_state()

                    if not local_state.get("recognized_person"):
                        local_state = make_last_known_person_state(last_known_person)

                    print(f"[FOLLOWUP] Grace mode active for {friendly_person}. Start talking now.")
                    ready_beep()
                    user_text = capture_user_turn()

                    if not user_text:
                        print("[AUDIO] No speech captured during follow-up grace.")
                        continue

                    print(f"[TRANSCRIPT] {user_text}")

                    # In grace mode, answer even short replies like yes/no/imagination.
                    print("[THINKING] Miguel is processing follow-up.")
                    keep_running = handle_user_turn_with_cached_state(user_text, local_state)
                    last_reply_time = time.time()

                else:
                    last_announced_person = None
                    print("[IDLE] No familiar person recognized. Wake phrase required.")
                    wake_detected = listen_for_wake(timeout_seconds=6)

                    # If the vision thread recognized someone while we were waiting,
                    # skip wake mode and return to the top of the loop.
                    latest_state = get_cached_local_state()
                    if not wake_detected and latest_state.get("recognized_person") and is_vision_state_fresh(latest_state):
                        print("[IDLE] Familiar person recognized during wake wait. Switching to natural mode.")
                        continue

                    if not wake_detected:
                        continue

                    speak("I'm listening now.")
                    print("[READY] Start talking now.")
                    ready_beep()

                    local_state = get_cached_local_state()
                    user_text = capture_user_turn()

                    if not user_text:
                        print("[AUDIO] No speech captured after wake phrase.")
                        continue

                    print(f"[TRANSCRIPT] {user_text}")
                    print("[THINKING] Miguel is processing.")
                    fresh_state = get_cached_local_state()
                    keep_running = handle_user_turn_with_cached_state(user_text, fresh_state)
                    last_reply_time = time.time()

        finally:
            STOP_EVENT.set()
            vision_thread.join(timeout=2)

    print("Miguel Cloud Brain V6 stopped.")


if __name__ == "__main__":
    run_v6_threaded_conversation()
