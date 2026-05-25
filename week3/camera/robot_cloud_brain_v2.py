import cv2
import depthai as dai
import json
import os
import re
import subprocess
import tempfile
import time
import numpy as np
from pathlib import Path
from openai import OpenAI
from vosk import Model, KaldiRecognizer

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

SPEECH_RMS_THRESHOLD = 350
SILENCE_SECONDS = 1.3
MIN_TURN_SECONDS = 1.0
MAX_TURN_SECONDS = 10.0

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

def speak(text: str):
    print(f"{ROBOT_NAME} says: {text}")
    wav_path = Path(tempfile.gettempdir()) / "miguel_speech.wav"
    subprocess.run(["espeak", "-w", str(wav_path), text], check=True)
    subprocess.run(["aplay", "-D", SPEAKER_DEVICE, str(wav_path)], check=True)


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

def listen_for_wake():
    print(f"\n{ROBOT_NAME} is idle. Say: 'Hey Miguel', 'Hey me go', or 'Mission Control'.")

    proc = open_raw_mic_stream()
    recognizer = KaldiRecognizer(vosk_model, AUDIO_RATE)

    idle_start = time.time()
    last_reset = time.time()

    try:
        while True:
            raw = proc.stdout.read(CHUNK_BYTES)
            if not raw:
                continue

            mono_bytes, rms = stereo_raw_to_mono_bytes(raw)
            if not mono_bytes:
                continue

            final_text = ""
            partial_text = ""

            if recognizer.AcceptWaveform(mono_bytes):
                result = json.loads(recognizer.Result())
                final_text = result.get("text", "").lower().strip()

                if final_text:
                    print(f"Idle final: {final_text}")

                    if text_contains_any(final_text, WAKE_PHRASES):
                        print("Wake phrase detected from final text.")
                        return True

                    # Reset after a final result that was not a wake phrase.
                    recognizer = KaldiRecognizer(vosk_model, AUDIO_RATE)
                    last_reset = time.time()

            else:
                partial_text = json.loads(recognizer.PartialResult()).get("partial", "").lower().strip()

                if partial_text:
                    print(f"Idle partial: {partial_text}")

                    if text_contains_any(partial_text, WAKE_PHRASES):
                        print("Wake phrase detected from partial text.")
                        return True

            # Avoid accumulating long garbage partials forever.
            if time.time() - last_reset > 6:
                recognizer = KaldiRecognizer(vosk_model, AUDIO_RATE)
                last_reset = time.time()

            # Optional quiet reset: if no meaningful speech for a while, reset.
            if rms < SPEECH_RMS_THRESHOLD and time.time() - idle_start > 12:
                recognizer = KaldiRecognizer(vosk_model, AUDIO_RATE)
                idle_start = time.time()

    finally:
        stop_stream(proc)


def capture_user_turn():
    print("Listening for your turn...")

    proc = open_raw_mic_stream()
    recognizer = KaldiRecognizer(vosk_model, AUDIO_RATE)

    start_time = time.time()
    last_voice_time = start_time
    speech_started = False
    final_parts = []
    current_text_snapshot = ""

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

            if rms > SPEECH_RMS_THRESHOLD:
                speech_started = True
                last_voice_time = time.time()

            if recognizer.AcceptWaveform(mono_bytes):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").lower().strip()
                if text:
                    final_parts.append(text)
                    current_text_snapshot = " ".join(final_parts)
                    print(f"Heard segment: {text}")
            else:
                partial = json.loads(recognizer.PartialResult()).get("partial", "").lower().strip()
                if partial:
                    current_text_snapshot = (" ".join(final_parts) + " " + partial).strip()

            if text_contains_any(current_text_snapshot, END_TURN_PHRASES):
                print("End-turn cue detected.")
                break

            if speech_started:
                turn_age = time.time() - start_time
                silence_age = time.time() - last_voice_time

                if turn_age >= MIN_TURN_SECONDS and silence_age >= SILENCE_SECONDS:
                    print("Silence timeout reached.")
                    break

        final = json.loads(recognizer.FinalResult())
        final_text = final.get("text", "").lower().strip()

        if final_text:
            final_parts.append(final_text)

        user_text = " ".join(p for p in final_parts if p).strip()
        print(f"Final heard: {user_text}")
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
        }

    # Most common face position
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
            "recognition_votes": recognition_votes,
        }

    sorted_votes = sorted(recognition_votes.items(), key=lambda item: item[1], reverse=True)
    best_name, best_votes = sorted_votes[0]
    second_votes = sorted_votes[1][1] if len(sorted_votes) > 1 else 0

    avg_distance = sum(recognition_distances[best_name]) / len(recognition_distances[best_name])

    print(f"Recognition votes: {recognition_votes}")
    print(f"Best recognition: {best_name}, votes={best_votes}, second_votes={second_votes}, avg_distance={avg_distance:.3f}")

    # Conservative decision:
    # Miguel only recognizes someone if the same person wins repeatedly.
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
    text = re.sub(r"[^a-z0-9 ]", "", text)
    words = [w for w in text.split() if w not in {"my", "name", "is", "i", "am", "im"}]

    if not words:
        return None

    return "_".join(words[:2])


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

    speak("I see a new face. Do you want to be my friend? Please say yes or no.")

    answer = capture_user_turn()
    answer_words = set(re.findall(r"\b\w+\b", answer.lower()))

    if answer_words & NO_WORDS:
        speak("Okay. I will not learn this face now.")
        return False

    if not (answer_words & YES_WORDS):
        speak("I am not sure if that was yes. I will wait for another time.")
        return False

    speak("Great. What is your name?")
    name_text = capture_user_turn()
    person_name = sanitize_person_name(name_text)

    if not person_name:
        person_name = f"friend_{int(time.time())}"

    saved = capture_face_samples(queue, person_name, sample_count=15)

    if saved >= 5:
        FACE_TEMPLATES = load_face_templates()
        speak(f"Thank you {person_name.replace('_', ' ')}. I learned your face.")
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
- Do not invent things you cannot sense.
- If asked what you see, only use local_state.
- If recognized_person is present, you may greet that person by name.
- Do not claim you can move unless local_state says motion hardware exists.
"""

    payload = {
        "user_text": user_text,
        "local_state": local_state,
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

    if words & {"quit", "stop", "exit", "shutdown"}:
        speak("Miguel is shutting down conversation mode. Mission saved.")
        return False

    local_state = detect_face_state(queue)
    print("Local state:", local_state)

    if local_state["face_detected"] and local_state["recognized_person"] is None:
        ask_to_enroll_unknown_face(queue)
        local_state = detect_face_state(queue)

    if local_state["recognized_person"] in CUSTOM_GREETINGS and words & {"hello", "hi", "hey", "look", "see", "who"}:
        speak(CUSTOM_GREETINGS[local_state["recognized_person"]])
        return True

    try:
        reply = ask_cloud_brain(user_text, local_state)
    except Exception as e:
        print("OpenAI API error:", e)
        reply = "My cloud brain is not reachable right now, but my local systems are still online."

    speak(reply)
    return True


# ============================================================
# Main
# ============================================================

print(f"Starting {ROBOT_NAME} Cloud Brain V2.")
print_loaded_faces()
print("Wake phrases:", ", ".join(WAKE_PHRASES))
print("End-turn phrases:", ", ".join(END_TURN_PHRASES))

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput((FRAME_W, FRAME_H))
    queue = cam_out.createOutputQueue(maxSize=4, blocking=True)

    pipeline.start()

    speak("Miguel cloud brain version two is online. Say Hey Miguel when you need me.")

    keep_running = True

    while keep_running and pipeline.isRunning():
        listen_for_wake()
        speak("I'm listening.")
        user_text = capture_user_turn()
        keep_running = handle_user_turn(user_text, queue)

print("Miguel Cloud Brain V2 stopped.")
