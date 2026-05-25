import cv2
import depthai as dai
import json
import re
import subprocess
import tempfile
import wave
import numpy as np
from pathlib import Path
from vosk import Model, KaldiRecognizer

FRAME_W = 640
FRAME_H = 480
FACE_SIZE = 160

MIC_DEVICE = "hw:0,0"
SPEAKER_DEVICE = "plughw:2,0"

BASE = Path.home() / "robot-project/week3"
AUDIO_DIR = BASE / "audio"
MODEL_PATH = BASE / "models/vosk-model-small-en-us-0.15"
FACES_DIR = BASE / "faces"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)

CUSTOM_GREETINGS = {
    "marquinho": "Hello Chief Engineer Marquinho. I recognize you.",
    "marco": "Hello Marco. Systems Engineer online.",
}

face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    raise RuntimeError(f"Could not load Haar cascade: {face_cascade_path}")

if not MODEL_PATH.exists():
    raise RuntimeError(f"Vosk model not found at: {MODEL_PATH}")

model = Model(str(MODEL_PATH))

def speak(text: str):
    print(f"Robot says: {text}")
    wav_path = Path(tempfile.gettempdir()) / "robot_speech.wav"
    subprocess.run(["espeak", "-w", str(wav_path), text], check=True)
    subprocess.run(["aplay", "-D", SPEAKER_DEVICE, str(wav_path)], check=True)

def preprocess_face(gray_frame, face_box):
    x, y, w, h = face_box

    # Add margin around face for better consistency
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
            cell = lbp[gy*cell_h:(gy+1)*cell_h, gx*cell_w:(gx+1)*cell_w]
            hist, _ = np.histogram(cell, bins=256, range=(0, 256))
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-6)
            features.append(hist)

    return np.concatenate(features)

def chi_square_distance(a, b):
    return 0.5 * np.sum(((a - b) ** 2) / (a + b + 1e-6))

def load_face_templates():
    templates = {}

    if not FACES_DIR.exists():
        return templates

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

print("Loaded face templates:")
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
        distances = [chi_square_distance(feature, t) for t in templates]
        distances = sorted(distances)

        # Average the best few matches for stability
        best_few = distances[:min(5, len(distances))]
        avg_distance = sum(best_few) / len(best_few)
        person_scores[person_name] = avg_distance

        if avg_distance < best_distance:
            best_distance = avg_distance
            best_name = person_name

    # Lower is better. This threshold may need tuning.
    # Typical starting range: 18-35 depending on lighting/samples.
    if best_distance <= 32.0:
        return best_name, best_distance, person_scores

    return None, best_distance, person_scores

def record_command(seconds=4):
    stereo_path = AUDIO_DIR / "voice_command_stereo.wav"
    mono_path = AUDIO_DIR / "voice_command_mono.wav"

    print(f"\nListening for {seconds} seconds...")
    subprocess.run([
        "arecord",
        "-D", MIC_DEVICE,
        "-f", "S16_LE",
        "-r", "16000",
        "-c", "2",
        "-d", str(seconds),
        str(stereo_path),
    ], check=True)

    subprocess.run([
        "sox",
        str(stereo_path),
        "-c", "1",
        "-r", "16000",
        str(mono_path),
    ], check=True)

    return mono_path

def transcribe(wav_path: Path):
    with wave.open(str(wav_path), "rb") as wf:
        recognizer = KaldiRecognizer(model, wf.getframerate())
        text_parts = []

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text_parts.append(result.get("text", ""))

        final_result = json.loads(recognizer.FinalResult())
        text_parts.append(final_result.get("text", ""))

    text = " ".join(t for t in text_parts if t).strip().lower()
    print(f"Heard: {text}")
    return text

def detect_and_identify_face(queue):
    best_faces = []
    best_frame_gray = None

    for _ in range(60):
        frame = queue.get().getCvFrame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(45, 45),
        )

        if len(faces) > 0:
            best_faces = faces
            best_frame_gray = gray
            break

    face_count = len(best_faces)

    if face_count == 0:
        return False, 0, "nowhere", None, 999.0, {}

    faces = sorted(best_faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    cx = x + w // 2

    if cx < FRAME_W * 0.35:
        side = "on my left"
    elif cx > FRAME_W * 0.65:
        side = "on my right"
    else:
        side = "in front of me"

    face_img = preprocess_face(best_frame_gray, faces[0])
    name, distance, scores = recognize_face(face_img)

    print(f"Recognition scores: {scores}")
    print(f"Recognition result: name={name}, distance={distance:.3f}")

    return True, face_count, side, name, distance, scores

def handle_command(text, queue):
    words = set(re.findall(r"\b\w+\b", text))

    if not text:
        speak("I did not understand. Please try again.")
        return True

    if "quit" in words or "exit" in words or "stop" in words:
        speak("Stopping named face vision test.")
        return False

    if "hello" in words or "hi" in words:
        speak("Hello Chief Engineer Marquinho. Hello Marco.")
        return True

    if "status" in words:
        speak("My microphone, speaker, camera, face memory, and robot brain are working.")
        return True

    if "look" in words or "see" in words or "face" in words or "who" in words:
        found, count, side, name, distance, scores = detect_and_identify_face(queue)

        if not found:
            speak("I do not see a face right now.")
            return True

        if name:
            greeting = CUSTOM_GREETINGS.get(name, f"Hello {name}. I recognize you.")
            speak(greeting)
        else:
            speak(f"I see a face {side}, but I do not recognize who it is yet.")

        return True

    speak("I heard you, but I do not know that command yet.")
    return True

print("Starting Marquinho Bot LBP named face recognition test.")
print("Say commands like: hello, status, look, who is this, do you see me, or quit.")

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput((FRAME_W, FRAME_H))
    queue = cam_out.createOutputQueue(maxSize=4, blocking=True)

    pipeline.start()

    speak("Named face recognition system ready.")

    keep_running = True

    while keep_running and pipeline.isRunning():
        input("\nPress ENTER, then speak a command...")
        wav_path = record_command(seconds=4)
        command_text = transcribe(wav_path)
        keep_running = handle_command(command_text, queue)

print("Named face vision test stopped.")
