import re
import cv2
import depthai as dai
import json
import subprocess
import tempfile
import wave
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
    face = gray_frame[y:y+h, x:x+w]
    face = cv2.resize(face, (FACE_SIZE, FACE_SIZE))
    face = cv2.equalizeHist(face)
    return face

def load_face_templates():
    templates = {}

    if not FACES_DIR.exists():
        return templates

    for person_dir in FACES_DIR.iterdir():
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        person_templates = []

        for img_path in sorted(person_dir.glob("face_*.png")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (FACE_SIZE, FACE_SIZE))
            img = cv2.equalizeHist(img)
            person_templates.append(img)

        if person_templates:
            templates[person_name] = person_templates

    return templates

FACE_TEMPLATES = load_face_templates()
print("Loaded face templates:")
for name, templates in FACE_TEMPLATES.items():
    print(f"  {name}: {len(templates)} samples")

def compare_face_to_templates(face_img):
    if not FACE_TEMPLATES:
        return None, 0.0

    best_name = None
    best_score = -1.0

    for person_name, templates in FACE_TEMPLATES.items():
        scores = []

        for template in templates:
            result = cv2.matchTemplate(face_img, template, cv2.TM_CCOEFF_NORMED)
            score = float(result[0][0])
            scores.append(score)

        # Use average of best few matches for stability
        scores = sorted(scores, reverse=True)
        top_scores = scores[:min(5, len(scores))]
        avg_score = sum(top_scores) / len(top_scores)

        if avg_score > best_score:
            best_score = avg_score
            best_name = person_name

    # Threshold tuned for simple prototype. Adjust if needed.
    if best_score >= 0.55:
        return best_name, best_score

    return None, best_score

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

    for _ in range(45):
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
        return False, 0, "nowhere", None, 0.0

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
    name, score = compare_face_to_templates(face_img)

    print(f"Recognition result: name={name}, score={score:.3f}")

    return True, face_count, side, name, score

def handle_command(text, queue):
    if not text:
        speak("I did not understand. Please try again.")
        return True

    if "quit" in text or "exit" in text or "stop" in text:
        speak("Stopping named face vision test.")
        return False

    words = set(re.findall(r"\b\w+\b", text))

    if "hello" in words or "hi" in words:
        speak("Hello Chief Engineer Marquinho. Hello Marco.")
        return True

    if "status" in text:
        speak("My microphone, speaker, camera, face memory, and robot brain are working.")
        return True

    if "look" in text or "see" in text or "face" in text or "who" in text:
        found, count, side, name, score = detect_and_identify_face(queue)

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

print("Starting Marquinho Bot named face recognition test.")
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
