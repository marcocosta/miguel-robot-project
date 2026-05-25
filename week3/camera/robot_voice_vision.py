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

MIC_DEVICE = "hw:CARD=Array,DEV=0"  # ReSpeaker XVF3800
SPEAKER_DEVICE = "plughw:CARD=Audio,DEV=0"  # USB audio adapter / Creative speaker

BASE = Path.home() / "robot-project/week3"
AUDIO_DIR = BASE / "audio"
MODEL_PATH = BASE / "models/vosk-model-small-en-us-0.15"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)

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

    # Convert ReSpeaker stereo capture to mono for Vosk.
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

def detect_face_once(queue):
    best_faces = []

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
            break

    face_count = len(best_faces)

    if face_count > 0:
        faces = sorted(best_faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        cx = x + w // 2

        if cx < FRAME_W * 0.35:
            side = "on my left"
        elif cx > FRAME_W * 0.65:
            side = "on my right"
        else:
            side = "in front of me"

        return True, face_count, side

    return False, 0, "nowhere"

def handle_command(text, queue):
    if not text:
        speak("I did not understand. Please try again.")
        return True

    if "quit" in text or "exit" in text or "stop" in text:
        speak("Stopping voice vision test.")
        return False

    if "hello" in text or "hi" in text:
        speak("Hello Chief Engineer Marquinho. Hello Marco.")
        return True

    if "status" in text:
        speak("My microphone, speaker, camera, and robot brain are working.")
        return True

    if "look" in text or "see" in text or "face" in text:
        found, count, side = detect_face_once(queue)

        if found:
            if count == 1:
                speak(f"I see one face {side}.")
            else:
                speak(f"I see {count} faces. The largest face is {side}.")
        else:
            speak("I do not see a face right now.")

        return True

    speak("I heard you, but I do not know that command yet.")
    return True

print("Starting Marquinho Bot voice command plus vision test.")
print("Say commands like: hello, status, look, do you see me, or quit.")

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput((FRAME_W, FRAME_H))
    queue = cam_out.createOutputQueue(maxSize=4, blocking=True)

    pipeline.start()

    speak("Voice command system ready.")

    keep_running = True

    while keep_running and pipeline.isRunning():
        input("\nPress ENTER, then speak a command...")
        wav_path = record_command(seconds=4)
        command_text = transcribe(wav_path)
        keep_running = handle_command(command_text, queue)

print("Voice vision test stopped.")
