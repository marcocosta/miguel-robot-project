import cv2
import depthai as dai
import json
import subprocess
import tempfile
import wave
from pathlib import Path

from openai import OpenAI
from vosk import Model, KaldiRecognizer

FRAME_W = 640
FRAME_H = 480

MIC_DEVICE = "hw:CARD=Array,DEV=0"
SPEAKER_DEVICE = "plughw:CARD=Audio,DEV=0"

BASE = Path.home() / "robot-project/week3"
AUDIO_DIR = BASE / "audio"
MODEL_PATH = BASE / "models/vosk-model-small-en-us-0.15"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI()

face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    raise RuntimeError(f"Could not load Haar cascade: {face_cascade_path}")

if not MODEL_PATH.exists():
    raise RuntimeError(f"Vosk model not found at: {MODEL_PATH}")

vosk_model = Model(str(MODEL_PATH))


def speak(text: str):
    print(f"Robot says: {text}")
    wav_path = Path(tempfile.gettempdir()) / "robot_speech.wav"
    subprocess.run(["espeak", "-w", str(wav_path), text], check=True)
    subprocess.run(["aplay", "-D", SPEAKER_DEVICE, str(wav_path)], check=True)


def record_command(seconds=4):
    stereo_path = AUDIO_DIR / "cloud_command_stereo.wav"
    mono_path = AUDIO_DIR / "cloud_command_mono.wav"

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
        recognizer = KaldiRecognizer(vosk_model, wf.getframerate())
        parts = []

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                parts.append(result.get("text", ""))

        final = json.loads(recognizer.FinalResult())
        parts.append(final.get("text", ""))

    text = " ".join(p for p in parts if p).strip().lower()
    print(f"Heard: {text}")
    return text


def detect_face_state(queue):
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

    if len(best_faces) == 0:
        return {
            "face_detected": False,
            "face_count": 0,
            "face_position": "none",
        }

    faces = sorted(best_faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    cx = x + w // 2

    if cx < FRAME_W * 0.35:
        position = "left"
    elif cx > FRAME_W * 0.65:
        position = "right"
    else:
        position = "center"

    return {
        "face_detected": True,
        "face_count": len(best_faces),
        "face_position": position,
    }


def ask_cloud_brain(user_text: str, local_state: dict):
    system_instructions = """
You are the high-level conversation brain for a small father-son robot project.
The robot was built by Chief Engineer Marquinho, age 9, and Systems Engineer Marco.
Respond like a friendly, curious robot assistant.

Rules:
- Keep replies short: 1 or 2 sentences.
- Use the local_state facts when relevant.
- Do not claim the robot can do things it cannot do.
- Do not invent sensor information.
- If asked what you see, only use local_state.
- Keep a fun mission-control tone, but not too exaggerated.
"""

    prompt = {
        "user_text": user_text,
        "local_state": local_state,
        "robot_capabilities": [
            "hear short voice commands through ReSpeaker",
            "speak through Creative speaker",
            "detect whether a face is visible",
            "estimate whether the largest face is left, center, or right",
            "hold a short conversation through ChatGPT API",
        ],
    }

    response = client.responses.create(
        model="gpt-5.2",
        instructions=system_instructions,
        input=json.dumps(prompt),
    )

    return response.output_text.strip()


print("Starting Marquinho Bot Cloud Brain.")
print("Press ENTER, speak, and the robot will answer using local sensors + ChatGPT.")
print("Say 'quit' or 'stop' to exit.")

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput((FRAME_W, FRAME_H))
    queue = cam_out.createOutputQueue(maxSize=4, blocking=True)

    pipeline.start()

    speak("Cloud brain conversation system ready.")

    while pipeline.isRunning():
        input("\nPress ENTER, then speak...")
        text = transcribe(record_command(seconds=4))

        if "quit" in text or "stop" in text or "exit" in text:
            speak("Cloud brain shutting down. Mission saved.")
            break

        if not text:
            speak("I did not understand. Please try again.")
            continue

        local_state = detect_face_state(queue)
        print("Local state:", local_state)

        try:
            reply = ask_cloud_brain(text, local_state)
        except Exception as e:
            print("OpenAI API error:", e)
            reply = "My cloud brain is not reachable right now, but my local systems are still online."

        speak(reply)

print("Cloud brain stopped.")
