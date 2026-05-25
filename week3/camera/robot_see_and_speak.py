import cv2
import depthai as dai
import subprocess
import tempfile
from pathlib import Path

FRAME_W = 640
FRAME_H = 480

OUTPUT_AUDIO_CARD = "plughw:2,0"  # AB13X / USB-C audio adapter from current working setup

face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    raise RuntimeError(f"Could not load Haar cascade: {face_cascade_path}")

def speak(text: str):
    print(f"Robot says: {text}")
    wav_path = Path(tempfile.gettempdir()) / "robot_speech.wav"

    # espeak creates a WAV, then aplay sends it to the USB speaker adapter.
    subprocess.run(["espeak", "-w", str(wav_path), text], check=True)
    subprocess.run(["aplay", "-D", OUTPUT_AUDIO_CARD, str(wav_path)], check=True)

def detect_face_once(queue):
    best_faces = []

    # Scan multiple frames instead of just one frame
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

print("Starting Marquinho Bot voice + vision test.")
print("Commands:")
print("  look")
print("  hello")
print("  status")
print("  quit")

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput((FRAME_W, FRAME_H))
    queue = cam_out.createOutputQueue(maxSize=4, blocking=True)

    pipeline.start()

    speak("Robot vision system ready.")

    while pipeline.isRunning():
        command = input("\nType command for robot: ").strip().lower()

        if command in ["quit", "exit", "q"]:
            speak("Shutting down vision test.")
            break

        elif command in ["hello", "hi"]:
            speak("Hello Chief Engineer Marquinho. Hello Marco.")

        elif command in ["status"]:
            speak("My camera, speaker, and robot brain are working.")

        elif command in ["look", "look at me", "do you see me"]:
            found, count, side = detect_face_once(queue)
            if found:
                if count == 1:
                    speak(f"I see one face {side}.")
                else:
                    speak(f"I see {count} faces. The largest face is {side}.")
            else:
                speak("I do not see a face right now.")

        else:
            speak("I do not know that command yet.")
