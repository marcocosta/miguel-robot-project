import cv2
import depthai as dai
from pathlib import Path
import sys

FRAME_W = 640
FRAME_H = 480
FACE_SIZE = 160

BASE = Path.home() / "robot-project/week3"
FACES_DIR = BASE / "faces"
FACES_DIR.mkdir(parents=True, exist_ok=True)

face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    raise RuntimeError(f"Could not load Haar cascade: {face_cascade_path}")

if len(sys.argv) < 2:
    print("Usage:")
    print("  python enroll_face.py marquinho")
    print("  python enroll_face.py marco")
    sys.exit(1)

person_name = sys.argv[1].strip().lower().replace(" ", "_")
person_dir = FACES_DIR / person_name
person_dir.mkdir(parents=True, exist_ok=True)

existing = list(person_dir.glob("face_*.png"))
capture_count = len(existing)

print(f"Enrolling face for: {person_name}")
print("Instructions:")
print("  - Look at the camera")
print("  - Keep good lighting")
print("  - Press C to capture a face")
print("  - Capture 10 to 20 samples")
print("  - Press Q to quit")

def preprocess_face(gray_frame, face_box):
    x, y, w, h = face_box
    face = gray_frame[y:y+h, x:x+w]
    face = cv2.resize(face, (FACE_SIZE, FACE_SIZE))
    face = cv2.equalizeHist(face)
    return face

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput((FRAME_W, FRAME_H))
    queue = cam_out.createOutputQueue(maxSize=4, blocking=True)

    pipeline.start()

    while pipeline.isRunning():
        frame = queue.get().getCvFrame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 60),
        )

        largest_face = None

        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            largest_face = faces[0]
            x, y, w, h = largest_face

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
            cv2.putText(frame, "FACE READY - PRESS C", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        else:
            cv2.putText(frame, "NO FACE FOUND", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.putText(frame, f"Person: {person_name}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Samples: {capture_count}", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, "C=capture  Q=quit", (20, FRAME_H - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        cv2.imshow("Marquinho Bot - Enroll Face", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("c") and largest_face is not None:
            face_img = preprocess_face(gray, largest_face)
            capture_count += 1
            out_path = person_dir / f"face_{capture_count:03d}.png"
            cv2.imwrite(str(out_path), face_img)
            print(f"Saved: {out_path}")

cv2.destroyAllWindows()
print(f"Enrollment complete for {person_name}. Samples saved: {capture_count}")
