import cv2
import depthai as dai
from datetime import datetime

FRAME_W = 640
FRAME_H = 480
CENTER_X = FRAME_W // 2
CENTER_Y = FRAME_H // 2

DEADZONE_X = 70
DEADZONE_Y = 55

print("Starting OAK-D Lite face position tracking...")
print("Press Q in the camera window to quit.")

face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    raise RuntimeError(f"Could not load Haar cascade: {face_cascade_path}")

last_direction = None

def get_direction(face_center_x, face_center_y):
    dx = face_center_x - CENTER_X
    dy = face_center_y - CENTER_Y

    if dx < -DEADZONE_X:
        horizontal = "LEFT"
    elif dx > DEADZONE_X:
        horizontal = "RIGHT"
    else:
        horizontal = "CENTER"

    if dy < -DEADZONE_Y:
        vertical = "UP"
    elif dy > DEADZONE_Y:
        vertical = "DOWN"
    else:
        vertical = "LEVEL"

    return horizontal, vertical, dx, dy

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
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60),
        )

        # Draw center target / deadzone
        cv2.line(frame, (CENTER_X, 0), (CENTER_X, FRAME_H), (255, 255, 255), 1)
        cv2.line(frame, (0, CENTER_Y), (FRAME_W, CENTER_Y), (255, 255, 255), 1)
        cv2.rectangle(
            frame,
            (CENTER_X - DEADZONE_X, CENTER_Y - DEADZONE_Y),
            (CENTER_X + DEADZONE_X, CENTER_Y + DEADZONE_Y),
            (255, 255, 255),
            1,
        )

        if len(faces) > 0:
            # Pick largest face
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]
            face_center_x = x + w // 2
            face_center_y = y + h // 2

            horizontal, vertical, dx, dy = get_direction(face_center_x, face_center_y)
            direction = f"{horizontal}-{vertical}"

            # Draw face box and vector
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.circle(frame, (face_center_x, face_center_y), 6, (255, 255, 255), -1)
            cv2.line(frame, (CENTER_X, CENTER_Y), (face_center_x, face_center_y), (255, 255, 255), 2)

            status = f"Face: {direction}  dx={dx} dy={dy}"

            if direction != last_direction:
                print(f"{datetime.now().isoformat(timespec='seconds')} - {status}")
                last_direction = direction

        else:
            status = "Face: NOT FOUND"
            if status != last_direction:
                print(f"{datetime.now().isoformat(timespec='seconds')} - {status}")
                last_direction = status

        cv2.putText(
            frame,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "Move face left/right/up/down. Press Q to quit.",
            (20, FRAME_H - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Marquinho Bot - Face Position Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
print("Face tracking stopped.")
