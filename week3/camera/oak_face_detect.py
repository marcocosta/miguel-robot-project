import cv2
import depthai as dai
from datetime import datetime

print("Starting OAK-D Lite face detection...")
print("Press Q in the camera window to quit.")

face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    raise RuntimeError(f"Could not load Haar cascade: {face_cascade_path}")

last_status = None

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput((640, 480))
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

        face_count = len(faces)

        for (x, y, w, h) in faces:
            cx = x + w // 2
            cy = y + h // 2

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
            cv2.putText(
                frame,
                "FACE",
                (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        status = f"Faces detected: {face_count}"

        cv2.putText(
            frame,
            status,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (20, 460),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        if status != last_status:
            print(f"{datetime.now().isoformat(timespec='seconds')} - {status}")
            last_status = status

        cv2.imshow("Marquinho Bot - Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
print("Face detection stopped.")
