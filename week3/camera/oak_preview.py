import cv2
import depthai as dai

print("Starting OAK-D Lite live preview...")
print("Press Q in the camera window to quit.")

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput((640, 480))
    queue = cam_out.createOutputQueue(maxSize=4, blocking=True)

    pipeline.start()

    while pipeline.isRunning():
        frame = queue.get().getCvFrame()

        cv2.putText(
            frame,
            "OAK-D Lite Live Preview - Press Q to quit",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Marquinho Bot - OAK-D Lite Preview", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
print("Preview stopped.")
