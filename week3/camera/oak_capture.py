import cv2
import depthai as dai
from pathlib import Path

output_path = Path.home() / "robot-project/week3/camera/oak_first_capture.jpg"

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput((640, 480))
    queue = cam_out.createOutputQueue(maxSize=4, blocking=True)

    pipeline.start()
    frame = queue.get().getCvFrame()

    cv2.imwrite(str(output_path), frame)
    print(f"Saved image to: {output_path}")
