import time
import depthai as dai

print("Starting DepthAI queue probe with pipeline.start()...")

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput((640, 480))
    q = cam_out.createOutputQueue(maxSize=1, blocking=False)

    pipeline.start()

    start = time.time()
    got = 0

    while time.time() - start < 5:
        msg = None

        if hasattr(q, "tryGetAll"):
            msgs = q.tryGetAll()
            if msgs:
                msg = msgs[-1]

        if msg is None and hasattr(q, "tryGet"):
            msg = q.tryGet()

        if msg is not None:
            frame = msg.getCvFrame()
            got += 1
            print(f"Frame {got}: shape={frame.shape}")
            time.sleep(0.2)
        else:
            print("No frame yet...")
            time.sleep(0.2)

print(f"Probe complete. Frames received: {got}")
