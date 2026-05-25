# ~/robot-project/week3/face/test_face_controller.py
# Last updated: 20260522

from __future__ import annotations

import time

from face_controller import FaceController


def main() -> None:
    face = FaceController(width=1024, height=600, fullscreen=True)
    face.start()

    try:
        while True:
            face.idle()
            time.sleep(2)

            face.listening()
            time.sleep(2)

            face.thinking()
            time.sleep(2)

            face.speaking("Miguel is speaking")
            time.sleep(4)

            face.happy("Hello Marquinho")
            time.sleep(2)

            face.confused("I am not sure")
            time.sleep(2)

    except KeyboardInterrupt:
        face.stop()


if __name__ == "__main__":
    main()
