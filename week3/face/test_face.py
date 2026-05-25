# ~/robot-project/week3/face/test_face.py
# Last updated: 20260522

from __future__ import annotations

import queue
import time

from face_state import FaceEvent, FaceMode
from face_thread import FaceThread


def main() -> None:
    face_events: "queue.Queue[FaceEvent]" = queue.Queue()

    face = FaceThread(
        event_queue=face_events,
        width=1024,
        height=600,
        fullscreen=True,
        fps=30,
    )

    face.start()

    sequence = [
        (FaceMode.IDLE, "Miguel is awake", 3),
        (FaceMode.LISTENING, "Listening", 3),
        (FaceMode.THINKING, "Thinking", 3),
        (FaceMode.SPEAKING, "Speaking", 5),
        (FaceMode.HAPPY, "Happy", 3),
        (FaceMode.CONFUSED, "Confused", 3),
        (FaceMode.SLEEPING, "", 3),
    ]

    try:
        while True:
            for mode, text, seconds in sequence:
                face_events.put(FaceEvent(mode=mode, text=text))
                time.sleep(seconds)

    except KeyboardInterrupt:
        face.stop()
        face.join(timeout=2)


if __name__ == "__main__":
    main()
