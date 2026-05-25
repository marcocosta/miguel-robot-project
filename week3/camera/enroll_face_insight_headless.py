import cv2
import depthai as dai
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis
import sys
import time

FRAME_W = 640
FRAME_H = 480

BASE = Path.home() / "robot-project/week3"
EMBED_DIR = BASE / "face_embeddings"
PREVIEW_DIR = BASE / "face_embedding_previews"

EMBED_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

if len(sys.argv) < 2:
    print("Usage:")
    print("  python enroll_face_insight_headless.py marco")
    print("  python enroll_face_insight_headless.py marquinho")
    sys.exit(1)

person_name = sys.argv[1].strip().lower().replace(" ", "_")
target_samples = int(sys.argv[2]) if len(sys.argv) >= 3 else 30

person_embed_dir = EMBED_DIR / person_name
person_preview_dir = PREVIEW_DIR / person_name
person_embed_dir.mkdir(parents=True, exist_ok=True)
person_preview_dir.mkdir(parents=True, exist_ok=True)

print("Loading InsightFace model...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))

existing = list(person_embed_dir.glob("emb_*.npy"))
capture_count = len(existing)
saved_this_run = 0

print(f"Headless enrollment for: {person_name}")
print(f"Target new samples: {target_samples}")
print("Instructions:")
print("  - Put ONLY this person in front of the camera")
print("  - Good front lighting")
print("  - Slowly move face: center, slight left, slight right, closer, farther")
print("  - Capturing starts in 5 seconds")
print("  - Press Ctrl+C to stop early")
print()

def normalize_embedding(embedding):
    emb = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb
    return emb / norm

for i in range(5, 0, -1):
    print(f"Starting in {i}...")
    time.sleep(1)

last_save = 0
start = time.time()

try:
    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build()
        cam_out = cam.requestOutput((FRAME_W, FRAME_H))
        queue = cam_out.createOutputQueue(maxSize=4, blocking=True)

        pipeline.start()

        while pipeline.isRunning() and saved_this_run < target_samples:
            frame = queue.get().getCvFrame()
            faces = app.get(frame)

            if not faces:
                print("No face found...")
                time.sleep(0.2)
                continue

            faces = sorted(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True,
            )
            face = faces[0]

            x1, y1, x2, y2 = face.bbox.astype(int)
            face_w = x2 - x1
            face_h = y2 - y1

            # Basic quality gate: face should not be tiny.
            if face_w < 80 or face_h < 80:
                print(f"Face too small: {face_w}x{face_h}. Move closer.")
                time.sleep(0.2)
                continue

            now = time.time()
            if now - last_save < 0.45:
                continue

            capture_count += 1
            saved_this_run += 1
            last_save = now

            emb = normalize_embedding(face.embedding)
            emb_path = person_embed_dir / f"emb_{capture_count:03d}.npy"
            np.save(str(emb_path), emb)

            preview = frame.copy()
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(FRAME_W, x2)
            y2 = min(FRAME_H, y2)
            cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 255, 255), 2)

            preview_path = person_preview_dir / f"face_{capture_count:03d}.jpg"
            cv2.imwrite(str(preview_path), preview)

            print(f"Saved {saved_this_run}/{target_samples}: {emb_path}")

except KeyboardInterrupt:
    print("Stopped early by user.")

elapsed = time.time() - start
print()
print(f"Enrollment complete for {person_name}.")
print(f"New samples saved: {saved_this_run}")
print(f"Total samples now: {capture_count}")
print(f"Elapsed seconds: {elapsed:.1f}")
