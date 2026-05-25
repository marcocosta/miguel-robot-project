import cv2
import depthai as dai
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis
import sys

FRAME_W = 640
FRAME_H = 480

BASE = Path.home() / "robot-project/week3"
EMBED_DIR = BASE / "face_embeddings"
PREVIEW_DIR = BASE / "face_embedding_previews"

EMBED_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

if len(sys.argv) < 2:
    print("Usage:")
    print("  python enroll_face_insight.py marco")
    print("  python enroll_face_insight.py marquinho")
    sys.exit(1)

person_name = sys.argv[1].strip().lower().replace(" ", "_")
person_embed_dir = EMBED_DIR / person_name
person_preview_dir = PREVIEW_DIR / person_name
person_embed_dir.mkdir(parents=True, exist_ok=True)
person_preview_dir.mkdir(parents=True, exist_ok=True)

print("Loading InsightFace model...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))

existing = list(person_embed_dir.glob("emb_*.npy"))
capture_count = len(existing)

print(f"Enrolling InsightFace embeddings for: {person_name}")
print("Instructions:")
print("  - Only this person should be visible")
print("  - Good front lighting")
print("  - Press C to capture embedding")
print("  - Capture 20-40 samples")
print("  - Press Q to quit")

def normalize_embedding(embedding):
    emb = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb
    return emb / norm

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput((FRAME_W, FRAME_H))
    queue = cam_out.createOutputQueue(maxSize=4, blocking=True)

    pipeline.start()

    while pipeline.isRunning():
        frame = queue.get().getCvFrame()
        faces = app.get(frame)

        largest_face = None

        if faces:
            faces = sorted(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True,
            )
            largest_face = faces[0]

            x1, y1, x2, y2 = largest_face.bbox.astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(FRAME_W, x2)
            y2 = min(FRAME_H, y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, "FACE READY - PRESS C", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
        else:
            cv2.putText(frame, "NO FACE FOUND", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

        cv2.putText(frame, f"Person: {person_name}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.putText(frame, f"Samples: {capture_count}", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.putText(frame, "C=capture  Q=quit", (20, FRAME_H - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        cv2.imshow("Miguel - InsightFace Enrollment", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("c") and largest_face is not None:
            capture_count += 1

            emb = normalize_embedding(largest_face.embedding)
            emb_path = person_embed_dir / f"emb_{capture_count:03d}.npy"
            np.save(str(emb_path), emb)

            preview_path = person_preview_dir / f"face_{capture_count:03d}.jpg"
            cv2.imwrite(str(preview_path), frame)

            print(f"Saved embedding: {emb_path}")

cv2.destroyAllWindows()
print(f"Enrollment complete for {person_name}. Samples saved: {capture_count}")
