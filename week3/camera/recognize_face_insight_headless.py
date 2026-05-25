import cv2
import depthai as dai
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis
import time

FRAME_W = 640
FRAME_H = 480

BASE = Path.home() / "robot-project/week3"
EMBED_DIR = BASE / "face_embeddings"

SIMILARITY_THRESHOLD = 0.42
MARGIN_THRESHOLD = 0.06

print("Loading InsightFace model...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))

def normalize_embedding(embedding):
    emb = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb
    return emb / norm

def load_embeddings():
    db = {}
    for person_dir in EMBED_DIR.iterdir():
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        embeddings = []

        for emb_path in sorted(person_dir.glob("emb_*.npy")):
            emb = np.load(str(emb_path))
            embeddings.append(normalize_embedding(emb))

        if embeddings:
            db[person_name] = embeddings

    return db

FACE_DB = load_embeddings()

print("Loaded embeddings:")
for name, embs in FACE_DB.items():
    print(f"  {name}: {len(embs)} samples")

def recognize_embedding(embedding):
    if not FACE_DB:
        return None, 0.0, {}

    emb = normalize_embedding(embedding)
    scores = {}

    for person_name, known_embeddings in FACE_DB.items():
        sims = [float(np.dot(emb, known)) for known in known_embeddings]
        sims = sorted(sims, reverse=True)
        top = sims[:min(5, len(sims))]
        scores[person_name] = sum(top) / len(top)

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_name, best_score = sorted_scores[0]
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
    margin = best_score - second_score

    if best_score >= SIMILARITY_THRESHOLD and margin >= MARGIN_THRESHOLD:
        return best_name, best_score, scores

    return None, best_score, scores

print()
print("Headless recognition test.")
print("Put ONE person in front of the camera.")
print("It will print 20 recognition attempts.")
print("Starting in 3 seconds...")
time.sleep(3)

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput((FRAME_W, FRAME_H))
    queue = cam_out.createOutputQueue(maxSize=4, blocking=True)

    pipeline.start()

    printed = 0

    while pipeline.isRunning() and printed < 20:
        frame = queue.get().getCvFrame()
        faces = app.get(frame)

        if not faces:
            print("No face")
            time.sleep(0.4)
            continue

        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )

        face = faces[0]
        name, score, scores = recognize_embedding(face.embedding)

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
        margin = score - second_score

        print(f"Result={name or 'unknown'} score={score:.3f} margin={margin:.3f} scores={scores}")

        printed += 1
        time.sleep(0.4)

print("Recognition test complete.")
