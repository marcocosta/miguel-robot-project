from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v5.py"
text = path.read_text()

marker = "# ============================================================\n# V4 Natural Conversation Mode"
idx = text.find(marker)

if idx == -1:
    raise SystemExit("Could not find V4 Natural Conversation Mode marker in V5 file.")

prefix = text[:idx]
suffix = text[idx:]

insight_block = r'''
# ============================================================
# InsightFace / ArcFace Offline Recognition Override
# This overrides the older LBP detect_face_state implementation.
# ============================================================

from insightface.app import FaceAnalysis

INSIGHT_EMBED_DIR = BASE / "face_embeddings"
INSIGHT_SIMILARITY_THRESHOLD = 0.50
INSIGHT_MARGIN_THRESHOLD = 0.08
INSIGHT_MIN_ACCEPTED_VOTES = 2
INSIGHT_SCAN_FRAMES = 6

print("Loading Miguel InsightFace recognizer...")
insight_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
insight_app.prepare(ctx_id=-1, det_size=(640, 640))


def insight_normalize_embedding(embedding):
    emb = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb
    return emb / norm


def load_insight_embeddings():
    db = {}

    if not INSIGHT_EMBED_DIR.exists():
        print(f"InsightFace embedding folder not found: {INSIGHT_EMBED_DIR}")
        return db

    for person_dir in INSIGHT_EMBED_DIR.iterdir():
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        embeddings = []

        for emb_path in sorted(person_dir.glob("emb_*.npy")):
            try:
                emb = np.load(str(emb_path))
                embeddings.append(insight_normalize_embedding(emb))
            except Exception as e:
                print(f"Could not load embedding {emb_path}: {e}")

        if embeddings:
            db[person_name] = embeddings

    return db


INSIGHT_FACE_DB = load_insight_embeddings()

print("Loaded InsightFace embeddings:")
if not INSIGHT_FACE_DB:
    print("  none")
else:
    for name, embs in INSIGHT_FACE_DB.items():
        print(f"  {name}: {len(embs)} samples")


def recognize_insight_embedding(embedding):
    if not INSIGHT_FACE_DB:
        return None, 0.0, 0.0, {}

    emb = insight_normalize_embedding(embedding)
    scores = {}

    for person_name, known_embeddings in INSIGHT_FACE_DB.items():
        sims = [float(np.dot(emb, known)) for known in known_embeddings]
        sims = sorted(sims, reverse=True)
        top = sims[:min(5, len(sims))]
        scores[person_name] = sum(top) / len(top)

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_name, best_score = sorted_scores[0]
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
    margin = best_score - second_score

    print(
        f"InsightFace scores: {scores} | "
        f"candidate={best_name} score={best_score:.3f} margin={margin:.3f}"
    )

    if best_score >= INSIGHT_SIMILARITY_THRESHOLD and margin >= INSIGHT_MARGIN_THRESHOLD:
        return best_name, best_score, margin, scores

    return None, best_score, margin, scores


def detect_face_state(queue, frames_to_scan=INSIGHT_SCAN_FRAMES):
    recognition_votes = {}
    recognition_scores = {}
    face_positions = []
    face_count_seen = 0
    best_overall_score = 0.0
    best_overall_margin = 0.0
    best_overall_scores = {}

    for _ in range(frames_to_scan):
        frame = queue.get().getCvFrame()
        faces = insight_app.get(frame)

        if not faces:
            continue

        face_count_seen = max(face_count_seen, len(faces))

        # Use largest face for identity and position.
        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )

        face = faces[0]
        x1, y1, x2, y2 = face.bbox.astype(int)
        cx = int((x1 + x2) / 2)

        if cx < FRAME_W * 0.35:
            position = "left"
        elif cx > FRAME_W * 0.65:
            position = "right"
        else:
            position = "center"

        face_positions.append(position)

        name, score, margin, scores = recognize_insight_embedding(face.embedding)

        if score > best_overall_score:
            best_overall_score = score
            best_overall_margin = margin
            best_overall_scores = scores

        if name:
            recognition_votes[name] = recognition_votes.get(name, 0) + 1
            recognition_scores.setdefault(name, []).append(score)

    if face_count_seen == 0:
        return {
            "face_detected": False,
            "face_count": 0,
            "face_position": "none",
            "recognized_person": None,
            "recognition_score": None,
            "recognition_margin": None,
            "recognition_votes": {},
            "recognition_scores": {},
            "recognizer": "insightface_arcface",
        }

    face_position = max(set(face_positions), key=face_positions.count) if face_positions else "center"

    if not recognition_votes:
        return {
            "face_detected": True,
            "face_count": face_count_seen,
            "face_position": face_position,
            "recognized_person": None,
            "recognition_score": best_overall_score,
            "recognition_margin": best_overall_margin,
            "recognition_votes": {},
            "recognition_scores": best_overall_scores,
            "recognizer": "insightface_arcface",
        }

    sorted_votes = sorted(recognition_votes.items(), key=lambda item: item[1], reverse=True)
    best_name, best_votes = sorted_votes[0]
    second_votes = sorted_votes[1][1] if len(sorted_votes) > 1 else 0

    avg_score = sum(recognition_scores[best_name]) / len(recognition_scores[best_name])

    print(
        f"InsightFace votes: {recognition_votes} | "
        f"best={best_name} votes={best_votes} second_votes={second_votes} avg_score={avg_score:.3f}"
    )

    # Conservative vote rule:
    # Accept if we have repeated accepted frames and winner is clearly ahead.
    if best_votes >= INSIGHT_MIN_ACCEPTED_VOTES and best_votes >= second_votes + 1:
        recognized_person = best_name
    else:
        recognized_person = None

    return {
        "face_detected": True,
        "face_count": face_count_seen,
        "face_position": face_position,
        "recognized_person": recognized_person,
        "recognition_score": avg_score,
        "recognition_margin": best_overall_margin,
        "recognition_votes": recognition_votes,
        "recognition_scores": best_overall_scores,
        "recognizer": "insightface_arcface",
    }

'''

path.write_text(prefix + insight_block + "\n" + suffix)
print(f"Patched InsightFace recognition into: {path}")
