from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"
text = path.read_text()

# ------------------------------------------------------------
# 1) Add identity hysteresis constant.
# ------------------------------------------------------------
if "IDENTITY_HYSTERESIS_SECONDS" not in text:
    text = text.replace(
        "VISION_STALE_SECONDS = 8.0",
        "VISION_STALE_SECONDS = 8.0\nIDENTITY_HYSTERESIS_SECONDS = 12.0"
    )

# ------------------------------------------------------------
# 2) Make InsightFace accept strong-margin lower-score matches.
# Old:
# if best_score >= INSIGHT_SIMILARITY_THRESHOLD and margin >= INSIGHT_MARGIN_THRESHOLD:
# New:
# normal accept OR lower-score strong-margin accept.
# ------------------------------------------------------------
old = """    if best_score >= INSIGHT_SIMILARITY_THRESHOLD and margin >= INSIGHT_MARGIN_THRESHOLD:
        return best_name, best_score, margin, scores

    return None, best_score, margin, scores
"""

new = """    # Normal confident match.
    if best_score >= INSIGHT_SIMILARITY_THRESHOLD and margin >= INSIGHT_MARGIN_THRESHOLD:
        return best_name, best_score, margin, scores

    # Low-score but very strong separation.
    # Useful for Marquinho when lighting/angle lowers absolute score but Marco is far away.
    if best_score >= 0.45 and margin >= 0.25:
        return best_name, best_score, margin, scores

    return None, best_score, margin, scores
"""

if old not in text:
    print("Warning: InsightFace accept block not found exactly.")
else:
    text = text.replace(old, new)

# ------------------------------------------------------------
# 3) Add hysteresis inside set_cached_local_state.
# If new frame is uncertain but previous identity was recent, preserve identity.
# ------------------------------------------------------------
old = """def set_cached_local_state(new_state):
    with STATE_LOCK:
        SHARED_LOCAL_STATE.clear()
        SHARED_LOCAL_STATE.update(new_state)
        SHARED_LOCAL_STATE["last_vision_update"] = time.time()
"""

new = """def set_cached_local_state(new_state):
    with STATE_LOCK:
        now = time.time()

        previous_person = SHARED_LOCAL_STATE.get("recognized_person")
        previous_update = SHARED_LOCAL_STATE.get("last_vision_update", 0.0)
        previous_age = now - previous_update

        # Identity hysteresis:
        # Do not drop Marco/Marquinho immediately because of one weak/uncertain frame.
        if (
            previous_person
            and not new_state.get("recognized_person")
            and new_state.get("face_detected")
            and previous_age <= IDENTITY_HYSTERESIS_SECONDS
        ):
            new_state = dict(new_state)
            new_state["recognized_person"] = previous_person
            new_state["recognizer"] = "insightface_arcface_hysteresis"
            new_state.setdefault("recognition_votes", {previous_person: 1})

        SHARED_LOCAL_STATE.clear()
        SHARED_LOCAL_STATE.update(new_state)
        SHARED_LOCAL_STATE["last_vision_update"] = now
"""

if old not in text:
    print("Warning: set_cached_local_state block not found exactly.")
else:
    text = text.replace(old, new)

# ------------------------------------------------------------
# 4) Make natural conversation more permissive.
# If familiar person is present and says a normal sentence, answer.
# ------------------------------------------------------------
old = """    if familiar_person and active_followup and len(words) >= 1:
        return True

    return False
"""

new = """    # Natural conversation mode:
    # If Miguel knows who is speaking, a normal sentence is probably intentional.
    if familiar_person and len(words) >= 3:
        return True

    if familiar_person and active_followup and len(words) >= 1:
        return True

    return False
"""

if old not in text:
    print("Warning: should_respond_naturally ending block not found exactly.")
else:
    text = text.replace(old, new)

path.write_text(text)
print(f"Patched V6.5 natural conversation: {path}")
