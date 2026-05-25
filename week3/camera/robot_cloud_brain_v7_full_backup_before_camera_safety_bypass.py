"""
Miguel Robot Cloud Brain V7 Full Core

This runner does not call the old V6 run loop.

Architecture:
- CameraManager owns the OAK queue.
- Face worker uses CameraManager only.
- Scene description uses CameraManager only.
- Main loop handles audio -> safety -> intent routing -> brain -> speech.
- V6 functions are reused as stable primitives, not as the orchestrator.
"""

import os
import sys
import threading
import time
from pathlib import Path

import depthai as dai

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import robot_cloud_brain_v6_threaded as v6

from v7.camera_manager import CameraManager
from v7.camera_intents import classify_camera_intent, is_scene_camera_request, is_identity_camera_request
from v7.safety_guard import SafetyGuard


ROBOT_NAME = "Miguel"
WAKE_PHRASES = [
    "hey miguel",
    "hey me go",
    "mission control",
    "miguel",
]


def has_wake_phrase(text: str) -> bool:
    t = str(text or "").lower()
    return any(p in t for p in WAKE_PHRASES)


def install_output_safety(safety: SafetyGuard):
    """
    Protect Miguel's spoken output while still using V6 speak/TTS.
    """
    original_speak = v6.speak

    def safe_speak(text: str):
        decision = safety.evaluate_assistant_reply(str(text))

        if not decision.allowed:
            print(
                f"[V7 FULL SAFETY] Blocked assistant reply. "
                f"category={decision.category} source={decision.source} reason={decision.reason}"
            )
            return original_speak(decision.safe_reply or "I can’t help with that.")

        return original_speak(text)

    v6.speak = safe_speak


def build_scene_reply(camera_manager: CameraManager) -> str:
    """
    Scene description from current CameraManager frame.
    Uses V6 cloud vision helper if available.
    """
    print("[V7 FULL] scene_camera -> CameraManager latest frame")

    image_path = camera_manager.save_latest_frame(prefix="v7_scene")

    if not image_path:
        return "I do not have a fresh camera frame right now."

    if hasattr(v6, "describe_scene_with_openai"):
        return v6.describe_scene_with_openai(image_path)

    # Fallback to legacy scene function using CameraManager as queue adapter.
    if hasattr(v6, "describe_scene_now"):
        return v6.describe_scene_now(camera_manager)

    return f"I captured a fresh camera frame at {image_path}, but scene description is not available."


def build_identity_reply(face_state: dict) -> str:
    """
    Identity reply from fresh face state only.
    """
    if not face_state.get("face_detected"):
        return "I checked the camera, and I do not see a confirmed face right now."

    person = face_state.get("recognized_person")
    if person:
        score = face_state.get("recognition_score")
        if score is not None:
            return f"I checked the camera and recognize one face as {person}, with confidence about {score:.2f}."
        return f"I checked the camera and recognize one face as {person}."

    return "I checked the camera and see a face, but I cannot confidently identify who it is."


def face_worker(camera_manager: CameraManager, stop_event: threading.Event):
    """
    Face recognition worker.
    It never touches the real OAK queue directly. It uses CameraManager's queue-like adapter.
    """
    print("[V7 FULL FACE] Face worker started.")

    while not stop_event.is_set():
        try:
            if hasattr(v6, "detect_face_state"):
                state = v6.detect_face_state(camera_manager)
            else:
                state = {
                    "face_detected": False,
                    "face_count": 0,
                    "recognized_person": None,
                    "recognizer": "detect_face_state_missing",
                }

            camera_manager.update_face_state(state)

            recognized = state.get("recognized_person")
            if recognized:
                print(
                    f"[V7 FULL FACE] recognized={recognized} "
                    f"score={state.get('recognition_score')}"
                )

        except Exception as e:
            # Do not spam logs while camera is warming up.
            msg = str(e)
            if "No fresh camera frame" not in msg:
                print("[V7 FULL FACE] error:", e)

        time.sleep(0.6)

    print("[V7 FULL FACE] Face worker stopped.")


def handle_user_turn(user_text: str, camera_manager: CameraManager, safety: SafetyGuard) -> bool:
    """
    Returns keep_running.
    """
    user_text = str(user_text or "").strip()
    if not user_text:
        return True

    print(f"[V7 FULL TRANSCRIPT] {user_text}")

    camera_intent = classify_camera_intent(user_text)
    if camera_intent != "none":
        print(f"[V7 FULL CAMERA INTENT] {camera_intent}: {user_text}")

    # Safety first, except robot-control commands are already handled inside SafetyGuard fast path.
    start = time.time()
    decision = safety.evaluate_user_text(user_text)
    elapsed = time.time() - start

    if elapsed > 1.0:
        print(
            f"[V7 FULL SAFETY] Safety check took {elapsed:.2f}s. "
            f"category={decision.category} source={decision.source}"
        )

    if not decision.allowed:
        v6.speak(decision.safe_reply or "I can’t help with that.")
        return True

    # Scene camera: always use CameraManager current frame.
    if is_scene_camera_request(user_text):
        reply = build_scene_reply(camera_manager)
        v6.speak(reply)
        try:
            v6.update_conversation_memory(user_text=user_text, assistant_reply=reply)
        except Exception:
            pass
        return True

    # Identity camera: always use fresh-ish face state from face worker.
    if is_identity_camera_request(user_text):
        face_state = camera_manager.get_face_state(max_age_seconds=2.0)
        reply = build_identity_reply(face_state)
        v6.speak(reply)
        try:
            v6.update_conversation_memory(user_text=user_text, assistant_reply=reply)
        except Exception:
            pass
        return True

    # Normal command/conversation path: use latest face state only as context.
    face_state = camera_manager.get_face_state(max_age_seconds=2.0)

    try:
        keep_running = v6.handle_user_turn_with_cached_state(user_text, face_state)
        return bool(keep_running)
    except Exception as e:
        print("[V7 FULL BRAIN] error:", e)
        v6.speak("I had a brain error while processing that.")
        return True


def create_camera_manager_from_live_pipeline(pipeline):
    """
    Create camera nodes inside a live DepthAI pipeline context.

    Important:
    This function must be called inside:
        with dai.Pipeline() as pipeline:
    Otherwise CameraManager will start but never receive frames.
    """
    cam = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput((640, 480))
    camera_queue = cam_out.createOutputQueue(maxSize=1, blocking=False)

    # Important: V6 requires explicit pipeline.start().
    # Without this, createOutputQueue exists but produces no frames.
    pipeline.start()

    manager = CameraManager(camera_queue, max_frame_age_seconds=1.0)
    return manager


def run_v7_full():
    print("======================================")
    print(" Miguel - Cloud Brain V7 FULL ")
    print("======================================")
    print("")
    print("V7 Full Core:")
    print("  - CameraManager owns OAK queue")
    print("  - Face worker uses CameraManager")
    print("  - Scene description uses CameraManager")
    print("  - Main V7 loop replaces V6 run loop")
    print("  - V6 remains fallback primitives only")
    print("")

    safety = SafetyGuard()
    install_output_safety(safety)

    stop_event = threading.Event()

    camera_manager = None

    try:
        with dai.Pipeline() as pipeline:
            camera_manager = create_camera_manager_from_live_pipeline(pipeline)
            camera_manager.start()

            # Give DepthAI a moment to deliver first frames before face worker starts.
            time.sleep(1.0)

            face_thread = threading.Thread(
                target=face_worker,
                args=(camera_manager, stop_event),
                daemon=True,
            )
            face_thread.start()

            v6.speak("Miguel V7 full core is online. Camera manager is active.")

            last_known_person = None

            while not stop_event.is_set():
                face_state = camera_manager.get_face_state(max_age_seconds=2.0)
                recognized = face_state.get("recognized_person")

                if recognized:
                    if recognized != last_known_person:
                        print(f"[V7 FULL READY] Familiar person present: {recognized}")
                        v6.speak(f"I see {recognized}. Conversation mode is ready.")
                        last_known_person = recognized

                    print("[V7 FULL READY] Start talking now.")
                    user_text = v6.capture_user_turn()

                    if not user_text:
                        print("[V7 FULL AUDIO] No speech captured.")
                        continue

                    keep_running = handle_user_turn(user_text, camera_manager, safety)
                    if not keep_running:
                        break

                else:
                    last_known_person = None
                    print("[V7 FULL IDLE] No familiar person. Say wake phrase or Miguel.")
                    user_text = v6.capture_user_turn()

                    if not user_text:
                        continue

                    if not has_wake_phrase(user_text):
                        print("[V7 FULL IDLE] Ignoring speech without wake phrase:", user_text)
                        continue

                    v6.speak("I'm listening now.")
                    user_text = v6.capture_user_turn()

                    if not user_text:
                        continue

                    keep_running = handle_user_turn(user_text, camera_manager, safety)
                    if not keep_running:
                        break

    except KeyboardInterrupt:
        print("[V7 FULL] Keyboard interrupt.")

    finally:
        stop_event.set()

        if camera_manager:
            camera_manager.stop()

        print("Miguel Cloud Brain V7 Full stopped.")


if __name__ == "__main__":
    run_v7_full()
