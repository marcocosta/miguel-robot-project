"""
Miguel Robot Cloud Brain V7 Full Core

Official run command:
    start-robot-cloud-v7-full

Face-disabled run command:
    MIGUEL_FACE_ENABLED=0 start-robot-cloud-v7-full

Syntax check only:
    python3 -m py_compile /home/marquinho/robot-project/week3/camera/robot_cloud_brain_v7_full.py

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
import robot_timer

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

face = None
last_face_mode = None


# Last updated: 20260522
# Optional display face subscriber. Failures here must never stop the robot runtime.
def init_optional_face():
    if os.getenv("MIGUEL_FACE_ENABLED", "1") == "0":
        print("[face] disabled by MIGUEL_FACE_ENABLED=0")
        return None

    try:
        face_dir = Path.home() / "robot-project" / "week3" / "face"
        if str(face_dir) not in sys.path:
            sys.path.insert(0, str(face_dir))

        from face_controller import FaceController

        controller = FaceController(width=1024, height=600, fullscreen=True, fps=30)
        controller.start()
        print("[face] enabled")
        return controller
    except Exception as exc:
        print(f"[face] disabled: {exc}")
        return None


# Last updated: 20260522
# Face helpers are subscriber-only: robot state changes enqueue expression updates.
def set_face_once(mode_name: str, callback) -> None:
    global last_face_mode
    if last_face_mode == mode_name:
        return

    last_face_mode = mode_name
    callback()


def face_idle() -> None:
    if face:
        try:
            set_face_once("idle", face.idle)
        except Exception as exc:
            print(f"[face] idle failed: {exc}")


def face_listening() -> None:
    if face:
        try:
            set_face_once("listening", face.listening)
        except Exception as exc:
            print(f"[face] listening failed: {exc}")


def face_thinking() -> None:
    if face:
        try:
            set_face_once("thinking", face.thinking)
        except Exception as exc:
            print(f"[face] thinking failed: {exc}")


def face_speaking(text: str = "Speaking") -> None:
    if face:
        try:
            set_face_once(f"speaking:{text}", lambda: face.speaking(text))
        except Exception as exc:
            print(f"[face] speaking failed: {exc}")


def face_happy(text: str = "Happy") -> None:
    if face:
        try:
            set_face_once(f"happy:{text}", lambda: face.happy(text))
        except Exception as exc:
            print(f"[face] happy failed: {exc}")


def face_confused(text: str = "Confused") -> None:
    if face:
        try:
            set_face_once(f"confused:{text}", lambda: face.confused(text))
        except Exception as exc:
            print(f"[face] confused failed: {exc}")


def face_error(text: str = "Error") -> None:
    if face:
        try:
            set_face_once(f"error:{text}", lambda: face.error(text))
        except Exception as exc:
            print(f"[face] error failed: {exc}")


def face_sleeping() -> None:
    if face:
        try:
            set_face_once("sleeping", face.sleeping)
        except Exception as exc:
            print(f"[face] sleeping failed: {exc}")


def face_status(interaction_state: str, status_text: str = "") -> None:
    if not face:
        return

    try:
        if hasattr(face, "status"):
            set_face_once(
                f"status:{interaction_state}:{status_text}",
                lambda: face.status(interaction_state, status_text),
            )
            return

        if interaction_state == "listening":
            face_listening()
        elif interaction_state in {"thinking", "heard", "looking"}:
            face_thinking()
        elif interaction_state == "speaking":
            face_speaking(status_text or "Speaking")
        elif interaction_state in {"sleeping", "shutdown_pending"}:
            face_sleeping()
        elif interaction_state == "error":
            face_error(status_text or "Error")
        elif interaction_state == "idle":
            face_idle()
    except Exception as exc:
        print(f"[face] status failed: {exc}")


def has_wake_phrase(text: str) -> bool:
    t = str(text or "").lower()
    return any(p in t for p in WAKE_PHRASES)


def is_global_idle_command(text: str) -> bool:
    """
    Shutdown/sleep/stop commands must work even before wake-phrase routing.
    Shutdown here means stopping Miguel's program only; the Jetson stays on.
    """
    t = str(text or "").lower().strip()

    phrases = [
        "shutdown",
        "shut down",
        "confirm shutdown",
        "confirm shut down",
        "stop robot",
        "stop robot program",
        "stop miguel program",
        "quit robot",
        "exit robot",
        "sleep mode",
        "go to sleep",
    ]

    return any(p in t for p in phrases)


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
            text_to_speak = decision.safe_reply or "I can’t help with that."
        else:
            text_to_speak = text

        # Last updated: 20260522
        # TTS remains behavior-owner; the display face only follows speaking state.
        face_speaking("Speaking")
        try:
            return original_speak(text_to_speak)
        finally:
            face_idle()

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



def is_local_robot_control_request(user_text: str) -> bool:
    """
    Commands about Miguel itself should not go through semantic safety first.
    They should be handled by local robot memory / mode / voice logic.
    """
    t = str(user_text or "").lower().strip()

    phrases = [
        # Voice controls.
        "voice",
        "deep voice",
        "robot voice",
        "robotic voice",
        "natural voice",
        "friendly voice",
        "story voice",
        "storyteller voice",
        "narrator voice",
        "change voice",
        "switch voice",
        "what voices",
        "voice options",
        "which voice",

        # Modes.
        "sleep mode",
        "wake up",
        "shutdown",
        "shut down",
        "stop robot",
        "stop robot program",
        "stop miguel program",
        "confirm shutdown",
        "confirm shut down",
        "mission control",

        # Simple robot utility/status.
        "status",
        "what time",
        "time is it",
        "current time",
    ]

    return any(p in t for p in phrases)


def handle_v7_local_utility(user_text: str) -> bool:
    """
    Very small V7-local utility layer for things V6 cloud brain may not know.
    Returns True if handled.
    """
    t = str(user_text or "").lower().strip()

    if handle_local_timer_command(user_text):
        return True

    if "what time" in t or "time is it" in t or "current time" in t:
        from datetime import datetime
        now = datetime.now().strftime("%I:%M %p").lstrip("0")
        v6.speak(f"The local Jetson time is {now}.")
        return True

    return False


def handle_local_timer_command(user_text: str) -> bool:
    command = robot_timer.parse_timer_command(user_text)
    if not command:
        return False

    intent = command.get("intent")
    if intent == "start_timer":
        seconds = int(command.get("seconds", 0) or 0)
        result = robot_timer.start_timer(seconds)
        v6.speak(f"Timer set for {_format_timer_duration(result['seconds'])}.")
        return True

    if intent == "cancel_timer":
        robot_timer.cancel_timer()
        v6.speak("Timer canceled.")
        return True

    if intent == "timer_status":
        status = robot_timer.get_timer_status()
        if not status.get("active"):
            v6.speak("No timer is running.")
            return True
        v6.speak(f"There are about {_format_timer_remaining(status['remaining_seconds'])} left.")
        return True

    return False


def _format_timer_duration(seconds: int) -> str:
    seconds = max(0, int(seconds))
    if seconds and seconds % 60 == 0:
        minutes = seconds // 60
        return f"{minutes} minute" + ("" if minutes == 1 else "s")
    return f"{seconds} second" + ("" if seconds == 1 else "s")


def _format_timer_remaining(seconds: int) -> str:
    seconds = max(0, int(seconds))
    minutes, remainder = divmod(seconds, 60)
    if minutes:
        return f"{minutes} minute{'s' if minutes != 1 else ''} and {remainder} second{'s' if remainder != 1 else ''}"
    return f"{remainder} second" + ("" if remainder == 1 else "s")


def check_local_timer_tick() -> None:
    expired = robot_timer.timer_tick()
    if not expired:
        return
    print("[V7 FULL TIMER] expired")
    face_happy("Time is up")
    v6.speak("Time is up.")


def route_camera_intent_now(user_text: str, camera_intent: str, camera_manager: CameraManager) -> bool:
    """
    Absolute V7 rule:
    Any camera/vision/see/look request uses camera truth directly,
    not semantic safety and not conversation memory.
    """
    t = str(user_text or "").lower()

    if is_identity_camera_request(user_text):
        face_state = camera_manager.get_face_state(max_age_seconds=2.0)
        reply = build_identity_reply(face_state)
        v6.speak(reply)
        try:
            v6.update_conversation_memory(user_text=user_text, assistant_reply=reply)
        except Exception:
            pass
        return True

    if is_scene_camera_request(user_text) or camera_intent == "camera_generic":
        # If generic wording includes "who", treat it as identity.
        if "who" in t or "identify" in t or "recognize" in t or "recognise" in t:
            face_state = camera_manager.get_face_state(max_age_seconds=2.0)
            reply = build_identity_reply(face_state)
        else:
            reply = build_scene_reply(camera_manager)

        v6.speak(reply)
        try:
            v6.update_conversation_memory(user_text=user_text, assistant_reply=reply)
        except Exception:
            pass
        return True

    return False


def handle_user_turn(user_text: str, camera_manager: CameraManager, safety: SafetyGuard) -> bool:
    """
    Returns keep_running.
    """
    user_text = str(user_text or "").strip()
    if not user_text:
        return True

    print(f"[V7 FULL TRANSCRIPT] {user_text}")
    face_thinking()

    camera_intent = classify_camera_intent(user_text)
    if camera_intent != "none":
        print(f"[V7 FULL CAMERA INTENT] {camera_intent}: {user_text}")

    # Local utility commands that should never require cloud safety.
    if handle_v7_local_utility(user_text):
        return True

    # Camera requests are robot-control/vision commands.
    # They must use live camera truth immediately.
    if route_camera_intent_now(user_text, camera_intent, camera_manager):
        return True

    # Miguel local controls like voice/sleep/shutdown should be handled by V6 local logic,
    # not semantic safety/cloud brain first.
    if is_local_robot_control_request(user_text):
        face_state = camera_manager.get_face_state(max_age_seconds=2.0)
        return bool(v6.handle_user_turn_with_cached_state(user_text, face_state))

    # Camera requests are robot-control/vision commands.
    # Do not send ordinary "what do you see?" / "who do you see?" / imperfect ASR
    # variants through cloud safety. They route directly to camera truth.
    if is_identity_camera_request(user_text):
        face_state = camera_manager.get_face_state(max_age_seconds=2.0)
        reply = build_identity_reply(face_state)
        v6.speak(reply)
        try:
            v6.update_conversation_memory(user_text=user_text, assistant_reply=reply)
        except Exception:
            pass
        return True

    if is_scene_camera_request(user_text) or camera_intent == "camera_generic":
        reply = build_scene_reply(camera_manager)
        v6.speak(reply)
        try:
            v6.update_conversation_memory(user_text=user_text, assistant_reply=reply)
        except Exception:
            pass
        return True

    # Safety for non-camera conversation/content requests.
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

    # Scene camera: already handled above.
    if is_scene_camera_request(user_text):
        reply = build_scene_reply(camera_manager)
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
    global face

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

    face = init_optional_face()
    face_happy("Miguel online")

    safety = SafetyGuard()
    install_output_safety(safety)

    stop_event = threading.Event()

    camera_manager = None
    face_thread = None

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
            face_happy("Miguel online")

            last_known_person = None

            while not stop_event.is_set():
                check_local_timer_tick()
                face_state = camera_manager.get_face_state(max_age_seconds=2.0)
                recognized = face_state.get("recognized_person")

                if recognized:
                    if recognized != last_known_person:
                        print(f"[V7 FULL READY] Familiar person present: {recognized}")
                        v6.speak(f"I see {recognized}. Conversation mode is ready.")
                        last_known_person = recognized

                    print("[V7 FULL READY] Start talking now.")
                    face_listening()
                    user_text = v6.capture_user_turn()

                    if not user_text:
                        print("[V7 FULL AUDIO] No speech captured.")
                        check_local_timer_tick()
                        face_idle()
                        continue

                    keep_running = handle_user_turn(user_text, camera_manager, safety)
                    check_local_timer_tick()
                    if not keep_running:
                        break

                else:
                    last_known_person = None
                    print("[V7 FULL IDLE] No familiar person. Say wake phrase or Miguel.")
                    # Last updated: 20260522
                    # This runs at the conversation loop boundary, not per camera frame.
                    if face_state.get("face_detected"):
                        face_confused("Who is there?")
                    else:
                        face_idle()
                    user_text = v6.capture_user_turn()

                    if not user_text:
                        check_local_timer_tick()
                        continue

                    if is_global_idle_command(user_text):
                        print("[V7 FULL IDLE] Handling global idle command:", user_text)
                        keep_running = handle_user_turn(user_text, camera_manager, safety)
                        check_local_timer_tick()
                        if not keep_running:
                            break
                        continue

                    if not has_wake_phrase(user_text):
                        print("[V7 FULL IDLE] Ignoring speech without wake phrase:", user_text)
                        continue

                    v6.speak("I'm listening now.")
                    face_listening()
                    user_text = v6.capture_user_turn()

                    if not user_text:
                        check_local_timer_tick()
                        face_idle()
                        continue

                    keep_running = handle_user_turn(user_text, camera_manager, safety)
                    check_local_timer_tick()
                    if not keep_running:
                        break

    except KeyboardInterrupt:
        print("[V7 FULL] Keyboard interrupt.")

    finally:
        stop_event.set()
        face_sleeping()

        # Give the face/ONNX worker a chance to leave before closing DepthAI resources.
        try:
            if face_thread and face_thread.is_alive():
                face_thread.join(timeout=1.5)
        except Exception as e:
            print("[V7 FULL] face thread join warning:", e)

        if camera_manager:
            camera_manager.stop()

        if face:
            try:
                face.stop()
            except Exception as e:
                print("[face] stop warning:", e)

        print("Miguel Cloud Brain V7 Full stopped.")


if __name__ == "__main__":
    run_v7_full()
