"""
Miguel Robot Cloud Brain V7

V7 goals:
- Single camera-owner architecture through CameraManager.
- SafetyGuard before and after brain responses.
- Keep V6 available as fallback.
- Start migration without breaking the working V6 robot.
"""

import os
import sys
import time
from pathlib import Path

# Make sure camera folder and v7 package are importable.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from v7.safety_guard import SafetyGuard
from v7.camera_intents import (
    classify_camera_intent,
    is_any_camera_request,
    is_identity_camera_request,
    is_scene_camera_request,
)

# Import stable V6 functions.
# V6 remains the source of truth for hardware startup until CameraManager is fully wired.
import robot_cloud_brain_v6_threaded as v6


ROBOT_NAME = "Miguel"
SAFETY = SafetyGuard()


def safety_check_user_text(user_text: str):
    start = time.time()
    decision = SAFETY.evaluate_user_text(user_text)
    elapsed = time.time() - start

    if elapsed > 1.0:
        print(f"[V7 SAFETY] Safety check took {elapsed:.2f}s. category={decision.category} source={decision.source}")

    if not decision.allowed:
        print(f"[V7 SAFETY] Blocked user text. category={decision.category} source={decision.source} reason={decision.reason}")
        return decision.safe_reply or "I can’t help with that topic."

    return None


def safety_check_reply(reply: str):
    decision = SAFETY.evaluate_assistant_reply(reply)

    if not decision.allowed:
        print(f"[V7 SAFETY] Blocked assistant reply. category={decision.category} source={decision.source} reason={decision.reason}")
        return decision.safe_reply or "I can’t help with that topic."

    return reply



def is_safe_robot_control_command(user_text: str) -> bool:
    """
    Commands that control Miguel itself should bypass content moderation.
    Example: "use story voice" is a harmless robot-control command, but moderation
    may falsely flag it because of wording/context.
    """
    text = str(user_text or "").lower().strip()

    phrases = [
        # Voice controls.
        "voice",
        "robot voice",
        "robotic voice",
        "natural voice",
        "friendly voice",
        "deep voice",
        "story voice",
        "storyteller voice",
        "narrator voice",
        "normal voice",
        "which voice",
        "what voice",
        "voice options",
        "what voices",

        # Robot operating modes.
        "sleep mode",
        "go to sleep",
        "wake up",
        "mission control",
        "what mode",
        "current mode",

        # Safe robot shutdown means stop program only.
        "shutdown",
        "shut down",
        "confirm shutdown",
        "confirm shut down",
        "confirme shutdown",
        "confirme shut down",
    ]

    return any(p in text for p in phrases)


def install_v7_safety_hooks():
    """
    Monkey-patch selected V6 functions so V7 can add safety without rewriting the whole robot yet.

    This keeps the migration small:
    - V6 still handles audio, face recognition, scene description, and conversation loop.
    - V7 adds input/output safety gates.
    """

    original_handle = v6.handle_user_turn_with_cached_state
    original_speak = v6.speak

    def safe_speak(text: str):
        safe_text = safety_check_reply(str(text))
        return original_speak(safe_text)

    def safe_handle_user_turn_with_cached_state(user_text, cached_local_state):
        camera_intent = classify_camera_intent(str(user_text))
        if camera_intent != "none":
            print(f"[V7 CAMERA INTENT] {camera_intent}: {user_text}")

        # Local robot-control commands are safe and should not be blocked by moderation.
        # The original V6 handler will route them to robot_memory.py.
        if is_safe_robot_control_command(str(user_text)):
            return original_handle(user_text, cached_local_state)

        safe_reply = safety_check_user_text(str(user_text))

        if safe_reply:
            safe_speak(safe_reply)
            try:
                v6.update_conversation_memory(user_text=user_text, assistant_reply=safe_reply)
            except Exception:
                pass
            return True

        return original_handle(user_text, cached_local_state)

    v6.speak = safe_speak
    v6.handle_user_turn_with_cached_state = safe_handle_user_turn_with_cached_state

    print("[V7] Safety hooks installed.")


def run_v7():
    print("======================================")
    print(" Miguel - Cloud Brain V7 ")
    print("======================================")
    print("")
    print("V7 Phase 1:")
    print("  - V6 conversation loop reused")
    print("  - V7 SafetyGuard enabled")
    print("  - Camera intent router loaded for Phase 2")
    print("  - Jetson shutdown disabled unless manually run from terminal")
    print("")

    install_v7_safety_hooks()

    # Run the known-good V6 loop with V7 safety hooks.
    return v6.run_v6_threaded_conversation()


if __name__ == "__main__":
    run_v7()
