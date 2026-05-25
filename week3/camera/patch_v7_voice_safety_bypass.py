from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v7.py"
text = path.read_text()

# Add helper before install_v7_safety_hooks.
marker = "def install_v7_safety_hooks():"
idx = text.find(marker)

if idx == -1:
    raise SystemExit("Could not find install_v7_safety_hooks().")

helper = r'''
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

'''

if "def is_safe_robot_control_command(" not in text:
    text = text[:idx] + helper + "\n" + text[idx:]


# Patch safety hook order: robot-control commands bypass moderation.
old = '''    def safe_handle_user_turn_with_cached_state(user_text, cached_local_state):
        safe_reply = safety_check_user_text(str(user_text))

        if safe_reply:
            safe_speak(safe_reply)
            try:
                v6.update_conversation_memory(user_text=user_text, assistant_reply=safe_reply)
            except Exception:
                pass
            return True

        return original_handle(user_text, cached_local_state)
'''

new = '''    def safe_handle_user_turn_with_cached_state(user_text, cached_local_state):
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
'''

if old not in text:
    raise SystemExit("Could not find safe_handle_user_turn_with_cached_state block.")

text = text.replace(old, new)

path.write_text(text)
print("Patched V7 safety bypass for safe robot-control commands.")
