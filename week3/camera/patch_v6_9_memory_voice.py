from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_memory.py"
text = path.read_text()

# Add voice_mode default if missing.
text = text.replace(
    '"personality_mode": "mission_control",',
    '"personality_mode": "mission_control",\n    "voice_mode": "natural_robot",'
)

# Ensure load_memory fills voice_mode.
text = text.replace(
    'data.setdefault("long_term_topics", {})',
    'data.setdefault("long_term_topics", {})\n    data.setdefault("voice_mode", "natural_robot")'
)

# Add voice helpers before handle_robot_mode_command.
marker = "def handle_robot_mode_command(user_text, recognized_person=None):"
idx = text.find(marker)
if idx == -1:
    raise SystemExit("Could not find handle_robot_mode_command.")

voice_helpers = r'''
def get_voice_mode():
    return load_memory().get("voice_mode", "natural_robot")


def set_voice_mode(mode):
    memory = load_memory()
    memory["voice_mode"] = mode
    save_memory(memory)

'''

if "def get_voice_mode():" not in text:
    text = text[:idx] + voice_helpers + "\n" + text[idx:]

# Add voice-mode command handling before personality modes.
marker2 = "    # Personality modes."
idx2 = text.find(marker2)
if idx2 == -1:
    raise SystemExit("Could not find personality modes marker.")

voice_command_block = r'''
    # Voice modes.
    if any(p in text for p in [
        "natural voice",
        "use natural voice",
        "more natural voice",
        "speak naturally",
    ]):
        memory["voice_mode"] = "natural_robot"
        save_memory(memory)
        return "Natural voice mode activated."

    if any(p in text for p in [
        "robot voice",
        "use robot voice",
        "classic robot voice",
        "local voice",
    ]):
        memory["voice_mode"] = "local_robot"
        save_memory(memory)
        return "Robot voice mode activated."

    if any(p in text for p in [
        "story voice",
        "story mode voice",
        "narrator voice",
    ]):
        memory["voice_mode"] = "story"
        save_memory(memory)
        return "Story voice activated."

    if any(p in text for p in [
        "deep voice",
        "deeper voice",
    ]):
        memory["voice_mode"] = "deep_robot"
        save_memory(memory)
        return "Deep robot voice activated."

    if any(p in text for p in [
        "friendly voice",
        "warmer voice",
    ]):
        memory["voice_mode"] = "friendly"
        save_memory(memory)
        return "Friendly voice activated."

    if any(p in text for p in [
        "what voice are you using",
        "current voice",
        "voice mode",
        "which voice",
    ]):
        return f"My current voice mode is {memory.get('voice_mode', 'natural_robot')}."

'''

if "Natural voice mode activated" not in text:
    text = text[:idx2] + voice_command_block + text[idx2:]

# Add voice to memory context.
text = text.replace(
    '"personality_mode": memory.get("personality_mode", "mission_control"),',
    '"personality_mode": memory.get("personality_mode", "mission_control"),\n        "voice_mode": memory.get("voice_mode", "natural_robot"),'
)

# Update mode list answer to mention voice.
text = text.replace(
    "sleep mode to stay silent until called, and shutdown mode to safely power off after confirmation.",
    "sleep mode to stay silent until called, shutdown mode to safely power off after confirmation, and voice modes like natural, robot, friendly, deep, and story voice."
)

path.write_text(text.replace("\t", "    "))
print("Patched robot_memory voice modes.")
