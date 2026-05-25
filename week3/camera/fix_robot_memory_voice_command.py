from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_memory.py"
text = path.read_text()

marker = "def handle_robot_mode_command(user_text, recognized_person=None):"
idx = text.find(marker)

if idx == -1:
    raise SystemExit("Could not find handle_robot_mode_command().")

voice_function = r'''
def handle_voice_mode_command(user_text):
    text = user_text.lower().strip()
    memory = load_memory()

    voice_map = [
        (
            ["natural voice", "use natural voice", "more natural voice", "speak naturally"],
            "natural_voice",
            "Natural voice activated.",
        ),
        (
            ["friendly voice", "warmer voice", "use friendly voice"],
            "friendly_voice",
            "Friendly voice activated.",
        ),
        (
            ["deep voice", "deeper voice", "use deep voice"],
            "deep_voice",
            "Deep voice activated.",
        ),
        (
            ["story voice", "narrator voice", "use story voice"],
            "story_voice",
            "Story voice activated.",
        ),
        (
            ["robot voice", "classic robot voice", "use robot voice", "local voice"],
            "robot_voice",
            "Robot voice activated.",
        ),
    ]

    for phrases, mode, reply in voice_map:
        if any(p in text for p in phrases):
            memory["voice_mode"] = mode
            save_memory(memory)
            return reply

    if any(p in text for p in [
        "what voice are you using",
        "current voice",
        "which voice are you using",
        "voice mode",
    ]):
        return f"My current voice mode is {memory.get('voice_mode', 'robot_voice')}."

    return None


'''

if "def handle_voice_mode_command(" not in text:
    text = text[:idx] + voice_function + "\n" + text[idx:]
else:
    print("handle_voice_mode_command already exists.")

path.write_text(text.replace("\t", "    "))
print("Fixed missing handle_voice_mode_command().")
