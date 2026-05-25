from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_memory.py"
text = path.read_text()

# Add imports if needed.
if "import time" not in text:
    text = text.replace("import json\n", "import json\nimport time\n")

# Add defaults.
text = text.replace(
    '"personality_mode": "mission_control",',
    '"personality_mode": "mission_control",\n    "voice_mode": "robot_voice",\n    "enrollment_unlock": {"active": False, "authorized_by": None, "expires_at": 0},'
)

# Ensure load_memory fills missing keys.
text = text.replace(
    'data.setdefault("voice_mode", "natural_robot")',
    'data.setdefault("voice_mode", "robot_voice")\n    data.setdefault("enrollment_unlock", {"active": False, "authorized_by": None, "expires_at": 0})'
)

if 'data.setdefault("voice_mode"' not in text:
    text = text.replace(
        'data.setdefault("active_topic_id", None)',
        'data.setdefault("active_topic_id", None)\n    data.setdefault("voice_mode", "robot_voice")\n    data.setdefault("enrollment_unlock", {"active": False, "authorized_by": None, "expires_at": 0})'
    )

# Insert helpers before handle_robot_mode_command.
marker = "def handle_robot_mode_command(user_text, recognized_person=None):"
idx = text.find(marker)
if idx == -1:
    raise SystemExit("Could not find handle_robot_mode_command().")

helpers = r'''
OWNER_NAMES = {"marco", "marquinho"}


def get_voice_mode():
    return load_memory().get("voice_mode", "robot_voice")


def set_voice_mode(mode):
    memory = load_memory()
    memory["voice_mode"] = mode
    save_memory(memory)


def is_owner_name(name):
    return normalize_person_name(name) in OWNER_NAMES


def unlock_enrollment(authorized_by, seconds=120):
    authorized_by = normalize_person_name(authorized_by)
    memory = load_memory()
    memory["enrollment_unlock"] = {
        "active": True,
        "authorized_by": authorized_by,
        "expires_at": time.time() + seconds,
    }
    save_memory(memory)


def clear_enrollment_unlock():
    memory = load_memory()
    memory["enrollment_unlock"] = {
        "active": False,
        "authorized_by": None,
        "expires_at": 0,
    }
    save_memory(memory)


def get_enrollment_unlock():
    memory = load_memory()
    unlock = memory.get("enrollment_unlock", {})
    if not unlock.get("active"):
        return {"active": False, "authorized_by": None, "expires_at": 0}

    if time.time() > float(unlock.get("expires_at", 0)):
        clear_enrollment_unlock()
        return {"active": False, "authorized_by": None, "expires_at": 0}

    return unlock


def enrollment_is_unlocked():
    return bool(get_enrollment_unlock().get("active"))


def get_enrollment_authorizer():
    return get_enrollment_unlock().get("authorized_by")


def handle_voice_mode_command(user_text):
    text = user_text.lower().strip()
    memory = load_memory()

    voice_map = [
        (["natural voice", "use natural voice", "more natural voice", "speak naturally"], "natural_voice",
         "Natural voice activated."),
        (["friendly voice", "warmer voice", "use friendly voice"], "friendly_voice",
         "Friendly voice activated."),
        (["deep voice", "deeper voice", "use deep voice"], "deep_voice",
         "Deep voice activated."),
        (["story voice", "narrator voice", "use story voice"], "story_voice",
         "Story voice activated."),
        (["robot voice", "classic robot voice", "use robot voice", "local voice"], "robot_voice",
         "Robot voice activated."),
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


def handle_enrollment_security_command(user_text, confirmed_camera_person=None):
    text = user_text.lower().strip()
    confirmed_camera_person = normalize_person_name(confirmed_camera_person)

    wants_unlock = any(p in text for p in [
        "allow new friend enrollment",
        "unlock enrollment",
        "enable enrollment",
        "allow face enrollment",
        "allow new faces",
        "authorize enrollment",
    ])

    if wants_unlock:
        if not is_owner_name(confirmed_camera_person):
            return (
                "Enrollment denied. Only Marco or Marquinho can unlock new friend enrollment, "
                "and I need to recognize them with the camera first."
            )

        unlock_enrollment(confirmed_camera_person, seconds=120)
        return (
            f"Enrollment unlocked by {confirmed_camera_person} for two minutes. "
            "A new friend may now say: enroll me as their name."
        )

    if any(p in text for p in [
        "lock enrollment",
        "disable enrollment",
        "cancel enrollment",
        "stop enrollment",
    ]):
        clear_enrollment_unlock()
        return "Enrollment is now locked."

    if any(p in text for p in [
        "is enrollment unlocked",
        "enrollment status",
        "can someone enroll",
    ]):
        unlock = get_enrollment_unlock()
        if unlock.get("active"):
            remaining = int(max(0, unlock.get("expires_at", 0) - time.time()))
            return f"Enrollment is unlocked by {unlock.get('authorized_by')} for about {remaining} more seconds."
        return "Enrollment is locked."

    return None

'''

if "def get_voice_mode():" not in text:
    text = text[:idx] + helpers + "\n" + text[idx:]

# Add voice command handling near top of handle_robot_mode_command, after memory = load_memory().
old = '''    memory = load_memory()

    # Questions ABOUT modes should not trigger shutdown/sleep.
'''
new = '''    memory = load_memory()

    voice_reply = handle_voice_mode_command(user_text)
    if voice_reply:
        return voice_reply

    # Questions ABOUT modes should not trigger shutdown/sleep.
'''

if old in text:
    text = text.replace(old, new)

# Include voice mode in memory context.
text = text.replace(
    '"personality_mode": memory.get("personality_mode", "mission_control"),',
    '"personality_mode": memory.get("personality_mode", "mission_control"),\n        "voice_mode": memory.get("voice_mode", "robot_voice"),'
)

path.write_text(text.replace("\t", "    "))
print("Patched robot_memory.py with voice modes and enrollment owner gate helpers.")
