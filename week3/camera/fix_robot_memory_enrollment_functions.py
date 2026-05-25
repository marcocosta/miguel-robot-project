from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_memory.py"
text = path.read_text()

# Ensure time import exists.
if "import time" not in text:
    text = text.replace("import json\n", "import json\nimport time\n")

# Add helper functions before handle_robot_mode_command.
marker = "def handle_robot_mode_command(user_text, recognized_person=None):"
idx = text.find(marker)

if idx == -1:
    raise SystemExit("Could not find handle_robot_mode_command().")

helpers = r'''
OWNER_NAMES = {"marco", "marquinho"}


def is_owner_name(name):
    return normalize_person_name(name) in OWNER_NAMES


def get_voice_mode():
    memory = load_memory()
    return memory.get("voice_mode", "robot_voice")


def set_voice_mode(mode):
    memory = load_memory()
    memory["voice_mode"] = mode
    save_memory(memory)


def _ensure_enrollment_unlock(memory):
    memory.setdefault("enrollment_unlock", {
        "active": False,
        "authorized_by": None,
        "expires_at": 0,
    })


def unlock_enrollment(authorized_by, seconds=120):
    memory = load_memory()
    _ensure_enrollment_unlock(memory)

    memory["enrollment_unlock"] = {
        "active": True,
        "authorized_by": normalize_person_name(authorized_by),
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
    _ensure_enrollment_unlock(memory)

    unlock = memory.get("enrollment_unlock", {})

    if not unlock.get("active"):
        return {
            "active": False,
            "authorized_by": None,
            "expires_at": 0,
        }

    if time.time() > float(unlock.get("expires_at", 0)):
        clear_enrollment_unlock()
        return {
            "active": False,
            "authorized_by": None,
            "expires_at": 0,
        }

    return unlock


def enrollment_is_unlocked():
    return bool(get_enrollment_unlock().get("active"))


def get_enrollment_authorizer():
    return get_enrollment_unlock().get("authorized_by")


def handle_enrollment_security_command(user_text, confirmed_camera_person=None):
    """
    Only Marco or Marquinho can unlock enrollment.
    confirmed_camera_person must come from real camera recognition, not conversation memory.
    """
    text = user_text.lower().strip()
    confirmed_camera_person = normalize_person_name(confirmed_camera_person)

    unlock_phrases = [
        "allow new friend enrollment",
        "unlock enrollment",
        "enable enrollment",
        "allow face enrollment",
        "allow new faces",
        "authorize enrollment",
    ]

    if any(p in text for p in unlock_phrases):
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

    lock_phrases = [
        "lock enrollment",
        "disable enrollment",
        "cancel enrollment",
        "stop enrollment",
    ]

    if any(p in text for p in lock_phrases):
        clear_enrollment_unlock()
        return "Enrollment is now locked."

    status_phrases = [
        "is enrollment unlocked",
        "enrollment status",
        "can someone enroll",
    ]

    if any(p in text for p in status_phrases):
        unlock = get_enrollment_unlock()

        if unlock.get("active"):
            remaining = int(max(0, unlock.get("expires_at", 0) - time.time()))
            return f"Enrollment is unlocked by {unlock.get('authorized_by')} for about {remaining} more seconds."

        return "Enrollment is locked."

    return None

'''

if "def handle_enrollment_security_command(" not in text:
    text = text[:idx] + helpers + "\n" + text[idx:]
else:
    print("Enrollment functions already exist; no insertion needed.")

# Ensure DEFAULT_MEMORY has keys if this exact string exists.
text = text.replace(
    '"personality_mode": "mission_control",',
    '"personality_mode": "mission_control",\n    "voice_mode": "robot_voice",\n    "enrollment_unlock": {"active": False, "authorized_by": None, "expires_at": 0},'
)

# Ensure load_memory fills keys.
if 'data.setdefault("voice_mode"' not in text:
    text = text.replace(
        'data.setdefault("long_term_topics", {})',
        'data.setdefault("long_term_topics", {})\n    data.setdefault("voice_mode", "robot_voice")\n    data.setdefault("enrollment_unlock", {"active": False, "authorized_by": None, "expires_at": 0})'
    )

path.write_text(text.replace("\t", "    "))
print("Fixed robot_memory enrollment + voice helper imports.")
