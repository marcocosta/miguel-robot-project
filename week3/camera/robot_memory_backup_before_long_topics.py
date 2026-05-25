import json
import re
import time
from pathlib import Path

MEMORY_DIR = Path.home() / "robot-project/week3/memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

MEMORY_PATH = MEMORY_DIR / "miguel_memory.json"

DEFAULT_MEMORY = {
    "robot_mode": "normal",
    "personality_mode": "mission_control",
    "pending_shutdown": False,
    "profiles": {
        "marco": {
            "role": "Systems Engineer",
            "preferences": [],
            "notes": [],
        },
        "marquinho": {
            "role": "Chief Engineer",
            "preferences": [],
            "notes": [
                "Marquinho wants Miguel to be creative and imaginative.",
            ],
        },
    },
    "topic_memory": [],
}


def load_memory():
    if not MEMORY_PATH.exists():
        save_memory(DEFAULT_MEMORY)
        return dict(DEFAULT_MEMORY)

    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = dict(DEFAULT_MEMORY)

    # Fill missing top-level keys.
    for k, v in DEFAULT_MEMORY.items():
        data.setdefault(k, v)

    return data


def save_memory(memory):
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)


def normalize_person_name(name):
    if not name:
        return "unknown"
    return name.lower().strip().replace(" ", "_")


def get_robot_mode():
    return load_memory().get("robot_mode", "normal")


def set_robot_mode(mode):
    memory = load_memory()
    memory["robot_mode"] = mode
    save_memory(memory)


def get_personality_mode():
    return load_memory().get("personality_mode", "mission_control")


def set_personality_mode(mode):
    memory = load_memory()
    memory["personality_mode"] = mode
    save_memory(memory)


def set_pending_shutdown(value):
    memory = load_memory()
    memory["pending_shutdown"] = bool(value)
    save_memory(memory)


def get_pending_shutdown():
    return bool(load_memory().get("pending_shutdown", False))


def add_profile_note(person, note):
    person = normalize_person_name(person)
    memory = load_memory()
    profiles = memory.setdefault("profiles", {})
    profile = profiles.setdefault(person, {"role": "", "preferences": [], "notes": []})

    item = {
        "text": note,
        "created_at": time.time(),
    }

    profile.setdefault("notes", []).append(item)
    save_memory(memory)


def add_preference(person, preference):
    person = normalize_person_name(person)
    memory = load_memory()
    profiles = memory.setdefault("profiles", {})
    profile = profiles.setdefault(person, {"role": "", "preferences": [], "notes": []})

    item = {
        "text": preference,
        "created_at": time.time(),
    }

    profile.setdefault("preferences", []).append(item)
    save_memory(memory)


def add_topic_memory(topic_text):
    memory = load_memory()
    memory.setdefault("topic_memory", []).append({
        "text": topic_text,
        "created_at": time.time(),
    })
    save_memory(memory)


def get_memory_context(person=None):
    memory = load_memory()
    person = normalize_person_name(person)

    profile = memory.get("profiles", {}).get(person, {})

    return {
        "robot_mode": memory.get("robot_mode", "normal"),
        "personality_mode": memory.get("personality_mode", "mission_control"),
        "profile": profile,
        "topic_memory": memory.get("topic_memory", [])[-10:],
    }


def extract_memory_text(user_text):
    text = user_text.strip()

    patterns = [
        r"remember that (.+)",
        r"remember this (.+)",
        r"note that (.+)",
        r"my preference is (.+)",
        r"i prefer (.+)",
        r"i like (.+)",
    ]

    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

    return None


def handle_robot_mode_command(user_text, recognized_person=None):
    """
    Returns a reply string if handled locally.
    Returns None if no mode/memory command was detected.
    """
    text = user_text.lower().strip()
    person = normalize_person_name(recognized_person)

    memory = load_memory()

    # Wake from sleep.
    if memory.get("robot_mode") == "sleep":
        wake_phrases = [
            "wake up",
            "miguel wake up",
            "come back",
            "mission control",
            "hey miguel",
            "hey me go",
        ]

        if any(p in text for p in wake_phrases):
            memory["robot_mode"] = "normal"
            save_memory(memory)
            return "I am awake again. Miguel is back online."

        # Silent ignore.
        return "__SILENT__"

    # Sleep / silence.
    if any(p in text for p in [
        "go to sleep",
        "sleep mode",
        "stay silent",
        "be quiet and wait",
        "be idle",
        "stop talking until i call you",
    ]):
        memory["robot_mode"] = "sleep"
        memory["pending_shutdown"] = False
        save_memory(memory)
        return "Sleep mode activated. I will stay quiet until you say Miguel wake up or Mission Control."

    # Shutdown two-step confirmation.
    if any(p in text for p in [
        "shutdown mode",
        "prepare shutdown",
        "turn off",
        "power down",
        "shut down",
    ]):
        memory["pending_shutdown"] = True
        save_memory(memory)
        return "Shutdown confirmation required. Say confirm shutdown if you want me to power off the Jetson."

    if memory.get("pending_shutdown") and any(p in text for p in [
        "confirm shutdown",
        "yes shutdown",
        "shutdown now",
        "yes turn off",
    ]):
        memory["pending_shutdown"] = False
        memory["robot_mode"] = "shutdown"
        save_memory(memory)
        return "__SHUTDOWN__"

    if memory.get("pending_shutdown") and any(p in text for p in [
        "cancel",
        "cancel shutdown",
        "no",
        "not now",
    ]):
        memory["pending_shutdown"] = False
        save_memory(memory)
        return "Shutdown cancelled."

    # Personality modes.
    mode_map = {
        "creative mode": "creative",
        "be creative": "creative",
        "more creative": "creative",
        "teacher mode": "teacher",
        "teach me": "teacher",
        "engineer mode": "engineer",
        "technical mode": "engineer",
        "mission control mode": "mission_control",
        "quiet mode": "quiet",
        "normal mode": "mission_control",
        "default mode": "mission_control",
    }

    for phrase, mode in mode_map.items():
        if phrase in text:
            memory["personality_mode"] = mode
            save_memory(memory)

            if mode == "creative":
                return "Creative mode activated. I will be more imaginative and propose ideas."
            if mode == "teacher":
                return "Teacher mode activated. I will explain slowly and step by step."
            if mode == "engineer":
                return "Engineer mode activated. I will be more technical and precise."
            if mode == "quiet":
                return "Quiet mode activated. I will keep answers shorter."
            return "Mission Control mode activated."

    # Memory commands.
    memory_text = extract_memory_text(user_text)
    if memory_text:
        if "prefer" in text or "like" in text or "preference" in text:
            add_preference(person, memory_text)
            return f"Got it. I saved that preference for {person}."
        else:
            add_profile_note(person, memory_text)
            return f"Got it. I saved that note for {person}."

    if "what do you remember about me" in text or "what is my profile" in text:
        ctx = get_memory_context(person)
        profile = ctx.get("profile", {})
        prefs = profile.get("preferences", [])[-5:]
        notes = profile.get("notes", [])[-5:]

        parts = [f"Your profile name is {person}."]

        if profile.get("role"):
            parts.append(f"Role: {profile['role']}.")

        if prefs:
            pref_text = "; ".join(p["text"] if isinstance(p, dict) else str(p) for p in prefs)
            parts.append(f"Recent preferences: {pref_text}.")

        if notes:
            note_text = "; ".join(n["text"] if isinstance(n, dict) else str(n) for n in notes)
            parts.append(f"Recent notes: {note_text}.")

        return " ".join(parts)

    if "what mode are you in" in text or "current mode" in text:
        return (
            f"Robot mode is {memory.get('robot_mode', 'normal')}. "
            f"Personality mode is {memory.get('personality_mode', 'mission_control')}."
        )

    return None
