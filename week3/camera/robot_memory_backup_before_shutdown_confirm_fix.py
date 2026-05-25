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
    "voice_mode": "natural_robot",
    "pending_shutdown": False,
    "active_topic_id": None,
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
                {
                    "text": "Marquinho wants Miguel to be creative and imaginative.",
                    "created_at": time.time(),
                }
            ],
        },
    },
    "topic_memory": [],
    "long_term_topics": {},
}


def now_ts():
    return time.time()


def slugify(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or f"topic_{int(time.time())}"


def load_memory():
    if not MEMORY_PATH.exists():
        save_memory(DEFAULT_MEMORY)
        return json.loads(json.dumps(DEFAULT_MEMORY))

    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = json.loads(json.dumps(DEFAULT_MEMORY))

    for k, v in DEFAULT_MEMORY.items():
        data.setdefault(k, v)

    data.setdefault("long_term_topics", {})
    data.setdefault("voice_mode", "natural_robot")
    data.setdefault("active_topic_id", None)

    return data


def save_memory(memory):
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
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
    profile.setdefault("notes", []).append({"text": note, "created_at": now_ts()})
    save_memory(memory)


def add_preference(person, preference):
    person = normalize_person_name(person)
    memory = load_memory()
    profiles = memory.setdefault("profiles", {})
    profile = profiles.setdefault(person, {"role": "", "preferences": [], "notes": []})
    profile.setdefault("preferences", []).append({"text": preference, "created_at": now_ts()})
    save_memory(memory)


def add_topic_memory(topic_text):
    memory = load_memory()
    memory.setdefault("topic_memory", []).append({
        "text": topic_text,
        "created_at": now_ts(),
    })
    save_memory(memory)


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


# ============================================================
# Long-term topic memory
# ============================================================

def guess_topic_title_from_text(text):
    t = text.lower()

    known = [
        ("time travel", ["time travel", "time traveling", "time machine"]),
        ("gravity", ["gravity", "gravity force"]),
        ("extreme life on Earth", ["extreme life", "extremophile", "life exist at its most extreme"]),
        ("Big Bang theory", ["big bang"]),
        ("creativity", ["creativity", "creative"]),
        ("robot personalities", ["personality", "personalities", "creative mode", "teacher mode"]),
        ("Miguel robot project", ["robot", "miguel", "face recognition", "voice recognition"]),
    ]

    for title, patterns in known:
        if any(p in t for p in patterns):
            return title

    cleaned = re.sub(r"^(miguel|hey miguel|mission control)[, ]*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"^(remember|save|keep talking about|resume|topic)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" .?!")

    words = cleaned.split()
    if not words:
        return "general ideas"

    return " ".join(words[:6])


def create_or_update_long_term_topic(title, summary=None, owner=None, source_text=None, activate=True):
    memory = load_memory()
    topics = memory.setdefault("long_term_topics", {})

    topic_id = slugify(title)
    existing = topics.get(topic_id, {})

    topic = {
        "id": topic_id,
        "title": existing.get("title", title),
        "summary": summary or existing.get("summary", ""),
        "owner": normalize_person_name(owner) if owner else existing.get("owner", "unknown"),
        "created_at": existing.get("created_at", now_ts()),
        "updated_at": now_ts(),
        "importance": existing.get("importance", "medium"),
        "status": existing.get("status", "active"),
        "turns": existing.get("turns", []),
    }

    if source_text:
        topic["turns"].append({
            "speaker": normalize_person_name(owner),
            "text": source_text,
            "created_at": now_ts(),
        })

    # Keep last 25 turns per topic to avoid huge JSON.
    topic["turns"] = topic["turns"][-25:]

    topics[topic_id] = topic

    if activate:
        memory["active_topic_id"] = topic_id

    save_memory(memory)
    return topic


def get_active_long_term_topic():
    memory = load_memory()
    topic_id = memory.get("active_topic_id")
    if not topic_id:
        return None
    return memory.get("long_term_topics", {}).get(topic_id)


def set_active_long_term_topic(title_or_id):
    memory = load_memory()
    topics = memory.get("long_term_topics", {})
    wanted = slugify(title_or_id)

    if wanted in topics:
        memory["active_topic_id"] = wanted
        save_memory(memory)
        return topics[wanted]

    # fuzzy title match
    for topic_id, topic in topics.items():
        if wanted in topic_id or wanted in slugify(topic.get("title", "")):
            memory["active_topic_id"] = topic_id
            save_memory(memory)
            return topic

    return None


def forget_long_term_topic(title_or_id):
    memory = load_memory()
    topics = memory.get("long_term_topics", {})
    wanted = slugify(title_or_id)

    found = None
    if wanted in topics:
        found = wanted
    else:
        for topic_id, topic in topics.items():
            if wanted in topic_id or wanted in slugify(topic.get("title", "")):
                found = topic_id
                break

    if not found:
        return None

    removed = topics.pop(found)

    if memory.get("active_topic_id") == found:
        memory["active_topic_id"] = None

    save_memory(memory)
    return removed


def list_long_term_topics(limit=10):
    memory = load_memory()
    topics = list(memory.get("long_term_topics", {}).values())
    topics.sort(key=lambda t: t.get("updated_at", 0), reverse=True)
    return topics[:limit]


def append_turn_to_active_topic(user_text=None, assistant_reply=None, person=None):
    memory = load_memory()
    topic_id = memory.get("active_topic_id")
    if not topic_id:
        return None

    topics = memory.setdefault("long_term_topics", {})
    topic = topics.get(topic_id)
    if not topic:
        return None

    if user_text:
        topic.setdefault("turns", []).append({
            "speaker": normalize_person_name(person),
            "text": user_text,
            "created_at": now_ts(),
        })

    if assistant_reply:
        topic.setdefault("turns", []).append({
            "speaker": "miguel",
            "text": assistant_reply,
            "created_at": now_ts(),
        })

    topic["updated_at"] = now_ts()
    topic["turns"] = topic["turns"][-25:]

    topics[topic_id] = topic
    save_memory(memory)
    return topic


def should_auto_attach_to_topic(user_text):
    topic = get_active_long_term_topic()
    if not topic:
        return False

    text = user_text.lower().strip()

    # Do not attach pure mode commands.
    if any(p in text for p in ["go to sleep", "shutdown", "what mode", "creative mode", "teacher mode"]):
        return False

    return len(text.split()) >= 3


def get_long_term_topic_context():
    active = get_active_long_term_topic()
    recent = list_long_term_topics(limit=5)

    return {
        "active_topic": active,
        "recent_topics": [
            {
                "id": t.get("id"),
                "title": t.get("title"),
                "summary": t.get("summary"),
                "owner": t.get("owner"),
                "updated_at": t.get("updated_at"),
            }
            for t in recent
        ],
    }


def handle_long_term_topic_command(user_text, recognized_person=None, conversation_memory=None):
    text = user_text.lower().strip()
    person = normalize_person_name(recognized_person)

    # Explicit save current topic.
    if any(p in text for p in [
        "remember this topic",
        "save this topic",
        "keep this topic",
        "keep talking about this",
    ]):
        last_topic = ""
        if conversation_memory:
            last_topic = conversation_memory.get("last_topic", "")

        title = last_topic or guess_topic_title_from_text(user_text)
        summary = f"{person} wants to keep discussing: {title}."
        topic = create_or_update_long_term_topic(
            title=title,
            summary=summary,
            owner=person,
            source_text=user_text,
            activate=True,
        )
        return f"Got it. I saved the long-term topic: {topic['title']}."

    # Save topic with explicit name.
    m = re.search(r"(save|remember|keep talking about)\s+(?:the\s+)?topic\s+(.+)", text)
    if not m:
        m = re.search(r"keep talking about\s+(.+)", text)

    if m:
        title = m.group(2).strip() if len(m.groups()) >= 2 else m.group(1).strip()
        title = title.strip(" .?!")
        topic = create_or_update_long_term_topic(
            title=title,
            summary=f"{person} wants to keep talking about {title}.",
            owner=person,
            source_text=user_text,
            activate=True,
        )
        return f"Saved. Our active long-term topic is now: {topic['title']}."

    # Resume topic.
    m = re.search(r"resume\s+(?:the\s+)?topic\s+(.+)", text)
    if not m:
        m = re.search(r"go back to\s+(.+)", text)

    if m:
        title = m.group(1).strip(" .?!")
        topic = set_active_long_term_topic(title)
        if topic:
            return f"Resuming topic: {topic['title']}. {topic.get('summary', '')}".strip()
        return f"I could not find a saved topic called {title}."

    # Forget topic.
    m = re.search(r"forget\s+(?:the\s+)?topic\s+(.+)", text)
    if m:
        title = m.group(1).strip(" .?!")
        removed = forget_long_term_topic(title)
        if removed:
            return f"Forgot the topic: {removed['title']}."
        return f"I could not find a saved topic called {title}."

    # List topics.
    if any(p in text for p in [
        "what topics do you remember",
        "list topics",
        "saved topics",
        "long term topics",
    ]):
        topics = list_long_term_topics(limit=8)
        if not topics:
            return "I do not have any long-term topics saved yet."

        parts = []
        for i, topic in enumerate(topics, start=1):
            parts.append(f"{i}. {topic.get('title')}")

        return "I remember these long-term topics: " + "; ".join(parts) + "."

    # Active topic.
    if any(p in text for p in [
        "what is our active topic",
        "current topic",
        "active topic",
    ]):
        topic = get_active_long_term_topic()
        if not topic:
            return "There is no active long-term topic right now."
        return f"Our active long-term topic is: {topic.get('title')}. {topic.get('summary', '')}".strip()

    return None


def get_memory_context(person=None):
    memory = load_memory()
    person = normalize_person_name(person)

    profile = memory.get("profiles", {}).get(person, {})

    return {
        "robot_mode": memory.get("robot_mode", "normal"),
        "personality_mode": memory.get("personality_mode", "mission_control"),
        "voice_mode": memory.get("voice_mode", "natural_robot"),
        "profile": profile,
        "topic_memory": memory.get("topic_memory", [])[-10:],
        "long_term_topic_context": get_long_term_topic_context(),
    }


# ============================================================
# Robot modes / profile commands
# ============================================================


def get_voice_mode():
    return load_memory().get("voice_mode", "natural_robot")


def set_voice_mode(mode):
    memory = load_memory()
    memory["voice_mode"] = mode
    save_memory(memory)


def handle_robot_mode_command(user_text, recognized_person=None):
    text = user_text.lower().strip()
    person = normalize_person_name(recognized_person)

    memory = load_memory()

    # Questions ABOUT modes should not trigger shutdown/sleep.
    if any(p in text for p in [
        "which modes do you have",
        "what modes do you have",
        "describe the modes",
        "explain the modes",
        "what about the modes",
        "tell me about the modes",
        "what about sleep",
        "what about shutdown",
        "sleep and shutdown",
    ]):
        return (
            "I have normal mode for regular conversation, creative mode for ideas, "
            "teacher mode for slow explanations, engineer mode for technical work, "
            "quiet mode for short answers, mission control mode for robot-project teamwork, "
            "sleep mode to stay silent until called, shutdown mode to safely power off after confirmation, and voice modes like natural, robot, friendly, deep, and story voice."
        )

    # If already sleeping, only wake phrases are handled.
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

        return "__SILENT__"

    # Sleep / silence mode.
    if any(p in text for p in [
        "go to sleep",
        "sleep mode",
        "sleepy mode",
        "stay silent",
        "be quiet and wait",
        "be idle",
        "stop talking until i call you",
    ]):
        memory["robot_mode"] = "sleep"
        memory["pending_shutdown"] = False
        save_memory(memory)
        return "Sleep mode activated. I will stay quiet until you say Miguel wake up or Mission Control."

    # Shutdown must be explicit, but normal phrases like "shut down" should work.
    # Do not trigger if the user is asking ABOUT shutdown/modes.
    shutdown_question_phrases = [
        "what about shutdown",
        "tell me about shutdown",
        "explain shutdown",
        "which modes",
        "what modes",
        "describe the modes",
    ]

    shutdown_request_phrases = [
        "prepare shutdown",
        "start shutdown",
        "turn yourself off",
        "power down",
        "power down now",
        "shut down",
        "shut down now",
        "shutdown",
        "shutdown now",
        "turn off",
    ]

    if not any(q in text for q in shutdown_question_phrases) and any(p in text for p in shutdown_request_phrases):
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

    # Profile memory commands.
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

        active_topic = get_active_long_term_topic()
        if active_topic:
            parts.append(f"Active long-term topic: {active_topic.get('title')}.")

        return " ".join(parts)

    if "what mode are you in" in text or "current mode" in text:
        return (
            f"Robot mode is {memory.get('robot_mode', 'normal')}. "
            f"Personality mode is {memory.get('personality_mode', 'mission_control')}."
        )

    return None
