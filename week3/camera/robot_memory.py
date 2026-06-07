import json
import re
import time
from pathlib import Path

MEMORY_DIR = Path.home() / "robot-project/week3/memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

MEMORY_PATH = MEMORY_DIR / "miguel_memory.json"
CONVERSATION_LOG_DIR = MEMORY_DIR / "conversation_logs"
CONVERSATION_ARCHIVE_SUMMARY_PATH = MEMORY_DIR / "conversation_log_archive_summaries.jsonl"
CONVERSATION_LOG_RETENTION_DAYS = 30
CONVERSATION_LOG_MAX_BYTES = 200 * 1024 * 1024

DEFAULT_MEMORY = {
    "memory_schema_version": 2,
    "robot_mode": "normal",
    "personality_mode": "mission_control",
    "enrollment_unlock": {"active": False, "authorized_by": None, "expires_at": 0},
    "voice_mode": "natural_voice",
    "pending_shutdown": False,
    "active_topic_id": None,
    "self_learning": {
        "enabled": True,
        "approved_facts": [],
        "pending_candidates": [],
        "last_reviewed_at": None,
    },
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

    data["memory_schema_version"] = max(int(data.get("memory_schema_version") or 1), DEFAULT_MEMORY["memory_schema_version"])
    data.setdefault("self_learning", {})
    for k, v in DEFAULT_MEMORY["self_learning"].items():
        data["self_learning"].setdefault(k, v)
    data.setdefault("long_term_topics", {})
    data["voice_mode"] = normalize_voice_mode(data.get("voice_mode", "natural_voice"))
    data.setdefault("enrollment_unlock", {"active": False, "authorized_by": None, "expires_at": 0})
    data.setdefault("active_topic_id", None)

    return data


def save_memory(memory):
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)


def _conversation_log_session_id():
    return time.strftime("%Y%m%d_%H%M%S")


def start_conversation_log_session():
    CONVERSATION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    session_id = _conversation_log_session_id()
    path = CONVERSATION_LOG_DIR / f"{session_id}.jsonl"
    event = {
        "type": "session_start",
        "session_id": session_id,
        "created_at": now_ts(),
        "retention": {
            "full_log_days": CONVERSATION_LOG_RETENTION_DAYS,
            "max_total_bytes": CONVERSATION_LOG_MAX_BYTES,
            "deleted_logs_keep_short_summary": True,
        },
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    cleanup_conversation_logs()
    return {"session_id": session_id, "path": str(path)}


def append_conversation_log_event(session_id, event_type, **payload):
    if not session_id:
        return
    CONVERSATION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONVERSATION_LOG_DIR / f"{session_id}.jsonl"
    event = {
        "type": event_type,
        "session_id": session_id,
        "created_at": now_ts(),
    }
    event.update(payload)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    events = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    except Exception:
        return []
    return events


def _summarize_log_for_archive(path):
    events = _read_jsonl(path)
    turns = [e for e in events if e.get("type") == "turn"]
    text = " ".join(
        str(e.get("user_text") or "") + " " + str(e.get("assistant_reply") or "")
        for e in turns
    )
    unsafe_markers = [
        "weapon", "bomb", "kill", "hurt", "self harm", "suicide",
        "password", "enroll", "shutdown", "unsafe", "inappropriate",
    ]
    flagged = [m for m in unsafe_markers if m in text.lower()]
    snippets = []
    for e in turns[:6]:
        user = re.sub(r"\s+", " ", str(e.get("user_text") or "")).strip()
        if user:
            snippets.append(user[:90])
    return {
        "type": "deleted_log_summary",
        "deleted_file": path.name,
        "deleted_at": now_ts(),
        "session_id": path.stem,
        "turn_count": len(turns),
        "first_event_at": events[0].get("created_at") if events else None,
        "last_event_at": events[-1].get("created_at") if events else None,
        "possible_sensitive_markers": flagged[:10],
        "short_summary": "; ".join(snippets)[:500] or "No turn text captured.",
    }


def _archive_and_delete_log(path):
    summary = _summarize_log_for_archive(path)
    with open(CONVERSATION_ARCHIVE_SUMMARY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def cleanup_conversation_logs():
    CONVERSATION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    now = now_ts()
    paths = sorted(CONVERSATION_LOG_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    cutoff = now - CONVERSATION_LOG_RETENTION_DAYS * 86400
    for path in list(paths):
        try:
            if path.stat().st_mtime < cutoff:
                _archive_and_delete_log(path)
        except FileNotFoundError:
            pass

    paths = sorted(CONVERSATION_LOG_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in paths if p.exists())
    for path in paths:
        if total <= CONVERSATION_LOG_MAX_BYTES:
            break
        try:
            size = path.stat().st_size
            _archive_and_delete_log(path)
            total -= size
        except FileNotFoundError:
            pass


def list_conversation_logs(limit=8):
    cleanup_conversation_logs()
    paths = sorted(CONVERSATION_LOG_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    results = []
    for path in paths[:limit]:
        events = _read_jsonl(path)
        turns = [e for e in events if e.get("type") == "turn"]
        topics = []
        for e in turns:
            topic = str(e.get("topic") or "").strip()
            if topic and topic not in topics:
                topics.append(topic)
        results.append({
            "session_id": path.stem,
            "path": str(path),
            "updated_at": path.stat().st_mtime,
            "turn_count": len(turns),
            "topics": topics[:5],
        })
    return results


def _conversation_log_date_token(offset_days=0):
    return time.strftime("%Y%m%d", time.localtime(now_ts() + (offset_days * 86400)))


def select_conversation_logs(query="", current_session_id=None, limit=3):
    """Select logs for memory recall/analysis without exposing file details to callers."""
    normalized = re.sub(r"\s+", " ", str(query or "").lower()).strip()
    logs = list_conversation_logs(limit=50)

    if current_session_id:
        current_session_id = str(current_session_id)

    selected = logs
    if "yesterday" in normalized:
        token = _conversation_log_date_token(-1)
        selected = [log for log in logs if str(log.get("session_id", "")).startswith(token)]
        if not selected:
            selected = [log for log in logs if log.get("session_id") != current_session_id]
    elif "today" in normalized:
        token = _conversation_log_date_token(0)
        selected = [log for log in logs if str(log.get("session_id", "")).startswith(token)]
    elif any(marker in normalized for marker in ("previous", "last conversation", "last log", "last interaction", "before this")):
        selected = [log for log in logs if log.get("session_id") != current_session_id]

    selected = [log for log in selected if int(log.get("turn_count") or 0) > 0]
    return [
        {
            "metadata": log,
            "events": _read_jsonl(Path(log["path"])),
        }
        for log in selected[:limit]
    ]


def load_conversation_log(session_id=None, limit=1):
    logs = list_conversation_logs(limit=max(1, limit if session_id is None else 50))
    if session_id:
        wanted = str(session_id).strip()
        logs = [log for log in logs if wanted in log["session_id"]]
    selected = logs[:limit]
    return [
        {
            "metadata": log,
            "events": _read_jsonl(Path(log["path"])),
        }
        for log in selected
    ]


def format_recent_conversation_topics(limit=5):
    logs = list_conversation_logs(limit=limit)
    if not logs:
        return "I do not have conversation logs yet."
    parts = []
    for log in logs:
        topics = ", ".join(log.get("topics") or []) or "general conversation"
        parts.append(f"{log['session_id']}: {topics}")
    return "Recent conversation topics: " + "; ".join(parts) + "."


def format_conversation_log_recall(query="", current_session_id=None, limit=3):
    logs = select_conversation_logs(query=query, current_session_id=current_session_id, limit=limit)
    if not logs:
        return "I do not have matching conversation logs yet."

    parts = []
    for log in logs:
        metadata = log.get("metadata", {})
        turns = [event for event in log.get("events", []) if event.get("type") == "turn"]
        snippets = []
        for event in turns[-4:]:
            user = re.sub(r"\s+", " ", str(event.get("user_text") or "")).strip()
            topic = str(event.get("topic") or "").strip()
            if user:
                snippets.append(user[:80])
            elif topic:
                snippets.append(topic[:80])
        topics = ", ".join(metadata.get("topics") or []) or "general conversation"
        if snippets:
            parts.append(f"{metadata.get('session_id')}: {topics}. Recent: " + "; ".join(snippets))
        else:
            parts.append(f"{metadata.get('session_id')}: {topics}")
    return "I found these conversation logs: " + " | ".join(parts) + "."


def add_learning_candidate(kind, text, source="conversation", metadata=None):
    text = str(text or "").strip()
    if not text:
        return None
    memory = load_memory()
    learning = memory.setdefault("self_learning", {})
    candidates = learning.setdefault("pending_candidates", [])
    record = {
        "kind": str(kind or "note"),
        "text": text,
        "source": str(source or "conversation"),
        "metadata": metadata or {},
        "created_at": now_ts(),
        "approved": False,
    }
    candidates.append(record)
    learning["pending_candidates"] = candidates[-100:]
    save_memory(memory)
    return record


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
        "voice_mode": memory.get("voice_mode", "robot_voice"),
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



OWNER_NAMES = {"marco", "marquinho"}


def is_owner_name(name):
    return normalize_person_name(name) in OWNER_NAMES


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




def normalize_voice_mode(mode):
    aliases = {
        "natural_robot": "natural_voice",
        "natural": "natural_voice",
        "normal": "natural_voice",
        "normal_voice": "natural_voice",
        "default_voice": "natural_voice",
        "friendly": "friendly_voice",
        "deep": "deep_voice",
        "story": "story_voice",
        "robot": "robot_voice",
        "robotic_voice": "robot_voice",
        "local_robot": "robot_voice",
    }
    return aliases.get(str(mode or "").strip(), str(mode or "robot_voice").strip())


def handle_voice_mode_command(user_text):
    text = user_text.lower().strip()
    memory = load_memory()

    # Normalize old stored values.
    memory["voice_mode"] = normalize_voice_mode(memory.get("voice_mode", "robot_voice"))

    voice_options_reply = (
        "I have five voice modes: robot voice, natural voice, friendly voice, "
        "deep voice, and story voice."
    )

    if any(p in text for p in [
        "what voice options",
        "which voice options",
        "what voices do you have",
        "which voices do you have",
        "voice options",
        "list voices",
        "available voices",
        "which options of voice",
        "what options of voice",
    ]):
        save_memory(memory)
        return voice_options_reply

    voice_map = [
        (
            [
                "robot voice",
                "robotic voice",
                "use robot voice",
                "use robotic voice",
                "go to robot voice",
                "go to robotic voice",
                "switch to robot voice",
                "switch to robotic voice",
                "classic robot voice",
                "local voice",
            ],
            "robot_voice",
            "Robot voice activated.",
        ),
        (
            [
                "natural voice",
                "use natural voice",
                "more natural voice",
                "speak naturally",
                "normal voice",
                "startup voice",
                "start voice",
                "go to normal voice",
                "use normal voice",
                "default voice",
            ],
            "natural_voice",
            "Natural voice activated.",
        ),
        (
            [
                "friendly voice",
                "warmer voice",
                "use friendly voice",
                "go to friendly voice",
            ],
            "friendly_voice",
            "Friendly voice activated.",
        ),
        (
            [
                "deep voice",
                "deeper voice",
                "use deep voice",
                "go to deep voice",
            ],
            "deep_voice",
            "Deep voice activated.",
        ),
        (
            [
                "story voice",
                "storyteller voice",
                "use storyteller voice",
                "narrator voice",
                "use story voice",
                "go to story voice",
            ],
            "story_voice",
            "Story voice activated.",
        ),
        (
            [
                "mission control voice",
                "space robot voice",
            ],
            "robot_voice",
            "Mission Control robot voice activated.",
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
        "which voice are you now",
        "what voice are you now",
        "voice mode",
    ]):
        mode = normalize_voice_mode(memory.get("voice_mode", "robot_voice"))
        memory["voice_mode"] = mode
        save_memory(memory)
        return f"My current voice mode is {mode}."

    save_memory(memory)
    return None


def handle_robot_mode_command(user_text, recognized_person=None):
    text = user_text.lower().strip()
    text = re.split(
        r"\nmiguel (?:response length|creative|long story|long explanation|language|live conversation) instruction:",
        text,
        maxsplit=1,
    )[0].strip()
    person = normalize_person_name(recognized_person)

    memory = load_memory()

    voice_reply = handle_voice_mode_command(user_text)
    if voice_reply:
        return voice_reply

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

    # Shutdown confirmation must be checked BEFORE new shutdown requests.
    # Otherwise "confirm shutdown" gets treated as another shutdown request.
    if memory.get("pending_shutdown") and any(p in text for p in [
        "confirm shutdown",
        "confirm shut down",
        "yes shutdown",
        "yes shut down",
        "shutdown confirmed",
        "shutdown confirmation",
        "shutdown confirmado",
        "confirme shutdown",
        "confirmar shutdown",
        "yes turn off",
    ]):
        memory["pending_shutdown"] = False
        memory["robot_mode"] = "normal"
        save_memory(memory)
        return "__STOP_ROBOT__"

    if memory.get("pending_shutdown"):
        cancel_phrases = [
            "cancel",
            "cancel shutdown",
            "cancel shut down",
            "no",
            "no shutdown",
            "not now",
        ]

        if text in cancel_phrases:
            memory["pending_shutdown"] = False
            save_memory(memory)
            return "Shutdown cancelled."

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
        return "Shutdown confirmation required. Say confirm shutdown if you want me to stop the robot program. The Jetson will stay on."

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
