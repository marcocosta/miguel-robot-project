from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_memory.py"
text = path.read_text()

marker = "def handle_robot_mode_command(user_text, recognized_person=None):"
idx = text.find(marker)

if idx == -1:
    raise SystemExit("Could not find handle_robot_mode_command().")

clean_function = r'''
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
            "sleep mode to stay silent until called, and shutdown mode to safely power off after confirmation."
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
        "stay silent",
        "be quiet and wait",
        "be idle",
        "stop talking until i call you",
    ]):
        memory["robot_mode"] = "sleep"
        memory["pending_shutdown"] = False
        save_memory(memory)
        return "Sleep mode activated. I will stay quiet until you say Miguel wake up or Mission Control."

    # Shutdown must be explicit. Do not trigger on questions ABOUT shutdown.
    if any(p in text for p in [
        "prepare shutdown",
        "start shutdown",
        "turn yourself off",
        "power down now",
        "shut down now",
        "shutdown now",
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
'''

# handle_robot_mode_command is the last function in this file, so replace to EOF.
text = text[:idx] + clean_function.strip() + "\n"

# Normalize tabs just in case.
text = text.replace("\t", "    ")

path.write_text(text)
print("Replaced handle_robot_mode_command cleanly.")
