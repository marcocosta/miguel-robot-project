from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"
text = path.read_text()

# Add conversation memory globals before handle_user_turn_with_cached_state.
marker = "def handle_user_turn_with_cached_state(user_text, cached_local_state):"
idx = text.find(marker)
if idx == -1:
    raise SystemExit("Could not find handle_user_turn_with_cached_state().")

memory_block = r'''
CONVERSATION_MEMORY = {
    "last_user_text": "",
    "last_assistant_reply": "",
    "last_topic": "",
}


def update_conversation_memory(user_text=None, assistant_reply=None, topic=None):
    if user_text is not None:
        CONVERSATION_MEMORY["last_user_text"] = user_text
    if assistant_reply is not None:
        CONVERSATION_MEMORY["last_assistant_reply"] = assistant_reply
    if topic is not None:
        CONVERSATION_MEMORY["last_topic"] = topic


def infer_simple_topic(text):
    t = text.lower()
    if "life" in t or "extreme" in t or "extremophile" in t:
        return "life in extreme environments"
    if "gravity" in t:
        return "gravity"
    if "big bang" in t:
        return "Big Bang theory"
    if "creativ" in t:
        return "creativity"
    if "robot" in t or "face recognition" in t or "voice" in t:
        return "Miguel robot project"
    return CONVERSATION_MEMORY.get("last_topic", "")


def is_short_acknowledgement(user_text):
    t = user_text.lower().strip()
    return t in {"ok", "okay", "yes", "yeah", "yep", "no", "nope", "não", "sim"}


def handle_short_acknowledgement(user_text):
    t = user_text.lower().strip()
    last_reply = CONVERSATION_MEMORY.get("last_assistant_reply", "")
    last_topic = CONVERSATION_MEMORY.get("last_topic", "")

    if t in {"ok", "okay"}:
        if last_topic:
            return f"Okay. We can keep going about {last_topic}, or you can ask me the next question."
        return "Okay. I'm ready for your next question."

    if t in {"no", "nope", "não"}:
        return "Okay, no problem. Tell me what you want instead."

    if t in {"yes", "yeah", "yep", "sim"}:
        if last_reply:
            return "Yes — continuing from what I just said. What part should I explain next?"
        return "Yes. Tell me what you want to do next."

    return None

'''

if "CONVERSATION_MEMORY = {" not in text:
    text = text[:idx] + memory_block + "\n" + text[idx:]


# Add memory into ask_cloud_brain payload and instructions.
old_payload = '''    payload = {
        "user_text": user_text,
        "local_state": local_state,
        "robot_capabilities": [
'''

new_payload = '''    payload = {
        "user_text": user_text,
        "local_state": local_state,
        "conversation_memory": CONVERSATION_MEMORY,
        "robot_capabilities": [
'''

if old_payload in text:
    text = text.replace(old_payload, new_payload)
else:
    print("Warning: payload block not found exactly.")

# Strengthen cloud brain rules.
old_rules = '''Rules:
- Use local_state when relevant.
- Do not invent things you cannot sense.
- If asked what you see, only use local_state.
- If recognized_person is present, you may greet that person by name.
- Do not claim you can move unless local_state says motion hardware exists.
'''

new_rules = '''Rules:
- Use local_state when relevant.
- Use conversation_memory to understand short follow-ups like ok, yes, no, repeat that, or I did not understand.
- Stay on the current topic unless the user clearly changes topics.
- Do not randomly return to robot testing, face enrollment, or voice testing unless the user asks about the robot.
- Speak in English unless the user clearly asks for Portuguese or uses a full Portuguese sentence.
- Do not invent things you cannot sense.
- If asked what you see, only use local_state.
- If recognized_person is present, you may greet that person by name, but do not turn every answer into a greeting.
- Do not claim you can move unless local_state says motion hardware exists.
'''

if old_rules in text:
    text = text.replace(old_rules, new_rules)
else:
    print("Warning: rules block not found exactly.")


# Patch handle_user_turn_with_cached_state with short-ack handling and memory updates.
old = '''    local_skill_reply = maybe_handle_local_skill(user_text, local_state)
    if local_skill_reply:
        speak(local_skill_reply)
        return True
'''

new = '''    topic = infer_simple_topic(user_text)
    update_conversation_memory(user_text=user_text, topic=topic)

    short_ack_reply = handle_short_acknowledgement(user_text)
    if short_ack_reply:
        speak(short_ack_reply)
        update_conversation_memory(assistant_reply=short_ack_reply)
        return True

    local_skill_reply = maybe_handle_local_skill(user_text, local_state)
    if local_skill_reply:
        speak(local_skill_reply)
        update_conversation_memory(assistant_reply=local_skill_reply)
        return True
'''

if old in text:
    text = text.replace(old, new)
else:
    print("Warning: local_skill block not found exactly.")


old = '''    if local_state.get("recognized_person") in CUSTOM_GREETINGS and is_simple_greeting(user_text):
        speak(CUSTOM_GREETINGS[local_state["recognized_person"]])
        return True
'''

new = '''    if local_state.get("recognized_person") in CUSTOM_GREETINGS and is_simple_greeting(user_text):
        greeting = CUSTOM_GREETINGS[local_state["recognized_person"]]
        speak(greeting)
        update_conversation_memory(assistant_reply=greeting)
        return True
'''

if old in text:
    text = text.replace(old, new)
else:
    print("Warning: greeting block not found exactly.")


old = '''    speak(reply)
    return True
'''

new = '''    speak(reply)
    update_conversation_memory(assistant_reply=reply)
    return True
'''

# This may replace multiple speak(reply) blocks, which is okay for this file.
text = text.replace(old, new)

path.write_text(text)
print(f"Patched V6.6 conversation memory: {path}")
