from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"
text = path.read_text()

# Extend imports.
if "handle_long_term_topic_command" not in text:
    text = text.replace(
        "handle_robot_mode_command,\n)",
        "handle_robot_mode_command,\n    handle_long_term_topic_command,\n    append_turn_to_active_topic,\n    should_auto_attach_to_topic,\n    get_long_term_topic_context,\n)"
    )

# Add long term topic context to payload.
old = '''        "miguel_memory": get_memory_context(local_state.get("recognized_person")),
        "personality_mode": get_personality_mode(),
'''

new = '''        "miguel_memory": get_memory_context(local_state.get("recognized_person")),
        "long_term_topic_context": get_long_term_topic_context(),
        "personality_mode": get_personality_mode(),
'''

if old in text:
    text = text.replace(old, new)

# Add cloud brain rule.
old = '''- Use miguel_memory for saved user preferences, profile notes, and selected personality mode.
'''

new = '''- Use miguel_memory for saved user preferences, profile notes, and selected personality mode.
- Use long_term_topic_context to resume or continue topics the user wants to discuss over days.
- If there is an active long-term topic, connect short follow-ups to that topic unless the user clearly changes topic.
'''

if old in text:
    text = text.replace(old, new)

# Insert topic command handling after mode command handling.
old = '''    if mode_reply:
        speak(mode_reply)
        return True

    if local_state.get("face_detected") and local_state.get("recognized_person") is None:
'''

new = '''    if mode_reply:
        speak(mode_reply)
        return True

    topic_reply = handle_long_term_topic_command(
        user_text,
        local_state.get("recognized_person"),
        CONVERSATION_MEMORY,
    )
    if topic_reply:
        speak(topic_reply)
        update_conversation_memory(assistant_reply=topic_reply)
        return True

    if local_state.get("face_detected") and local_state.get("recognized_person") is None:
'''

if old in text:
    text = text.replace(old, new)
else:
    print("Warning: mode/topic insert location not found exactly.")

# After speaking local skill replies, attach to active topic.
old = '''        speak(local_skill_reply)
        update_conversation_memory(assistant_reply=local_skill_reply)
        return True
'''

new = '''        speak(local_skill_reply)
        update_conversation_memory(assistant_reply=local_skill_reply)
        if should_auto_attach_to_topic(user_text):
            append_turn_to_active_topic(user_text, local_skill_reply, local_state.get("recognized_person"))
        return True
'''

if old in text:
    text = text.replace(old, new)

# After speaking cloud reply, attach to active topic.
old = '''    speak(reply)
    update_conversation_memory(assistant_reply=reply)
    return True
'''

new = '''    speak(reply)
    update_conversation_memory(assistant_reply=reply)
    if should_auto_attach_to_topic(user_text):
        append_turn_to_active_topic(user_text, reply, local_state.get("recognized_person"))
    return True
'''

if old in text:
    text = text.replace(old, new)
else:
    print("Warning: cloud reply memory attach location not found exactly.")

path.write_text(text)
print(f"Patched V6.8 long-term topics: {path}")
