from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"
text = path.read_text()

# Add imports.
if "from robot_memory import" not in text:
    marker = "from robot_skills import maybe_handle_local_skill"
    replacement = """from robot_skills import maybe_handle_local_skill
from robot_memory import (
    get_robot_mode,
    get_personality_mode,
    get_memory_context,
    handle_robot_mode_command,
)
"""
    if marker in text:
        text = text.replace(marker, replacement)
    else:
        # fallback insert after imports
        text = text.replace(
            "from openai import OpenAI\n",
            "from openai import OpenAI\nfrom robot_memory import get_robot_mode, get_personality_mode, get_memory_context, handle_robot_mode_command\n",
        )

# Add memory context to cloud payload.
old_payload = '''    payload = {
        "user_text": user_text,
        "local_state": local_state,
        "conversation_memory": CONVERSATION_MEMORY,
        "robot_capabilities": [
'''

new_payload = '''    payload = {
        "user_text": user_text,
        "local_state": local_state,
        "conversation_memory": CONVERSATION_MEMORY,
        "miguel_memory": get_memory_context(local_state.get("recognized_person")),
        "personality_mode": get_personality_mode(),
        "robot_capabilities": [
'''

if old_payload in text:
    text = text.replace(old_payload, new_payload)
else:
    print("Warning: payload block not found or already patched.")

# Add personality rules into cloud brain instructions.
old_rules = '''- Use conversation_memory to understand short follow-ups like ok, yes, no, repeat that, or I did not understand.
- Stay on the current topic unless the user clearly changes topics.
'''

new_rules = '''- Use conversation_memory to understand short follow-ups like ok, yes, no, repeat that, or I did not understand.
- Use miguel_memory for saved user preferences, profile notes, and selected personality mode.
- If personality_mode is creative, be more imaginative and propose ideas.
- If personality_mode is teacher, explain slowly with simple examples.
- If personality_mode is engineer, be more technical and precise.
- If personality_mode is quiet, keep the answer very short.
- If personality_mode is mission_control, use a friendly father-son robot project tone.
- Stay on the current topic unless the user clearly changes topics.
'''

if old_rules in text:
    text = text.replace(old_rules, new_rules)
else:
    print("Warning: rules block not found or already patched.")

# Add local mode command handling inside handle_user_turn_with_cached_state after local_state is assigned.
old_block = '''    local_state = cached_local_state
    print("[BRAIN] Using cached local state:", local_state)

    if local_state.get("face_detected") and local_state.get("recognized_person") is None:
'''

new_block = '''    local_state = cached_local_state
    print("[BRAIN] Using cached local state:", local_state)

    mode_reply = handle_robot_mode_command(user_text, local_state.get("recognized_person"))
    if mode_reply == "__SILENT__":
        print("[MODE] Sleep mode: silently ignored user text.")
        return True

    if mode_reply == "__SHUTDOWN__":
        speak("Confirmed. Miguel is shutting down the Jetson now. Mission saved.")
        subprocess.Popen(["sudo", "shutdown", "now"])
        return False

    if mode_reply:
        speak(mode_reply)
        return True

    if local_state.get("face_detected") and local_state.get("recognized_person") is None:
'''

if old_block in text:
    text = text.replace(old_block, new_block)
else:
    print("Warning: handle_user_turn block not found exactly.")

# Add sleep-mode branch at top of main loop after local_state is obtained.
old_loop = '''                local_state = get_cached_local_state()
                recognized_person = local_state.get("recognized_person")
                active_followup = (time.time() - last_reply_time) < ACTIVE_FOLLOWUP_SECONDS
                vision_fresh = is_vision_state_fresh(local_state)
'''

new_loop = '''                local_state = get_cached_local_state()
                recognized_person = local_state.get("recognized_person")
                active_followup = (time.time() - last_reply_time) < ACTIVE_FOLLOWUP_SECONDS
                vision_fresh = is_vision_state_fresh(local_state)

                if get_robot_mode() == "sleep":
                    print("[SLEEP] Miguel is quiet. Wake phrase required.")
                    wake_detected = listen_for_wake(timeout_seconds=3)

                    if wake_detected:
                        # Route the wake phrase through mode command handling.
                        mode_reply = handle_robot_mode_command("Miguel wake up", recognized_person)
                        if mode_reply and mode_reply != "__SILENT__":
                            speak(mode_reply)
                        last_reply_time = time.time()

                    continue
'''

if old_loop in text:
    text = text.replace(old_loop, new_loop)
else:
    print("Warning: main loop state block not found exactly.")

path.write_text(text)
print(f"Patched V6.7 modes + memory: {path}")
