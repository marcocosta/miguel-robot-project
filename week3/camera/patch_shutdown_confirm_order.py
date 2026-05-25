from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_memory.py"
text = path.read_text()

marker = "    # Shutdown must be explicit"
idx = text.find(marker)

if idx == -1:
    raise SystemExit("Could not find shutdown section marker.")

next_marker = "    # Personality modes."
end = text.find(next_marker, idx)

if end == -1:
    raise SystemExit("Could not find personality modes marker after shutdown section.")

new_shutdown_block = r'''    # Shutdown confirmation must be checked BEFORE new shutdown requests.
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
        memory["robot_mode"] = "shutdown"
        save_memory(memory)
        return "__SHUTDOWN__"

    if memory.get("pending_shutdown") and any(p in text for p in [
        "cancel",
        "cancel shutdown",
        "cancel shut down",
        "no",
        "not now",
    ]):
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
        return "Shutdown confirmation required. Say confirm shutdown if you want me to power off the Jetson."

'''

text = text[:idx] + new_shutdown_block + text[end:]
path.write_text(text.replace("\t", "    "))
print("Patched shutdown confirmation order.")
