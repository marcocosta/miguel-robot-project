from pathlib import Path
import re

path = Path.home() / "robot-project/week3/camera/robot_memory.py"
text = path.read_text()

# Find shutdown section and make it strict but usable.
old = '''    # Shutdown must be explicit. Do not trigger on questions ABOUT shutdown.
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
'''

new = '''    # Shutdown must be explicit, but normal phrases like "shut down" should work.
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

if old in text:
    text = text.replace(old, new)
else:
    print("Shutdown block not found exactly; appending fallback guard near personality modes.")
    marker = "    # Personality modes."
    idx = text.find(marker)
    if idx == -1:
        raise SystemExit("Could not find insertion marker.")
    text = text[:idx] + new + "\n" + text[idx:]

path.write_text(text.replace("\\t", "    "))
print("Patched shutdown mode.")
