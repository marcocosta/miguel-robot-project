from pathlib import Path

path = Path.home() / "robot-project/week3/camera/robot_memory.py"
text = path.read_text()

# Replace the cancel block so "now" does not match "no".
old = '''    if memory.get("pending_shutdown") and any(p in text for p in [
        "cancel",
        "cancel shutdown",
        "cancel shut down",
        "no",
        "not now",
    ]):
        memory["pending_shutdown"] = False
        save_memory(memory)
        return "Shutdown cancelled."
'''

new = '''    if memory.get("pending_shutdown"):
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
'''

if old not in text:
    print("Cancel block not found exactly. Trying safer fallback replacement.")
else:
    text = text.replace(old, new)

path.write_text(text.replace("\\t", "    "))
print("Patched shutdown cancel word matching.")
