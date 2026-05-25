from pathlib import Path
import re

path = Path.home() / "robot-project/week3/camera/robot_skills.py"
text = path.read_text()

old = '''    if "time" in words:
        return get_time_text()
'''

new = '''    # Time skill: only answer clock-time questions.
    # Do NOT trigger for "time travel", "time machine", "spacetime", etc.
    blocked_time_topics = [
        "time travel",
        "time traveling",
        "time machine",
        "space time",
        "spacetime",
        "time dilation",
    ]

    time_phrases = [
        "what time is it",
        "what is the time",
        "current time",
        "time now",
        "what time now",
        "tell me the time",
    ]

    if any(blocked in text for blocked in blocked_time_topics):
        return None

    if any(phrase in text for phrase in time_phrases):
        return get_time_text()
'''

if old in text:
    text = text.replace(old, new)
else:
    print("Warning: simple time rule not found. It may already be patched.")

# Avoid "what is time travel" being treated as calculator.
text = text.replace(
    'if "calculate" in words or "what is" in text or "how much" in text:',
    'if "calculate" in words or "how much" in text:'
)

path.write_text(text)
print(f"Patched robot_skills time intent: {path}")
