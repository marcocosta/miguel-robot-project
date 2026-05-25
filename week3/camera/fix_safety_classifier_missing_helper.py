from pathlib import Path

path = Path.home() / "robot-project/week3/camera/v7/safety_classifier.py"
text = path.read_text()

helper = r'''
    def _is_harmless_meaning_question(self, text: str) -> bool:
        t = str(text or "").lower().strip()

        starters = [
            "what does ",
            "what do ",
            "what is the meaning of",
            "what's the meaning of",
            "explain the phrase",
            "explain what",
            "what means",
        ]

        if not any(s in t for s in starters):
            return False

        risky_intent = [
            "instructions",
            "teach me how",
            "show me how",
            "where can i find",
            "how can i get",
            "help me get",
            "help me use",
        ]

        if any(r in t for r in risky_intent):
            return False

        return True

'''

if "def _is_harmless_meaning_question" in text:
    print("Helper already exists.")
else:
    marker = "    def _is_robot_control(self, text: str) -> bool:"
    idx = text.find(marker)

    if idx == -1:
        raise SystemExit("Could not find _is_robot_control insertion point.")

    text = text[:idx] + helper + "\n" + text[idx:]
    path.write_text(text)
    print("Inserted missing _is_harmless_meaning_question helper.")
