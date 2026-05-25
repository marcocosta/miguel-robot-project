from pathlib import Path

path = Path.home() / "robot-project/week3/camera/v7/safety_classifier.py"
text = path.read_text()

helper = r'''
    def _is_obviously_safe_smalltalk(self, text: str) -> bool:
        t = str(text or "").lower().strip()

        safe_exact = [
            "hi",
            "hello",
            "hey",
            "hey miguel",
            "hi miguel",
            "hello miguel",
            "how are you",
            "how are you doing",
            "good morning",
            "good afternoon",
            "good evening",
            "thank you",
            "thanks",
            "ok good",
            "okay good",
            "that's good",
        ]

        if t in safe_exact:
            return True

        safe_starts = [
            "hi miguel",
            "hello miguel",
            "hey miguel",
            "how are you",
            "how are you doing",
        ]

        if any(t.startswith(p + ",") or t.startswith(p + " ") for p in safe_starts):
            return True

        risky = [
            "instructions",
            "teach me",
            "show me",
            "where can i find",
            "how can i get",
            "help me get",
            "hurt",
            "harm",
            "porn",
            "drug",
            "weapon",
            "kill",
        ]

        if any(r in t for r in risky):
            return False

        return False

'''

if "def _is_obviously_safe_smalltalk" in text:
    print("Helper already exists.")
else:
    marker = "    def _is_harmless_meaning_question(self, text: str) -> bool:"
    idx = text.find(marker)

    if idx == -1:
        marker = "    def _is_robot_control(self, text: str) -> bool:"
        idx = text.find(marker)

    if idx == -1:
        raise SystemExit("Could not find insertion point for smalltalk helper.")

    text = text[:idx] + helper + "\n" + text[idx:]
    path.write_text(text)
    print("Inserted missing _is_obviously_safe_smalltalk helper.")
