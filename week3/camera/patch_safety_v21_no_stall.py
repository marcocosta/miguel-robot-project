from pathlib import Path

path = Path.home() / "robot-project/week3/camera/v7/safety_classifier.py"
text = path.read_text()

# Add timeout-enabled OpenAI client.
text = text.replace(
    "self.client = OpenAI()",
    'self.client = OpenAI(timeout=float(os.getenv("MIGUEL_SAFETY_TIMEOUT_SECONDS", "2.5")))'
)

# Add fast allow for harmless phrase/meaning questions before semantic classifier calls.
old = '''        # Local quick allow for robot-control commands. This avoids silly false positives.
        if self._is_robot_control(user_text):
'''

new = '''        # Fast allow for harmless language/meaning questions.
        # Example: "what does kill time mean?", "what does keep the time mean?"
        if self._is_harmless_meaning_question(user_text):
            return SemanticSafetyResult(
                decision="ALLOW",
                category="idiom_or_language",
                confidence=0.98,
                reason="User is asking for the meaning of a phrase or expression.",
            )

        # Local quick allow for robot-control commands. This avoids silly false positives.
        if self._is_robot_control(user_text):
'''

if old not in text:
    raise SystemExit("Could not find insertion point in classify().")

text = text.replace(old, new)

# Add helper before _is_robot_control.
marker = "    def _is_robot_control(self, text: str) -> bool:"
idx = text.find(marker)

if idx == -1:
    raise SystemExit("Could not find _is_robot_control().")

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

        # If the user asks for instructions/access/help doing something risky,
        # do not fast-allow it as a harmless definition.
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

if "_is_harmless_meaning_question" not in text:
    text = text[:idx] + helper + text[idx:]

path.write_text(text)
print("Patched SafetyClassifier V2.1 with fast allow + timeout.")
