import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI


@dataclass
class SemanticSafetyResult:
    decision: str
    category: str
    confidence: float
    reason: str
    safe_reply: Optional[str] = None
    source: str = "semantic_classifier"


class SemanticSafetyClassifier:
    """
    Miguel SafetyGuard V2 semantic classifier.

    This is not a word-list blocker.
    It classifies intent and requested action.

    Decisions:
      ALLOW
      SAFE_EDUCATIONAL
      REFUSE
      CRISIS_REDIRECT
      ASK_CLARIFY
    """

    def __init__(self):
        self.client = OpenAI(timeout=float(os.getenv("MIGUEL_SAFETY_TIMEOUT_SECONDS", "2.5")))
        self.model = os.getenv("MIGUEL_SAFETY_CLASSIFIER_MODEL", "gpt-4o-mini")
        self.enabled = os.getenv("MIGUEL_SEMANTIC_SAFETY_ENABLED", "1") != "0"

    def classify(self, user_text: str, recent_context: str = "") -> SemanticSafetyResult:
        if not self.enabled:
            return SemanticSafetyResult(
                decision="ALLOW",
                category="disabled",
                confidence=1.0,
                reason="Semantic safety disabled.",
            )

        user_text = str(user_text or "").strip()
        recent_context = str(recent_context or "").strip()

        if not user_text:
            return SemanticSafetyResult(
                decision="ALLOW",
                category="empty",
                confidence=1.0,
                reason="Empty input.",
            )

        if self._is_obviously_safe_smalltalk(user_text):
            return SemanticSafetyResult(
                decision="ALLOW",
                category="normal_conversation",
                confidence=0.98,
                reason="User is making normal small talk or a greeting.",
            )

        # Fast allow for harmless language/meaning questions.
        # Example: "what does kill time mean?", "what does keep the time mean?"
        if self._is_harmless_meaning_question(user_text):
            return SemanticSafetyResult(
                decision="ALLOW",
                category="idiom_or_language",
                confidence=0.98,
                reason="User is asking for the meaning of a phrase or expression.",
            )

        # Fast allow for harmless language/meaning questions.
        # Example: "what does kill time mean?", "what does keep the time mean?"
        # Local quick allow for robot-control commands. This avoids silly false positives.
        if self._is_robot_control(user_text):
            return SemanticSafetyResult(
                decision="ALLOW",
                category="robot_control",
                confidence=0.99,
                reason="User is controlling Miguel's modes, voice, camera, or shutdown.",
            )

        # Local quick allow for clearly benign project/debugging talk.
        if self._is_obviously_benign_project_context(user_text):
            return SemanticSafetyResult(
                decision="ALLOW",
                category="robot_project_debugging",
                confidence=0.95,
                reason="User is discussing robot project debugging or normal development work.",
            )

        try:
            return self._classify_with_model(user_text, recent_context)
        except Exception as e:
            print("[V7 SAFETY V2] Semantic classifier failed:", e)

            # Fail-safe strategy:
            # Use a conservative but not paranoid fallback.
            # If the input contains a direct request for harmful/explicit/illegal help,
            # refuse. Otherwise allow normal conversation.
            fallback = self._fallback_intent_check(user_text)
            fallback.reason = f"Fallback after classifier failure: {fallback.reason}"
            return fallback

    def _classify_with_model(self, user_text: str, recent_context: str) -> SemanticSafetyResult:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["ALLOW", "SAFE_EDUCATIONAL", "REFUSE", "CRISIS_REDIRECT", "ASK_CLARIFY"],
                },
                "category": {
                    "type": "string",
                    "enum": [
                        "normal_conversation",
                        "robot_control",
                        "robot_project_debugging",
                        "idiom_or_language",
                        "safe_education",
                        "health_safety",
                        "self_harm",
                        "sexual_minors",
                        "explicit_sexual",
                        "illegal_drug_facilitation",
                        "drug_health_safety",
                        "real_world_violence",
                        "weapons_harm",
                        "harassment_or_abuse",
                        "ambiguous",
                        "other",
                    ],
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "reason": {
                    "type": "string",
                },
                "safe_reply": {
                    "type": ["string", "null"],
                },
            },
            "required": ["decision", "category", "confidence", "reason", "safe_reply"],
        }

        policy = """
You are Miguel's safety classifier for a small father-son robot.

Classify the USER'S INTENT, not isolated words.

Allowed:
- Normal conversation.
- Robot debugging and software/hardware bugs.
- Harmless idioms such as "kill time".
- Safe educational discussion about risks, meanings, health, law, ethics, or safety.
- Robot controls: voice mode, sleep mode, shutdown mode, camera tests.

Refuse:
- Requests for instructions, access, encouragement, or practical help for self-harm.
- Requests involving sexual content with minors.
- Requests for pornography or explicit sexual content.
- Requests to obtain, use, make, sell, hide, or facilitate illegal drugs or intoxication.
- Requests to harm people or use weapons for harm.

Crisis redirect:
- User expresses self-harm intent, asks for self-harm instructions, or appears in immediate danger.

Ask clarify:
- Ambiguous message where intent is unclear and a safe clarification would resolve it.

Output only JSON matching the schema.
Keep safe_reply brief, spoken, and suitable for a family robot.
"""

        input_text = (
            f"Recent context, if useful:\n{recent_context}\n\n"
            f"Current user text:\n{user_text}"
        )

        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": policy},
                {"role": "user", "content": input_text},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "miguel_safety_decision",
                    "strict": True,
                    "schema": schema,
                }
            },
        )

        raw = getattr(response, "output_text", "") or ""
        data = json.loads(raw)

        return SemanticSafetyResult(
            decision=data["decision"],
            category=data["category"],
            confidence=float(data["confidence"]),
            reason=data["reason"],
            safe_reply=data.get("safe_reply"),
        )



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


    def _is_robot_control(self, text: str) -> bool:
        t = text.lower()

        phrases = [
            "use robot voice",
            "use robotic voice",
            "use natural voice",
            "use friendly voice",
            "use deep voice",
            "use story voice",
            "voice mode",
            "which voice",
            "what voice",
            "sleep mode",
            "go to sleep",
            "wake up",
            "mission control",
            "shutdown",
            "shut down",
            "confirm shutdown",
            "camera",
            "what do you see",
            "who do you see",
            "describe what you see",
        ]

        return any(p in t for p in phrases)

    def _is_obviously_benign_project_context(self, text: str) -> bool:
        t = text.lower()

        benign_patterns = [
            r"\bbug\b",
            r"\bbugs\b",
            r"\bdebug\b",
            r"\bdebugging\b",
            r"\bfix\b.*\bbug\b",
            r"\bsafety guard\b",
            r"\brobot project\b",
            r"\bmiguel\b.*\bproject\b",
            r"\bjetson\b",
            r"\boak-d\b",
            r"\brealsense\b",
            r"\brespeaker\b",
            r"\bvoice mode\b",
            r"\bcamera queue\b",
            r"\bstale camera\b",
            r"\bcertificate\b",
        ]

        if any(re.search(p, t) for p in benign_patterns):
            return not self._contains_direct_harmful_request(t)

        return False

    def _contains_direct_harmful_request(self, text: str) -> bool:
        t = text.lower()

        intent_markers = [
            "teach me how to",
            "show me how to",
            "give me instructions",
            "where can i find",
            "how can i get",
            "how do i get",
            "help me get",
            "help me use",
        ]

        risky_domains = [
            "self-harm",
            "hurt myself",
            "harm myself",
            "porn",
            "sexual content involving minors",
            "illegal drugs",
            "get high",
            "weapon",
            "attack",
        ]

        return any(i in t for i in intent_markers) and any(d in t for d in risky_domains)

    def _fallback_intent_check(self, user_text: str) -> SemanticSafetyResult:
        t = user_text.lower()

        # Keep this fallback tiny. It is not the main safety brain.
        risky_request = self._contains_direct_harmful_request(t)

        if risky_request:
            return SemanticSafetyResult(
                decision="REFUSE",
                category="fallback_risky_request",
                confidence=0.75,
                reason="Fallback detected a direct risky request.",
                safe_reply="I can’t help with that, but I can help with something safe.",
                source="fallback",
            )

        return SemanticSafetyResult(
            decision="ALLOW",
            category="fallback_allow",
            confidence=0.55,
            reason="No direct risky request detected in fallback.",
            source="fallback",
        )
