import os
import re
from dataclasses import dataclass
from typing import Dict, Optional

from openai import OpenAI


@dataclass
class SafetyDecision:
    allowed: bool
    category: str = "safe"
    reason: str = ""
    safe_reply: Optional[str] = None
    source: str = "local"


class SafetyGuard:
    """
    Context-aware safety layer for Miguel.

    Design:
    - Do not rely on simple word blocking.
    - Use local intent heuristics for obvious safe/unsafe cases.
    - Use OpenAI moderation for contextual classification.
    - Fail safely if classification errors occur.
    """

    def __init__(self):
        self.client = OpenAI()
        self.moderation_model = os.getenv("MIGUEL_MODERATION_MODEL", "omni-moderation-latest")
        self.enabled = os.getenv("MIGUEL_SAFETY_ENABLED", "1") != "0"

    def evaluate_user_text(self, text: str) -> SafetyDecision:
        if not self.enabled:
            return SafetyDecision(allowed=True, source="disabled")

        normalized = self._normalize(text)

        if not normalized:
            return SafetyDecision(allowed=True, source="empty")

        local = self._local_context_check(normalized)
        if local is not None:
            return local

        return self._moderation_check(text)

    def evaluate_assistant_reply(self, reply: str) -> SafetyDecision:
        """
        Output guard. If Miguel's generated answer drifts unsafe, replace it.
        """
        if not self.enabled:
            return SafetyDecision(allowed=True, source="disabled")

        normalized = self._normalize(reply)
        if not normalized:
            return SafetyDecision(allowed=True, source="empty")

        local = self._local_output_check(normalized)
        if local is not None:
            return local

        return self._moderation_check(reply)

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").lower()).strip()

    def _local_context_check(self, text: str) -> Optional[SafetyDecision]:
        """
        Conservative intent checks. These are phrase + context checks, not simple word bans.
        """

        # Safe idioms / benign contexts.
        benign_phrases = [
            "kill time",
            "killer app",
            "kill the process",
            "kill a program",
            "kill the terminal",
            "drug discovery",
            "drug store",
            "prescription from a doctor",
            "safety rules",
            "why is it dangerous",
            "what does it mean",
            "explain why it is unsafe",
        ]

        if any(p in text for p in benign_phrases):
            return SafetyDecision(allowed=True, category="safe_benign_context", source="local")

        # Direct unsafe request patterns.
        unsafe_patterns: Dict[str, list[str]] = {
            "self_harm": [
                r"\bhow (do|can) i (hurt|harm) myself\b",
                r"\bhelp me (hurt|harm) myself\b",
                r"\bi want to (hurt|harm) myself\b",
                r"\binstructions? for self[- ]harm\b",
            ],
            "sexual_minors": [
                r"\bsexual\b.*\b(minor|child|kid|teen)\b",
                r"\bminor\b.*\bsexual\b",
                r"\bchild\b.*\bporn\b",
            ],
            "explicit_sexual": [
                r"\bshow me porn\b",
                r"\bfind porn\b",
                r"\bexplicit sexual\b.*\bcontent\b",
            ],
            "illegal_drug_facilitation": [
                r"\bhow (do|can) i (buy|get|make|sell) illegal drugs\b",
                r"\bwhere (do|can) i buy illegal drugs\b",
                r"\bmake meth\b",
                r"\bcook meth\b",
            ],
            "harmful_violence": [
                r"\bhow (do|can) i kill\b",
                r"\bhow (do|can) i hurt someone\b",
                r"\bhow (do|can) i make a weapon\b",
                r"\bhelp me attack\b",
            ],
        }

        for category, patterns in unsafe_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return SafetyDecision(
                        allowed=False,
                        category=category,
                        reason=f"Matched unsafe intent pattern: {category}",
                        safe_reply=self._safe_redirect(category),
                        source="local",
                    )

        return None

    def _local_output_check(self, text: str) -> Optional[SafetyDecision]:
        unsafe_output_patterns = [
            r"\bstep[- ]by[- ]step\b.*\b(harm|attack|illegal drugs|porn)\b",
            r"\binstructions\b.*\b(self[- ]harm|illegal drugs|weapon)\b",
        ]

        for pattern in unsafe_output_patterns:
            if re.search(pattern, text):
                return SafetyDecision(
                    allowed=False,
                    category="unsafe_output",
                    reason="Assistant reply appeared to include unsafe procedural content.",
                    safe_reply="I can’t help with that, but I can help with a safer question.",
                    source="local_output",
                )

        return None

    def _moderation_check(self, text: str) -> SafetyDecision:
        try:
            response = self.client.moderations.create(
                model=self.moderation_model,
                input=text,
            )

            result = response.results[0]
            categories = getattr(result, "categories", None)
            flagged = bool(getattr(result, "flagged", False))

            category_dict = {}
            if categories is not None:
                if hasattr(categories, "model_dump"):
                    category_dict = categories.model_dump()
                elif isinstance(categories, dict):
                    category_dict = categories
                else:
                    category_dict = dict(categories)

            blocking_categories = [
                "sexual/minors",
                "self-harm/intent",
                "self-harm/instructions",
                "illicit/violent",
                "violence/graphic",
            ]

            for cat in blocking_categories:
                if bool(category_dict.get(cat, False)):
                    return SafetyDecision(
                        allowed=False,
                        category=cat,
                        reason=f"Moderation category blocked: {cat}",
                        safe_reply=self._safe_redirect(cat),
                        source="moderation",
                    )

            # General flagged content: block if clearly sensitive. For broad categories,
            # use a safer redirect but avoid overblocking benign conversation.
            if flagged:
                return SafetyDecision(
                    allowed=False,
                    category="moderation_flagged",
                    reason="Moderation flagged content.",
                    safe_reply="I can’t help with that topic, but I can help with something safe or educational.",
                    source="moderation",
                )

            return SafetyDecision(allowed=True, category="safe", source="moderation")

        except Exception as e:
            # Fail safe, but do not kill the robot.
            return SafetyDecision(
                allowed=False,
                category="safety_check_failed",
                reason=str(e),
                safe_reply="My safety check is not available right now, so I should not answer that. Please ask something else.",
                source="error",
            )

    def _safe_redirect(self, category: str) -> str:
        if "self-harm" in category or category == "self_harm":
            return (
                "I can’t help with self-harm. If someone might be in danger, "
                "please talk to a trusted adult or emergency help right now."
            )

        if category in {"sexual_minors", "sexual/minors"}:
            return "I can’t help with sexual content involving minors."

        if category == "explicit_sexual":
            return "I can’t help with explicit sexual content."

        if "illicit" in category or "drug" in category:
            return "I can’t help with illegal drug instructions or access. I can discuss health and safety in a safe way."

        if "violence" in category or "harmful_violence" in category:
            return "I can’t help with harming people. I can help with safety, conflict resolution, or emergency planning."

        return "I can’t help with that topic, but I can help with something safe."
