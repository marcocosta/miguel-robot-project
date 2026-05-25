import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from .safety_classifier import SemanticSafetyClassifier, SemanticSafetyResult


@dataclass
class SafetyDecision:
    allowed: bool
    category: str = "safe"
    reason: str = ""
    safe_reply: Optional[str] = None
    source: str = "safety_v2"


class SafetyGuard:
    """
    SafetyGuard V2 for Miguel.

    Architecture:
      1. Semantic classifier decides intent.
      2. Moderation endpoint is used as secondary hard-category signal.
      3. Deterministic policy router decides final action.
      4. Output guard is light and only blocks procedural unsafe replies.

    This avoids endless keyword lists.
    """

    def __init__(self):
        self.client = OpenAI()
        self.classifier = SemanticSafetyClassifier()
        self.moderation_model = os.getenv("MIGUEL_MODERATION_MODEL", "omni-moderation-latest")
        self.enabled = os.getenv("MIGUEL_SAFETY_ENABLED", "1") != "0"
        self.use_moderation_signal = os.getenv("MIGUEL_USE_MODERATION_SIGNAL", "1") != "0"
        self.strict_output_guard = os.getenv("MIGUEL_STRICT_OUTPUT_GUARD", "0") == "1"

    def evaluate_user_text(self, text: str) -> SafetyDecision:
        if not self.enabled:
            return SafetyDecision(allowed=True, source="disabled")

        semantic = self.classifier.classify(text)

        # Moderation is secondary. It can hard-block the most severe categories,
        # but generic flagged=True must not auto-block normal conversation.
        moderation_block = None
        if self.use_moderation_signal:
            moderation_block = self._moderation_hard_block(text)

        if moderation_block is not None:
            # If semantic says robot_control or normal project debug, trust semantic
            # unless moderation category is a hard child-safety or self-harm category.
            if semantic.category in {"robot_control", "robot_project_debugging", "idiom_or_language", "normal_conversation"}:
                if moderation_block.category not in {"sexual/minors", "self-harm/intent", "self-harm/instructions"}:
                    print(
                        "[V7 SAFETY V2] Moderation hard-ish signal ignored due semantic allow:",
                        moderation_block.category,
                        semantic.category,
                    )
                else:
                    return moderation_block
            else:
                return moderation_block

        return self._route_semantic_decision(semantic)

    def evaluate_assistant_reply(self, reply: str) -> SafetyDecision:
        if not self.enabled:
            return SafetyDecision(allowed=True, source="disabled")

        text = str(reply or "").lower()

        # Light output guard: only catch procedural unsafe content.
        unsafe_markers = [
            "step-by-step instructions for self-harm",
            "instructions to get illegal drugs",
            "where to find sexual content involving minors",
            "how to harm someone",
        ]

        if any(m in text for m in unsafe_markers):
            return SafetyDecision(
                allowed=False,
                category="unsafe_output",
                reason="Assistant reply appeared to include unsafe procedural content.",
                safe_reply="I can’t help with that, but I can help with something safe.",
                source="output_guard",
            )

        if not self.strict_output_guard:
            return SafetyDecision(allowed=True, category="safe_output", source="output_guard")

        semantic = self.classifier.classify(reply)
        return self._route_semantic_decision(semantic)

    def _route_semantic_decision(self, semantic: SemanticSafetyResult) -> SafetyDecision:
        decision = semantic.decision.upper().strip()

        if decision in {"ALLOW", "SAFE_EDUCATIONAL"}:
            return SafetyDecision(
                allowed=True,
                category=semantic.category,
                reason=semantic.reason,
                source=semantic.source,
            )

        if decision == "ASK_CLARIFY":
            return SafetyDecision(
                allowed=False,
                category=semantic.category,
                reason=semantic.reason,
                safe_reply=semantic.safe_reply or "Can you clarify what you mean in a safe way?",
                source=semantic.source,
            )

        if decision == "CRISIS_REDIRECT":
            return SafetyDecision(
                allowed=False,
                category=semantic.category,
                reason=semantic.reason,
                safe_reply=semantic.safe_reply or (
                    "I can’t help with that. If someone might be in danger, "
                    "please talk to a trusted adult or emergency help right now."
                ),
                source=semantic.source,
            )

        if decision == "REFUSE":
            return SafetyDecision(
                allowed=False,
                category=semantic.category,
                reason=semantic.reason,
                safe_reply=semantic.safe_reply or "I can’t help with that, but I can help with something safe.",
                source=semantic.source,
            )

        # Unknown classifier result: safe clarification, not hard block.
        return SafetyDecision(
            allowed=False,
            category="unknown_safety_decision",
            reason=f"Unknown semantic decision: {semantic.decision}",
            safe_reply="I’m not sure how to handle that safely. Can you ask another way?",
            source=semantic.source,
        )

    def _moderation_hard_block(self, text: str) -> Optional[SafetyDecision]:
        try:
            response = self.client.moderations.create(
                model=self.moderation_model,
                input=text,
            )

            result = response.results[0]
            categories = getattr(result, "categories", None)

            category_dict = {}
            if categories is not None:
                if hasattr(categories, "model_dump"):
                    category_dict = categories.model_dump()
                elif isinstance(categories, dict):
                    category_dict = categories
                else:
                    try:
                        category_dict = dict(categories)
                    except Exception:
                        category_dict = {}

            hard_block_categories = [
                "sexual/minors",
                "self-harm/intent",
                "self-harm/instructions",
                "illicit/violent",
                "violence/graphic",
            ]

            for cat in hard_block_categories:
                if bool(category_dict.get(cat, False)):
                    return SafetyDecision(
                        allowed=False,
                        category=cat,
                        reason=f"Moderation hard category: {cat}",
                        safe_reply=self._safe_redirect(cat),
                        source="moderation_hard_block",
                    )

            return None

        except Exception as e:
            print("[V7 SAFETY V2] Moderation unavailable, relying on semantic classifier:", e)
            return None

    def _safe_redirect(self, category: str) -> str:
        if "self-harm" in category:
            return (
                "I can’t help with self-harm. If someone might be in danger, "
                "please talk to a trusted adult or emergency help right now."
            )

        if category == "sexual/minors":
            return "I can’t help with sexual content involving minors."

        if "illicit" in category or "drug" in category:
            return "I can’t help with illegal drug instructions or access. I can discuss health and safety in a safe way."

        if "violence" in category:
            return "I can’t help with harming people. I can help with safety or conflict resolution."

        return "I can’t help with that topic, but I can help with something safe."
