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
    - Do not block only because a single sensitive word appears.
    - Allow normal robot/project/debugging/profanity/frustration.
    - Block actual requests for harmful, explicit, exploitative, or illegal facilitation.
    - Use moderation as a helper, not as an automatic generic kill switch.
    """

    def __init__(self):
        self.client = OpenAI()
        self.moderation_model = os.getenv("MIGUEL_MODERATION_MODEL", "omni-moderation-latest")
        self.enabled = os.getenv("MIGUEL_SAFETY_ENABLED", "1") != "0"
        self.strict_output_guard = os.getenv("MIGUEL_STRICT_OUTPUT_GUARD", "0") == "1"

    def evaluate_user_text(self, text: str) -> SafetyDecision:
        if not self.enabled:
            return SafetyDecision(allowed=True, source="disabled")

        normalized = self._normalize(text)
        if not normalized:
            return SafetyDecision(allowed=True, source="empty")

        safe_local = self._safe_context_allow(normalized)
        if safe_local:
            return SafetyDecision(allowed=True, category=safe_local, source="local_allow")

        unsafe_local = self._local_unsafe_intent_check(normalized)
        if unsafe_local is not None:
            return unsafe_local

        return self._moderation_check(text, is_output=False)

    def evaluate_assistant_reply(self, reply: str) -> SafetyDecision:
        """
        Output guard should be lighter than input guard.

        Most generated replies from Miguel are normal conversation. Do not call
        moderation on every harmless sentence unless strict output guard is enabled.
        """
        if not self.enabled:
            return SafetyDecision(allowed=True, source="disabled")

        normalized = self._normalize(reply)
        if not normalized:
            return SafetyDecision(allowed=True, source="empty")

        unsafe_output = self._local_output_check(normalized)
        if unsafe_output is not None:
            return unsafe_output

        # Default: do not run cloud moderation on every assistant reply.
        # This prevents false positives like normal project/bug conversation.
        if not self.strict_output_guard:
            return SafetyDecision(allowed=True, category="safe_output", source="output_local")

        return self._moderation_check(reply, is_output=True)

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").lower()).strip()

    def _safe_context_allow(self, text: str) -> Optional[str]:
        """
        Explicitly allowed benign contexts.
        These are not unsafe requests even if they contain sensitive-looking words.
        """
        benign_patterns = [
            # Idioms / technical uses.
            r"\bkill time\b",
            r"\bkiller app\b",
            r"\bkill the process\b",
            r"\bkill a process\b",
            r"\bkill the terminal\b",
            r"\bkill the program\b",

            # Robot/project/debugging context.
            r"\bbug\b",
            r"\bbugs\b",
            r"\bsoftware bug\b",
            r"\bdebug\b",
            r"\bdebugging\b",
            r"\bfix the bug\b",
            r"\brobot project\b",
            r"\bmiguel\b.*\bbug\b",
            r"\bwe fixed it\b",
            r"\bcertificate\b",
            r"\bjetson\b",
            r"\boak-d\b",
            r"\bcamera\b.*\bbug\b",

            # Safe educational / meaning questions.
            r"\bwhat does .* mean\b",
            r"\bwhy is .* dangerous\b",
            r"\bexplain why .* unsafe\b",
            r"\bhealth and safety\b",

            # Profanity/frustration without harmful request.
            r"\bputa\b",
            r"\bcaralho\b",
            r"\bmerda\b",
            r"\bdamn\b",
            r"\bshit\b",
        ]

        for pattern in benign_patterns:
            if re.search(pattern, text):
                # Do not allow profanity if it also contains a direct unsafe request.
                if self._contains_request_for_harm_or_illicit(text):
                    return None
                return "benign_context"

        return None

    def _contains_request_for_harm_or_illicit(self, text: str) -> bool:
        risky_verbs = [
            "teach me how to",
            "show me how to",
            "give me instructions",
            "instructions for",
            "where can i find",
            "where do i buy",
            "how do i buy",
            "how can i buy",
            "how to make",
            "how do i make",
            "help me make",
        ]

        risky_topics = [
            "self harm",
            "hurt myself",
            "harm myself",
            "illegal drug",
            "illegal drugs",
            "weapon",
            "attack",
            "porn",
            "pornography",
            "sexual content involving minors",
        ]

        return any(v in text for v in risky_verbs) and any(t in text for t in risky_topics)

    def _local_unsafe_intent_check(self, text: str) -> Optional[SafetyDecision]:
        """
        Block intent, not vocabulary.
        """

        unsafe_patterns: Dict[str, list[str]] = {
            "self_harm": [
                r"\bhow (do|can) i (hurt|harm) myself\b",
                r"\bhelp me (hurt|harm) myself\b",
                r"\bi want to (hurt|harm) myself\b",
                r"\bteach me how to .* (hurt|harm|kill) me\b",
                r"\buse .* to .* kill me\b",
            ],
            "sexual_minors": [
                r"\bwhere can i find\b.*\b(pedophilia|child porn|minor sexual)\b",
                r"\bwebsites?\b.*\b(pedophilia|child porn|minor sexual)\b",
                r"\bsexual\b.*\b(minor|child|kid)\b",
                r"\bminor\b.*\bsexual\b",
                r"\bchild\b.*\bporn\b",
            ],
            "explicit_sexual_access": [
                r"\bshow me\b.*\bporn\b",
                r"\bfind\b.*\bporn\b",
                r"\bwhere can i find\b.*\bporn\b",
                r"\bwebsite\b.*\bporn\b",
            ],
            "illegal_drug_facilitation": [
                r"\bgive me\b.*\billegal drug instructions\b",
                r"\bhow (do|can) i (buy|get|make|sell) illegal drugs\b",
                r"\bwhere (do|can) i buy illegal drugs\b",
                r"\bmake meth\b",
                r"\bcook meth\b",
            ],
            "harmful_violence": [
                r"\bhow (do|can) i kill\b",
                r"\bhow (do|can) i hurt someone\b",
                r"\bhelp me attack\b",
                r"\bteach me how to use .* to kill\b",
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
        """
        Only block assistant output if it appears to contain unsafe procedural help.
        Do not block normal friendly conversation.
        """
        unsafe_output_patterns = [
            r"\bstep[- ]by[- ]step\b.*\b(self[- ]harm|illegal drugs|weapon|attack)\b",
            r"\binstructions\b.*\b(self[- ]harm|illegal drugs|weapon|attack)\b",
            r"\bwhere to buy\b.*\billegal drugs\b",
            r"\bhow to make\b.*\billegal drugs\b",
            r"\bhow to access\b.*\bchild porn\b",
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

    def _moderation_check(self, text: str, is_output: bool = False) -> SafetyDecision:
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
                        reason=f"Moderation category blocked: {cat}",
                        safe_reply=self._safe_redirect(cat),
                        source="moderation",
                    )

            # Sexual content: block access to explicit content, but do not block
            # safe educational or parent-context discussion automatically.
            if bool(category_dict.get("sexual", False)):
                normalized = self._normalize(text)
                if any(p in normalized for p in ["show me porn", "find porn", "porn website", "explicit sexual"]):
                    return SafetyDecision(
                        allowed=False,
                        category="explicit_sexual_access",
                        reason="Explicit sexual access request.",
                        safe_reply=self._safe_redirect("explicit_sexual_access"),
                        source="moderation",
                    )

            # Important change:
            # Generic flagged=True is NOT enough to block.
            # It is logged as suspicious but allowed unless a hard category or local
            # unsafe intent is present.
            if flagged:
                print("[V7 SAFETY] Moderation flagged but no hard-block category; allowing with caution.")

            return SafetyDecision(allowed=True, category="safe", source="moderation")

        except Exception as e:
            # For normal conversation, fail open with caution.
            # Local checks already caught obvious dangerous intent.
            print("[V7 SAFETY] Moderation check failed; allowing after local checks:", e)
            return SafetyDecision(
                allowed=True,
                category="moderation_unavailable_after_local_checks",
                reason=str(e),
                source="error_fail_open",
            )

    def _safe_redirect(self, category: str) -> str:
        if "self-harm" in category or category == "self_harm":
            return (
                "I can’t help with self-harm. If someone might be in danger, "
                "please talk to a trusted adult or emergency help right now."
            )

        if category in {"sexual_minors", "sexual/minors"}:
            return "I can’t help with sexual content involving minors."

        if category in {"explicit_sexual_access", "explicit_sexual"}:
            return "I can’t help with pornography or explicit sexual content."

        if "illicit" in category or "drug" in category:
            return "I can’t help with illegal drug instructions or access. I can discuss health and safety in a safe way."

        if "violence" in category or "harmful_violence" in category:
            return "I can’t help with harming people. I can help with safety or conflict resolution."

        return "I can’t help with that topic, but I can help with something safe."
