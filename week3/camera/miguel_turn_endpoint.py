"""Deterministic Stage-1 endpointing, isolated for later ML replacement."""

from dataclasses import dataclass, field
import re
import time
import unicodedata
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class ConversationConfig:
    candidate_silence_ms: int = 300
    fast_complete_ms: int = 500
    normal_complete_ms: int = 750
    incomplete_wait_ms: int = 1400
    repair_consider_ms: int = 2200
    repair_commit_ms: int = 3000
    engagement_accept_threshold: float = 0.70
    engagement_decay_per_second: float = 0.008
    expected_reply_seconds: float = 20.0
    enable_xvf_monitor: bool = True
    enable_adaptive_endpoint: bool = True
    sensor_log_interval_seconds: float = 0.5

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "ConversationConfig":
        import os
        source = os.environ if env is None else env

        def integer(name: str, default: int) -> int:
            try:
                return int(source.get(name, default))
            except (TypeError, ValueError):
                return default

        def number(name: str, default: float) -> float:
            try:
                return float(source.get(name, default))
            except (TypeError, ValueError):
                return default

        def boolean(name: str, default: bool) -> bool:
            value = str(source.get(name, str(default))).strip().lower()
            return value not in {"0", "false", "no", "off"}

        return cls(
            candidate_silence_ms=integer("MIGUEL_EOT_CANDIDATE_SILENCE_MS", 300),
            fast_complete_ms=integer("MIGUEL_EOT_FAST_COMPLETE_MS", 500),
            normal_complete_ms=integer("MIGUEL_EOT_NORMAL_COMPLETE_MS", 750),
            incomplete_wait_ms=integer("MIGUEL_EOT_INCOMPLETE_WAIT_MS", 1400),
            repair_consider_ms=integer("MIGUEL_EOT_REPAIR_CONSIDER_MS", 2200),
            repair_commit_ms=integer("MIGUEL_EOT_REPAIR_COMMIT_MS", 3000),
            engagement_accept_threshold=number("MIGUEL_ENGAGEMENT_ACCEPT_THRESHOLD", 0.70),
            engagement_decay_per_second=number("MIGUEL_ENGAGEMENT_DECAY_PER_SECOND", 0.008),
            expected_reply_seconds=number("MIGUEL_EXPECTED_REPLY_SECONDS", 20.0),
            enable_xvf_monitor=boolean("MIGUEL_ENABLE_XVF_MONITOR", True),
            enable_adaptive_endpoint=boolean("MIGUEL_ENABLE_ADAPTIVE_ENDPOINT", True),
        )


@dataclass(frozen=True)
class CompletionEvidence:
    score: float
    reason_codes: tuple[str, ...] = ()
    unfinished: bool = False


@dataclass(frozen=True)
class EndpointDecision:
    commit: bool
    reason: str
    silence_ms: int
    evidence: CompletionEvidence
    repair_consider: bool = False


_RULES = {
    "en": {
        "conjunctions": {"and", "but", "or", "because", "if", "when", "so", "then", "with", "to"},
        "fillers": {"um", "uh", "erm", "hmm"},
        "question_starts": {"who", "what", "where", "when", "why", "how", "can", "could", "would", "will", "do", "does", "did", "is", "are"},
    },
    "pt": {
        "conjunctions": {"e", "mas", "ou", "porque", "se", "quando", "entao", "com", "para"},
        "fillers": {"hum", "hmm", "ahn", "eh"},
        "question_starts": {"quem", "que", "qual", "onde", "quando", "por", "como", "pode", "voce", "e"},
    },
    "fr": {
        "conjunctions": {"et", "mais", "ou", "parce", "si", "quand", "donc", "avec", "pour"},
        "fillers": {"euh", "hum", "hmm"},
        "question_starts": {"qui", "que", "quel", "ou", "quand", "pourquoi", "comment", "peux", "est"},
    },
}


def _normalized_words(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode().lower()
    return re.findall(r"[a-z0-9']+", folded)


def score_semantic_completion(partial_text: str, language: str = "en", dialogue_context: Any = None) -> CompletionEvidence:
    words = _normalized_words(partial_text)
    if not words:
        return CompletionEvidence(0.0, ("empty",), True)
    rules = _RULES.get(language.split("-")[0].lower(), _RULES["en"])
    reasons: list[str] = []
    score = 0.55
    unfinished = False
    last = words[-1]
    normalized_text = " ".join(words)
    if last in rules["conjunctions"]:
        score -= 0.45
        unfinished = True
        reasons.append("trailing_conjunction")
    if last in rules["fillers"] or (len(words) <= 2 and all(w in rules["fillers"] for w in words)):
        score -= 0.45
        unfinished = True
        reasons.append("filler_or_hesitation")
    unfinished_prompts = {
        "en": {"can you tell me", "could you tell me", "i was wondering", "what about"},
        "pt": {"voce pode me dizer", "pode me dizer", "eu estava pensando", "e sobre"},
        "fr": {"peux tu me dire", "pouvez vous me dire", "je me demandais", "et concernant"},
    }
    language_key = language.split("-")[0].lower()
    if str(partial_text).rstrip().endswith("...") or normalized_text in unfinished_prompts.get(language_key, set()):
        score -= 0.45
        unfinished = True
        reasons.append("unfinished_clause")
    if str(partial_text).rstrip().endswith(("?", ".", "!")):
        score += 0.25
        reasons.append("terminal_punctuation")
    if words[0] in rules["question_starts"] and len(words) >= 3 and not unfinished:
        score += 0.25
        reasons.append("complete_question")
    context = dialogue_context or {}
    expected = bool(context.get("expected_reply")) if isinstance(context, Mapping) else False
    yes_no = {"yes", "no", "yeah", "nope", "sim", "nao", "oui", "non"}
    if expected and words[-1] in yes_no:
        score = max(score, 0.96)
        reasons.append("expected_yes_no")
    commands = {"stop", "pause", "shutdown", "cancel", "repeat", "continue", "quiet", "wait", "para", "arrete"}
    if len(words) <= 4 and any(word in commands for word in words):
        score = max(score, 0.92)
        reasons.append("clear_short_local_command")
    if isinstance(context, Mapping) and context.get("stable_repeated_partial"):
        score += 0.15
        reasons.append("stable_repeated_partial")
    if isinstance(context, Mapping) and context.get("recent_partial_change"):
        score -= 0.20
        reasons.append("recent_partial_change")
    if len(words) >= 4 and not unfinished:
        score += 0.10
        reasons.append("complete_statement_shape")
    return CompletionEvidence(max(0.0, min(1.0, score)), tuple(reasons or ["neutral"]), unfinished)


class AdaptiveEndpointDetector:
    def __init__(self, config: Optional[ConversationConfig] = None):
        self.config = config or ConversationConfig()
        self.reset()

    def reset(self) -> None:
        self.partial_text = ""
        self.language = "en"
        self.speech_start_monotonic: Optional[float] = None
        self.last_voice_monotonic: Optional[float] = None
        self.silence_start_monotonic: Optional[float] = None
        self.last_partial_change_monotonic: Optional[float] = None
        self.repeated_partial_count = 0

    def speech_started(self, timestamp: Optional[float] = None) -> None:
        now = time.monotonic() if timestamp is None else timestamp
        if self.speech_start_monotonic is None:
            self.speech_start_monotonic = now
        self.last_voice_monotonic = now
        self.silence_start_monotonic = None

    def voice_detected(self, timestamp: Optional[float] = None) -> bool:
        resumed = self.silence_start_monotonic is not None
        self.speech_started(timestamp)
        return resumed

    def silence_detected(self, timestamp: Optional[float] = None) -> None:
        now = time.monotonic() if timestamp is None else timestamp
        if self.silence_start_monotonic is None:
            self.silence_start_monotonic = now

    def update_partial(self, text: str, timestamp: Optional[float] = None, language: str = "en") -> None:
        now = time.monotonic() if timestamp is None else timestamp
        normalized = str(text or "").strip()
        if normalized == self.partial_text and normalized:
            self.repeated_partial_count += 1
        elif normalized:
            self.partial_text = normalized
            self.last_partial_change_monotonic = now
            self.repeated_partial_count = 0
        self.language = language

    def evaluate(self, timestamp: Optional[float] = None, dialogue_context: Optional[Mapping[str, Any]] = None) -> EndpointDecision:
        now = time.monotonic() if timestamp is None else timestamp
        if self.silence_start_monotonic is None:
            return EndpointDecision(False, "speech_active", 0, CompletionEvidence(0.0, ("speech_active",), True))
        silence_ms = max(0, round((now - self.silence_start_monotonic) * 1000))
        context = dict(dialogue_context or {})
        context["stable_repeated_partial"] = self.repeated_partial_count >= 1
        context["recent_partial_change"] = bool(
            self.last_partial_change_monotonic is not None
            and now - self.last_partial_change_monotonic < 0.25
        )
        evidence = score_semantic_completion(self.partial_text, self.language, context)
        cfg = self.config
        if silence_ms < cfg.candidate_silence_ms:
            return EndpointDecision(False, "below_candidate", silence_ms, evidence)
        # Stage 1 only marks a clearly incomplete utterance for repair at 2.2s;
        # it must not route the fragment as an ordinary completed turn yet.
        if silence_ms >= cfg.repair_commit_ms and evidence.unfinished:
            return EndpointDecision(True, "repair_timeout", silence_ms, evidence, True)
        if silence_ms >= cfg.repair_consider_ms and evidence.unfinished:
            return EndpointDecision(False, "repair_consider", silence_ms, evidence, True)
        # Vosk does not always emit a partial for very short or noisy turns.
        # Avoid an unnecessary repair-length wait while retaining a longer
        # silence than the normal semantic path.
        if silence_ms >= cfg.incomplete_wait_ms and not self.partial_text:
            return EndpointDecision(True, "no_partial_fallback", silence_ms, evidence)
        if silence_ms >= cfg.incomplete_wait_ms and not evidence.unfinished:
            return EndpointDecision(True, "ambiguous_complete", silence_ms, evidence)
        if silence_ms >= cfg.normal_complete_ms and not evidence.unfinished and evidence.score >= 0.55:
            return EndpointDecision(True, "normal_complete", silence_ms, evidence)
        if silence_ms >= cfg.fast_complete_ms and not evidence.unfinished and evidence.score >= 0.80:
            return EndpointDecision(True, "fast_complete", silence_ms, evidence)
        return EndpointDecision(False, "wait", silence_ms, evidence)
