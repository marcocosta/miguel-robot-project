"""Stage-1 owner of interaction state, engagement, floor and endpointing."""

import re
import threading
import time
from typing import Callable, Optional

from miguel_conversation_events import FloorOwner, InteractionState
from miguel_spatial_audio import SpatialAudioEvidence, UNKNOWN
from miguel_turn_endpoint import AdaptiveEndpointDetector, ConversationConfig, EndpointDecision


class ConversationManager:
    def __init__(
        self,
        config: Optional[ConversationConfig] = None,
        logger: Callable[[str], None] = print,
        state_observer: Optional[Callable[[InteractionState, str], None]] = None,
    ):
        self.config = config or ConversationConfig.from_env()
        self.log = logger
        self.state_observer = state_observer
        self.state = InteractionState.IDLE
        self.floor_owner = FloorOwner.NONE
        self.engagement_score = 0.0
        self.last_engagement_update = time.monotonic()
        self.robot_just_asked_question = False
        self.expected_reply_person: Optional[str] = None
        self.expected_reply_until = 0.0
        self.endpoint = AdaptiveEndpointDetector(self.config)
        self._candidate_logged = False
        self._last_candidate_log = 0.0
        self.turn_timing: dict[str, object] = {}
        self._repair_logged = False
        self._spatial_lock = threading.Lock()
        self.latest_spatial_audio_evidence = SpatialAudioEvidence(
            0.0, False, False, None, None, False, 0.0, 0, UNKNOWN
        )
        self.turn_spatial_first_stable: Optional[SpatialAudioEvidence] = None
        self.turn_spatial_last_stable: Optional[SpatialAudioEvidence] = None

    def note_spatial_audio_evidence(self, evidence: SpatialAudioEvidence) -> None:
        """Store Stage 2B evidence without changing conversation behavior."""
        with self._spatial_lock:
            self.latest_spatial_audio_evidence = evidence
            if (
                self.state in {InteractionState.LISTENING, InteractionState.END_CANDIDATE}
                and evidence.available
                and evidence.vad_active
                and evidence.stable
                and not evidence.suppressed
            ):
                if self.turn_spatial_first_stable is None:
                    self.turn_spatial_first_stable = evidence
                self.turn_spatial_last_stable = evidence

    def spatial_audio_evidence(self) -> SpatialAudioEvidence:
        with self._spatial_lock:
            return self.latest_spatial_audio_evidence

    def turn_spatial_audio_evidence(self) -> Optional[SpatialAudioEvidence]:
        """Return the final stable snapshot latched for this listening turn."""
        with self._spatial_lock:
            return self.turn_spatial_last_stable

    def _log_addressee_shadow(self, baseline_accept: bool) -> None:
        evidence = self.turn_spatial_audio_evidence()
        evidence_source = "turn_latched"
        if evidence is None:
            evidence_source = "no_evidence"
            evidence = SpatialAudioEvidence(
                0.0, False, False, None, None, False, 0.0, 0, UNKNOWN
            )
        self.log(
            "[ADDRESSEE_SHADOW] "
            f"baseline_accept={str(baseline_accept).lower()} "
            f"evidence_source={evidence_source} "
            f"vad={str(evidence.vad_active).lower()} raw_doa={evidence.raw_doa_deg} "
            f"relative_doa={evidence.relative_doa_deg} stable={str(evidence.stable).lower()} "
            f"resultant={evidence.circular_resultant:.3f} samples={evidence.sample_count} "
            f"spatial={evidence.classification} suppressed={str(evidence.suppressed).lower()}"
        )

    def _transition(self, target: InteractionState, reason: str) -> None:
        if target == self.state:
            return
        old = self.state
        self.state = target
        self.log(f"[CONV_STATE] from={old.value} to={target.value} reason={reason}")
        if self.state_observer is not None:
            try:
                self.state_observer(target, reason)
            except Exception as exc:
                self.log(f"[CONV_STATE] observer_error={type(exc).__name__}: {exc}")

    def _floor(self, owner: FloorOwner, reason: str) -> None:
        if owner == self.floor_owner:
            return
        old = self.floor_owner
        self.floor_owner = owner
        self.log(f"[FLOOR] from={old.value} to={owner.value} reason={reason}")

    def decay_engagement(self, timestamp: Optional[float] = None) -> float:
        now = time.monotonic() if timestamp is None else timestamp
        elapsed = max(0.0, now - self.last_engagement_update)
        old = self.engagement_score
        self.engagement_score = max(0.0, old - elapsed * self.config.engagement_decay_per_second)
        self.last_engagement_update = now
        return self.engagement_score

    def reinforce_engagement(self, amount: float, reason: str, timestamp: Optional[float] = None) -> None:
        self.decay_engagement(timestamp)
        old = self.engagement_score
        self.engagement_score = min(1.0, max(0.0, self.engagement_score + amount))
        self.log(f"[ENGAGEMENT] from={old:.3f} to={self.engagement_score:.3f} reason={reason}")

    def on_wake(self, phrase: str = "miguel", person: Optional[str] = None, timestamp: Optional[float] = None) -> None:
        now = time.monotonic() if timestamp is None else timestamp
        old = self.engagement_score
        self.engagement_score = 1.0
        self.last_engagement_update = now
        self.log(f"[ENGAGEMENT] from={old:.3f} to=1.000 reason=wake person={person or 'unknown'}")
        self._transition(InteractionState.ENGAGED, "wake")

    def begin_listening(self, timestamp: Optional[float] = None) -> None:
        now = time.monotonic() if timestamp is None else timestamp
        self.endpoint.reset()
        self.turn_timing = {"capture_start_monotonic": now}
        self._candidate_logged = False
        self._repair_logged = False
        with self._spatial_lock:
            self.turn_spatial_first_stable = None
            self.turn_spatial_last_stable = None
        self._floor(FloorOwner.NONE, "listen_started")
        self._transition(InteractionState.LISTENING, "listen_started")

    def on_voice(self, active: bool, timestamp: Optional[float] = None) -> None:
        now = time.monotonic() if timestamp is None else timestamp
        self.log(f"[VAD] active={str(active).lower()} source=audio_rms")
        if active:
            first = self.endpoint.speech_start_monotonic is None
            self._floor(FloorOwner.HUMAN, "voice_started" if first else "voice_resumed")
            resumed = self.endpoint.voice_detected(now)
            if first:
                self.turn_timing["speech_start_monotonic"] = now
            self.turn_timing["last_detected_voice_monotonic"] = now
            if resumed:
                self.log("[EOT_CANCEL] reason=speech_resumed")
                self._transition(InteractionState.LISTENING, "speech_resumed")
                self._candidate_logged = False
        else:
            self.endpoint.silence_detected(now)
            self.turn_timing["silence_start_monotonic"] = now

    def note_voice_activity(self, timestamp: Optional[float] = None) -> None:
        """Refresh voice timing without emitting one VAD log per audio frame."""
        now = time.monotonic() if timestamp is None else timestamp
        self.endpoint.last_voice_monotonic = now
        self.turn_timing["last_detected_voice_monotonic"] = now

    def on_partial(self, text: str, timestamp: Optional[float] = None, language: str = "en") -> None:
        now = time.monotonic() if timestamp is None else timestamp
        previous = self.endpoint.partial_text
        self.endpoint.update_partial(text, now, language)
        if text and text != previous:
            self.log(f"[ASR_PARTIAL] text={text!r} language={language}")

    def evaluate_endpoint(self, timestamp: Optional[float] = None) -> EndpointDecision:
        now = time.monotonic() if timestamp is None else timestamp
        decision = self.endpoint.evaluate(now, {"expected_reply": self.expects_reply(now)})
        if decision.silence_ms >= self.config.candidate_silence_ms and not self._candidate_logged:
            self._candidate_logged = True
            self._transition(InteractionState.END_CANDIDATE, "silence_candidate")
        if self._candidate_logged and (now - self._last_candidate_log >= 0.25 or decision.commit):
            self._last_candidate_log = now
            marker = "EOT_COMMIT" if decision.commit else "EOT_CANDIDATE"
            self.log(
                f"[{marker}] silence_ms={decision.silence_ms} partial={self.endpoint.partial_text!r} "
                f"semantic_score={decision.evidence.score:.2f} unfinished={str(decision.evidence.unfinished).lower()} "
                f"decision={'commit' if decision.commit else 'wait'} reason={decision.reason}"
            )
        if decision.repair_consider and not decision.commit and not self._repair_logged:
            self._repair_logged = True
            self.turn_timing["repair_consider_monotonic"] = now
            self.log(
                f"[EOT_REPAIR] silence_ms={decision.silence_ms} partial={self.endpoint.partial_text!r} "
                f"semantic_score={decision.evidence.score:.2f} decision=offer_continuation"
            )
        if decision.commit:
            self.turn_timing["turn_commit_monotonic"] = now
            self.turn_timing["turn_commit_reason"] = decision.reason
            self.turn_timing["repair_consider"] = decision.repair_consider
            self._floor(FloorOwner.NONE, "human_turn_committed")
            self._transition(InteractionState.PREPARING, decision.reason)
        return decision

    def final_asr(self, text: str, timestamp: Optional[float] = None) -> None:
        now = time.monotonic() if timestamp is None else timestamp
        self.turn_timing["final_asr_monotonic"] = now

    def finish_listening_without_turn(self, reason: str = "no_speech") -> None:
        """Release the human floor after timeout, empty ASR, or capture failure."""
        self._floor(FloorOwner.NONE, reason)
        target = InteractionState.ENGAGED if self.decay_engagement() > 0 else InteractionState.IDLE
        self._transition(target, reason)

    def legacy_commit(self, timestamp: Optional[float] = None, silence_ms: int = 0) -> None:
        now = time.monotonic() if timestamp is None else timestamp
        self.turn_timing["turn_commit_monotonic"] = now
        self.log(f"[EOT_COMMIT] silence_ms={silence_ms} decision=commit reason=legacy_endpoint")
        self._floor(FloorOwner.NONE, "legacy_turn_committed")
        self._transition(InteractionState.PREPARING, "legacy_endpoint")

    def accepted_turn(self, timestamp: Optional[float] = None) -> None:
        self.reinforce_engagement(0.20, "successful_turn", timestamp)

    def accept_human_turn(
        self,
        text: str,
        explicit_wake: bool = False,
        timestamp: Optional[float] = None,
        person: Optional[str] = None,
    ) -> bool:
        now = time.monotonic() if timestamp is None else timestamp
        if explicit_wake:
            self.on_wake(text, timestamp=now)
            baseline_accept = True
        elif self.expects_reply(now):
            if self.expected_reply_person and person:
                baseline_accept = self.expected_reply_person.casefold() == person.casefold()
            else:
                baseline_accept = True
        else:
            baseline_accept = self.decay_engagement(now) >= self.config.engagement_accept_threshold
        self._log_addressee_shadow(baseline_accept)
        return baseline_accept

    def expects_reply(self, timestamp: Optional[float] = None) -> bool:
        now = time.monotonic() if timestamp is None else timestamp
        if now > self.expected_reply_until:
            self.robot_just_asked_question = False
            self.expected_reply_person = None
        return self.robot_just_asked_question

    def response_ready(self, timestamp: Optional[float] = None) -> None:
        self.turn_timing["brain_response_ready_monotonic"] = time.monotonic() if timestamp is None else timestamp
        self._transition(InteractionState.PREPARING, "response_ready")

    def grant_robot_floor(self, text: str, person: Optional[str] = None, timestamp: Optional[float] = None) -> bool:
        if self.floor_owner == FloorOwner.HUMAN:
            self.log("[FLOOR] grant=MIGUEL allowed=false reason=human_has_floor")
            return False
        now = time.monotonic() if timestamp is None else timestamp
        self._floor(FloorOwner.MIGUEL, "tts_start")
        self._transition(InteractionState.SPEAKING, "tts_start")
        return True

    def robot_first_audio(self, timestamp: Optional[float] = None) -> None:
        self.turn_timing["tts_first_audio_monotonic"] = time.monotonic() if timestamp is None else timestamp

    def robot_speech_ended(self, text: str, person: Optional[str] = None, timestamp: Optional[float] = None) -> None:
        now = time.monotonic() if timestamp is None else timestamp
        asked = bool(re.search(r"\?\s*$", text.strip())) or text.lower().strip().startswith(
            ("do you ", "would you ", "can you ", "what ", "who ", "where ", "when ", "why ", "how ")
        )
        self._floor(FloorOwner.NONE, "tts_ended")
        if asked:
            self.robot_just_asked_question = True
            self.expected_reply_person = person
            self.expected_reply_until = now + self.config.expected_reply_seconds
            self.reinforce_engagement(0.35, "robot_question", now)
            self._transition(InteractionState.ENGAGED, "awaiting_reply")
        else:
            self._transition(InteractionState.ENGAGED if self.decay_engagement(now) > 0 else InteractionState.IDLE, "tts_ended")

    def close_conversation(self, reason: str = "explicit_closure") -> None:
        old = self.engagement_score
        self.engagement_score = 0.0
        self.robot_just_asked_question = False
        self.log(f"[ENGAGEMENT] from={old:.3f} to=0.000 reason={reason}")
        self._floor(FloorOwner.NONE, reason)
        self._transition(InteractionState.IDLE, reason)

    def latency_fields(self) -> dict:
        values = dict(self.turn_timing)
        eot = values.get("turn_commit_monotonic")
        speech = values.get("last_detected_voice_monotonic")
        if eot is not None and speech is not None:
            values["speech_to_eot_delay_ms"] = max(0.0, (eot - speech) * 1000)
        for name, stamp in (
            ("eot_to_final_asr_ms", values.get("final_asr_monotonic")),
            ("eot_to_brain_request_ms", values.get("brain_request_start_monotonic")),
            ("eot_to_response_ready_ms", values.get("brain_response_ready_monotonic")),
            ("eot_to_first_audio_ms", values.get("tts_first_audio_monotonic")),
        ):
            if eot is not None and stamp is not None:
                values[name] = max(0.0, (stamp - eot) * 1000)
        return values
