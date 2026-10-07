from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from .schemas import ConversationState, Evidence, EvidenceHistoryEntry
from .config import DIMENSIONS, DEFAULT_SCALE
from .scoring import normalize_raw_state, credibility_score


class StateManager:
    """
    Central State Manager for PTP evidence.
    Maintains dynamic continuous signed raw state, normalized soft saturation,
    evidence history audit trail, and trajectory tracking.
    """

    def __init__(self):
        self.state = ConversationState()

    def reset(self):
        self.state = ConversationState()

    def update(self, evidence: Evidence, speaker: str, text: str) -> ConversationState:
        s = self.state

        # Resolve speaker role
        effective_speaker = evidence.detected_speaker or speaker
        if effective_speaker == "auto":
            effective_speaker = "borrower"

        # 1. Repetition penalty / diminishing returns for duplicate utterances
        clean_text = text.strip().lower()
        repeat_count = sum(
            1 for entry in s.transcript
            if entry.get("text", "").strip().lower() == clean_text
        )
        repetition_decay = 1.0 / (1.0 + 0.85 * repeat_count)

        # 2. Process dimensional signals and apply signed deltas
        timestamp_str = datetime.now(timezone.utc).strftime("%H:%M:%S")

        for dim, sig in evidence.signals.items():
            if dim in DIMENSIONS and sig.direction != 0:
                effective_conf = sig.confidence * repetition_decay
                delta = sig.direction * sig.strength * effective_conf

                state_before = s.raw_state.get(dim, 0.0)
                state_after = state_before + delta
                s.raw_state[dim] = state_after

                # Audit trail entry
                s.evidence_history.append(
                    EvidenceHistoryEntry(
                        timestamp=timestamp_str,
                        speaker=effective_speaker,
                        text=text,
                        dimension=dim,
                        direction=sig.direction,
                        strength=sig.strength,
                        confidence=round(effective_conf, 3),
                        delta=round(delta, 3),
                        state_before=round(state_before, 3),
                        state_after=round(state_after, 3),
                    )
                )

        # 3. Soft-saturation normalization via tanh(raw / SCALE)
        s.normalized_state = normalize_raw_state(s.raw_state, DEFAULT_SCALE)

        # 4. Extract concrete amount/date entities
        if evidence.amount is not None:
            s.amount = evidence.amount
            s.amount_confirmed = True
        if evidence.date is not None:
            s.date = evidence.date
            s.date_confirmed = True

        # 5. Track speakers
        if effective_speaker not in s.active_speakers:
            s.active_speakers.append(effective_speaker)
        if evidence.third_party_background and "third_party_background" not in s.active_speakers:
            s.active_speakers.append("third_party_background")

        # 6. Append to transcript
        s.transcript.append({
            "speaker": effective_speaker,
            "input_speaker": speaker,
            "detected_speaker": evidence.detected_speaker or effective_speaker,
            "speaker_confidence": evidence.speaker_confidence or 1.0,
            "speaker_rationale": evidence.speaker_rationale or "",
            "is_background": evidence.third_party_background,
            "background_details": evidence.background_details or "",
            "text": text,
            "timestamp": timestamp_str,
        })

        # 7. Record score and dimension trajectories
        current_score = credibility_score(s)
        s.score_history.append(current_score)
        for dim in DIMENSIONS:
            s.dimension_history[dim].append(round(s.raw_state[dim], 2))

        # 8. Synchronize backward-compatibility convenience flags
        s.ptp_detected = (
            s.ptp_detected
            or evidence.ptp_detected
            or (s.raw_state.get("commitment", 0.0) > 0.0)
            or (s.raw_state.get("specificity", 0.0) > 0.0)
        )
        s.borrower_initiated = s.raw_state.get("borrower_initiation", 0.0) > 0.0
        s.explicit_confirmation = s.raw_state.get("confirmation", 0.0) > 0.0
        s.conditional = s.raw_state.get("conditionality", 0.0) < 0.0
        s.hardship = s.raw_state.get("hardship", 0.0) < 0.0
        s.agent_pushed = s.raw_state.get("agent_pressure", 0.0) < 0.0
        s.third_party = s.raw_state.get("third_party", 0.0) < 0.0
        s.escape_language = s.raw_state.get("escape_signal", 0.0) < 0.0
        s.third_party_background = s.third_party_background or evidence.third_party_background
        s.background_coaching = s.background_coaching or evidence.background_coaching

        for field in [
            "borrower_initiated", "explicit_confirmation", "conditional",
            "hardship", "agent_pushed", "third_party", "escape_language"
        ]:
            if getattr(s, field, False):
                s.observed_evidence[field] = True

        return s

    def snapshot(self) -> Dict[str, Any]:
        return self.state.model_dump()
