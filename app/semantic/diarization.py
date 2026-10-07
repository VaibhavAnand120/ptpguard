"""
Speaker Diarizer & Role Classifier Bridge
Uses NVIDIA Streaming Sortformer (nvidia/diar_streaming_sortformer_4spk-v2.1)
and Persistent RoleResolver to separate speaker identity (speaker_0..speaker_3)
from speaker role (agent, borrower, third_party).
"""

import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from ..voice.sortformer import StreamingSortformerDiarizer
from ..voice.roles import RoleResolver, AGENT_PATTERNS, BORROWER_PATTERNS, THIRD_PARTY_PATTERNS


@dataclass
class DiarizationResult:
    primary_speaker: str  # "agent", "borrower", "third_party", "third_party_background"
    confidence: float     # 0.0 - 1.0
    rationale: str
    has_background_speech: bool = False
    background_speaker_type: Optional[str] = None  # "wife", "husband", "family_member", "parent", "unknown"
    background_transcript: Optional[str] = None
    background_coaching: bool = False              # coaching borrower to evade or delay
    clean_text: str = ""
    speaker_id: Optional[str] = None              # "speaker_0", "speaker_1", "speaker_2", "speaker_3"
    role_status: Optional[str] = None             # "provisional", "confirmed"


# Explicit background annotations e.g. "(peeche se patni: bolo paise nahi hai)"
BACKGROUND_ANNOTATION_RE = re.compile(
    r"(?:\(|\[|\*)\s*(?:peeche\s+se|background|bg|wife|patni|family|whisper|side\s+voice)[^:\)\-\*]*[:\-–]?\s*([^\]\)\*]+)(?:\)|\]|\*)",
    re.IGNORECASE
)

COACHING_PATTERNS = [
    re.compile(r"\b(?:bol\s*do|bolo|kaho|keh\s*do|kehdo)\b.*?\b(?:paise\s+nahi|salary\s+nahi|salary\s+late|nahi\s+hai|kal\b|baad\s+me|baad\s+mein|abhi\s+nahi|phone\s+kaat|phone\s+rakh|mat\s+do|call\s+karein|baat\s+karenge|cut\s+karo|nahi\s+denge)\b", re.I),
    re.compile(r"\b(?:unko|collector\s+ko|bank\s+walo\s+ko|unhe)\s+bolo\b", re.I),
    re.compile(r"\bphone\s+(?:kaat\s*do|rakh\s*do|cut\s*karo|disconnect\s*karo|band\s*karo|end\s*karo)\b", re.I),
    re.compile(r"\b(?:mat\s+do\s+abhi|paise\s+mat\s+dena|mat\s+bhejo|commitment\s+mat\s+do|mat\s+do\s+paise)\b", re.I),
    re.compile(r"\bration\s+(?:ka\s+paisa|lana\s+hai|lena\s+hai|ka\s+kharach)\b", re.I),
    re.compile(r"\btell\s+(?:him|them|the\s+agent)\b.*?\b(?:we\s+don't\s+have|no\s+money|call\s+later|hang\s+up|not\s+at\s+home|cannot\s+pay|not\s+here)\b", re.I),
    re.compile(r"\b(?:hang\s+up|cut\s+the\s+call|don't\s+pay|don't\s+give\s+any\s+money|don't\s+promise)\b", re.I),
    re.compile(r"\b(?:kyun\s+phone\s+uthaya|unko\s+mat\s+batao|mana\s+kar\s+do)\b", re.I),
]


class SpeakerDiarizer:
    """
    Adapter bridging call turns to NVIDIA Streaming Sortformer and RoleResolver.
    """

    _sortformer = StreamingSortformerDiarizer()
    _role_resolver = RoleResolver()

    @classmethod
    def reset(cls):
        cls._sortformer.reset()
        cls._role_resolver.reset()

    @classmethod
    def detect(
        cls,
        text: str,
        hint_speaker: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        force_reset: bool = False
    ) -> DiarizationResult:
        if force_reset:
            cls.reset()

        raw_text = text.strip()
        t = raw_text.lower()

        # Step 1: Detect explicit background annotations
        bg_match = BACKGROUND_ANNOTATION_RE.search(raw_text)
        has_annotation_bg = bool(bg_match)
        bg_transcript = bg_match.group(1).strip() if bg_match else None

        clean_text = BACKGROUND_ANNOTATION_RE.sub("", raw_text).strip()
        if not clean_text and bg_transcript:
            clean_text = bg_transcript

        # Step 2: Check coaching patterns
        has_coaching = any(p.search(t) for p in COACHING_PATTERNS)

        # Step 3: Speaker type identifiers
        bg_type = None
        if any(w in t for w in ["wife", "patni", "biwi"]):
            bg_type = "wife"
        elif any(w in t for w in ["husband", "pati"]):
            bg_type = "husband"
        elif any(w in t for w in ["mother", "mummy", "maa", "father", "papa", "pita"]):
            bg_type = "parent"
        elif any(w in t for w in ["bhai", "brother", "sister", "behan"]):
            bg_type = "family_member"
        elif has_annotation_bg or has_coaching:
            bg_type = "family_member"

        is_pure_background = False
        if has_annotation_bg and (not clean_text or clean_text == bg_transcript or len(clean_text) < 10):
            is_pure_background = True
        elif has_coaching and (hint_speaker in (None, "", "auto", "third_party_background") or not clean_text):
            is_pure_background = True
        elif hint_speaker == "third_party_background":
            is_pure_background = True

        if is_pure_background:
            return DiarizationResult(
                primary_speaker="third_party_background",
                confidence=0.96,
                rationale=f"Background speech detected on borrower's side ({bg_type or 'family member'})",
                has_background_speech=True,
                background_speaker_type=bg_type,
                background_transcript=bg_transcript or raw_text,
                background_coaching=has_coaching,
                clean_text=clean_text or raw_text,
                speaker_id="speaker_2",
                role_status="confirmed"
            )

        # Step 4: Run NVIDIA Streaming Sortformer (identifies stable speaker ID)
        spk_segments = cls._sortformer.process_utterance_event(
            text=clean_text or raw_text,
            start_time=cls._sortformer.current_time,
            end_time=cls._sortformer.current_time + max(1.0, len(raw_text.split()) * 0.35),
            speaker_hint=hint_speaker,
            background_hint=has_annotation_bg or has_coaching
        )
        assigned_speaker_id = spk_segments[0].speaker_id if spk_segments else "speaker_0"

        # Step 5: Run Persistent RoleResolver (resolves role decoupled from speaker ID)
        role_state = cls._role_resolver.resolve_role(
            speaker_id=assigned_speaker_id,
            text=clean_text or raw_text,
            is_background=has_annotation_bg or has_coaching,
            manual_override=hint_speaker if hint_speaker != "auto" else None
        )

        return DiarizationResult(
            primary_speaker=role_state.role,
            confidence=role_state.confidence,
            rationale=role_state.last_rationale,
            has_background_speech=has_annotation_bg or has_coaching,
            background_speaker_type=bg_type,
            background_transcript=bg_transcript,
            background_coaching=has_coaching,
            clean_text=clean_text or raw_text,
            speaker_id=assigned_speaker_id,
            role_status=role_state.status,
        )
