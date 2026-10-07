from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from .config import DIMENSIONS


class Signal(BaseModel):
    direction: int = Field(default=0, description="+1 (credibility supporting), -1 (credibility reducing), 0 (neutral)")
    strength: int = Field(default=1, ge=0, le=5, description="1 (minor), 2 (moderate), 3 (strong)")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0, description="0.0 to 1.0 confidence score")
    rationale: Optional[str] = None


class EvidenceHistoryEntry(BaseModel):
    timestamp: str
    speaker: str
    text: str
    dimension: str
    direction: int
    strength: int
    confidence: float
    delta: float
    state_before: float
    state_after: float
    speaker_id: Optional[str] = None
    speaker_role: Optional[str] = None
    source: Optional[str] = None

class DeadlineEvent(BaseModel):
    timestamp: str
    speaker: str
    speaker_id: str
    deadline: str                        # e.g. "10 October"
    days_diff: Optional[int] = None      # Difference in days compared to previous (e.g. +10, -5, 0)
    event_type: str                      # "initial", "postponed", "accelerated", "clarification", "agent_proposed", "confirmed_agent_proposal"
    credibility_penalty_pct: int = 0     # e.g. -8 for -8%
    penalty_raw: float = 0.0             # raw state delta applied to commitment
    rationale: str
    agent_pressure: bool = False
    hardship_present: bool = False
    borrower_initiated: bool = True


class DeadlineState(BaseModel):
    current_deadline: Optional[str] = None
    original_deadline: Optional[str] = None
    previous_deadline: Optional[str] = None
    confidence: float = 0.0              # e.g. 0.94 (94%)
    days_diff: Optional[int] = None      # Current shift magnitude (+10, -5)
    last_penalty_pct: int = 0            # Current change impact in % (e.g. -8)
    postpone_count: int = 0
    history: List[DeadlineEvent] = Field(default_factory=list)
    agent_proposed_deadline: Optional[str] = None
    borrower_stated_deadline: Optional[str] = None
    borrower_confirmed_deadline: Optional[str] = None


class Utterance(BaseModel):
    speaker: str = Field(default="auto", description="agent, borrower, third_party, third_party_background, or auto")
    text: str
    mode: str = Field(default="auto", description="live_audio, auto_text, manual_simulation")
    speaker_id: Optional[str] = None
    detected_speaker: Optional[str] = None
    speaker_confidence: Optional[float] = None
    speaker_rationale: Optional[str] = None
    is_background_speech: bool = False
    background_speaker_info: Optional[str] = None
    background_transcript: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    voice: Optional[str] = None
    voice_id: Optional[str] = None
    voice_features: Optional[Dict[str, Any]] = None
    chunk_id: Optional[str] = None


class Evidence(BaseModel):
    # Semantic directional signals across dimensions (Source of Truth for updates)
    signals: Dict[str, Signal] = Field(default_factory=dict)

    # Concrete entities
    amount: Optional[float] = None
    date: Optional[str] = None

    # Track agent proposed vs borrower stated separately
    agent_proposed_date: Optional[str] = None
    agent_proposed_amount: Optional[float] = None
    borrower_stated_date: Optional[str] = None
    borrower_stated_amount: Optional[float] = None

    # Processing mode
    mode: str = "auto"

    # Speaker identity and role separation
    speaker_id: Optional[str] = None
    speaker_role: Optional[str] = None
    detected_speaker: Optional[str] = None
    speaker_confidence: Optional[float] = None
    speaker_rationale: Optional[str] = None
    third_party_background: bool = False
    background_coaching: bool = False
    background_details: Optional[str] = None

    evidence_text: List[str] = Field(default_factory=list)

    # Backward compatibility fields (Derived/convenience, do NOT drive scoring)
    ptp_detected: bool = False
    amount_concrete: bool = False
    date_concrete: bool = False
    borrower_initiated: bool = False
    explicit_confirmation: bool = False
    negotiated_commitment: bool = False
    immediate_payment: bool = False
    conditional: bool = False
    hedging: bool = False
    vague_date: bool = False
    vague_amount: bool = False
    hardship: bool = False
    agent_pushed: bool = False
    third_party: bool = False
    escape_language: bool = False


class ConversationState(BaseModel):
    # Continuous raw signed state across 10 dimensions (can grow unbounded)
    raw_state: Dict[str, float] = Field(default_factory=lambda: {d: 0.0 for d in DIMENSIONS})

    # Normalized soft-saturated state via tanh(raw / SCALE) in [-1.0, +1.0]
    normalized_state: Dict[str, float] = Field(default_factory=lambda: {d: 0.0 for d in DIMENSIONS})

    # Historical score & dimension trajectories
    score_history: List[int] = Field(default_factory=list)
    dimension_history: Dict[str, List[float]] = Field(default_factory=lambda: {d: [] for d in DIMENSIONS})

    # Complete audit trail of evidence deltas
    evidence_history: List[EvidenceHistoryEntry] = Field(default_factory=list)

    # Entities
    amount: Optional[float] = None
    amount_confirmed: bool = False
    date: Optional[str] = None
    date_confirmed: bool = False

    # Distinct tracking of agent proposed vs borrower stated terms
    agent_proposed_date: Optional[str] = None
    agent_proposed_amount: Optional[float] = None
    borrower_stated_date: Optional[str] = None
    borrower_stated_amount: Optional[float] = None

    active_speakers: List[str] = Field(default_factory=list)
    speaker_roles: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    transcript: List[Dict[str, Any]] = Field(default_factory=list)
    deadline_state: DeadlineState = Field(default_factory=DeadlineState)

    # Backward compatibility indicators
    ptp_detected: bool = False
    borrower_initiated: bool = False
    explicit_confirmation: bool = False
    negotiated_commitment: bool = False
    immediate_payment: bool = False
    conditional: bool = False
    hedging: bool = False
    vague_date: bool = False
    vague_amount: bool = False
    hardship: bool = False
    agent_pushed: bool = False
    third_party: bool = False
    third_party_background: bool = False
    background_coaching: bool = False
    escape_language: bool = False
    observed_evidence: Dict[str, bool] = Field(default_factory=dict)


class AnalysisResponse(BaseModel):
    score: int
    ptp_type: str
    action: str
    action_message: str
    evidence: Evidence
    reasons: List[str]
    raw_state: Dict[str, float]
    normalized_state: Dict[str, float]
    score_history: List[int]
    evidence_history: List[EvidenceHistoryEntry]
    speaker_id: Optional[str] = None
    speaker_role: Optional[str] = None
    role_status: Optional[str] = None
    role_confidence: Optional[float] = None
    mode: Optional[str] = None
    speaker_roles: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    detected_acoustic_speakers: int = 0
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    sortformer_tracks: Dict[str, Any] = Field(default_factory=dict)
    audio_intake: Dict[str, Any] = Field(default_factory=dict)
    deadline_state: DeadlineState = Field(default_factory=DeadlineState)
    telemetry: Dict[str, Any] = Field(default_factory=dict)
    state: Dict[str, Any]
