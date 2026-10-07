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


class Utterance(BaseModel):
    speaker: str = Field(default="auto", description="agent, borrower, third_party, third_party_background, or auto")
    text: str
    detected_speaker: Optional[str] = None
    speaker_confidence: Optional[float] = None
    speaker_rationale: Optional[str] = None
    is_background_speech: bool = False
    background_speaker_info: Optional[str] = None
    background_transcript: Optional[str] = None


class Evidence(BaseModel):
    # Semantic directional signals across dimensions (Source of Truth for updates)
    signals: Dict[str, Signal] = Field(default_factory=dict)

    # Concrete entities
    amount: Optional[float] = None
    date: Optional[str] = None

    # Speaker separation metadata
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

    active_speakers: List[str] = Field(default_factory=list)
    transcript: List[Dict[str, Any]] = Field(default_factory=list)

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
    state: Dict[str, Any]
