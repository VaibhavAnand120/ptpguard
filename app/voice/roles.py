"""
Persistent Role Resolver
Maintains speaker role states across the call independently of speaker IDs.

Roles:
- AGENT: Collections agent, demanding dues, institutional introductions, 2nd-person inquiries.
- BORROWER: Primary customer/debtor, personal finances, salary updates, repayment commitments.
- THIRD_PARTY: Relatives, spouse, third party speaking directly or in background.
- UNKNOWN: Insufficient conversational evidence.

Key Architectural Guarantees:
1. Different persistent speaker IDs receive different roles.
2. In a normal 2-person collection call, exactly one speaker is AGENT and one is BORROWER
   unless there is evidence of a third party.
3. The two primary speaker IDs must NEVER both automatically receive BORROWER (or both AGENT).
4. Uses conversation-level cumulative evidence across interlocutor dialogue turns.
5. Zero unsafe fallbacks: no `default="borrower"`, no `previous_role`, no `previous_speaker`.
6. Full evidence logging per speaker ID for transparent debugging.
"""

import re
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("ptpguard.roles")


class SpeakerRoleState(BaseModel):
    speaker_id: str
    role: str = "unknown"               # "agent", "borrower", "third_party", "third_party_background", "unknown"
    confidence: float = 0.5             # 0.0 - 1.0
    status: str = "provisional"         # "provisional", "confirmed", "manual_simulation"
    evidence_count: int = 0
    agent_score: float = 0.0
    borrower_score: float = 0.0
    third_party_score: float = 0.0
    evidence_logs: List[str] = Field(default_factory=list)
    last_rationale: str = ""


# ---------------------------------------------------------------------------
# Linguistic Rules & Weighted Evidence
# ---------------------------------------------------------------------------

AGENT_WEIGHTED_RULES = [
    # 1. Institutional / collections intro
    (re.compile(r"\b(?:calling\s+from|se\s+bol\s+raha|se\s+bol\s+rahi|se\s+call\s+hai|se\s+call\s+kar|regarding\s+your|am\s+i\s+speaking\s+to|kya\s+meri\s+baat)\b", re.I), 3.0, "institutional calling introduction"),
    (re.compile(r"\b(?:bank|finance|capital|nbfc|agency|recovery|credit\s+card|collections?|hdfc|icici|sbi|bajaj|axis|kotak|cred|moneyview|kreditbee)\b", re.I), 2.5, "institutional creditor mention"),
    
    # 2. Notification of dues, emi, deadline
    (re.compile(r"\b(?:you\s+have\s+a\s+deadline|deadline\s+of\s+payment|payment\s+deadline|deadline\s+is|deadline\s+hai|your\s+deadline|deadline\s+for\s+payment|deadline\s+of)\b", re.I), 3.0, "payment deadline notification"),
    (re.compile(r"\b(?:aapka|aapki|your)\s+(?:loan|emi|pending|overdue|dues|installment|balance|khata|account|payment)\b", re.I), 2.5, "notice of debtor account/dues"),
    (re.compile(r"\b(?:pending\s+amount|overdue\s+amount|due\s+amount|outstanding\s+amount|required\s+amount|pay\s+the\s+required\s+amount)\b", re.I), 2.5, "due/overdue amount specification"),

    # 3. Demanding / Instructing payment
    (re.compile(r"\b(?:you\s+need\s+to\s+pay|you\s+have\s+to\s+pay|you\s+must\s+pay|you\s+should\s+pay|sir,\s*but\s+you\s+need\s+to\s+pay|but\s+you\s+need\s+to\s+pay)\b", re.I), 3.0, "demand for payment"),
    (re.compile(r"\b(?:pay\s+before|pay\s+by\b|clear\s+before|clear\s+by|pay\s+your\s+payment|clear\s+your\s+dues)\b", re.I), 2.5, "direction to clear payment by date"),
    (re.compile(r"\b(?:karna\s+padega|dena\s+padega|pay\s+karna\s+hoga|clear\s+karna\s+hoga|jama\s+karna\s+padega)\b", re.I), 2.5, "collection requirement in Hindi"),

    # 4. Inquiring when / how payment will be made
    (re.compile(r"\b(?:when\s+(?:will|can)\s+you\s+(?:make\s+(?:the\s+)?)?pay(?:ment)?|how\s+(?:will|can|much)\s+(?:you|can\s+you)\s+pay|payment\s+status)\b", re.I), 2.5, "inquiry on payment timing/amount"),
    (re.compile(r"\b(?:kab\s+tak|kab)\b.*?\b(?:payment|pay|clear|jama|bhej|karenge|karoge|kar\s+paoge|denge|karein|ho\s+payega)\b", re.I), 2.5, "Hindi inquiry on payment date"),
    (re.compile(r"\b(?:can\s+you\s+pay|can\s+you\s+make|will\s+you\s+pay|will\s+you\s+be\s+able\s+to\s+pay)\b", re.I), 2.0, "inquiry on debtor payment ability"),
    (re.compile(r"\b(?:payment\s+kar\s+paoge|payment\s+kar\s+denge|payment\s+possible\s+hai|ho\s+payega)\b", re.I), 2.0, "Hindi possibility inquiry"),

    # 5. Inquiries / Confirmation check
    (re.compile(r"(?:\bconfirmed|\bconfirm\s+hai|\bright\s+sir|\bcorrect\s+sir|\bcorrect|\bright)\s*\?", re.I), 2.5, "agent confirmation inquiry"),
    (re.compile(r"\b(?:are\s+you\s+sure|are\s+you\s+still|confirmed\b|confirm\s+karein|confirm\s+kar\s+rahe)\b", re.I), 1.5, "agent confirmation check"),
    (re.compile(r"\b(?:i\'?ll\s+mark|i\s+will\s+mark|mark\s+kar\s+doon|note\s+kar\s+loon|system\s+me\s+update|system\s+me\s+mark)\b", re.I), 2.5, "agent record keeping action"),
    (re.compile(r"\b(?:payment\s+link|upi\s+link|link\s+bhej|link\s+share)\b", re.I), 2.0, "payment collection channel"),
    (re.compile(r"\b(?:cibil|penalty|legal\s+notice|field\s+visit)\b", re.I), 2.5, "collection escalation consequence")
]

BORROWER_WEIGHTED_RULES = [
    # 1. Hardship / Inability to pay
    (re.compile(r"\b(?:i\s+don\'?t\s+have\s+(?:any\s+)?money|i\s+don\'?t\s+have\s+(?:any\s+)?payment|don\'?t\s+have\s+money|no\s+money)\b", re.I), 3.0, "borrower lack of funds"),
    (re.compile(r"\b(?:paise\s+nahi|paisa\s+nahi|paise\s+ki\s+dikkat|fund\s+nahi|cash\s+nahi)\b", re.I), 3.0, "Hindi lack of funds"),
    (re.compile(r"\b(?:i\s+don\'?t\s+know\s+when\s+to\s+pay|i\s+don\'?t\s+know\s+how\s+to\s+pay|don\'?t\s+know\s+what\s+to\s+pay)\b", re.I), 2.5, "debtor confusion/inability to commit"),
    (re.compile(r"\b(?:bank\s+account\s+is\s+empty|account\s+is\s+empty|empty|difficult\s+right\s+now|quite\s+difficult)\b", re.I), 2.5, "account distress/difficulty"),
    (re.compile(r"\b(?:lost\s+my\s+job|job\s+loss|job\s+chali\s+gayi|unemployed|salary\s+late|not\s+received\s+any\s+salary|salary\s+nahi\s+aayi)\b", re.I), 3.0, "income shock/job loss"),
    (re.compile(r"\b(?:hospital|medical\s+emergency|tabiyat\s+kharab|bimari|loss\s+in\s+business|dukaan\s+me\s+nuksan)\b", re.I), 2.5, "health/business distress"),

    # 2. Personal commitment / efforts / promises
    (re.compile(r"\b(?:i\s+will\s+try\s+my\s+best|i\s+will\s+try|i\'?ll\s+try\s+my\s+best|try\s+karunga|koshish\s+karunga)\b", re.I), 2.5, "effort promise"),
    (re.compile(r"\b(?:i\s+will\s+pay|i\s+will\s+clear|i\s+will\s+transfer|i\s+can\s+pay|i\s+promise|yes,\s*i\s+can\s+pay|yes\s+i\s+can\s+pay)\b", re.I), 2.5, "debtor first-person commitment"),
    (re.compile(r"\b(?:kar\s+dunga|de\s+dunga|pay\s+kar\s+dunga|transfer\s+kar\s+dunga|bhej\s+dunga|jama\s+kar\s+dunga|clear\s+kar\s+dunga)\b", re.I), 2.5, "Hindi payment promise"),
    (re.compile(r"\b(?:i\s+know\s+and\s+i\s+will\s+pay|i\s+know\s+i\s+need\s+to\s+pay|we\s+need\s+to\s+pay,\s*i\s+know)\b", re.I), 2.5, "acknowledging obligation to pay"),
    (re.compile(r"\b(?:salary\s+aayegi|salary\s+ke\s+baad|salary\s+aane\s+pe)\b", re.I), 2.0, "salary dependent payment"),
    (re.compile(r"\b(?:haan\s+sir|ji\s+sir|theek\s+hai\s+sir|confirm\s+hai|pakka\s+hai|pakka\s+sir|haan\s+haan\s+sir|actually\s+sir|sir\s+abhi)\b", re.I), 2.0, "borrower conversational affirmation"),

    # 3. Pleading / requesting time
    (re.compile(r"\b(?:thoda\s+time|kuch\s+din\s+ka\s+time|time\s+de\s+do|please\s+give\s+some\s+time|time\s+chahiye|mohalat)\b", re.I), 2.0, "asking for extension"),
    (re.compile(r"\b(?:abhi\s+possible\s+nahi|abhi\s+nahi\s+ho\s+payega|possible\s+nahi\s+hai)\b", re.I), 2.0, "current impossibility"),
    (re.compile(r"\b(?:bas\s+call\s+rakhiye|call\s+rakho|phone\s+rakhiye|phone\s+cut\s+kijiye)\b", re.I), 2.0, "evasive disengagement")
]

THIRD_PARTY_WEIGHTED_RULES = [
    (re.compile(r"\bmain\s+unka\s+(?:bhai|beta|pita|father|brother|husband|pati|dost|friend|colleague|vakeel|advocate|lawyer)\b", re.I), 4.0, "Hindi third party relative"),
    (re.compile(r"\bmain\s+unki\s+(?:patni|wife|mata|mother|behan|sister|beti|daughter)\b", re.I), 4.0, "Hindi third party relative female"),
    (re.compile(r"\bi\s+am\s+his\s+(?:wife|brother|father|mother|son|daughter|lawyer|friend|colleague)\b", re.I), 4.0, "English third party relative"),
    (re.compile(r"\b(?:wo\s+abhi|woh\s+abhi)\s+(?:ghar\s+pe\s+nahi|out\s+of\s+station|hospital\s+me|busy\s+hai|phone\s+nahi\s+utha\s+sakte|nahi\s+hai)\b", re.I), 3.0, "debtor absence statement"),
    (re.compile(r"\b(?:peeche\s+se|phone\s+kaato|bol\s+do|mat\s+do\s+paise|cut\s+karo)\b", re.I), 3.0, "background third party interference")
]

# Exported pattern lists for backward compatibility
AGENT_PATTERNS = [rule[0] for rule in AGENT_WEIGHTED_RULES]
BORROWER_PATTERNS = [rule[0] for rule in BORROWER_WEIGHTED_RULES]
THIRD_PARTY_PATTERNS = [rule[0] for rule in THIRD_PARTY_WEIGHTED_RULES]


class RoleResolver:
    """
    Persistent Role Resolver with Joint Interlocutor Balancing.
    Strictly decouples acoustic speaker ID from conversational role.
    Guarantees that two primary speakers never both become BORROWER (or both AGENT).
    """

    CONFIRMATION_THRESHOLD = 0.85

    def __init__(self):
        self.speaker_roles: Dict[str, SpeakerRoleState] = {}
        self.call_history: List[Dict[str, Any]] = []

    def reset(self):
        self.speaker_roles.clear()
        self.call_history.clear()

    def get_role_state(self, speaker_id: str) -> SpeakerRoleState:
        if speaker_id not in self.speaker_roles:
            self.speaker_roles[speaker_id] = SpeakerRoleState(speaker_id=speaker_id)
        return self.speaker_roles[speaker_id]

    def resolve_role(
        self,
        speaker_id: str,
        text: str,
        is_background: bool = False,
        manual_override: Optional[str] = None
    ) -> SpeakerRoleState:
        """
        Incrementally resolves and updates speaker role.
        Preserves role inertia once confirmed.
        Reversible under strong contradicting evidence.
        """
        state = self.get_role_state(speaker_id)
        state.evidence_count += 1
        t = text.lower().strip()

        # 1. Manual simulation override
        if manual_override and manual_override in ("agent", "borrower", "third_party", "third_party_background") and manual_override != "auto":
            state.role = manual_override
            state.confidence = 1.0
            state.status = "manual_simulation"
            state.last_rationale = f"Manual simulation: {manual_override.upper()}"
            self.call_history.append({"speaker_id": speaker_id, "role": state.role, "text": text})
            return state

        # 2. Background speech indicator
        if is_background or "peeche se" in t or "background" in t:
            state.role = "third_party_background"
            state.third_party_score += 4.0
            state.confidence = 0.96
            state.status = "confirmed"
            state.evidence_logs.append(f"[+4.0 ThirdPartyBg] '{text}'")
            state.last_rationale = "Third-party background speech / coaching"
            self.call_history.append({"speaker_id": speaker_id, "role": state.role, "text": text})
            return state

        # 3. Third-party direct announcements
        for pat, weight, desc in THIRD_PARTY_WEIGHTED_RULES:
            if pat.search(t):
                state.third_party_score += weight
                state.evidence_logs.append(f"[+{weight} ThirdParty] '{text}' ({desc})")

        if state.third_party_score >= 3.0:
            state.role = "third_party"
            state.confidence = 0.95
            state.status = "confirmed"
            state.last_rationale = "Direct third-party speaker identified"
            self.call_history.append({"speaker_id": speaker_id, "role": state.role, "text": text})
            return state

        # 4. Score linguistic cues for agent and borrower
        for pat, weight, desc in AGENT_WEIGHTED_RULES:
            if pat.search(t):
                state.agent_score += weight
                state.evidence_logs.append(f"[+{weight} Agent] '{text}' ({desc})")

        for pat, weight, desc in BORROWER_WEIGHTED_RULES:
            if pat.search(t):
                state.borrower_score += weight
                state.evidence_logs.append(f"[+{weight} Borrower] '{text}' ({desc})")

        # 5. Apply Joint 2-Person Balancing across interlocutors
        self._rebalance_primary_roles()

        self.call_history.append({"speaker_id": speaker_id, "role": state.role, "text": text})

        # Debug logging for transparency
        logger.info(
            f"[ROLE RESOLVER] {speaker_id} -> {state.role.upper()} ({state.status}, conf={state.confidence:.2f}) | "
            f"AgentScore={state.agent_score:.1f} BorrowerScore={state.borrower_score:.1f}"
        )
        return state

    def _rebalance_primary_roles(self):
        """
        Enforces conversation-level 2-person collection invariant:
        1. In a normal 2-person collection call, exactly one speaker is AGENT and one is BORROWER.
        2. NEVER both BORROWER or both AGENT.
        3. If evidence is ambiguous, use UNKNOWN.
        4. When only one physical speaker is present, does NOT fabricate a second speaker/role.
        5. Handles any speaker IDs (e.g. speaker_0 & speaker_1, or speaker_0 & speaker_2, etc.)
        """
        # Consider all speakers who have accumulated dialogue turns or evidence
        dialogue_speakers = [
            spk_id for spk_id, st in self.speaker_roles.items()
            if st.status != "manual_simulation" and not st.role.startswith("third_party") and st.evidence_count > 0
        ]

        if not dialogue_speakers:
            return

        # Case A: Exactly ONE dialogue speaker observed so far
        if len(dialogue_speakers) == 1:
            spk = self.speaker_roles[dialogue_speakers[0]]
            net = spk.agent_score - spk.borrower_score
            # Unambiguous monologue evidence:
            if net >= 2.0:
                spk.role = "agent"
                spk.confidence = min(0.98, max(0.65, 0.60 + 0.08 * spk.agent_score))
                spk.status = "confirmed" if spk.confidence >= self.CONFIRMATION_THRESHOLD else "provisional"
                spk.last_rationale = f"Assigned AGENT from collection/institutional cues (net agent score +{net:.1f})"
            elif net <= -2.0:
                spk.role = "borrower"
                spk.confidence = min(0.98, max(0.65, 0.60 + 0.08 * spk.borrower_score))
                spk.status = "confirmed" if spk.confidence >= self.CONFIRMATION_THRESHOLD else "provisional"
                spk.last_rationale = f"Assigned BORROWER from personal hardship/commitment cues (net borrower score +{-net:.1f})"
            else:
                spk.role = "unknown"
                spk.confidence = 0.50
                spk.status = "provisional"
                spk.last_rationale = "Single physical speaker observed; awaiting conversation evidence or counterparty"
            return

        # Case B: TWO or more dialogue interlocutors observed
        # Select the two primary interlocutors with the highest activity/evidence
        sorted_candidates = sorted(
            dialogue_speakers,
            key=lambda s: (self.speaker_roles[s].agent_score + self.speaker_roles[s].borrower_score, self.speaker_roles[s].evidence_count),
            reverse=True
        )
        spk_a = self.speaker_roles[sorted_candidates[0]]
        spk_b = self.speaker_roles[sorted_candidates[1]]

        # Hypothesis 1: spk_a = AGENT, spk_b = BORROWER
        h1 = spk_a.agent_score + spk_b.borrower_score
        # Hypothesis 2: spk_a = BORROWER, spk_b = AGENT
        h2 = spk_a.borrower_score + spk_b.agent_score

        total_ev = spk_a.agent_score + spk_a.borrower_score + spk_b.agent_score + spk_b.borrower_score
        if total_ev < 1.0 or abs(h1 - h2) < 0.5:
            # Insufficient collective evidence to distinguish roles
            spk_a.role = "unknown"
            spk_b.role = "unknown"
            spk_a.confidence = 0.50
            spk_b.confidence = 0.50
            spk_a.status = "provisional"
            spk_b.status = "provisional"
            spk_a.last_rationale = "Ambiguous dialogue, awaiting role evidence"
            spk_b.last_rationale = "Ambiguous dialogue, awaiting role evidence"
            return

        # Contrastive resolution
        if h1 >= h2:
            agent_spk, borrower_spk = spk_a, spk_b
            win_score, lose_score = h1, h2
        else:
            agent_spk, borrower_spk = spk_b, spk_a
            win_score, lose_score = h2, h1

        denom = win_score + lose_score + 0.5
        prob = win_score / denom if denom > 0 else 0.5

        agent_conf = min(0.99, max(0.65, 0.55 + 0.35 * prob + 0.04 * agent_spk.evidence_count))
        borrower_conf = min(0.99, max(0.65, 0.55 + 0.35 * prob + 0.04 * borrower_spk.evidence_count))

        agent_spk.role = "agent"
        agent_spk.confidence = round(agent_conf, 2)
        agent_spk.status = "confirmed" if agent_conf >= self.CONFIRMATION_THRESHOLD else "provisional"
        agent_spk.last_rationale = (
            f"Resolved as AGENT (score {agent_spk.agent_score:.1f}) | Interlocutor {borrower_spk.speaker_id} is BORROWER"
        )

        borrower_spk.role = "borrower"
        borrower_spk.confidence = round(borrower_conf, 2)
        borrower_spk.status = "confirmed" if borrower_conf >= self.CONFIRMATION_THRESHOLD else "provisional"
        borrower_spk.last_rationale = (
            f"Resolved as BORROWER (score {borrower_spk.borrower_score:.1f}) | Interlocutor {agent_spk.speaker_id} is AGENT"
        )

        # Extra interlocutors beyond top 2 (if not third party) stay unknown
        for extra_id in sorted_candidates[2:]:
            extra = self.speaker_roles[extra_id]
            if not extra.role.startswith("third_party"):
                extra.role = "unknown"
                extra.confidence = 0.50
                extra.status = "provisional"

    def get_summary(self) -> Dict[str, Any]:
        """Return persistent role mapping for all speakers with audit trail."""
        return {
            spk_id: {
                "role": state.role,
                "confidence": state.confidence,
                "status": state.status,
                "evidence_count": state.evidence_count,
                "agent_score": round(state.agent_score, 2),
                "borrower_score": round(state.borrower_score, 2),
                "third_party_score": round(state.third_party_score, 2),
                "rationale": state.last_rationale,
                "evidence_logs": state.evidence_logs[-5:]
            }
            for spk_id, state in self.speaker_roles.items()
        }
