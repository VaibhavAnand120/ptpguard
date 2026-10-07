"""
Dynamic PTP Deadline Tracking Engine
Tracks:
- agent_proposed_deadline
- borrower_stated_deadline
- borrower_confirmed_deadline
- dynamic deadline shifts (postponements, acceleration, clarifications)
- context-aware, non-arbitrary deadline change penalties
- fairness safeguards: hardship is not dishonesty
- agent pressure context tracking
"""

import re
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field

from ..schemas import DeadlineEvent, DeadlineState


MONTH_OFFSETS = {
    "jan": 0, "january": 0,
    "feb": 31, "february": 31,
    "mar": 59, "march": 59,
    "apr": 90, "april": 90,
    "may": 120,
    "jun": 151, "june": 151,
    "jul": 181, "july": 181,
    "aug": 212, "august": 212,
    "sep": 243, "september": 243,
    "oct": 273, "october": 273,
    "nov": 304, "november": 304,
    "dec": 334, "december": 334,
}

PAYMENT_COMMITMENT_CUES = [
    re.compile(r"\b(?:pay\s+by|pay\s+before|pay\s+on|pay\s+on\s+the|will\s+pay|can\s+pay|i\'?ll\s+pay)\b", re.I),
    re.compile(r"\b(?:de\s+dunga|kar\s+dunga|clear\s+kar|transfer\s+kar|bhej\s+dunga|jama\s+kar)\b", re.I),
    re.compile(r"\b(?:by\s+\d+|on\s+\d+|before\s+\d+|\d+\s+tareekh\s+ko|tareekh\s+tak)\b", re.I),
    re.compile(r"\b(?:yes|haan|ji|okay|theek\s+hai|sure|confirm)\b.*?\b(?:\d+|today|tomorrow|october|november|december|tareekh)\b", re.I),
    re.compile(r"\b(?:i\'?ll\s+say|say\s+\d+|make\s+it\s+\d+|can\s+do\s+\d+)\b", re.I),
]

NON_PAYMENT_DATE_CUES = [
    re.compile(r"\b(?:bought|purchased|started|bounced|taken\s+on|statement\s+of|last\s+month|previous\s+month|called\s+on|applied\s+on)\b", re.I),
    re.compile(r"\b(?:birthday|anniversary|joined|meeting\s+on)\b", re.I),
]

AGENT_PRESSURE_PHRASES = [
    re.compile(r"\b(?:earlier|pehle|cannot\s+accept|not\s+possible|too\s+late|need\s+to\s+pay\s+earlier|pay\s+earlier)\b", re.I),
    re.compile(r"\b(?:today\s+itself|aaj\s+hi|kal\s+hi|immediately|urgent|no,\s*you\s+need)\b", re.I),
    re.compile(r"\b(?:can\s+you\s+pay\s+earlier|earlier\s+possible|can\s+you\s+do\s+earlier)\b", re.I),
]

AGENT_ALTERNATIVE_DATE_PROPOSALS = [
    re.compile(r"\b(?:can\s+you\s+pay\s+on\s+(.+?)\s+instead|what\s+about\s+(.+?)\?|pay\s+on\s+(.+?)\s+instead)\b", re.I),
]


def parse_deadline_day(date_str: str) -> Optional[int]:
    """
    Parses a conversational date expression into an integer day offset (day of year or relative index).
    Enables precise difference calculations: e.g. "20 October" - "10 October" = +10 days.
    """
    if not date_str:
        return None

    clean = date_str.lower().strip()

    if "today" in clean:
        return 273 + 1  # Base reference
    if "tomorrow" in clean:
        return 273 + 2

    # Match Month + Day: e.g. "10 October", "October 10", "20th Oct"
    m_match = re.search(
        r"(?:(?:before|by|on|after)\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)",
        clean
    )
    if m_match:
        day_num = int(m_match.group(1))
        month_name = m_match.group(2)
        for m_prefix, offset in MONTH_OFFSETS.items():
            if month_name.startswith(m_prefix):
                return offset + day_num

    m_match_rev = re.search(
        r"([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?",
        clean
    )
    if m_match_rev:
        month_name = m_match_rev.group(1)
        day_num = int(m_match_rev.group(2))
        for m_prefix, offset in MONTH_OFFSETS.items():
            if month_name.startswith(m_prefix):
                return offset + day_num

    # Match Day only: e.g. "10th", "10 tareekh", "20th"
    d_match = re.search(r"(\d{1,2})(?:st|nd|rd|th|\s+tareekh)?", clean)
    if d_match:
        day_num = int(d_match.group(1))
        # Default assume October base for standard month comparisons
        return 273 + day_num

    return None


def calculate_days_difference(prev_date: str, new_date: str) -> Optional[int]:
    """Compute difference in days: positive = postponed, negative = accelerated."""
    p1 = parse_deadline_day(prev_date)
    p2 = parse_deadline_day(new_date)
    if p1 is not None and p2 is not None:
        return p2 - p1
    return None


CANDIDATE_DATE_RE = re.compile(
    r"\b(?:(?:before|by|on|after)\s+)?(\d{1,2}(?:st|nd|rd|th)?\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?))\b|"
    r"\b((?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?)\b|"
    r"\b(\d{1,2}\s+tareekh|\d{1,2}(?:st|nd|rd|th))\b",
    re.I
)

NEGATION_CUES = re.compile(r"\b(?:can\'?t\s+pay|cannot\s+pay|not\s+pay|not\s+on|won\'?t\s+pay|nahi\s+kar|nahi\s+de|possible\s+nahi)\b", re.I)
HISTORICAL_NON_PAYMENT_CUES = re.compile(r"\b(?:bought|purchased|started|bounced|taken\s+on|birthday|anniversary|statement|called\s+on|applied\s+on)\b", re.I)
FORWARD_COMMITMENT_CUES = re.compile(r"\b(?:can\s+pay|will\s+pay|pay\s+by|pay\s+on|i\'?ll\s+pay|kar\s+dunga|de\s+dunga|clear|yes|haan|confirm|pakka|i\'?ll\s+say|say)\b", re.I)


def get_clause_around_span(text: str, start: int, end: int) -> str:
    """Finds the local clause bounded by punctuation or coordinating conjunctions."""
    delims = ['.', ',', ';', ' but ', ' so ']
    clause_start = 0
    for delim in delims:
        idx = text.rfind(delim, 0, start)
        if idx != -1:
            clause_start = max(clause_start, idx + len(delim))

    clause_end = len(text)
    for delim in delims:
        idx = text.find(delim, end)
        if idx != -1:
            clause_end = min(clause_end, idx)

    return text[clause_start:clause_end].strip().lower()


def extract_payment_deadline_from_text(text: str, speaker_role: str) -> Optional[str]:
    """
    Extracts the genuine forward payment deadline from text, properly resolving:
    - Multiple dates (e.g., historical loan date vs future payment date)
    - Negated dates (e.g., 'can't pay on 10 Oct, can pay on 20 Oct')
    - Non-payment incidental dates (e.g., 'bought phone on 10 Oct')
    """
    matches = list(CANDIDATE_DATE_RE.finditer(text))
    if not matches:
        return None

    candidates = []
    for m in matches:
        date_str = (m.group(1) or m.group(2) or m.group(3)).strip()
        clause = get_clause_around_span(text, m.start(), m.end())

        is_negated = bool(NEGATION_CUES.search(clause))
        is_non_payment = bool(HISTORICAL_NON_PAYMENT_CUES.search(clause))
        is_positive = bool(FORWARD_COMMITMENT_CUES.search(clause))
        candidates.append((date_str, is_negated, is_non_payment, is_positive))

    # 1. High priority: Positive forward commitment candidate that is neither negated nor historical
    for d, is_neg, is_non, is_pos in candidates:
        if is_pos and not is_neg and not is_non:
            return d

    # 2. Medium priority: Candidate that is neither negated nor historical
    for d, is_neg, is_non, is_pos in candidates:
        if not is_neg and not is_non:
            return d

    return None


def is_payment_commitment_date(text: str, date_str: str, speaker_role: str) -> bool:
    """
    Validates whether a candidate date represents a forward payment deadline commitment.
    Rejects historical, non-payment, or incidental dates.
    """
    t = text.lower().strip()

    # Re-extract with full clause context
    best = extract_payment_deadline_from_text(text, speaker_role)
    if best is not None:
        return True

    # If spoken by borrower, check if it expresses payment commitment intent
    if speaker_role == "borrower":
        has_payment_cue = any(p.search(t) for p in PAYMENT_COMMITMENT_CUES)
        has_short_confirm = (
            len(t.split()) <= 6
            and any(w in t for w in ["yes", "haan", "okay", "i will", "i can", "sure", "pakka"])
            and any(char.isdigit() for char in t)
        )
        return has_payment_cue or has_short_confirm

    # If spoken by agent
    if speaker_role == "agent":
        return any(w in t for w in ["pay", "mark", "clear", "possible", "kab", "can you", "link"])

    return True



def deadline_change_penalty(
    original_date: str,
    previous_date: str,
    new_date: str,
    days_diff: Optional[int],
    pressure_detected: bool,
    borrower_initiated_change: bool,
    hardship_present: bool = False,
    postpone_count: int = 1
) -> Tuple[float, int, str]:
    """
    Calculates context-aware, non-arbitrary credibility penalty for deadline shift.
    
    Formula:
    - Postponement (days_diff > 0):
        base_raw = min(3.5, 0.5 + 0.15 * days_diff)
        recurrence_multiplier = 1.0 + 0.45 * max(0, postpone_count - 1)
        hardship_discount = 0.5 if hardship_present else 1.0 (FAIRNESS RULE)
        penalty_raw = base_raw * recurrence_multiplier * hardship_discount
        penalty_pct = round(penalty_raw * 6.5)
    - Accelerated (days_diff < 0):
        Moving date earlier indicates high commitment (0 penalty, or positive boost)
    - Clarification (days_diff == 0):
        0 penalty
    """
    diff = days_diff if days_diff is not None else 0

    if diff > 0:
        # Postponement into the future
        base_raw = min(3.0, 0.40 + 0.14 * diff)
        recurrence_mult = 1.0 + 0.40 * max(0, postpone_count - 1)
        
        # Fairness safeguard: Temper penalty if legitimate financial hardship forced the change
        hardship_mult = 0.50 if hardship_present else 1.0
        
        penalty_raw = round(base_raw * recurrence_mult * hardship_mult, 2)
        penalty_pct = min(35, int(round(penalty_raw * 7.5)))

        rationale_parts = [f"Deadline postponed by +{diff} days"]
        if postpone_count > 1:
            rationale_parts.append(f"repeated postponement #{postpone_count}")
        if hardship_present:
            rationale_parts.append("penalty tempered due to documented hardship")
        if pressure_detected:
            rationale_parts.append("change followed agent discussion")

        rationale = "; ".join(rationale_parts)
        return penalty_raw, penalty_pct, rationale

    elif diff < 0:
        # Accelerated: borrower moves deadline earlier!
        rationale = f"Deadline moved earlier by {abs(diff)} days (commitment positive)"
        return 0.0, 0, rationale

    else:
        # Same day / clarification
        return 0.0, 0, "Deadline clarification without calendar shift"


class DeadlineTracker:
    """
    Maintains dynamic state and history of all PTP deadlines during the call.
    """

    def __init__(self):
        self.state = DeadlineState()

    def reset(self):
        self.state = DeadlineState()

    def process_utterance(
        self,
        text: str,
        speaker_role: str,
        speaker_id: str,
        date_extracted: Optional[str],
        timestamp: str,
        history: Optional[List[Dict[str, Any]]] = None,
        hardship_active: bool = False
    ) -> Optional[DeadlineEvent]:
        """
        Processes an utterance to update deadline state.
        Ensures strict borrower-only confirmation rules and tracks agent pressure.
        """
        # Resolve target date expression using full clause context
        clause_date = extract_payment_deadline_from_text(text, speaker_role)
        if clause_date:
            target_date = clause_date
        elif CANDIDATE_DATE_RE.search(text):
            # Candidate dates were present but all were rejected as non-payment/negated
            return None
        else:
            target_date = date_extracted

        if not target_date:
            return None

        # Check if the mentioned date is actually a payment commitment date
        if not is_payment_commitment_date(text, target_date, speaker_role):
            return None

        date_extracted = target_date

        t = text.lower().strip()
        s = self.state

        # Check if previous agent turn exerted pressure
        agent_pressured_recent = False
        agent_proposed_recently = False
        recent_agent_date = None

        if history and len(history) > 0:
            for past in reversed(history[-3:]):
                if past.get("role") == "agent" or past.get("speaker") == "agent":
                    past_t = past.get("text", "").lower()
                    if any(p.search(past_t) for p in AGENT_PRESSURE_PHRASES):
                        agent_pressured_recent = True
                    for prop_re in AGENT_ALTERNATIVE_DATE_PROPOSALS:
                        if prop_re.search(past_t):
                            agent_proposed_recently = True
                    break

        # -------------------------------------------------------------
        # CASE 1: AGENT SPEECH
        # -------------------------------------------------------------
        if speaker_role == "agent":
            s.agent_proposed_deadline = date_extracted
            # Does NOT set borrower deadline!
            event = DeadlineEvent(
                timestamp=timestamp,
                speaker="agent",
                speaker_id=speaker_id,
                deadline=date_extracted,
                days_diff=None,
                event_type="agent_proposed",
                credibility_penalty_pct=0,
                penalty_raw=0.0,
                rationale=f"Agent proposed payment deadline ({date_extracted})",
                agent_pressure=agent_pressured_recent,
                hardship_present=hardship_active,
                borrower_initiated=False
            )
            s.history.append(event)
            return event

        # -------------------------------------------------------------
        # CASE 2: BORROWER SPEECH
        # -------------------------------------------------------------
        if speaker_role == "borrower":
            # Check if borrower is confirming an agent-proposed date
            is_confirming_agent = (
                s.agent_proposed_deadline is not None
                and parse_deadline_day(s.agent_proposed_deadline) == parse_deadline_day(date_extracted)
                and any(w in t for w in ["yes", "haan", "okay", "confirm", "ji", "can pay"])
            )

            if is_confirming_agent:
                s.borrower_confirmed_deadline = date_extracted

            s.borrower_stated_deadline = date_extracted

            # First deadline stated by borrower
            if not s.current_deadline:
                s.original_deadline = date_extracted
                s.current_deadline = date_extracted
                s.confidence = 0.94

                event = DeadlineEvent(
                    timestamp=timestamp,
                    speaker="borrower",
                    speaker_id=speaker_id,
                    deadline=date_extracted,
                    days_diff=0,
                    event_type="initial" if not is_confirming_agent else "confirmed_agent_proposal",
                    credibility_penalty_pct=0,
                    penalty_raw=0.0,
                    rationale=f"Initial borrower PTP deadline established: {date_extracted}",
                    agent_pressure=agent_pressured_recent,
                    hardship_present=hardship_active,
                    borrower_initiated=not is_confirming_agent
                )
                s.history.append(event)
                return event

            # Deadline update / change!
            prev_date = s.current_deadline
            days_diff = calculate_days_difference(prev_date, date_extracted)
            if days_diff is None:
                days_diff = 0

            # If same date, treat as reassurance / clarification
            if days_diff == 0 and prev_date.lower() == date_extracted.lower():
                s.confidence = min(0.99, s.confidence + 0.02)
                return None

            # Date actually shifted!
            if days_diff > 0:
                s.postpone_count += 1
                event_type = "postponed"
            elif days_diff < 0:
                event_type = "accelerated"
            else:
                event_type = "clarification"

            # Calculate dynamic penalty
            penalty_raw, penalty_pct, rationale = deadline_change_penalty(
                original_date=s.original_deadline or prev_date,
                previous_date=prev_date,
                new_date=date_extracted,
                days_diff=days_diff,
                pressure_detected=agent_pressured_recent,
                borrower_initiated_change=not is_confirming_agent,
                hardship_present=hardship_active,
                postpone_count=s.postpone_count
            )

            # Update state
            s.previous_deadline = prev_date
            s.current_deadline = date_extracted
            s.days_diff = days_diff
            s.last_penalty_pct = penalty_pct
            s.confidence = 0.92

            event = DeadlineEvent(
                timestamp=timestamp,
                speaker="borrower",
                speaker_id=speaker_id,
                deadline=date_extracted,
                days_diff=days_diff,
                event_type=event_type,
                credibility_penalty_pct=penalty_pct,
                penalty_raw=penalty_raw,
                rationale=rationale,
                agent_pressure=agent_pressured_recent,
                hardship_present=hardship_active,
                borrower_initiated=not is_confirming_agent
            )
            s.history.append(event)
            return event

        return None
