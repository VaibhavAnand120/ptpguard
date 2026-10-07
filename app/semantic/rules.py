import re
from typing import Optional, List, Dict, Any
from .base import SemanticAnalyzer
from .diarization import SpeakerDiarizer
from .deadline import extract_payment_deadline_from_text, CANDIDATE_DATE_RE
from ..schemas import Evidence, Signal

AMOUNT_RE = re.compile(r"(?:₹|rs\.?|inr)?\s*([0-9]{2,7})(?:\s*(?:rs|rupees))?", re.I)

DATE_RE = re.compile(
    r"\b(?:(?:before|by|on|after)\s+)?(\d{1,2}(?:st|nd|rd|th)?\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?))\b|"
    r"\b(\d{1,2}\s+tareekh|\d{1,2}(?:st|nd|rd|th)|today|tomorrow|month\s+end|next\s+week|salary\s+ke\s+baad)\b",
    re.I
)


class RulesAnalyzer(SemanticAnalyzer):
    """
    Deterministic rule-based semantic analyzer producing directional evidence signals
    across all 10 PTP credibility dimensions.

    Separates:
    - agent_proposed_date / agent_proposed_amount
    - borrower_stated_date / borrower_stated_amount
    Ensures agent questions or unilateral statements do NOT create borrower commitment.
    """

    async def analyze(
        self,
        speaker: str,
        text: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Evidence:
        raw_text = text.strip()
        t = raw_text.lower()

        # Step 1: Speaker Diarization Resolution (only if speaker is 'auto' or empty)
        if speaker and speaker not in ("auto", ""):
            effective_speaker = speaker
            diar_speaker_id = "speaker_0" if speaker == "agent" else ("speaker_2" if "third_party" in speaker else "speaker_1")
            diar_conf = 0.96
            diar_rationale = f"Role: {speaker.upper()}"
            has_bg = "third_party_background" in speaker or "peeche se" in t or "background" in t
            bg_transcript = None
            bg_coaching = False
            clean_text = raw_text
        else:
            diarization = SpeakerDiarizer.detect(text, hint_speaker="auto", history=history)
            effective_speaker = diarization.primary_speaker
            diar_speaker_id = diarization.speaker_id
            diar_conf = diarization.confidence
            diar_rationale = diarization.rationale
            has_bg = diarization.has_background_speech
            bg_transcript = diarization.background_transcript
            bg_coaching = diarization.background_coaching
            clean_text = diarization.clean_text or raw_text

        clean_lower = clean_text.lower()

        # Step 2: Extract amounts and dates
        amount_match = AMOUNT_RE.search(clean_lower)
        amount = float(amount_match.group(1)) if amount_match else None

        date = extract_payment_deadline_from_text(clean_text, effective_speaker)
        if not date and not CANDIDATE_DATE_RE.search(clean_text):
            date_words = [
                "today", "tomorrow", "10 tareekh", "15 tareekh", "month end",
                "salary ke baad", "next week"
            ]
            date = next((d for d in date_words if d in clean_lower), None)

        # Step 3: Separate Agent Proposed vs Borrower Stated Entities
        agent_proposed_date = None
        agent_proposed_amount = None
        borrower_stated_date = None
        borrower_stated_amount = None

        if effective_speaker == "agent":
            agent_proposed_amount = amount
            agent_proposed_date = date
        elif effective_speaker == "borrower":
            borrower_stated_amount = amount
            borrower_stated_date = date

        signals: Dict[str, Signal] = {}

        # 1. COMMITMENT (Borrower-side only)
        commitment_positive_words = [
            "kar dunga", "de dunga", "pay karunga", "pay kar dunga", "transfer karunga",
            "transfer kar dunga", "bhej dunga", "jama kar dunga", "i will pay",
            "i will transfer", "definitely pay", "pakka de dunga", "clear kar dunga",
            "i can pay", "can pay", "yes, i can pay", "yes i can pay", "will pay",
            "i can pay before", "yes, i can"
        ]
        commitment_negative_words = [
            "nahi karunga", "nahi de sakta", "nahi ho payega", "pay nahi kar sakta",
            "cannot pay", "won't pay", "nahi de paunga", "don't have money", "no money"
        ]

        # Check if borrower statement is an independent commitment or merely passive "okay"
        is_passive_okay = clean_lower in ["okay", "ok", "ji", "theek hai", "hmm"]
        recent_agent_push = False
        if history and len(history) > 0:
            last_turn = history[-1]
            last_text = last_turn.get("text", "").lower()
            if any(w in last_text for w in ["mark", "note", "correct?", "right?"]):
                recent_agent_push = True

        if effective_speaker == "borrower":
            if any(w in clean_lower for w in commitment_positive_words):
                signals["commitment"] = Signal(
                    direction=1, strength=2, confidence=0.94,
                    rationale="Borrower stated explicit commitment to pay"
                )
            elif any(w in clean_lower for w in commitment_negative_words) and not any(w in clean_lower for w in ["yes, i can pay", "i can pay"]):
                # If only stating inability without commitment
                pass
            elif is_passive_okay and recent_agent_push:
                # Passive okay to an agent push is NOT strong borrower commitment
                pass

        # 2. SPECIFICITY (Borrower-side concrete date/amount)
        has_concrete_amount = borrower_stated_amount is not None
        has_concrete_date = borrower_stated_date is not None and "next week" not in borrower_stated_date
        is_vague = any(w in clean_lower for w in ["next week", "soon", "later", "baad me", "thoda", "some amount", "kuch din"])

        if effective_speaker == "borrower":
            if has_concrete_amount and has_concrete_date:
                signals["specificity"] = Signal(
                    direction=1, strength=3, confidence=0.96,
                    rationale=f"Borrower stated concrete amount (₹{borrower_stated_amount}) and date ({borrower_stated_date})"
                )
            elif has_concrete_amount or has_concrete_date:
                signals["specificity"] = Signal(
                    direction=1, strength=2, confidence=0.92,
                    rationale=f"Borrower confirmed concrete payment term: {borrower_stated_date or borrower_stated_amount}"
                )
            elif is_vague:
                signals["specificity"] = Signal(
                    direction=-1, strength=2, confidence=0.88,
                    rationale="Vague timeline or amount provided"
                )

        # 3. BORROWER INITIATION vs AGENT PRESSURE
        agent_push_words = [
            "right sir?", "correct sir?", "kar denge, correct", "mark kar raha hoon",
            "confirm kijiye", "i will mark", "i'll mark", "mark kar doon", "note kar loon"
        ]

        if effective_speaker == "borrower" and (any(w in clean_lower for w in commitment_positive_words) or borrower_stated_amount is not None):
            signals["borrower_initiation"] = Signal(
                direction=1, strength=2, confidence=0.90,
                rationale="Borrower voluntarily proposed payment details"
            )
        elif effective_speaker == "agent":
            if any(w in clean_lower for w in agent_push_words) or ("will mark" in clean_lower or "i'll mark" in clean_lower):
                signals["agent_pressure"] = Signal(
                    direction=-1, strength=2, confidence=0.92,
                    rationale="Agent unilaterally recording or pushing payment terms"
                )
                signals["borrower_initiation"] = Signal(
                    direction=-1, strength=2, confidence=0.90,
                    rationale="Agent driving payment terms, not borrower initiated"
                )
            elif amount is not None or date is not None:
                signals["borrower_initiation"] = Signal(
                    direction=-1, strength=1, confidence=0.85,
                    rationale="Agent proposing payment terms"
                )

        # 4. CONFIRMATION
        confirmation_words = [
            "confirmed", "confirm hai", "confirm", "pakka", "sure", "definitely",
            "haan sir, confirm", "ji sir", "yes, i can", "yes i can", "yes, i can pay",
            "yes, i will", "yes"
        ]
        if effective_speaker == "borrower":
            if any(w in clean_lower for w in confirmation_words):
                signals["confirmation"] = Signal(
                    direction=1, strength=2, confidence=0.94,
                    rationale="Explicit borrower confirmation"
                )
            elif any(w in clean_lower for w in ["pata nahi", "shayad", "not sure", "dekhunga"]):
                signals["confirmation"] = Signal(
                    direction=-1, strength=2, confidence=0.88,
                    rationale="Borrower avoided explicit confirmation"
                )

        # 5. FEASIBILITY
        feasibility_positive_words = [
            "salary aa gayi", "salary has come", "already credited", "salary credited",
            "paise arrange ho gaye", "fund available", "paise aa gaye"
        ]
        feasibility_negative_words = [
            "paise nahi", "no money", "paisa nahi hai", "salary nahi aayi",
            "salary late", "loss", "hospital", "not received any salary", "empty",
            "difficult right now", "quite difficult", "don't have money", "bank account is empty"
        ]
        if any(w in clean_lower for w in feasibility_positive_words):
            signals["feasibility"] = Signal(
                direction=1, strength=2, confidence=0.92,
                rationale="Evidence borrower has funds / salary credited"
            )
        elif any(w in clean_lower for w in feasibility_negative_words):
            signals["feasibility"] = Signal(
                direction=-1, strength=2, confidence=0.90,
                rationale="Financial strain / lack of funds currently"
            )

        # 6. CONDITIONALITY
        conditional_words = [
            "if", "agar", "salary aayi toh", "salary aaye", "salary aayegi toh",
            "koshish", "try karunga", "dekhta hoon", "maybe"
        ]
        condition_cleared_words = [
            "salary aa gayi hai already", "salary aa gayi", "salary credited",
            "already credited", "without condition", "unconditional", "confirm hai",
            "salary has already come", "salary has come", "already come", "paise aa gaye",
            "funds ready", "account me paise"
        ]
        if any(w in clean_lower for w in condition_cleared_words):
            signals["conditionality"] = Signal(
                direction=1, strength=2, confidence=0.92,
                rationale="Condition cleared / unconditional commitment"
            )
        elif any(w in clean_lower for w in conditional_words):
            signals["conditionality"] = Signal(
                direction=-1, strength=2, confidence=0.92,
                rationale="Conditional promise depending on future event"
            )

        # 7. HARDSHIP
        hardship_words = [
            "paise nahi", "money nahi", "no money", "financial problem", "dikkat",
            "hospital", "bimari", "tabiyat kharab", "loss in business", "job chali gayi",
            "lost my job", "job loss", "lost job", "lost work", "unemployed",
            "difficult", "quite difficult", "not received any salary", "not received salary",
            "bank account is empty", "account is empty", "empty", "don't have money right now",
            "don't have money"
        ]
        hardship_resolved_words = [
            "salary aa gayi", "sab theek hai", "paise arrange ho gaye", "no problem now"
        ]
        if any(w in clean_lower for w in hardship_resolved_words):
            signals["hardship"] = Signal(
                direction=1, strength=2, confidence=0.90,
                rationale="Financial hardship resolved / funds cleared"
            )
        elif any(w in clean_lower for w in hardship_words):
            signals["hardship"] = Signal(
                direction=-1, strength=2, confidence=0.94,
                rationale="Financial hardship indicators present"
            )

        # 8. THIRD PARTY
        is_third_party = (
            effective_speaker in ("third_party", "third_party_background")
            or has_bg
            or bg_coaching
        )
        borrower_personal_words = ["main khud", "mera loan", "main personal", "i will pay myself"]
        if is_third_party:
            signals["third_party"] = Signal(
                direction=-1, strength=3, confidence=0.96,
                rationale="Third party or background family speaking/coaching"
            )
        elif any(w in clean_lower for w in borrower_personal_words) and effective_speaker == "borrower":
            signals["third_party"] = Signal(
                direction=1, strength=2, confidence=0.90,
                rationale="Borrower confirming personal accountability"
            )

        # 9. ESCAPE SIGNAL
        escape_words = ["bas call rakhiye", "call rakho", "phone cut", "end the call", "haan haan kar dunga"]
        if any(w in clean_lower for w in escape_words):
            signals["escape_signal"] = Signal(
                direction=-1, strength=3, confidence=0.94,
                rationale="Evasive / call termination language detected"
            )
        elif any(w in clean_lower for w in commitment_positive_words) and not any(w in clean_lower for w in escape_words):
            signals["escape_signal"] = Signal(
                direction=1, strength=1, confidence=0.85,
                rationale="Constructive engagement without escape signals"
            )

        evidence_list = [raw_text]
        if bg_transcript:
            evidence_list.append(f"Background: {bg_transcript}")

        ptp_detected = "commitment" in signals and signals["commitment"].direction > 0
        borrower_initiated = "borrower_initiation" in signals and signals["borrower_initiation"].direction > 0
        explicit_conf = "confirmation" in signals and signals["confirmation"].direction > 0
        cond = "conditionality" in signals and signals["conditionality"].direction < 0
        hard = "hardship" in signals and signals["hardship"].direction < 0
        agent_pushed = "agent_pressure" in signals and signals["agent_pressure"].direction < 0
        tp = "third_party" in signals and signals["third_party"].direction < 0
        escape = "escape_signal" in signals and signals["escape_signal"].direction < 0

        return Evidence(
            signals=signals,
            amount=borrower_stated_amount or agent_proposed_amount,
            date=borrower_stated_date or agent_proposed_date,
            agent_proposed_date=agent_proposed_date,
            agent_proposed_amount=agent_proposed_amount,
            borrower_stated_date=borrower_stated_date,
            borrower_stated_amount=borrower_stated_amount,
            speaker_id=diar_speaker_id,
            speaker_role=effective_speaker,
            detected_speaker=effective_speaker,
            speaker_confidence=diar_conf,
            speaker_rationale=diar_rationale,
            third_party_background=has_bg,
            background_coaching=bg_coaching,
            background_details=bg_transcript,
            evidence_text=evidence_list,
            ptp_detected=ptp_detected,
            amount_concrete=has_concrete_amount,
            date_concrete=has_concrete_date,
            borrower_initiated=borrower_initiated,
            explicit_confirmation=explicit_conf,
            conditional=cond,
            hardship=hard,
            agent_pushed=agent_pushed,
            third_party=tp,
            escape_language=escape,
            vague_date="next week" in clean_lower,
            vague_amount="thoda" in clean_lower or "some amount" in clean_lower,
        )
