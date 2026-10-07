import re
from typing import Optional, List, Dict, Any
from .base import SemanticAnalyzer
from .diarization import SpeakerDiarizer
from ..schemas import Evidence, Signal

AMOUNT_RE = re.compile(r"(?:₹|rs\.?|inr)?\s*([0-9]{2,7})(?:\s*(?:rs|rupees))?", re.I)

class RulesAnalyzer(SemanticAnalyzer):
    """
    Deterministic rule-based semantic analyzer producing directional evidence signals
    across all 10 PTP credibility dimensions.
    """

    async def analyze(
        self,
        speaker: str,
        text: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Evidence:
        raw_text = text.strip()
        t = raw_text.lower()

        # Step 1: Run Speaker Diarization
        diarization = SpeakerDiarizer.detect(text, hint_speaker=speaker, history=history)
        effective_speaker = diarization.primary_speaker
        clean_text = diarization.clean_text or raw_text
        clean_lower = clean_text.lower()

        # Step 2: Extract amounts and dates
        amount_match = AMOUNT_RE.search(clean_lower)
        amount = float(amount_match.group(1)) if amount_match else None

        date_words = [
            "today", "tomorrow", "10th", "11th", "12th", "13th", "14th",
            "15th", "16th", "17th", "18th", "19th", "20th", "21st",
            "22nd", "23rd", "24th", "25th", "26th", "27th", "28th",
            "29th", "30th", "31st", "10 tareekh", "15 tareekh", "month end",
            "salary ke baad", "next week"
        ]
        date = next((d for d in date_words if d in clean_lower), None)

        signals: Dict[str, Signal] = {}

        # 1. COMMITMENT
        commitment_positive_words = [
            "kar dunga", "de dunga", "pay karunga", "pay kar dunga", "transfer karunga",
            "transfer kar dunga", "bhej dunga", "jama kar dunga", "i will pay",
            "i will transfer", "definitely pay", "pakka de dunga", "clear kar dunga"
        ]
        commitment_negative_words = [
            "nahi karunga", "nahi de sakta", "nahi ho payega", "pay nahi kar sakta",
            "cannot pay", "won't pay", "nahi de paunga"
        ]
        if any(w in clean_lower for w in commitment_positive_words) and effective_speaker == "borrower":
            signals["commitment"] = Signal(direction=1, strength=2, confidence=0.92, rationale="Borrower stated commitment to pay")
        elif any(w in clean_lower for w in commitment_negative_words) and effective_speaker == "borrower":
            signals["commitment"] = Signal(direction=-1, strength=3, confidence=0.95, rationale="Borrower refused or stated inability to commit")

        # 2. SPECIFICITY
        has_concrete_amount = amount is not None
        has_concrete_date = date is not None and "next week" not in date
        is_vague = any(w in clean_lower for w in ["next week", "soon", "later", "baad me", "thoda", "some amount", "kuch din"])

        if has_concrete_amount and has_concrete_date:
            signals["specificity"] = Signal(direction=1, strength=3, confidence=0.96, rationale=f"Concrete amount (₹{amount}) and date ({date}) provided")
        elif has_concrete_amount or has_concrete_date:
            signals["specificity"] = Signal(direction=1, strength=2, confidence=0.90, rationale="Concrete amount or date provided")
        elif is_vague:
            signals["specificity"] = Signal(direction=-1, strength=2, confidence=0.88, rationale="Vague timeline or amount provided")

        # 3. BORROWER INITIATION
        if effective_speaker == "borrower" and (any(w in clean_lower for w in commitment_positive_words) or amount is not None):
            signals["borrower_initiation"] = Signal(direction=1, strength=2, confidence=0.90, rationale="Borrower voluntarily proposed payment details")
        elif effective_speaker == "agent" and (amount is not None or date is not None):
            signals["borrower_initiation"] = Signal(direction=-1, strength=2, confidence=0.88, rationale="Agent driving the terms and amount")

        # 4. CONFIRMATION
        confirmation_words = ["confirmed", "confirm hai", "confirm", "pakka", "sure", "definitely", "haan sir, confirm", "ji sir"]
        if any(w in clean_lower for w in confirmation_words) and effective_speaker == "borrower":
            signals["confirmation"] = Signal(direction=1, strength=2, confidence=0.94, rationale="Explicit borrower confirmation")
        elif any(w in clean_lower for w in ["pata nahi", "shayad", "not sure", "dekhunga"]):
            signals["confirmation"] = Signal(direction=-1, strength=2, confidence=0.88, rationale="Borrower avoided explicit confirmation")

        # 5. FEASIBILITY (ability to pay)
        feasibility_positive_words = [
            "salary aa gayi", "salary has come", "already credited", "salary credited",
            "paise arrange ho gaye", "fund available", "paise aa gaye"
        ]
        feasibility_negative_words = [
            "paise nahi", "no money", "paisa nahi hai", "salary nahi aayi",
            "salary late", "loss", "hospital"
        ]
        if any(w in clean_lower for w in feasibility_positive_words):
            signals["feasibility"] = Signal(direction=1, strength=2, confidence=0.92, rationale="Evidence borrower has funds / salary credited")
        elif any(w in clean_lower for w in feasibility_negative_words):
            signals["feasibility"] = Signal(direction=-1, strength=2, confidence=0.88, rationale="Financial strain / lack of funds currently")

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
            signals["conditionality"] = Signal(direction=1, strength=2, confidence=0.92, rationale="Condition cleared / unconditional commitment")
        elif any(w in clean_lower for w in conditional_words):
            signals["conditionality"] = Signal(direction=-1, strength=2, confidence=0.92, rationale="Conditional promise depending on future event")

        # 7. HARDSHIP
        hardship_words = [
            "paise nahi", "money nahi", "no money", "financial problem", "dikkat",
            "hospital", "bimari", "tabiyat kharab", "loss in business", "job chali gayi"
        ]
        hardship_resolved_words = [
            "salary aa gayi", "sab theek hai", "paise arrange ho gaye", "no problem now"
        ]
        if any(w in clean_lower for w in hardship_resolved_words):
            signals["hardship"] = Signal(direction=1, strength=2, confidence=0.90, rationale="Financial hardship resolved / funds cleared")
        elif any(w in clean_lower for w in hardship_words):
            signals["hardship"] = Signal(direction=-1, strength=2, confidence=0.92, rationale="Financial hardship indicators present")

        # 8. AGENT PRESSURE
        agent_push_words = ["right sir?", "correct sir?", "kar denge, correct", "mark kar raha hoon", "confirm kijiye"]
        if effective_speaker == "agent" and any(w in clean_lower for w in agent_push_words):
            signals["agent_pressure"] = Signal(direction=-1, strength=2, confidence=0.90, rationale="Agent exerting pressure on borrower to agree")
        elif effective_speaker == "borrower" and amount is not None:
            signals["agent_pressure"] = Signal(direction=1, strength=1, confidence=0.80, rationale="Borrower responding without heavy agent coercion")

        # 9. THIRD PARTY
        is_third_party = (
            effective_speaker in ("third_party", "third_party_background")
            or diarization.has_background_speech
            or diarization.background_coaching
        )
        borrower_personal_words = ["main khud", "mera loan", "main personal", "i will pay myself"]
        if is_third_party:
            signals["third_party"] = Signal(direction=-1, strength=3, confidence=0.96, rationale="Third party or background family speaking/coaching")
        elif any(w in clean_lower for w in borrower_personal_words) and effective_speaker == "borrower":
            signals["third_party"] = Signal(direction=1, strength=2, confidence=0.90, rationale="Borrower confirming personal accountability")

        # 10. ESCAPE SIGNAL
        escape_words = ["bas call rakhiye", "call rakho", "phone cut", "end the call", "haan haan kar dunga"]
        if any(w in clean_lower for w in escape_words):
            signals["escape_signal"] = Signal(direction=-1, strength=3, confidence=0.94, rationale="Evasive / call termination language detected")
        elif any(w in clean_lower for w in commitment_positive_words) and not any(w in clean_lower for w in escape_words):
            signals["escape_signal"] = Signal(direction=1, strength=1, confidence=0.85, rationale="Constructive engagement without escape signals")

        evidence_list = [raw_text]
        if diarization.background_transcript:
            evidence_list.append(f"Background ({diarization.background_speaker_type or 'family'}): {diarization.background_transcript}")

        # Backward compatibility flags
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
            amount=amount,
            date=date,
            detected_speaker=effective_speaker,
            speaker_confidence=diarization.confidence,
            speaker_rationale=diarization.rationale,
            third_party_background=diarization.has_background_speech,
            background_coaching=diarization.background_coaching,
            background_details=diarization.background_transcript,
            evidence_text=evidence_list,
            # Backward compatibility booleans
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
