import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple


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


# 1. Regex patterns to capture explicit background annotations/brackets in transcript
BACKGROUND_ANNOTATION_RE = re.compile(
    r"(?:\(|\[|\*)\s*(?:peeche\s+se|background|bg|wife|patni|family|whisper|side\s+voice)[^:\)\-\*]*[:\-–]?\s*([^\]\)\*]+)(?:\)|\]|\*)",
    re.IGNORECASE
)

# 2. Coaching / prompting cues directed at the borrower (telling borrower what to say to lender)
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

# 3. Direct third-party phrases (speaking directly on the line)
DIRECT_THIRD_PARTY_PATTERNS = [
    re.compile(r"\bmain\s+unka\s+(?:bhai|beta|pita|father|brother|husband|pati|dost|friend|colleague|vakeel|advocate|lawyer)\b", re.I),
    re.compile(r"\bmain\s+unki\s+(?:patni|wife|mata|mother|behan|sister|beti|daughter)\b", re.I),
    re.compile(r"\bi\s+am\s+his\s+(?:wife|brother|father|mother|son|daughter|lawyer|friend|colleague)\b", re.I),
    re.compile(r"\bi\s+am\s+her\s+(?:husband|brother|father|mother|son|daughter|lawyer|friend|colleague)\b", re.I),
    re.compile(r"\b(?:wo\s+abhi|woh\s+abhi)\s+(?:ghar\s+pe\s+nahi|out\s+of\s+station|hospital\s+me|busy\s+hai|phone\s+nahi\s+utha\s+sakte|nahi\s+hai)\b", re.I),
    re.compile(r"\bhe\s+is\s+(?:not\s+available|not\s+at\s+home|in\s+hospital|out\s+of\s+station|busy)\b", re.I),
    re.compile(r"\b(?:inka|unka)\s+phone\s+mere\s+paas\s+hai\b", re.I),
    re.compile(r"\bmain\s+unke\s+behalf\s+pe\s+bol\s+raha\b", re.I),
]

# 4. Lender / Agent phrases (collections executive, financial institution, demanding dues, 2nd-person inquiries)
LENDER_PATTERNS = [
    # Demanding payment or timeline (inquiries with 'kab', 'kitna', 'when')
    re.compile(r"\b(?:kab\s+tak|kab)\b.*?\b(?:payment|pay|clear|jama|bhej|karenge|karoge|kar\s+paoge|denge|karein|ho\s+payega)\b", re.I),
    re.compile(r"\b(?:payment|emi|overdue|amount|balance)\b.*?\b(?:kab|kitna|clear|jama|karenge|karoge|kar\s+paoge)\b", re.I),
    re.compile(r"\b(?:kab\s+karenge|kab\s+karoge|kab\s+kar\s+paoge|kab\s+tak\s+denge|kab\s+de\s+paoge)\b", re.I),
    re.compile(r"\b(?:when\s+will\s+you\s+pay|when\s+can\s+you|payment\s+status|how\s+will\s+you\s+pay)\b", re.I),
    re.compile(r"\b(?:kitna\s+amount|kitna\s+pay|kitna\s+kar\s+paoge|kitna\s+karoge)\b", re.I),
    re.compile(r"\b(?:payment\s+kar\s+paoge\?|payment\s+kar\s+denge\?|payment\s+possible\s+hai\?|ho\s+payega\?)\b", re.I),
    # Institutional greetings & context
    re.compile(r"\b(?:calling\s+from|se\s+bol\s+raha|se\s+bol\s+rahi|se\s+call\s+hai|se\s+call\s+kar)\b", re.I),
    re.compile(r"\b(?:bank|finance|capital|nbfc|agency|recovery|credit\s+card|collections?|hdfc|icici|sbi|bajaj|axis|kotak|cred|moneyview|kreditbee)\b", re.I),
    re.compile(r"\b(?:aapka|aapki)\s+(?:loan|emi|pending|overdue|dues|installment|balance|khata|account)\b", re.I),
    # Confirmation of commitment / agent recording
    re.compile(r"\b(?:confirmed\?|confirm\s+hai\?|right\?|correct\?|right\s+sir\?|correct\s+sir\?|confirm\s+kar\s+rahe|kar\s+denge,\s*correct|mark\s+kar\s+doon|note\s+kar\s+loon)", re.I),
    re.compile(r"\b(?:mark\s+kar|note\s+kar|system\s+me\s+update|system\s+me\s+mark|i\'?ll\s+mark)\b", re.I),
    re.compile(r"\b(?:payment\s+link|upi\s+link|link\s+bhej|link\s+share)\b", re.I),
    re.compile(r"\b(?:cibil|penalty|legal\s+notice|field\s+visit)\b", re.I),
]

# 5. Borrower phrases (personal financial constraints, 1st person repayment commitments, requests for time)
BORROWER_PATTERNS = [
    # 1st person future commitment verbs (-dunga, -dungi, -karunga, -paunga)
    re.compile(r"\b(?:kar\s+dunga|de\s+dunga|pay\s+kar\s+dunga|transfer\s+kar\s+dunga|bhej\s+dunga|jama\s+kar\s+dunga|clear\s+kar\s+dunga)\b", re.I),
    re.compile(r"\b(?:kar\s+dungi|de\s+dungi|pay\s+kar\s+dungi|transfer\s+kar\s+dungi|bhej\s+dungi|jama\s+kar\s+dungi)\b", re.I),
    re.compile(r"\b(?:pay\s+karunga|transfer\s+karunga|jama\s+karunga|de\s+paunga|kar\s+paunga|de\s+paungi|kar\s+paungi|koshish\s+karunga|try\s+karunga|bhejunga)\b", re.I),
    re.compile(r"\b(?:i\s+will\s+pay|i\s+will\s+clear|i\s+will\s+transfer|i\s+can\s+pay|i\s+promise|i\s+will\s+arrange)\b", re.I),
    # Hardship & Financial condition
    re.compile(r"\b(?:paise\s+nahi|paisa\s+nahi|paise\s+ki\s+dikkat|fund\s+nahi|cash\s+nahi|paise\s+arrange|no\s+money|can\'?t\s+pay|cannot\s+pay)\b", re.I),
    re.compile(r"\b(?:salary\s+aayegi|salary\s+aane\s+pe|salary\s+ke\s+baad|salary\s+late|salary\s+nahi\s+aayi|tankhwah)\b", re.I),
    re.compile(r"\b(?:meri|mera)\s+(?:salary|job|tabiyat|kamai|dukaan|paisa|dikkat|problem|kharcha)\b", re.I),
    re.compile(r"\b(?:my\s+salary|lost\s+my\s+job|financial\s+problem|medical\s+emergency)\b", re.I),
    re.compile(r"\b(?:hospital|tabiyat\s+kharab|ilaj|bimari|loss\s+in\s+business|dukaan\s+me\s+nuksan)\b", re.I),
    # Borrower negotiation / asking for time
    re.compile(r"\b(?:thoda\s+time|thoda\s+samay|do\s+din\s+ka\s+time|kuch\s+din|mohalat|time\s+de\s+do|time\s+chahiye)\b", re.I),
    re.compile(r"\b(?:bas\s+call\s+rakhiye|call\s+rakho|phone\s+rakhiye|phone\s+cut\s+kijiye)\b", re.I),
    re.compile(r"\b(?:abhi\s+possible\s+nahi|abhi\s+nahi\s+ho\s+payega|possible\s+nahi\s+hai)\b", re.I),
    # Borrower affirmative response / confirmation
    re.compile(r"\b(?:haan\s+sir|ji\s+sir|theek\s+hai\s+sir|confirm\s+hai|pakka\s+hai|pakka\s+sir|haan\s+haan\s+sir)\b", re.I),
]


class SpeakerDiarizer:
    @staticmethod
    def detect(
        text: str,
        hint_speaker: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> DiarizationResult:
        """
        Classifies the speaker into:
        - 'agent': Lender / Collection agent
        - 'borrower': Primary customer / debtor
        - 'third_party': Third party speaking directly on call
        - 'third_party_background': Third person (wife, family) speaking/coaching in background on borrower's side
        """
        raw_text = text.strip()
        t = raw_text.lower()

        # Step 1: Detect explicit background annotations e.g. "(peeche se patni: bolo paise nahi hai)"
        bg_match = BACKGROUND_ANNOTATION_RE.search(raw_text)
        has_annotation_bg = bool(bg_match)
        bg_transcript = bg_match.group(1).strip() if bg_match else None

        # Clean text without background annotation if present
        clean_text = BACKGROUND_ANNOTATION_RE.sub("", raw_text).strip()
        if not clean_text and bg_transcript:
            clean_text = bg_transcript

        # Step 2: Check coaching patterns (imperatives telling borrower what to say)
        has_coaching = any(p.search(t) for p in COACHING_PATTERNS)
        
        # Step 3: Check speaker type identifiers in background or text
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

        # Check if entire utterance is background third-party speech
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
                rationale=f"Background speech detected on borrower's side ({bg_type or 'family member'}{' coaching borrower to evade/delay' if has_coaching else ''})",
                has_background_speech=True,
                background_speaker_type=bg_type,
                background_transcript=bg_transcript or raw_text,
                background_coaching=has_coaching,
                clean_text=clean_text or raw_text,
            )

        # Step 4: Check direct third-party patterns
        is_direct_third_party = any(p.search(t) for p in DIRECT_THIRD_PARTY_PATTERNS) or hint_speaker == "third_party"
        if is_direct_third_party:
            return DiarizationResult(
                primary_speaker="third_party",
                confidence=0.94,
                rationale="Speaker identifies as third party (relative, spouse, or representative) speaking on the call",
                has_background_speech=has_annotation_bg,
                background_speaker_type=bg_type,
                background_transcript=bg_transcript,
                background_coaching=has_coaching,
                clean_text=clean_text or raw_text,
            )

        # Step 5: Check explicit hint_speaker (preserve manual developer/user selection if not auto)
        if hint_speaker in ("agent", "borrower") and hint_speaker != "auto":
            return DiarizationResult(
                primary_speaker=hint_speaker,
                confidence=0.95,
                rationale=f"Specified as {hint_speaker}",
                has_background_speech=has_annotation_bg or has_coaching,
                background_speaker_type=bg_type,
                background_transcript=bg_transcript,
                background_coaching=has_coaching,
                clean_text=clean_text or raw_text,
            )

        # Step 6: Full Auto-Diarization (Score Lender vs Borrower signals)
        clean_lower = (clean_text or raw_text).lower()
        lender_score = sum(1 for p in LENDER_PATTERNS if p.search(clean_lower))
        borrower_score = sum(1 for p in BORROWER_PATTERNS if p.search(clean_lower))

        if lender_score > borrower_score:
            primary = "agent"
            confidence = min(0.98, 0.75 + 0.10 * lender_score)
            rationale = "Lender/Agent collections inquiries, institutional greetings, or overdue demands detected"
        elif borrower_score > lender_score:
            primary = "borrower"
            confidence = min(0.98, 0.75 + 0.10 * borrower_score)
            rationale = "Borrower first-person financial situation, payment promise, or hardship explanation detected"
        else:
            # Neutral / Tie-breaker:
            last_speaker = history[-1].get("speaker") if history and len(history) > 0 else None
            
            # Questions are overwhelmingly asked by the Lender/Agent in collections calls
            if "?" in raw_text or "kab" in clean_lower or "kitna" in clean_lower:
                primary = "agent"
                confidence = 0.75
                rationale = "Inferred as Lender/Agent due to inquiry/question structure"
            # Short affirmations (e.g. 'Ji.', 'Haan.') in response to Agent are Borrower
            elif last_speaker == "agent":
                primary = "borrower"
                confidence = 0.75
                rationale = "Inferred as Borrower responding to preceding Lender turn"
            elif last_speaker in ("borrower", "third_party", "third_party_background"):
                primary = "agent"
                confidence = 0.70
                rationale = "Inferred as Lender following up on borrower turn"
            else:
                # Call opening defaults to Agent if greeting, otherwise Borrower answering
                if any(w in clean_lower for w in ["namaste", "hello", "good morning", "good afternoon"]):
                    primary = "agent"
                    confidence = 0.70
                    rationale = "Inferred as Lender initiating the call greeting"
                else:
                    primary = "borrower"
                    confidence = 0.60
                    rationale = "Defaulted to borrower"

        return DiarizationResult(
            primary_speaker=primary,
            confidence=confidence,
            rationale=rationale,
            has_background_speech=has_annotation_bg or has_coaching,
            background_speaker_type=bg_type,
            background_transcript=bg_transcript,
            background_coaching=has_coaching,
            clean_text=clean_text or raw_text,
        )
