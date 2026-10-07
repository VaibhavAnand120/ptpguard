import json
from typing import Optional, List, Dict, Any
from google import genai
from google.genai import types
from .base import SemanticAnalyzer
from .diarization import SpeakerDiarizer
from ..schemas import Evidence, Signal
from ..config import GEMINI_API_KEY, GEMINI_MODEL, DIMENSIONS

PROMPT = """
You are the semantic evidence extractor for PTPGuard, a collections-call
decision-support prototype.

Extract directional evidence signals from the current utterance across the 10 PTP credibility dimensions.

SIGN CONVENTION RULE (VERY IMPORTANT):
- POSITIVE (+1) = evidence supporting a credible, firm, feasible PTP
- NEGATIVE (-1) = evidence reducing credibility, introducing conditionality/hardship/evasion
- NEUTRAL (0)   = no evidence observed for this dimension

The 10 Dimensions:
1. commitment: +1 firm commitment, -1 refusal or backing out
2. specificity: +1 concrete amount/date/action, -1 vague commitment
3. borrower_initiation: +1 borrower voluntarily proposes, -1 agent driving terms
4. confirmation: +1 explicit borrower confirmation, -1 avoidance/evasion
5. feasibility: +1 evidence borrower can afford/salary credited, -1 lack of funds/inability
6. conditionality: +1 unconditional / condition resolved, -1 conditional ("if salary comes", "agar")
7. hardship: +1 no financial distress / hardship resolved, -1 severe financial hardship
8. agent_pressure: +1 low pressure / voluntary, -1 strong agent coercion / pushing
9. third_party: +1 borrower confirms personal responsibility, -1 third party promise / background chatter
10. escape_signal: +1 constructive engagement, -1 avoidance ("bas call rakhiye", end call)

Speaker input role: {speaker}
Utterance:
{utterance}

Return ONLY valid JSON matching this schema:
{{
  "signals": {{
    "dimension_name": {{
      "direction": 1|-1|0,
      "strength": 1|2|3,
      "confidence": 0.0-1.0
    }}
  }},
  "amount": number|null,
  "date": string|null,
  "detected_speaker": "agent"|"borrower"|"third_party"|"third_party_background",
  "speaker_confidence": number,
  "speaker_rationale": string,
  "third_party_background": boolean,
  "background_coaching": boolean,
  "background_details": string|null,
  "evidence_text": [string]
}}

Code-mixed Hindi/English is expected.
"""

class GeminiAnalyzer(SemanticAnalyzer):
    def __init__(self):
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is missing")
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    async def analyze(
        self,
        speaker: str,
        text: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Evidence:
        pre_diarization = SpeakerDiarizer.detect(text, hint_speaker=speaker, history=history)

        try:
            response = await self.client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=PROMPT.format(speaker=speaker, utterance=text),
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
            data = json.loads(response.text)

            # Convert signals dict to Signal models
            raw_signals = data.get("signals", {})
            parsed_signals = {}
            for dim, sig_data in raw_signals.items():
                if dim in DIMENSIONS and isinstance(sig_data, dict):
                    parsed_signals[dim] = Signal(
                        direction=sig_data.get("direction", 0),
                        strength=sig_data.get("strength", 1),
                        confidence=sig_data.get("confidence", 0.9),
                    )

            if "detected_speaker" not in data or not data["detected_speaker"]:
                data["detected_speaker"] = pre_diarization.primary_speaker
            if "speaker_confidence" not in data:
                data["speaker_confidence"] = pre_diarization.confidence
            if "speaker_rationale" not in data:
                data["speaker_rationale"] = pre_diarization.rationale

            data["signals"] = parsed_signals
            return Evidence(**data)
        except Exception:
            from .rules import RulesAnalyzer
            return await RulesAnalyzer().analyze(speaker, text, history)
