import json
from typing import Optional, List, Dict, Any
import httpx
from .base import SemanticAnalyzer
from .diarization import SpeakerDiarizer
from ..schemas import Evidence, Signal
from ..config import OLLAMA_BASE_URL, OLLAMA_MODEL, DIMENSIONS

PROMPT = """
Extract directional evidence signals from this collections-call utterance across 10 dimensions.
Sign: +1 = supports credibility, -1 = reduces credibility, 0 = neutral.

Dimensions:
commitment, specificity, borrower_initiation, confirmation, feasibility,
conditionality, hardship, agent_pressure, third_party, escape_signal.

Speaker: {speaker}
Utterance: {utterance}

Return ONLY JSON:
{{
  "signals": {{
    "dimension_name": {{
      "direction": 1|-1|0,
      "strength": 1|2|3,
      "confidence": 0.9
    }}
  }},
  "amount": null,
  "date": null,
  "detected_speaker": "agent"|"borrower"|"third_party"|"third_party_background",
  "speaker_confidence": 0.9,
  "speaker_rationale": "...",
  "third_party_background": false,
  "background_coaching": false,
  "background_details": null,
  "evidence_text": []
}}
"""

class OllamaAnalyzer(SemanticAnalyzer):
    async def analyze(
        self,
        speaker: str,
        text: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Evidence:
        pre_diarization = SpeakerDiarizer.detect(text, hint_speaker=speaker, history=history)

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": PROMPT.format(speaker=speaker, utterance=text),
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
                r.raise_for_status()
                data = r.json()
            parsed = json.loads(data["response"])

            raw_signals = parsed.get("signals", {})
            parsed_signals = {}
            for dim, sig_data in raw_signals.items():
                if dim in DIMENSIONS and isinstance(sig_data, dict):
                    parsed_signals[dim] = Signal(
                        direction=sig_data.get("direction", 0),
                        strength=sig_data.get("strength", 1),
                        confidence=sig_data.get("confidence", 0.9),
                    )

            if "detected_speaker" not in parsed or not parsed["detected_speaker"]:
                parsed["detected_speaker"] = pre_diarization.primary_speaker

            parsed["signals"] = parsed_signals
            return Evidence(**parsed)
        except Exception:
            from .rules import RulesAnalyzer
            return await RulesAnalyzer().analyze(speaker, text, history)
