"""
CreditNirvana - Real-Time Fake PTP Detection
Module: Gemini Flash Conversational Recovery Agent & Credibility Engine
Empowers an ultra-realistic, persuasive, empathetic debt collections tele-calling agent
that negotiates with borrowers in natural Hinglish to recover overdue EMI or secure instant token payments,
while continuously evaluating credibility and enforcing RBI Fair Practices regulations.
"""

import os
import json
import re
import sys
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# High-availability, low-latency model cascade (0.9s - 1.5s latency)
MODEL_CASCADE = [
    "gemini-flash-lite-latest",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest"
]

SYSTEM_INSTRUCTION = """You are a senior, highly experienced debt collections tele-calling officer representing Axis Bank / CreditNirvana Collections Unit in India.
You are on an ACTIVE LIVE PHONE CALL with a borrower who has an overdue retail loan EMI.

YOUR MISSION ON THIS LIVE CALL:
Convince the borrower to resolve their overdue loan today by either:
1. Making an immediate full payment,
2. Paying an immediate token amount (₹1,000–₹2,000) via instant WhatsApp/SMS link right now while on the call to prevent delinquency escalation, or
3. Committing to a firm, feasible, verified PTP date aligned with their salary or cashflow under RBI Fair Practices guidelines.

NEGOTIATION PLAYBOOK & CONVERSATIONAL STRATEGY:
- Sound 100% like a real, experienced, empathetic yet assertive Indian collections professional speaking in natural colloquial Hinglish/English (respectful 'Aap', polite but firm).
- Actively adapt your next words based on whatever specific situation, constraint, or excuse the borrower presents:
  * If borrower is EVASIVE / ESCAPING ("driving kar raha hoon", "busy hoon", "meeting", "agle hafte dekhta hoon", "baad me call karo"):
    Acknowledge their immediate situation respectfully (e.g. driving safety), but firmly highlight the immediate consequence: the account will roll forward into delinquency today, attracting late penalties and CIBIL credit score downgrade. Persuade them to pay a small token payment of ₹1,000–₹2,000 via WhatsApp UPI link right now (or as soon as they pull over) to freeze escalation for 48 hours.
  * If borrower reports GENUINE HARDSHIP / DISTRESS ("hospital me hoon", "accident ho gaya", "salary delay", "shop me chori ho gayi", "paise nahi hain"):
    Express genuine empathy and strictly adhere to RBI Fair Practices Code (zero pressure or harassment). Offer compassionate relief by scheduling their payment to their salary/income date (e.g. 10th or 15th) or proposing EMI restructuring. Gently ask if a small nominal token can be arranged today to maintain active status.
  * If borrower makes a VAGUE PROMISE ("haan kar dunga", "dekh lunga", "ho jayega", "chinta mat karo"):
    Do not accept ambiguity! Firm up the commitment by politely probing for the exact hour ("Aap kal subah 11 baje tak karenge ya shaam 4 baje?") and payment rail ("Google Pay UPI se karenge ya netbanking se?").
  * If borrower is a CHRONIC BREAKER / REPEAT PROMISER:
    Diplomatically remind them that prior promises were missed and system risk policy cannot grant an extended grace window without an immediate partial token payment today.
  * If a THIRD PARTY / FAMILY MEMBER answers:
    Remain professional and adhere to DPDP privacy. Inform them of an urgent bank matter and politely request a callback time or ask them to notify the borrower to check the payment link on their registered phone.
  * If borrower CONFIRMS A SPECIFIC PAYMENT ("kal subah 10 baje GPay se kar dunga"):
    Confirm the exact date, time, and amount. Inform them that a pre-due reminder and UPI link have been scheduled, and thank them courteously.

CRITICAL RULES:
- Never use robotic, generic canned templates.
- Write natural conversational dialogue in 'agent_firming_reply' that directly speaks to the borrower as your next spoken line in the call.
- Compute continuous calibrated P(Keep) ∈ [0.01, 0.99] reflecting genuine repayment credibility.

Output STRICT JSON ONLY:
{
  "predicted_ptp_type": "GENUINE_FEASIBLE" | "GENUINE_INFEASIBLE" | "ESCAPE_PROMISE" | "THIRD_PARTY_PROMISE" | "REPEAT_PROMISER" | "AGENT_PUSHED",
  "confidence": 0.88,
  "calibrated_keep_probability": 0.65,
  "class_distribution": {
    "GENUINE_FEASIBLE": 0.65,
    "GENUINE_INFEASIBLE": 0.12,
    "ESCAPE_PROMISE": 0.13,
    "THIRD_PARTY_PROMISE": 0.04,
    "REPEAT_PROMISER": 0.04,
    "AGENT_PUSHED": 0.02
  },
  "recommended_action": {
    "action_code": "REQUEST_INSTANT_TOKEN_PAYMENT" | "OFFER_RESTRUCTURING_POST_SALARY" | "PARK_AND_REMIND" | "TRIGGER_BORROWER_VERIFICATION_IVR" | "SHORTEN_PARKING_WINDOW_ESCALATE" | "FLAG_FOR_INTEGRITY_AUDIT",
    "nudge_text": "Actionable 1-line guidance for tele-caller",
    "recommended_parking_window_days": 1 | 2 | 5 | 7
  },
  "agent_firming_reply": "Realistic, natural, persuasive Hinglish dialogue spoken directly by the agent to convince/negotiate with the borrower.",
  "compliance_and_audit": {
    "rbi_hardship_protected": true | false,
    "dpdp_biometric_voice_purged": true,
    "integrity_review_required": false,
    "audit_explanations": [
      "Specific analysis point grounded in borrower's words",
      "Specific analysis point grounded in timing/intent"
    ]
  }
}
"""


class GeminiPTPClassifier:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = None
        self._init_client()

    def _init_client(self):
        active_key = self.api_key or os.environ.get("GEMINI_API_KEY")
        if active_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=active_key)
                self.api_key = active_key
            except Exception as e:
                print(f"[WARN] Could not initialize google.genai: {e}")
                self.client = None

    def classify_turn(
        self,
        transcript_history: str,
        borrower_latest_utterance: str,
        pause_latency_sec: float = 0.8,
        overdue_amount: float = 14500.0,
        past_ptps_given: int = 2,
        past_ptps_kept: int = 1,
        salary_credit_day: int = 7,
        api_key_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Classifies borrower's response and dynamically synthesizes the next persuasive conversational turn.
        Uses Google Gemini Flash Neural cascade for 100% unscripted, realistic human dialogue.
        """
        active_key = api_key_override or self.api_key or os.environ.get("GEMINI_API_KEY")
        if active_key and not self.client:
            self.api_key = active_key
            self._init_client()

        # Route to Gemini Flash Cascade
        if active_key and self.client:
            from google.genai import types
            
            prompt = f"""LIVE PHONE CALL INTERACTION - AXIS BANK COLLECTIONS

[BORROWER ACCOUNT CONTEXT]
- Overdue Loan EMI: ₹{overdue_amount:,.2f}
- Historical Repayment History: {past_ptps_kept} kept out of {past_ptps_given} promises given
- Borrower Known Salary Date: {salary_credit_day}th of month

[LIVE ACOUSTIC / BEHAVIORAL CUES]
- Response Latency / Hesitation: {pause_latency_sec:.2f} seconds before answering

[CALL TRANSCRIPT UP TO THIS MOMENT]
{transcript_history}
Borrower: {borrower_latest_utterance}

You are the collections agent on this live call.
Respond directly to the borrower's statement: "{borrower_latest_utterance}".
Analyze their true intent and constraints.
Determine what to say next to persuade them to resolve the loan today (pay full amount, pay an immediate token of ₹1,000–₹2,000 via WhatsApp link, or schedule a verified date under RBI guidelines).
Output STRICT JSON ONLY."""

            # Try cascade of fast, high-availability models
            for model_name in MODEL_CASCADE:
                try:
                    config = types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        response_mime_type="application/json",
                        temperature=0.25
                    )
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=config
                    )
                    text = response.text.strip()
                    if text.startswith("```json"):
                        text = text[7:]
                    if text.startswith("```"):
                        text = text[3:]
                    if text.endswith("```"):
                        text = text[:-3]
                    
                    parsed = json.loads(text.strip())
                    parsed["source"] = f"Gemini Flash Neural ({model_name})"
                    
                    if "calibrated_keep_probability" in parsed:
                        parsed["calibrated_keep_probability"] = float(parsed["calibrated_keep_probability"])
                    return parsed
                except Exception as ex:
                    print(f"[INFO] Cascade model {model_name} attempt: {type(ex).__name__}")
                    continue

        # Dynamic Open-Vocabulary Fallback (Continuous Probabilities, Semantic Extraction)
        print("[INFO] Utilizing local calibrated semantic feature engine")
        from ptp_engine import PTPCredibilityEngine
        engine = PTPCredibilityEngine()
        engine.load()

        full_transcript = f"{transcript_history}\nBorrower: {borrower_latest_utterance}".strip()
        record = {
            "transcript_raw": full_transcript,
            "past_ptps_given": past_ptps_given,
            "past_ptps_kept": past_ptps_kept,
            "overdue_amount": overdue_amount,
            "conversational_dynamics": {
                "borrower_initiated_date": True,
                "borrower_initiated_amount": False,
                "agent_speaking_ratio": 0.45,
                "mean_pause_duration_sec": pause_latency_sec,
                "pitch_jitter": 0.05,
                "hedging_score": 0.5
            }
        }
        res = engine.predict_stream(record)

        # Calculate continuous probability based on exact words and pause
        lower_txt = borrower_latest_utterance.lower()
        hedging_words = sum(1 for w in ["dekhta", "try", "koshish", "shayad", "maybe", "agle", "baad", "dekh", "sochta", "busy", "chalega"] if w in lower_txt)
        commit_words = sum(1 for w in ["kal", "subah", "pakka", "schedule", "11", "pay", "upi", "imps", "done", "clear", "karunga", "guarantee"] if w in lower_txt)
        
        raw_prob = res["calibrated_keep_probability"]
        word_shift = (commit_words * 0.08) - (hedging_words * 0.12) - (max(0.0, pause_latency_sec - 1.0) * 0.08)
        smooth_prob = float(min(0.96, max(0.04, raw_prob + word_shift)))

        res["calibrated_keep_probability"] = round(smooth_prob, 3)
        res["agent_firming_reply"] = None
        res["source"] = "Local Calibrated Engine (Fallback)"
        return res
