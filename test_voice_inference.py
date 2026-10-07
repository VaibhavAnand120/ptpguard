"""
CreditNirvana - Voice & Phrase Inference Test Runner
Demonstrates how user voice inputs in Hindi/Hinglish/English are scored in <10ms.
"""

import requests
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

URL = "http://127.0.0.1:8000/api/analyze"

test_phrases = [
    {
        "name": "Test 1: Escape Promise (Hedging & Avoidance in Hinglish)",
        "phrase": "Haan bhaiya dekhta hoon, try karunga next week Friday ko, abhi driving kar raha hoon.",
        "pause_sec": 2.2,
        "past_given": 2,
        "past_kept": 0
    },
    {
        "name": "Test 2: Genuine Feasible (Firm Commitment with specific time & mode)",
        "phrase": "Maine already schedule kar diya hai, kal subah 11 baje office se nikalte hi IMPS se pay kar dunga pakka.",
        "pause_sec": 0.4,
        "past_given": 2,
        "past_kept": 2
    },
    {
        "name": "Test 3: Financial Hardship (Medical emergency & salary delay)",
        "phrase": "Hospital me mummy admit hain isliye cash tight chal raha hai, salary 10th ko aayegi tabhi poora bhar paunga.",
        "pause_sec": 1.6,
        "past_given": 1,
        "past_kept": 0
    },
    {
        "name": "Test 4: Third Party (Spouse answering)",
        "phrase": "Main unki wife bol rahi hoon, woh abhi bahar gaye hain, unko inform kar dungi kal baat karenge.",
        "pause_sec": 0.8,
        "past_given": 1,
        "past_kept": 0
    }
]

print("="*80)
print("CREDITNIRVANA: LIVE VOICE & PHRASE SCORING TEST")
print("="*80)

for t in test_phrases:
    payload = {
        "transcript_raw": f"Agent: EMI payment kab clear hoga?\nBorrower: {t['phrase']}",
        "past_ptps_given": t["past_given"],
        "past_ptps_kept": t["past_kept"],
        "overdue_amount": 14500.0,
        "conversational_dynamics": {
            "borrower_initiated_date": True,
            "borrower_initiated_amount": False,
            "agent_speaking_ratio": 0.4,
            "mean_pause_duration_sec": t["pause_sec"],
            "pitch_jitter": 0.05,
            "hedging_score": 0.5
        }
    }
    
    resp = requests.post(URL, json=payload)
    res = resp.json()
    
    print(f"\n[TEST CASE] {t['name']}")
    print(f"[SPOKEN] \"{t['phrase']}\"")
    print(f"[PREDICTION] Archetype: {res['predicted_ptp_type']} (Confidence: {res['confidence']*100:.1f}%)")
    print(f"[CALIBRATED P(Keep)] {res['calibrated_keep_probability']*100:.1f}%")
    print(f"[ACTION NUDGE] {res['recommended_action']['nudge_text']}")
    print(f"[RBI HARDSHIP PROTECTED] {res['compliance_and_audit']['rbi_hardship_protected']}")
    print(f"[AUDIT RATIONALE] {', '.join(res['compliance_and_audit']['audit_explanations'])}")
    print("-" * 80)
