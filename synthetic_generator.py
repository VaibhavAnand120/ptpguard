"""
CreditNirvana - Real-Time Fake PTP Detection
Module: Synthetic Data & Dialogue Simulation Engine
Generates high-fidelity bilingual (Hinglish/English) debt collection call transcripts,
conversational dynamics, acoustic prosody features, and account credit metadata.
"""

import json
import random
import os
from typing import List, Dict, Any

# 6 Canonical PTP Archetypes from Problem Statement
PTP_TYPES = [
    "GENUINE_FEASIBLE",       # Borrower means it and has cash flow
    "GENUINE_INFEASIBLE",     # Borrower means it, but date/amount mismatches salary/cashflow
    "ESCAPE_PROMISE",         # Borrower insincerely says yes to end the call
    "THIRD_PARTY_PROMISE",    # Spouse/parent/relative promises on borrower's behalf
    "REPEAT_PROMISER",        # Chronic breaker with long history of broken PTPs
    "AGENT_PUSHED"            # Agent forces/logs PTP borrower never actually committed to
]

DELINQUENCY_BUCKETS = ["X (Current)", "1-30 DPD", "31-60 DPD", "61-90 DPD", "90+ / NPA"]

LOAN_PRODUCTS = ["Personal Loan", "Credit Card", "Two-Wheeler", "MSME Unsecured", "Microfinance"]

# Lexical templates & conversational markers for Hinglish debt collection calls
CONVERSATION_TEMPLATES = {
    "GENUINE_FEASIBLE": [
        {
            "turns": [
                ("Agent", "Namaste {name} ji, main Axis Bank collections team se baat kar raha hoon. Aapka EMI overdue hai ₹{amount}."),
                ("Borrower", "Haanji sir, I am aware. Actually I was travelling last week isliye miss ho gaya."),
                ("Agent", "Toh aap kab tak payment clear kar sakte hain?"),
                ("Borrower", "Kal subah 11 baje tak IMPS se pay kar dunga. Maine already reminder set kar diya hai."),
                ("Agent", "Theek hai sir, main kal subah ka date mark kar raha hoon ₹{amount} ke liye. Dhanyawad.")
            ],
            "borrower_initiated_date": True,
            "borrower_initiated_amount": True,
            "hedging_score": 0.08,
            "hardship_flag": False,
            "third_party_flag": False,
            "agent_speaking_ratio": 0.48,
            "mean_pause_duration": 0.45,
            "pitch_jitter": 0.02,
            "ground_truth_keep": 0.92
        },
        {
            "turns": [
                ("Agent", "Hello {name}, your loan EMI of ₹{amount} was due on the 5th. When can we expect the payment?"),
                ("Borrower", "Haanji, mere account me balance aa chuka hai. Main aaj sham ko 6 baje office se nikalte hi UPI se pay kar deta hoon."),
                ("Agent", "Can you make it now via payment link?"),
                ("Borrower", "Abhi client meeting me hoon, but 6 baje pakka link se kar dunga. Don't worry.")
            ],
            "borrower_initiated_date": True,
            "borrower_initiated_amount": True,
            "hedging_score": 0.12,
            "hardship_flag": False,
            "third_party_flag": False,
            "agent_speaking_ratio": 0.45,
            "mean_pause_duration": 0.50,
            "pitch_jitter": 0.03,
            "ground_truth_keep": 0.88
        }
    ],
    "GENUINE_INFEASIBLE": [
        {
            "turns": [
                ("Agent", "{name} ji, aapka ₹{amount} pending hai. 3 din se bounce ho raha hai."),
                ("Borrower", "Sir main dena chahta hoon, lekin company me salary delay ho gayi hai iss month."),
                ("Agent", "Toh kab tak clear hoga? Aaj kar sakte hain?"),
                ("Borrower", "Aaj bilkul nahi ho payega sir. 10th ko hamari salary aayegi, ussi din 10 baje main poora clear kar dunga."),
                ("Agent", "Lekin system me aaj ya kal ka hi PTP le sakte hain."),
                ("Borrower", "Agar main kal bol doon toh jhooth hoga sir, paise hi 10th ko aayenge. Please 10th ka mark kijiye.")
            ],
            "borrower_initiated_date": True,
            "borrower_initiated_amount": False,
            "hedging_score": 0.35,
            "hardship_flag": True,
            "third_party_flag": False,
            "agent_speaking_ratio": 0.52,
            "mean_pause_duration": 1.25,
            "pitch_jitter": 0.08,
            "ground_truth_keep": 0.38
        },
        {
            "turns": [
                ("Agent", "Sir aapka credit card minimum due ₹{amount} unpaid hai."),
                ("Borrower", "Bhaiya iss bar hospital ka thoda kharcha aa gaya tha family me, so cash tight hai."),
                ("Agent", "Poora amount nahi toh part payment ₹2,000 abhi kar dijiye."),
                ("Borrower", "Abhi account me ₹500 hi bache hain. 7 tareekh ko bonus aayega tabhi ₹{amount} pay kar paunga. Try karunga usse pehle kuch arrange ho sake.")
            ],
            "borrower_initiated_date": True,
            "borrower_initiated_amount": False,
            "hedging_score": 0.42,
            "hardship_flag": True,
            "third_party_flag": False,
            "agent_speaking_ratio": 0.50,
            "mean_pause_duration": 1.40,
            "pitch_jitter": 0.09,
            "ground_truth_keep": 0.30
        }
    ],
    "ESCAPE_PROMISE": [
        {
            "turns": [
                ("Agent", "{name} ji, aapka overdue amount ₹{amount} hai. Call kyun disconnect kar rahe the?"),
                ("Borrower", "Arre sir driving kar raha hoon..."),
                ("Agent", "Payment kab hoga sir? Aaj evening 5 PM?"),
                ("Borrower", "Haan haan theek hai, kar dunga."),
                ("Agent", "Confirm 5 PM kar denge na? PTP log kar doon?"),
                ("Borrower", "Haan bhai dekhta hoon, try karunga sham tak ho jaye. Rakhta hoon abhi.")
            ],
            "borrower_initiated_date": False,
            "borrower_initiated_amount": False,
            "hedging_score": 0.88,
            "hardship_flag": False,
            "third_party_flag": False,
            "agent_speaking_ratio": 0.65,
            "mean_pause_duration": 2.10,
            "pitch_jitter": 0.12,
            "ground_truth_keep": 0.07
        },
        {
            "turns": [
                ("Agent", "Hello sir, payment kab clear kar rahe ho? 4 din se aap call tal rahe ho."),
                ("Borrower", "Arey kar denge na sir, bhag thodi na rahe hain."),
                ("Agent", "Toh exact date bataiye, system me log karna hai."),
                ("Borrower", "Agle hafte dekh lo... Monday ya Tuesday kar denge."),
                ("Agent", "Monday 10 AM kar doon? ₹{amount}?"),
                ("Borrower", "Haan haan chalega, Monday kar dena, abhi busy hoon.")
            ],
            "borrower_initiated_date": False,
            "borrower_initiated_amount": False,
            "hedging_score": 0.82,
            "hardship_flag": False,
            "third_party_flag": False,
            "agent_speaking_ratio": 0.60,
            "mean_pause_duration": 1.95,
            "pitch_jitter": 0.11,
            "ground_truth_keep": 0.10
        }
    ],
    "THIRD_PARTY_PROMISE": [
        {
            "turns": [
                ("Agent", "Hello, kya meri baat {name} ji se ho rahi hai?"),
                ("Borrower", "Nahi, main unki wife bol rahi hoon. Woh abhi bahar gaye hain."),
                ("Agent", "Mam unka ₹{amount} loan payment pending hai. Kya aap pay kar sakti hain ya kab aayenge?"),
                ("Borrower", "Woh sham ko 8 baje aate hain. Main unko bol dungi kal subah kar denge."),
                ("Agent", "Kal subah 10 baje pakka kar denge na?"),
                ("Borrower", "Haan unko bata dungi, woh kal kar denge.")
            ],
            "borrower_initiated_date": True,
            "borrower_initiated_amount": False,
            "hedging_score": 0.55,
            "hardship_flag": False,
            "third_party_flag": True,
            "agent_speaking_ratio": 0.55,
            "mean_pause_duration": 0.90,
            "pitch_jitter": 0.05,
            "ground_truth_keep": 0.22
        },
        {
            "turns": [
                ("Agent", "Namaskar, {name} ji se baat karni hai loan repayment ke regarding."),
                ("Borrower", "Ye unka office colleague hai. Unka phone yahan desk pe chhoota hai."),
                ("Agent", "Aap unhe inform kar sakte hain ₹{amount} pay karne ke liye?"),
                ("Borrower", "Theek hai, main bol dunga unhe call back karne ko ya payment karne ko.")
            ],
            "borrower_initiated_date": False,
            "borrower_initiated_amount": False,
            "hedging_score": 0.60,
            "hardship_flag": False,
            "third_party_flag": True,
            "agent_speaking_ratio": 0.58,
            "mean_pause_duration": 0.85,
            "pitch_jitter": 0.04,
            "ground_truth_keep": 0.15
        }
    ],
    "REPEAT_PROMISER": [
        {
            "turns": [
                ("Agent", "{name} ji, pichle hafte bhi aapne 2 bar promise kiya tha payment ka, but bounce ho gaya."),
                ("Borrower", "Haan sir I know, thoda technical issue ho gaya tha netbanking me."),
                ("Agent", "Lekin ab 45 din overdue ho chuka hai ₹{amount}."),
                ("Borrower", "Aap tension mat lijiye, kal pakka 2 baje se pehle ho jayega. 100% guarantee."),
                ("Agent", "Aapne pichli bar bhi yahi bola tha sir."),
                ("Borrower", "Iss bar pakka sir, branch jaake cheque drop karunga.")
            ],
            "borrower_initiated_date": True,
            "borrower_initiated_amount": False,
            "hedging_score": 0.40,
            "hardship_flag": False,
            "third_party_flag": False,
            "agent_speaking_ratio": 0.52,
            "mean_pause_duration": 0.60,
            "pitch_jitter": 0.07,
            "ground_truth_keep": 0.18
        }
    ],
    "AGENT_PUSHED": [
        {
            "turns": [
                ("Agent", "{name} ji, loan account number 4092. Aapko ₹{amount} aaj hi pay karna hai."),
                ("Borrower", "Sir abhi toh arrange nahi ho payega..."),
                ("Agent", "Aise kaise nahi hoga? Kal subah tak arrange kijiye. Main 11 AM ka PTP daal raha hoon."),
                ("Borrower", "...hmmm..."),
                ("Agent", "11 AM theek hai na? Confirm boliye taaki penalty na lage."),
                ("Borrower", "Abhi dekhiye main kya bolu... theek hai dekh lijiye."),
                ("Agent", "Okay logging PTP for tomorrow 11 AM.")
            ],
            "borrower_initiated_date": False,
            "borrower_initiated_amount": False,
            "hedging_score": 0.75,
            "hardship_flag": False,
            "third_party_flag": False,
            "agent_speaking_ratio": 0.82,  # Agent dominates conversation
            "mean_pause_duration": 2.45,
            "pitch_jitter": 0.14,
            "ground_truth_keep": 0.04
        }
    ]
}

NAMES = ["Rahul Sharma", "Pooja Verma", "Vikram Patel", "Sunita Devi", "Mohammed Arif", 
         "Amit Kumar", "Deepak Joshi", "Kavita Nair", "Suresh Gounder", "Ananya Sen"]

def generate_synthetic_record(record_id: int) -> Dict[str, Any]:
    ptp_type = random.choices(
        PTP_TYPES,
        weights=[0.30, 0.20, 0.25, 0.08, 0.10, 0.07],  # Realistic distribution
        k=1
    )[0]

    template = random.choice(CONVERSATION_TEMPLATES[ptp_type])
    name = random.choice(NAMES)
    amount = random.choice([2500, 4800, 8500, 12000, 18500, 25000, 45000])
    dpd_bucket = random.choice(DELINQUENCY_BUCKETS)
    loan_type = random.choice(LOAN_PRODUCTS)

    # Historical repayment metrics
    if ptp_type == "GENUINE_FEASIBLE":
        past_ptps_given = random.randint(1, 4)
        past_ptps_kept = past_ptps_given - random.choice([0, 1])
    elif ptp_type == "REPEAT_PROMISER":
        past_ptps_given = random.randint(4, 8)
        past_ptps_kept = random.randint(0, 1)
    else:
        past_ptps_given = random.randint(1, 5)
        past_ptps_kept = random.randint(0, past_ptps_given // 2)

    # Format turns
    turns = []
    accumulated_text = ""
    for speaker, text in template["turns"]:
        formatted_text = text.format(name=name, amount=amount)
        turns.append({"speaker": speaker, "text": formatted_text})
        accumulated_text += f"{speaker}: {formatted_text}\n"

    # Add realistic noise to continuous features
    mean_pause = max(0.2, template["mean_pause_duration"] + random.gauss(0, 0.1))
    pitch_jitter = max(0.01, template["pitch_jitter"] + random.gauss(0, 0.01))
    hedging = min(1.0, max(0.0, template["hedging_score"] + random.gauss(0, 0.05)))
    agent_ratio = min(0.95, max(0.3, template["agent_speaking_ratio"] + random.gauss(0, 0.03)))

    return {
        "call_id": f"CALL_{record_id:05d}",
        "customer_name": name,
        "loan_type": loan_type,
        "delinquency_bucket": dpd_bucket,
        "overdue_amount": amount,
        "past_ptps_given": past_ptps_given,
        "past_ptps_kept": past_ptps_kept,
        "past_keep_rate": round(past_ptps_kept / max(1, past_ptps_given), 3),
        "transcript_raw": accumulated_text.strip(),
        "turns": turns,
        "conversational_dynamics": {
            "borrower_initiated_date": template["borrower_initiated_date"],
            "borrower_initiated_amount": template["borrower_initiated_amount"],
            "agent_speaking_ratio": round(agent_ratio, 3),
            "mean_pause_duration_sec": round(mean_pause, 2),
            "pitch_jitter": round(pitch_jitter, 3),
            "hedging_score": round(hedging, 3),
            "hardship_flag": template["hardship_flag"],
            "third_party_flag": template["third_party_flag"]
        },
        "ground_truth": {
            "ptp_type": ptp_type,
            "ptp_kept": random.random() < template["ground_truth_keep"],
            "keep_probability": template["ground_truth_keep"]
        }
    }

def generate_dataset(num_samples: int = 1000, output_file: str = "data/synthetic_ptp_calls.json"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    records = [generate_synthetic_record(i + 1) for i in range(num_samples)]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[SUCCESS] Generated {num_samples} synthetic multi-modal collection call records -> {output_file}")
    return records

if __name__ == "__main__":
    generate_dataset(1200)
