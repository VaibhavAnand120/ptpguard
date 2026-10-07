import math
import pytest
from app.schemas import ConversationState, Evidence, Signal, Utterance
from app.state import StateManager
from app.scoring import credibility_score, classify_ptp, normalize_raw_state
from app.policy import policy
from app.config import DIMENSIONS, DEFAULT_SCALE
from fastapi.testclient import TestClient
from app.main import app


def test_positive_evidence_increases_state():
    mgr = StateManager()
    ev = Evidence(
        signals={
            "commitment": Signal(direction=1, strength=2, confidence=0.9),
            "specificity": Signal(direction=1, strength=2, confidence=0.9),
        }
    )
    s = mgr.update(ev, speaker="borrower", text="I will definitely pay ₹2000 on the 10th.")
    assert s.raw_state["commitment"] > 1.5
    assert s.raw_state["specificity"] > 1.5


def test_negative_evidence_decreases_state():
    mgr = StateManager()
    ev = Evidence(
        signals={
            "conditionality": Signal(direction=-1, strength=2, confidence=0.95),
            "hardship": Signal(direction=-1, strength=2, confidence=0.90),
        }
    )
    s = mgr.update(ev, speaker="borrower", text="If salary comes, maybe I will pay.")
    assert s.raw_state["conditionality"] < -1.5
    assert s.raw_state["hardship"] < -1.5


def test_contradictory_evidence_reverses_previous_state():
    mgr = StateManager()
    # Turn 1: Negative conditionality
    ev1 = Evidence(signals={"conditionality": Signal(direction=-1, strength=2, confidence=0.95)})
    s1 = mgr.update(ev1, speaker="borrower", text="If salary comes, I will pay.")
    cond1 = s1.raw_state["conditionality"]
    assert cond1 < -1.8

    # Turn 2: Positive conditionality reversal
    ev2 = Evidence(signals={"conditionality": Signal(direction=1, strength=2, confidence=0.90)})
    s2 = mgr.update(ev2, speaker="borrower", text="Salary has already come, payment is confirmed.")
    cond2 = s2.raw_state["conditionality"]
    # State should now be near neutral (-1.9 + 1.8 = -0.1)
    assert abs(cond2) < 0.5
    assert cond2 > cond1


def test_repeated_identical_evidence_has_diminishing_effect():
    mgr = StateManager()
    sentence = "Main 10 ko pay kar dunga."
    deltas = []
    for _ in range(4):
        ev = Evidence(signals={"commitment": Signal(direction=1, strength=2, confidence=0.90)})
        mgr.update(ev, speaker="borrower", text=sentence)
        entry = mgr.state.evidence_history[-1]
        deltas.append(entry.delta)

    # Each subsequent repetition must contribute strictly less delta
    assert deltas[0] > deltas[1] > deltas[2] > deltas[3]


def test_raw_state_not_restricted_to_minus_two_plus_two():
    mgr = StateManager()
    for i in range(5):
        ev = Evidence(signals={"commitment": Signal(direction=1, strength=3, confidence=0.95)})
        mgr.update(ev, speaker="borrower", text=f"Novel firm promise {i}")
    # Raw state can grow well beyond +2.0
    assert mgr.state.raw_state["commitment"] > 6.0


def test_soft_saturation_prevents_unlimited_score_growth():
    raw_5 = {"commitment": 5.0}
    raw_20 = {"commitment": 20.0}
    norm_5 = normalize_raw_state(raw_5, DEFAULT_SCALE)["commitment"]
    norm_20 = normalize_raw_state(raw_20, DEFAULT_SCALE)["commitment"]

    # Both values are bounded asymptotically below 1.0
    assert norm_5 < 1.0
    assert norm_20 < 1.0
    # Difference between raw 5 and raw 20 is tiny due to tanh saturation
    assert abs(norm_20 - norm_5) < 0.05


def test_score_can_move_upward_and_downward():
    mgr = StateManager()
    # Positive utterance increases score
    ev1 = Evidence(signals={
        "commitment": Signal(direction=1, strength=2, confidence=0.9),
        "specificity": Signal(direction=1, strength=2, confidence=0.9),
        "borrower_initiation": Signal(direction=1, strength=2, confidence=0.9),
        "confirmation": Signal(direction=1, strength=2, confidence=0.9),
    })
    s1 = mgr.update(ev1, speaker="borrower", text="I will pay 2000 on 10th.")
    score1 = credibility_score(s1)
    assert score1 >= 65

    # Negative / evasive utterance drops score
    ev2 = Evidence(signals={
        "escape_signal": Signal(direction=-1, strength=3, confidence=0.95),
        "commitment": Signal(direction=-1, strength=2, confidence=0.90),
    })
    s2 = mgr.update(ev2, speaker="borrower", text="Bas call rakhiye sir, nahi ho payega.")
    score2 = credibility_score(s2)
    assert score2 < score1


def test_boolean_flags_not_source_of_truth():
    # Construct state where legacy boolean flags are contrary to numeric state
    s = ConversationState(
        conditional=True,  # Old boolean flag
        hardship=True,     # Old boolean flag
        raw_state={dim: 0.0 for dim in DIMENSIONS}
    )
    # Set numeric state to strongly positive
    s.raw_state["commitment"] = 4.0
    s.raw_state["specificity"] = 4.0
    s.raw_state["confirmation"] = 4.0
    s.raw_state["borrower_initiation"] = 4.0
    s.raw_state["feasibility"] = 3.0
    s.raw_state["conditionality"] = 3.0  # Actually unconditional in numeric state
    s.raw_state["hardship"] = 2.0        # Actually no hardship in numeric state
    s.normalized_state = normalize_raw_state(s.raw_state, DEFAULT_SCALE)

    score = credibility_score(s)
    classification = classify_ptp(s)

    # Numeric state drives the outcome, not the stale booleans
    assert score >= 75
    assert classification == "GENUINE_FEASIBLE"


def test_classification_based_on_current_state():
    mgr = StateManager()
    # Step 1: Escape promise
    ev1 = Evidence(signals={"escape_signal": Signal(direction=-1, strength=3, confidence=0.95)})
    s1 = mgr.update(ev1, speaker="borrower", text="End the call, will see later.")
    assert classify_ptp(s1) == "ESCAPE_PROMISE"

    # Step 2: Genuine commitment reverses escape signal
    ev2 = Evidence(signals={
        "escape_signal": Signal(direction=1, strength=3, confidence=0.95),
        "commitment": Signal(direction=1, strength=3, confidence=0.95),
        "specificity": Signal(direction=1, strength=3, confidence=0.95),
        "confirmation": Signal(direction=1, strength=3, confidence=0.95),
    })
    s2 = mgr.update(ev2, speaker="borrower", text="Wait, I confirm 2000 on 10th definitely.")
    assert classify_ptp(s2) == "GENUINE_FEASIBLE"


def test_agent_pushed_can_later_become_borrower_initiated():
    mgr = StateManager()
    # Agent pushes
    ev1 = Evidence(signals={"agent_pressure": Signal(direction=-1, strength=2, confidence=0.9)})
    s1 = mgr.update(ev1, speaker="agent", text="Pay ₹5000 on 10th right?")
    assert classify_ptp(s1) == "AGENT_RECORDED_OR_PUSHED"

    # Borrower actively confirms and takes ownership
    ev2 = Evidence(signals={
        "borrower_initiation": Signal(direction=1, strength=2, confidence=0.9),
        "confirmation": Signal(direction=1, strength=2, confidence=0.95),
        "commitment": Signal(direction=1, strength=2, confidence=0.9),
        "specificity": Signal(direction=1, strength=2, confidence=0.9),
    })
    s2 = mgr.update(ev2, speaker="borrower", text="Yes, I confirm I will pay 5000 on 10th.")
    assert classify_ptp(s2) != "AGENT_RECORDED_OR_PUSHED"


def test_hardship_can_weaken_based_on_later_evidence():
    mgr = StateManager()
    # Hardship reported
    ev1 = Evidence(signals={"hardship": Signal(direction=-1, strength=2, confidence=0.9)})
    s1 = mgr.update(ev1, speaker="borrower", text="Medical problem, no funds right now.")
    hard1 = s1.raw_state["hardship"]
    assert hard1 < -1.5

    # Hardship resolved
    ev2 = Evidence(signals={"hardship": Signal(direction=1, strength=2, confidence=0.9)})
    s2 = mgr.update(ev2, speaker="borrower", text="Medical issue cleared, salary has come.")
    hard2 = s2.raw_state["hardship"]
    assert hard2 > -0.5
    assert hard2 > hard1


def test_third_party_evidence_can_be_corrected_later():
    mgr = StateManager()
    # Third party initially speaks
    ev1 = Evidence(signals={"third_party": Signal(direction=-1, strength=3, confidence=0.95)})
    s1 = mgr.update(ev1, speaker="third_party", text="I am his brother.")
    assert classify_ptp(s1) == "THIRD_PARTY_PROMISE"

    # Borrower takes over and confirms personal responsibility
    ev2 = Evidence(signals={
        "third_party": Signal(direction=1, strength=3, confidence=0.95),
        "commitment": Signal(direction=1, strength=2, confidence=0.9),
        "specificity": Signal(direction=1, strength=2, confidence=0.9),
        "confirmation": Signal(direction=1, strength=2, confidence=0.9),
    })
    s2 = mgr.update(ev2, speaker="borrower", text="This is the borrower speaking, I confirm my payment.")
    assert classify_ptp(s2) != "THIRD_PARTY_PROMISE"


def test_specificity_does_not_imply_feasibility():
    mgr = StateManager()
    # Large specific amount, but no feasibility evidence
    ev = Evidence(signals={"specificity": Signal(direction=1, strength=3, confidence=0.95)})
    s = mgr.update(ev, speaker="borrower", text="I will pay ₹1,00,000 on 10th.")
    assert s.raw_state["specificity"] > 2.0
    # Feasibility must remain neutral, not automatically positive
    assert s.raw_state["feasibility"] == 0.0


def test_hardship_safety_override_works_independently():
    s = ConversationState(
        raw_state={
            "commitment": 4.0,
            "specificity": 4.0,
            "confirmation": 3.0,
            "hardship": -3.5,  # Severe hardship
        }
    )
    s.normalized_state = normalize_raw_state(s.raw_state, DEFAULT_SCALE)
    score = credibility_score(s)
    # Score may still be moderate/high because commitment is concrete
    action, msg = policy(s, score)
    # Safety override forces HUMAN_OR_RESTRUCTURE regardless of score
    assert action == "HUMAN_OR_RESTRUCTURE"


def test_existing_api_endpoints_work():
    client = TestClient(app)
    # Reset
    res_reset = client.post("/api/call/reset")
    assert res_reset.status_code == 200

    # Utterance
    res_utt = client.post("/api/call/utterance", json={"speaker": "borrower", "text": "10 ko 2000 de dunga"})
    assert res_utt.status_code == 200
    data = res_utt.json()
    assert "score" in data
    assert "raw_state" in data
    assert "normalized_state" in data
    assert "score_history" in data
    assert "evidence_history" in data

    # Health
    res_health = client.get("/api/health")
    assert res_health.status_code == 200


def test_reversal_demo_scenario():
    client = TestClient(app)
    res = client.get("/api/demo/reversal")
    assert res.status_code == 200
    steps = res.json()["steps"]
    assert len(steps) == 4

    # Step 1: Agent asks
    # Step 2: Borrower gives conditional promise ("If salary comes...")
    cond_step2 = steps[1]["raw_state"]["conditionality"]
    assert cond_step2 < -1.0  # Conditionality negative

    # Step 3: Borrower clarifies ("Salary has already come...")
    cond_step3 = steps[2]["raw_state"]["conditionality"]
    assert cond_step3 > cond_step2  # Conditionality reversed upward

    # Final step should have high credibility and feasible classification
    final_step = steps[-1]
    assert final_step["score"] >= 65
    assert final_step["ptp_type"] == "GENUINE_FEASIBLE"
