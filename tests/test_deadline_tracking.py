import pytest
from app.semantic.rules import RulesAnalyzer
from app.semantic.deadline import (
    DeadlineTracker,
    parse_deadline_day,
    calculate_days_difference,
    deadline_change_penalty,
    extract_payment_deadline_from_text,
    is_payment_commitment_date
)
from app.state import StateManager
from app.scoring import classify_ptp
from fastapi.testclient import TestClient
from app.main import app


# ---------------------------------------------------------------------------
# 1. Borrower states first deadline
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_borrower_states_first_deadline():
    analyzer = RulesAnalyzer()
    sm = StateManager()

    u = "I will pay by 10 October."
    ev = await analyzer.analyze("borrower", u)
    state = sm.update(ev, "borrower", u)

    ds = state.deadline_state
    assert ds.current_deadline == "10 October"
    assert ds.original_deadline == "10 October"
    assert ds.borrower_stated_deadline == "10 October"
    assert ds.agent_proposed_deadline is None
    assert ds.borrower_confirmed_deadline is None
    assert ds.confidence >= 0.90
    assert len(ds.history) == 1
    assert ds.history[0].event_type == "initial"
    assert ds.history[0].borrower_initiated is True


# ---------------------------------------------------------------------------
# 2. Agent proposes a deadline but borrower does not confirm
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_agent_proposes_deadline_borrower_does_not_confirm():
    analyzer = RulesAnalyzer()
    sm = StateManager()

    # Step 1: Agent proposes date
    u1 = "Can you pay on 10 October?"
    ev1 = await analyzer.analyze("agent", u1)
    state1 = sm.update(ev1, "agent", u1)

    ds1 = state1.deadline_state
    assert ds1.agent_proposed_deadline == "10 October"
    assert ds1.current_deadline is None
    assert ds1.borrower_stated_deadline is None
    assert ds1.borrower_confirmed_deadline is None

    # Step 2: Borrower says passive "okay" without confirming date
    u2 = "Okay."
    ev2 = await analyzer.analyze("borrower", u2)
    state2 = sm.update(ev2, "borrower", u2)

    ds2 = state2.deadline_state
    assert ds2.current_deadline is None
    assert ds2.borrower_confirmed_deadline is None

    # Step 3: Agent unilaterally says "I'll mark 10 October"
    u3 = "I'll mark 10 October."
    ev3 = await analyzer.analyze("agent", u3)
    state3 = sm.update(ev3, "agent", u3)

    ds3 = state3.deadline_state
    assert ds3.current_deadline is None  # Still NO borrower deadline created!


# ---------------------------------------------------------------------------
# 3. Borrower confirms agent-proposed deadline
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_borrower_confirms_agent_proposed_deadline():
    analyzer = RulesAnalyzer()
    sm = StateManager()

    # Agent proposes
    u1 = "Can you pay on 20 October instead?"
    ev1 = await analyzer.analyze("agent", u1)
    sm.update(ev1, "agent", u1)

    # Borrower confirms explicitly
    u2 = "Yes, 20 October."
    ev2 = await analyzer.analyze("borrower", u2)
    state2 = sm.update(ev2, "borrower", u2)

    ds = state2.deadline_state
    assert ds.agent_proposed_deadline == "20 October"
    assert ds.borrower_confirmed_deadline == "20 October"
    assert ds.current_deadline == "20 October"
    assert len(ds.history) >= 2
    last_event = ds.history[-1]
    assert last_event.event_type == "confirmed_agent_proposal"
    assert last_event.borrower_initiated is False


# ---------------------------------------------------------------------------
# 4. Borrower postpones deadline (+10 days)
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_borrower_postpones_deadline():
    analyzer = RulesAnalyzer()
    sm = StateManager()

    # Original commitment
    u1 = "I will pay by 10 October."
    ev1 = await analyzer.analyze("borrower", u1)
    sm.update(ev1, "borrower", u1)

    # Agent asks
    u2 = "Can you pay earlier?"
    ev2 = await analyzer.analyze("agent", u2)
    sm.update(ev2, "agent", u2)

    # Borrower postpones to 20 October
    u3 = "Okay, I'll pay by 20 October."
    ev3 = await analyzer.analyze("borrower", u3)
    state3 = sm.update(ev3, "borrower", u3)

    ds = state3.deadline_state
    assert ds.current_deadline == "20 October"
    assert ds.previous_deadline == "10 October"
    assert ds.original_deadline == "10 October"
    assert ds.days_diff == 10
    assert ds.postpone_count == 1
    assert ds.last_penalty_pct > 0
    assert len(ds.history) == 2
    assert ds.history[-1].event_type == "postponed"


# ---------------------------------------------------------------------------
# 5. Borrower moves deadline earlier (-5 days)
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_borrower_moves_deadline_earlier():
    analyzer = RulesAnalyzer()
    sm = StateManager()

    # Original
    u1 = "I will pay by 10 October."
    ev1 = await analyzer.analyze("borrower", u1)
    sm.update(ev1, "borrower", u1)

    # Moves earlier
    u2 = "Actually, I can pay on 5 October."
    ev2 = await analyzer.analyze("borrower", u2)
    state2 = sm.update(ev2, "borrower", u2)

    ds = state2.deadline_state
    assert ds.current_deadline == "5 October"
    assert ds.previous_deadline == "10 October"
    assert ds.days_diff == -5
    assert ds.last_penalty_pct == 0  # No penalty for paying earlier!
    assert ds.history[-1].event_type == "accelerated"


# ---------------------------------------------------------------------------
# 6. Borrower repeatedly postpones deadline
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_borrower_repeatedly_postpones_deadline():
    analyzer = RulesAnalyzer()
    sm = StateManager()

    # Initial: 10 October
    u1 = "I will pay by 10 October."
    sm.update(await analyzer.analyze("borrower", u1), "borrower", u1)

    # Postpone 1: 15 October (+5 days)
    u2 = "I need more time, I will pay on 15 October."
    st2 = sm.update(await analyzer.analyze("borrower", u2), "borrower", u2)
    pen1 = st2.deadline_state.last_penalty_pct

    # Postpone 2: 25 October (+10 days)
    u3 = "Sorry, please extend, I will pay on 25 October."
    st3 = sm.update(await analyzer.analyze("borrower", u3), "borrower", u3)
    pen2 = st3.deadline_state.last_penalty_pct

    assert st3.deadline_state.postpone_count == 2
    assert pen2 > pen1  # Recurrence multiplier increases penalty!


# ---------------------------------------------------------------------------
# 7. Deadline changes because of hardship
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_deadline_changes_because_of_hardship():
    analyzer = RulesAnalyzer()
    sm = StateManager()

    # Initial
    u1 = "I will pay by 10 October."
    sm.update(await analyzer.analyze("borrower", u1), "borrower", u1)

    # Hardship change: "I lost my job, so I can't pay on 10 October. I can pay on 20 October."
    u2 = "I lost my job, so I can't pay on 10 October. I can pay on 20 October."
    ev2 = await analyzer.analyze("borrower", u2)
    st2 = sm.update(ev2, "borrower", u2)

    ds = st2.deadline_state
    assert ds.current_deadline == "20 October"
    assert ds.days_diff == 10

    # Hardship evidence is active
    assert st2.raw_state.get("hardship", 0.0) < 0.0

    # Fairness rule: Hardship does NOT classify as ESCAPE_PROMISE
    ptp_type = classify_ptp(st2)
    assert ptp_type != "ESCAPE_PROMISE"
    assert ptp_type in ("GENUINE_NOT_FEASIBLE", "GENUINE_FEASIBLE")

    # Penalty is tempered due to hardship
    last_event = ds.history[-1]
    assert last_event.hardship_present is True
    assert "hardship" in last_event.rationale.lower()


# ---------------------------------------------------------------------------
# 8. Deadline changes after agent pressure
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_deadline_changes_after_agent_pressure():
    analyzer = RulesAnalyzer()
    sm = StateManager()

    # Initial
    u1 = "I can pay on 10 October."
    sm.update(await analyzer.analyze("borrower", u1), "borrower", u1)

    # Agent pressures
    u2 = "No, you need to pay earlier."
    sm.update(await analyzer.analyze("agent", u2), "agent", u2)

    # Borrower responds
    u3 = "Okay, I'll say 5 October."
    ev3 = await analyzer.analyze("borrower", u3)
    st3 = sm.update(ev3, "borrower", u3)

    ds = st3.deadline_state
    assert ds.current_deadline == "5 October"
    assert ds.original_deadline == "10 October"

    # Context records agent pressure and borrower change
    last_event = ds.history[-1]
    assert last_event.agent_pressure is True
    assert last_event.borrower_initiated is True


# ---------------------------------------------------------------------------
# 9. Borrower mentions a date that is not a payment commitment
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_borrower_mentions_date_not_payment_commitment():
    analyzer = RulesAnalyzer()
    sm = StateManager()

    # Historical purchase date
    u1 = "I bought the phone on 10 October, but I cannot pay right now."
    ev1 = await analyzer.analyze("borrower", u1)
    st1 = sm.update(ev1, "borrower", u1)

    assert st1.deadline_state.current_deadline is None
    assert st1.deadline_state.original_deadline is None

    # Incidental event date
    u2 = "My birthday was on 15 October."
    ev2 = await analyzer.analyze("borrower", u2)
    st2 = sm.update(ev2, "borrower", u2)

    assert st2.deadline_state.current_deadline is None


# ---------------------------------------------------------------------------
# 10. Multiple dates mentioned in the same conversation
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_multiple_dates_in_same_conversation():
    analyzer = RulesAnalyzer()
    sm = StateManager()

    # Sentence with both loan start date and future payment date
    u = "The loan started on 5 September, but I will definitely pay by 15 October."
    ev = await analyzer.analyze("borrower", u)
    st = sm.update(ev, "borrower", u)

    # The forward commitment date must be chosen, not the historical loan date
    assert st.deadline_state.current_deadline == "15 October"


# ---------------------------------------------------------------------------
# 11. Test Demo API Endpoints
# ---------------------------------------------------------------------------
def test_demo_deadline_scenarios():
    client = TestClient(app)

    # Demo: Postponed
    res_postponed = client.get("/api/demo/deadline_postponed")
    assert res_postponed.status_code == 200
    steps = res_postponed.json()["steps"]
    final_dl = steps[-1]["state"]["deadline_state"]
    assert final_dl["current_deadline"] == "20 October"
    assert final_dl["previous_deadline"] == "10 October"
    assert final_dl["days_diff"] == 10
    assert final_dl["last_penalty_pct"] > 0

    # Demo: Accelerated
    res_acc = client.get("/api/demo/deadline_accelerated")
    assert res_acc.status_code == 200
    steps_acc = res_acc.json()["steps"]
    final_dl_acc = steps_acc[-1]["state"]["deadline_state"]
    assert final_dl_acc["current_deadline"] == "5 October"
    assert final_dl_acc["days_diff"] == -10
    assert final_dl_acc["last_penalty_pct"] == 0

    # Demo: Hardship Change
    res_hardship = client.get("/api/demo/deadline_hardship")
    assert res_hardship.status_code == 200
    steps_hardship = res_hardship.json()["steps"]
    final_dl_hardship = steps_hardship[-1]["state"]["deadline_state"]
    assert final_dl_hardship["current_deadline"] == "20 October"
    assert steps_hardship[-1]["ptp_type"] != "ESCAPE_PROMISE"
