from app.schemas import ConversationState
from app.scoring import credibility_score, classify_ptp

def test_no_ptp_is_zero():
    s = ConversationState()
    assert credibility_score(s) == 0
    assert classify_ptp(s) == "NO_PTP"

def test_genuine_feasible():
    s = ConversationState(
        ptp_detected=True,
        amount_confirmed=True,
        date_confirmed=True,
        borrower_initiated=True,
        explicit_confirmation=True
    )
    assert credibility_score(s) >= 70
    assert classify_ptp(s) == "GENUINE_FEASIBLE"

def test_hardship():
    s = ConversationState(
        ptp_detected=True,
        amount_confirmed=True,
        date_confirmed=True,
        borrower_initiated=True,
        hardship=True
    )
    assert classify_ptp(s) == "GENUINE_NOT_FEASIBLE"
