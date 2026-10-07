from app.schemas import ConversationState
from app.policy import policy

def test_hardship_overrides():
    s = ConversationState(ptp_detected=True, hardship=True)
    action, _ = policy(s, 10)
    assert action == "HUMAN_OR_RESTRUCTURE"

def test_agent_push():
    s = ConversationState(ptp_detected=True, agent_pushed=True)
    action, _ = policy(s, 60)
    assert action == "CONFIRM_PTP"
