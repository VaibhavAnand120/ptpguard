from .schemas import ConversationState
from .scoring import sync_legacy_booleans
from .config import THRESHOLDS


def policy(state: ConversationState, score: int):
    """
    Evaluates business and compliance policy actions based on the current
    evidence state and credibility score.
    """
    sync_legacy_booleans(state)
    norm = state.normalized_state

    # 1. Third-party background presence / coaching override: verify borrower privacy
    if (
        state.third_party_background
        or state.background_coaching
        or norm.get("third_party", 0.0) < -0.60
    ):
        return (
            "VERIFY_BORROWER_PRIVACY",
            "Third-party background speech detected (spouse/family member speaking or coaching in background). Verify borrower privacy, ensure borrower is answering without external coercion, and confirm direct commitment."
        )

    # 2. Safety override: Hardship must NOT lead directly to aggressive treatment
    if norm.get("hardship", 0.0) < THRESHOLDS["strong_negative"] or state.hardship:
        return (
            "HUMAN_OR_RESTRUCTURE",
            "Hardship indicators detected. Confirm a realistic date or consider restructuring/partial payment. Avoid aggressive treatment."
        )

    # 3. Third-party commitment confirmation
    if norm.get("third_party", 0.0) < THRESHOLDS["strong_negative"] or state.third_party:
        return (
            "CONFIRM_WITH_BORROWER",
            "A third party appears to be making the promise. Confirm the commitment with the borrower before parking the account."
        )

    # 4. Agent-pushed commitment without borrower confirmation
    if (
        (norm.get("agent_pressure", 0.0) < THRESHOLDS["strong_negative"] and norm.get("confirmation", 0.0) <= 0.0)
        or (state.agent_pushed and not state.explicit_confirmation)
    ):
        return (
            "CONFIRM_PTP",
            "The agent appears to have supplied the amount/date. Obtain explicit borrower confirmation before treating it as a firm PTP."
        )

    # 5. Escape language or low credibility
    if (
        norm.get("escape_signal", 0.0) < THRESHOLDS["strong_negative"]
        or state.escape_language
        or score < 40
    ):
        return (
            "FIRM_UP_PROMISE",
            "Ask for a concrete amount and date, and consider an immediate partial payment if appropriate."
        )

    # 6. Unconfirmed / moderate credibility
    if score < 70:
        return (
            "CLARIFY_PROMISE",
            "Confirm the promised amount, date and payment capability before parking the account."
        )

    # 7. Highly concrete and credible
    return (
        "NORMAL_PARKING",
        "PTP is sufficiently concrete. Park according to the normal reminder policy."
    )
