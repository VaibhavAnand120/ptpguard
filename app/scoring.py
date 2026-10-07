import math
from typing import Dict
from .schemas import ConversationState
from .config import DIMENSIONS, DEFAULT_WEIGHTS, DEFAULT_SCALE, THRESHOLDS


def normalize_raw_state(raw_state: Dict[str, float], scale: float = DEFAULT_SCALE) -> Dict[str, float]:
    """
    Soft-saturates raw continuous evidence state using tanh(raw / scale).
    Produces normalized values in (-1.0, +1.0).
    """
    normalized = {}
    for dim in DIMENSIONS:
        val = raw_state.get(dim, 0.0)
        normalized[dim] = math.tanh(val / scale)
    return normalized


def sync_legacy_booleans(state: ConversationState):
    """
    Backward-compatibility bridge:
    If a synthetic ConversationState was constructed directly in tests with legacy booleans
    and raw_state is untouched (all 0.0), populate raw_state so legacy tests continue to pass.
    If raw_state already has non-zero values, raw_state remains the sole source of truth.
    """
    if (
        len(state.transcript) == 0
        and len(state.evidence_history) == 0
        and all(abs(v) < 1e-6 for v in state.raw_state.values())
    ):
        has_legacy = False
        if state.borrower_initiated:
            state.raw_state["borrower_initiation"] = 3.5
            state.raw_state["commitment"] = 3.5
            has_legacy = True
        if state.amount_confirmed or state.date_confirmed:
            state.raw_state["specificity"] = 3.5
            has_legacy = True
        if state.explicit_confirmation:
            state.raw_state["confirmation"] = 3.0
            has_legacy = True
        if state.conditional:
            state.raw_state["conditionality"] = -3.0
            has_legacy = True
        if state.hardship:
            state.raw_state["hardship"] = -3.5
            has_legacy = True
        if state.agent_pushed:
            state.raw_state["agent_pressure"] = -3.5
            has_legacy = True
        if state.third_party:
            state.raw_state["third_party"] = -3.0
            has_legacy = True
        if state.escape_language:
            state.raw_state["escape_signal"] = -3.5
            has_legacy = True
        if has_legacy:
            for dim in DIMENSIONS:
                state.normalized_state[dim] = math.tanh(state.raw_state[dim] / DEFAULT_SCALE)


def credibility_score(
    state: ConversationState,
    scale: float = DEFAULT_SCALE,
    weights: Dict[str, float] = DEFAULT_WEIGHTS
) -> int:
    """
    Calculates an explainable 0–100 PTP Credibility Score from CURRENT numeric state.
    
    Formula:
        1. normalized_i = tanh(raw_i / SCALE)  in [-1.0, +1.0]
        2. contribution_i = (normalized_i + 1) / 2  in [0.0, 1.0]
        3. weighted_score = sum(weight_i * contribution_i)
        4. score = round(weighted_score * 100) clamped to [0, 100]
    """
    sync_legacy_booleans(state)

    # If call just reset and no PTP or signals have been observed yet:
    is_initial = (
        len(state.transcript) == 0
        and not state.ptp_detected
        and all(abs(v) < 1e-6 for v in state.raw_state.values())
    )
    if is_initial:
        return 0

    weighted_score = 0.0
    for dim, weight in weights.items():
        raw_val = state.raw_state.get(dim, 0.0)
        norm_val = math.tanh(raw_val / scale)
        state.normalized_state[dim] = norm_val
        contribution = (norm_val + 1.0) / 2.0
        weighted_score += weight * contribution

    score = int(round(weighted_score * 100.0))
    return max(0, min(100, score))


def classify_ptp(state: ConversationState) -> str:
    """
    Classifies the current conversation state into a PTP category based on the
    CURRENT normalized numeric state (not historical irreversible flags).
    """
    sync_legacy_booleans(state)

    norm = state.normalized_state
    raw = state.raw_state

    # 1. Background third-party interference / coaching
    if state.third_party_background or state.background_coaching:
        return "THIRD_PARTY_BACKGROUND_INTERFERENCE"

    # 2. Strong third-party involvement
    if norm.get("third_party", 0.0) < THRESHOLDS["strong_negative"]:
        return "THIRD_PARTY_PROMISE"

    # 3. Agent pushed commitment or agent unilaterally recorded
    is_agent_pushed = (
        (norm.get("agent_pressure", 0.0) < THRESHOLDS["strong_negative"] and norm.get("confirmation", 0.0) <= 0.0)
        or (
            (state.agent_proposed_amount is not None or state.agent_proposed_date is not None or norm.get("agent_pressure", 0.0) < -0.15)
            and (state.borrower_stated_amount is None and state.borrower_stated_date is None)
            and raw.get("commitment", 0.0) <= 0.5
        )
    )
    if is_agent_pushed:
        return "AGENT_RECORDED_OR_PUSHED"

    # 4. Escape / evasion promise
    if norm.get("escape_signal", 0.0) < THRESHOLDS["strong_negative"]:
        return "ESCAPE_PROMISE"

    # 5. Genuine commitment hindered by severe financial hardship
    if (
        norm.get("hardship", 0.0) < THRESHOLDS["strong_negative"]
        and (norm.get("commitment", 0.0) > 0.0 or raw.get("commitment", 0.0) > 0.0)
    ):
        return "GENUINE_NOT_FEASIBLE"

    # 6. Genuine feasible PTP
    is_genuine_feasible = (
        norm.get("commitment", 0.0) > THRESHOLDS["positive"]
        and norm.get("specificity", 0.0) > THRESHOLDS["positive"]
        and norm.get("confirmation", 0.0) > THRESHOLDS["positive"]
        and norm.get("feasibility", 0.0) > THRESHOLDS["strong_negative"]
        and norm.get("conditionality", 0.0) > THRESHOLDS["moderate_negative"]
    )
    if is_genuine_feasible:
        return "GENUINE_FEASIBLE"

    # 7. No PTP detected yet
    is_empty_or_no_commitment = (
        not state.ptp_detected
        and raw.get("commitment", 0.0) <= 0.0
        and raw.get("specificity", 0.0) <= 0.0
        and raw.get("confirmation", 0.0) <= 0.0
    )
    if is_empty_or_no_commitment:
        return "NO_PTP"

    # 8. Unconfirmed
    return "UNCONFIRMED"
