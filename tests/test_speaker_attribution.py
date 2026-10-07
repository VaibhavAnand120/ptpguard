"""
Comprehensive Speaker Attribution & Role Resolution Tests
Validates:
1. Agent → Borrower → Agent → Borrower alternation
2. Consecutive utterances from the same speaker
3. Very short utterances ("yes", "okay", "hmm")
4. Similar vocabulary between agent and borrower (numbers/dates do not flip roles)
5. Agent proposing date/amount does NOT create borrower commitment
6. Borrower independently stating date/amount produces strong commitment
7. Third-party speaker detection and separation
8. Ambiguous role resolution does NOT lock into borrower
9. Auto mode ("auto") runs full Sortformer + RoleResolver without manual override
10. Manual text simulation mode preserves explicit user selection
11. Verification that an ambiguous utterance does NOT automatically inherit previous speaker
12. Section 8 exact benchmark scenario
13. Section 9 exact failure case (Agent pushed / unilateral mark)
"""

import pytest
import asyncio
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import Utterance
from app.voice.pipeline import ParallelVoicePipeline
from app.voice.sortformer import StreamingSortformerDiarizer
from app.voice.roles import RoleResolver
from app.semantic.rules import RulesAnalyzer


@pytest.fixture
def pipeline():
    p = ParallelVoicePipeline()
    p.reset()
    return p


# 1. Agent → Borrower → Agent → Borrower Alternation
@pytest.mark.anyio
async def test_agent_borrower_alternation(pipeline):
    conversation = [
        "Namaste sir, calling from Bajaj Finance regarding pending EMI.",
        "Sir abhi paise nahi hain, salary late chal rahi hai.",
        "Aap payment kab tak clear karenge sir?",
        "Main 15 tareekh ko payment kar dunga pakka."
    ]
    speakers = []
    roles = []
    for text in conversation:
        res = await pipeline.process_utterance(Utterance(speaker="auto", text=text))
        speakers.append(res["speaker_id"])
        roles.append(res["speaker_role"])

    assert speakers == ["speaker_0", "speaker_1", "speaker_0", "speaker_1"]
    assert roles == ["agent", "borrower", "agent", "borrower"]


# 2. Consecutive Utterances From Same Speaker
@pytest.mark.anyio
async def test_consecutive_utterances_same_speaker(pipeline):
    # Agent sends two consecutive utterances
    res1 = await pipeline.process_utterance(Utterance(speaker="auto", text="Hello sir."))
    res2 = await pipeline.process_utterance(Utterance(speaker="auto", text="I am calling from HDFC Bank regarding your overdue loan EMI."))
    assert res1["speaker_id"] == "speaker_0"
    assert res2["speaker_id"] == "speaker_0"
    assert res1["speaker_role"] in ("agent", "unknown")
    assert res2["speaker_role"] == "agent"

    # Borrower responds with two consecutive utterances
    res3 = await pipeline.process_utterance(Utterance(speaker="auto", text="Actually sir, my dukaan suffered heavy losses."))
    res4 = await pipeline.process_utterance(Utterance(speaker="auto", text="Meri salary bhi nahi aayi hai abhi tak."))
    assert res3["speaker_id"] == "speaker_1"
    assert res4["speaker_id"] == "speaker_1"
    assert res3["speaker_role"] == "borrower"
    assert res4["speaker_role"] == "borrower"


# 3. Very Short Utterances ("yes", "okay", "hmm")
@pytest.mark.anyio
async def test_short_utterances_not_attributed_to_self(pipeline):
    # Agent asks a question
    res1 = await pipeline.process_utterance(Utterance(speaker="auto", text="Can you pay on 10th?"))
    assert res1["speaker_id"] == "speaker_0"
    assert res1["speaker_role"] == "agent"

    # Borrower says short "yes"
    res2 = await pipeline.process_utterance(Utterance(speaker="auto", text="Yes."))
    assert res2["speaker_id"] == "speaker_1"
    assert res2["speaker_role"] == "borrower"

    # Agent says short "okay"
    res3 = await pipeline.process_utterance(Utterance(speaker="auto", text="Okay."))
    assert res3["speaker_id"] == "speaker_0"
    assert res3["speaker_role"] == "agent"


# 4. Similar Vocabulary (Borrower stating amounts/dates does not flip into Agent)
@pytest.mark.anyio
async def test_similar_vocabulary_role_inertia(pipeline):
    await pipeline.process_utterance(Utterance(speaker="auto", text="Hello sir, when will you pay your overdue EMI?"))
    await pipeline.process_utterance(Utterance(speaker="auto", text="Actually sir, I have financial difficulties."))

    # Borrower mentions financial terms and dates
    res3 = await pipeline.process_utterance(Utterance(speaker="auto", text="I can pay ₹5,000 on 10 October."))
    assert res3["speaker_id"] == "speaker_1"
    assert res3["speaker_role"] == "borrower"
    assert res3["role_status"] == "confirmed"


# 5. Agent Proposing Date/Amount Does NOT Create Borrower Commitment
@pytest.mark.anyio
async def test_agent_proposing_date_amount_no_borrower_commitment(pipeline):
    res1 = await pipeline.process_utterance(Utterance(speaker="auto", text="Can you pay ₹10,000 before 15th?"))
    assert res1["speaker_role"] == "agent"
    raw_state = res1["raw_state"]
    assert raw_state.get("commitment", 0.0) <= 0.0
    assert raw_state.get("specificity", 0.0) <= 0.0


# 6. Borrower Independently Stating Date/Amount
@pytest.mark.anyio
async def test_borrower_independent_statement(pipeline):
    await pipeline.process_utterance(Utterance(speaker="auto", text="When will you pay?"))
    res = await pipeline.process_utterance(Utterance(speaker="auto", text="Yes, I can pay ₹3,000 on 15th."))
    assert res["speaker_role"] == "borrower"
    raw_state = res["raw_state"]
    assert raw_state.get("commitment", 0.0) > 0.0
    assert raw_state.get("specificity", 0.0) > 0.0
    assert raw_state.get("confirmation", 0.0) > 0.0


# 7. Third-Party Speaker
@pytest.mark.anyio
async def test_third_party_speaker(pipeline):
    res1 = await pipeline.process_utterance(Utterance(speaker="auto", text="Hello sir, kab payment karenge?"))
    res2 = await pipeline.process_utterance(Utterance(speaker="auto", text="Main unka bhai bol raha hoon, hospital me hain wo."))
    assert res2["speaker_id"] == "speaker_2"
    assert "third_party" in res2["speaker_role"]
    assert res2["raw_state"].get("third_party", 0.0) < 0.0


# 8. Ambiguous Utterance Does NOT Automatically Inherit Previous Speaker
@pytest.mark.anyio
async def test_ambiguous_utterance_does_not_inherit_previous_speaker(pipeline):
    # Turn 1: Borrower speaks
    res1 = await pipeline.process_utterance(Utterance(speaker="borrower", text="I cannot pay today."))
    assert res1["speaker_id"] == "speaker_1"
    assert res1["speaker_role"] == "borrower"

    # Turn 2: Ambiguous utterance from other side
    res2 = await pipeline.process_utterance(Utterance(speaker="auto", text="Can you pay tomorrow?"))
    assert res2["speaker_id"] == "speaker_0"
    assert res2["speaker_role"] == "agent"


# 9. Auto Mode Uses speaker="auto" Without Overrides
@pytest.mark.anyio
async def test_auto_mode_no_spurious_override(pipeline):
    res = await pipeline.process_utterance(Utterance(speaker="auto", text="Namaste, calling from SBI Card."))
    assert res["mode"] == "auto"
    assert res["speaker_role"] == "agent"


# 10. Manual Text Simulation Mode
@pytest.mark.anyio
async def test_manual_text_simulation_mode(pipeline):
    res = await pipeline.process_utterance(Utterance(speaker="agent", text="I will record your response."))
    assert res["mode"] == "manual_simulation"
    assert res["speaker_role"] == "agent"
    assert res["role_status"] == "manual_simulation"


# 11. Section 8 Exact Benchmark Test
@pytest.mark.anyio
async def test_section_8_exact_scenario(pipeline):
    turns = [
        "Hello sir, when will you pay the required amount?",
        "I know I need to pay the amount but I don't have money right now.",
        "Can you pay before 10 October?",
        "Yes, I can pay before 10 October."
    ]

    r1 = await pipeline.process_utterance(Utterance(speaker="auto", text=turns[0]))
    assert r1["speaker_id"] == "speaker_0"
    assert r1["speaker_role"] == "agent"

    r2 = await pipeline.process_utterance(Utterance(speaker="auto", text=turns[1]))
    assert r2["speaker_id"] == "speaker_1"
    assert r2["speaker_role"] == "borrower"
    # Hardship detected on borrower
    assert r2["raw_state"]["hardship"] < 0.0

    r3 = await pipeline.process_utterance(Utterance(speaker="auto", text=turns[2]))
    assert r3["speaker_id"] == "speaker_0"
    assert r3["speaker_role"] == "agent"
    # Agent's question must NOT itself create borrower commitment
    assert r3["raw_state"]["commitment"] <= 0.0

    r4 = await pipeline.process_utterance(Utterance(speaker="auto", text=turns[3]))
    assert r4["speaker_id"] == "speaker_1"
    assert r4["speaker_role"] == "borrower"

    # Expected evidence: Hardship negative, Commitment positive, Specificity positive, Confirmation positive
    raw = r4["raw_state"]
    assert raw["hardship"] < 0.0
    assert raw["commitment"] > 0.0
    assert raw["specificity"] > 0.0
    assert raw["confirmation"] > 0.0

    # Credibility score is explainable and updated
    assert r4["score"] > 0
    assert r4["ptp_type"] == "GENUINE_NOT_FEASIBLE"


# 12. Section 9 Exact Failure Case Test (Agent push without borrower commitment)
@pytest.mark.anyio
async def test_section_9_failure_case(pipeline):
    turns = [
        "I will mark ₹5,000 on 10 October.",
        "Okay."
    ]

    r1 = await pipeline.process_utterance(Utterance(speaker="auto", text=turns[0]))
    assert r1["speaker_id"] == "speaker_0"
    assert r1["speaker_role"] == "agent"

    r2 = await pipeline.process_utterance(Utterance(speaker="auto", text=turns[1]))
    assert r2["speaker_id"] == "speaker_1"
    assert r2["speaker_role"] == "borrower"

    # Borrower did NOT independently state date/amount or give strong commitment
    raw = r2["raw_state"]
    assert raw["commitment"] <= 0.5
    # Must be categorized as AGENT_RECORDED_OR_PUSHED
    assert r2["ptp_type"] == "AGENT_RECORDED_OR_PUSHED"


# 13. API Endpoint Demos for Section 8 & Failure Case
def test_demo_section8_and_failure_case():
    client = TestClient(app)

    res8 = client.get("/api/demo/section8")
    assert res8.status_code == 200
    data8 = res8.json()
    assert len(data8["steps"]) == 4
    last8 = data8["steps"][-1]
    assert last8["speaker_role"] == "borrower"
    assert last8["raw_state"]["hardship"] < 0.0
    assert last8["raw_state"]["commitment"] > 0.0
    assert last8["ptp_type"] == "GENUINE_NOT_FEASIBLE"

    res_f = client.get("/api/demo/failure_case")
    assert res_f.status_code == 200
    data_f = res_f.json()
    assert len(data_f["steps"]) == 2
    last_f = data_f["steps"][-1]
    assert last_f["ptp_type"] == "AGENT_RECORDED_OR_PUSHED"


# 14. Exact User Scenario: Payment Deadline vs Can't Pay
@pytest.mark.anyio
async def test_deadline_payment_conversation_speaker_roles(pipeline):
    conversation = [
        "Hello sir, you have a deadline of payment of ₹5000 by 10th October.",
        "Yes sir, I will try my best but I don't know when to pay.",
        "You need to pay before 10 October.",
        "I don't have money right now."
    ]
    results = []
    for text in conversation:
        res = await pipeline.process_utterance(Utterance(speaker="auto", text=text))
        results.append(res)

    spk_roles = pipeline.role_resolver.speaker_roles
    assert "speaker_0" in spk_roles
    assert "speaker_1" in spk_roles

    role_0 = spk_roles["speaker_0"].role
    role_1 = spk_roles["speaker_1"].role

    # In a 2-person collection call, exactly one is AGENT and one is BORROWER
    # NEVER both BORROWER or both AGENT
    assert not (role_0 == "borrower" and role_1 == "borrower")
    assert not (role_0 == "agent" and role_1 == "agent")
    assert role_0 == "agent"
    assert role_1 == "borrower"
    assert spk_roles["speaker_0"].confidence >= 0.85
    assert spk_roles["speaker_1"].confidence >= 0.85


# 15. Reversed Speaker IDs Test
@pytest.mark.anyio
async def test_role_resolver_reversed_speaker_ids():
    resolver = RoleResolver()
    dialogue = [
        ("speaker_1", "Hello sir, you have a deadline of payment of ₹5000 by 10th October."),
        ("speaker_0", "Yes sir, I will try my best but I don't know when to pay."),
        ("speaker_1", "You need to pay before 10 October."),
        ("speaker_0", "I don't have money right now.")
    ]
    for spk, text in dialogue:
        resolver.resolve_role(spk, text)

    assert resolver.speaker_roles["speaker_1"].role == "agent"
    assert resolver.speaker_roles["speaker_0"].role == "borrower"
    assert not (resolver.speaker_roles["speaker_0"].role == "borrower" and resolver.speaker_roles["speaker_1"].role == "borrower")


# 16. Ambiguous Utterances Remain UNKNOWN
@pytest.mark.anyio
async def test_role_resolver_ambiguous_stays_unknown():
    resolver = RoleResolver()
    resolver.resolve_role("speaker_0", "Hello.")
    resolver.resolve_role("speaker_1", "Yes.")
    assert resolver.speaker_roles["speaker_0"].role == "unknown"
    assert resolver.speaker_roles["speaker_1"].role == "unknown"

