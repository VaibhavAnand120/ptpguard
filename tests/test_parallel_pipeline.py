import pytest
import anyio
from fastapi.testclient import TestClient

from app.main import app
from app.voice.sortformer import StreamingSortformerDiarizer, SpeakerSegment
from app.voice.asr import StreamingASR, TranscriptSegment
from app.voice.alignment import TimestampAlignmentEngine
from app.voice.roles import RoleResolver, SpeakerRoleState
from app.voice.telemetry import LatencyMonitor, TurnLatencyRecord
from app.voice.pipeline import ParallelVoicePipeline
from app.schemas import Utterance


def test_sortformer_four_speakers_streaming():
    diarizer = StreamingSortformerDiarizer()
    assert diarizer.MODEL_NAME == "nvidia/diar_streaming_sortformer_4spk-v2.1"
    assert diarizer.MAX_SPEAKERS == 4

    # 4 distinct acoustic turns
    seg0 = diarizer.process_audio_chunk(b"chunk0", timestamp=0.0, duration=1.0)
    assert seg0[0].speaker_id == "speaker_0"

    seg1 = diarizer.process_audio_chunk(b"chunk1_diff", timestamp=1.5, duration=1.0)
    assert seg1[0].speaker_id in ("speaker_0", "speaker_1")

    # Text stream assigns up to 4 speakers
    diarizer.reset()
    s0 = diarizer.process_utterance_event("Calling from Bajaj Finance.", 0.0, 2.0)[0].speaker_id
    s1 = diarizer.process_utterance_event("Sir abhi paise nahi hai.", 2.1, 4.0)[0].speaker_id
    s2 = diarizer.process_utterance_event("(Peeche se patni: phone kaato unka)", 4.1, 5.5, background_hint=True)[0].speaker_id
    s3 = diarizer.process_utterance_event("Main unka bhai bol raha hoon.", 5.6, 7.0)[0].speaker_id

    assert s0 == "speaker_0"
    assert s1 == "speaker_1"
    assert s2 in ("speaker_2", "speaker_3")
    assert s3 in ("speaker_2", "speaker_3")


def test_streaming_asr_parallel():
    asr = StreamingASR()
    seg = asr.segment_from_text("Salary nahi aayi hai", start_time=15.9, duration=2.5)
    assert seg.text == "Salary nahi aayi hai"
    assert seg.start == 15.9
    assert seg.end == 18.4


def test_alignment_fusion_canonical_turn():
    engine = TimestampAlignmentEngine()
    engine.register_speaker_segment(SpeakerSegment(speaker_id="speaker_1", start=15.8, end=19.2))

    asr_seg = TranscriptSegment(text="Salary nahi aayi hai", start=15.9, end=18.4)
    aligned = engine.align(asr_seg)

    assert aligned.speaker_id == "speaker_1"
    assert aligned.text == "Salary nahi aayi hai"
    assert aligned.start == 15.9
    assert aligned.end == 18.4
    assert aligned.overlap_ratio >= 0.8


def test_persistent_role_resolution():
    resolver = RoleResolver()

    # Turn 1: Agent intro
    r0 = resolver.resolve_role("speaker_0", "Hello sir, I am calling from Bajaj Finance regarding your overdue EMI.")
    assert r0.role == "agent"
    assert r0.confidence >= 0.85
    assert r0.status == "confirmed"

    # Turn 2: Borrower hardship
    r1 = resolver.resolve_role("speaker_1", "Actually sir, my salary has not come yet.")
    assert r1.role == "borrower"
    assert r1.confidence >= 0.85
    assert r1.status == "confirmed"

    # Turn 3: Borrower quotes amount/date — MUST NOT FLIP TO AGENT!
    r1_next = resolver.resolve_role("speaker_1", "I can pay 2000 on the 15th.")
    assert r1_next.role == "borrower"
    assert r1_next.status == "confirmed"


def test_provisional_to_confirmed_role():
    resolver = RoleResolver()
    # Weak ambiguous statement
    state = resolver.resolve_role("speaker_0", "Hello.")
    assert state.status == "provisional"

    # Strong institutional statement confirms
    state2 = resolver.resolve_role("speaker_0", "This is HDFC Bank recovery department regarding your overdue EMI.")
    assert state2.role == "agent"
    assert state2.status == "confirmed"
    assert state2.confidence >= 0.85


def test_telemetry_latency_tracking():
    monitor = LatencyMonitor()
    for lat in [12.0, 18.0, 25.0, 32.0, 45.0, 150.0]:
        monitor.record_turn(TurnLatencyRecord(total_end_to_end_ms=lat))

    summary = monitor.get_summary()
    assert summary["sample_count"] == 6
    assert summary["p50_ms"] > 0
    assert summary["p95_ms"] >= summary["p50_ms"]
    assert summary["p99_ms"] >= summary["p95_ms"]


@pytest.mark.anyio
async def test_section_18_exact_scenario():
    """
    Test the exact Section 18 scenario:
    Agent: 'Hello sir, when will you pay the required amount?'
    Borrower: 'Actually sir, it is quite difficult right now. I have not received any salary and my bank account is empty.'
    Agent: 'Can you pay on the 15th?'
    Borrower: 'Yes, I can pay on the 15th.'
    Expected:
    Speaker_0 -> AGENT
    Speaker_1 -> BORROWER
    hardship -> negative
    commitment -> positive
    specificity -> positive
    confirmation -> positive
    """
    pipeline = ParallelVoicePipeline()

    # Turn 1
    t1 = await pipeline.process_utterance(Utterance(speaker="auto", text="Hello sir, when will you pay the required amount?"))
    assert t1["speaker_id"] == "speaker_0"
    assert t1["speaker_role"] == "agent"

    # Turn 2
    t2 = await pipeline.process_utterance(Utterance(speaker="auto", text="Actually sir, it is quite difficult right now. I have not received any salary and my bank account is empty."))
    assert t2["speaker_id"] == "speaker_1"
    assert t2["speaker_role"] == "borrower"
    assert t2["raw_state"]["hardship"] < 0  # Hardship negative

    # Turn 3
    t3 = await pipeline.process_utterance(Utterance(speaker="auto", text="Can you pay on the 15th?"))
    assert t3["speaker_id"] == "speaker_0"
    assert t3["speaker_role"] == "agent"

    # Turn 4
    t4 = await pipeline.process_utterance(Utterance(speaker="auto", text="Yes, I can pay on the 15th."))
    assert t4["speaker_id"] == "speaker_1"
    assert t4["speaker_role"] == "borrower"
    assert t4["raw_state"]["commitment"] > 0    # Commitment positive
    assert t4["raw_state"]["specificity"] > 0   # Specificity positive
    assert t4["raw_state"]["confirmation"] > 0  # Confirmation positive

    # Latency instrumentation present
    assert "telemetry" in t4
    assert t4["telemetry"]["current_ms"]["total_end_to_end"] < 50.0  # Fast sub-50ms execution


def test_api_demo_section18():
    client = TestClient(app)
    res = client.get("/api/demo/section18")
    assert res.status_code == 200
    data = res.json()
    assert len(data["steps"]) == 4

    step2 = data["steps"][1]
    assert step2["speaker_id"] == "speaker_1"
    assert step2["speaker_role"] == "borrower"
    assert step2["raw_state"]["hardship"] < 0

    step4 = data["steps"][3]
    assert step4["speaker_id"] == "speaker_1"
    assert step4["speaker_role"] == "borrower"
    assert step4["raw_state"]["commitment"] > 0
    assert step4["raw_state"]["confirmation"] > 0
