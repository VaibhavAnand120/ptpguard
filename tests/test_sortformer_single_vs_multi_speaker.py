"""
Unit & Integration Tests for Sortformer Acoustic Diarization & Role Resolution:
1. Single physical voice:
   - Voice A speaking multiple utterances remains on the exact same speaker ID.
   - detected_acoustic_speakers == 1.
   - No second speaker or second role is artificially fabricated.
2. Two distinct physical voices:
   - Voice A -> speaker_0 -> AGENT
   - Voice B -> speaker_1 -> BORROWER
   - When Voice A speaks again -> returns to speaker_0.
   - When Voice B speaks again -> returns to speaker_1.
   - detected_acoustic_speakers == 2.
3. Silence gating:
   - Chunks of silence before any speech do not create spurious speaker tracks.
4. Diagnostic fields:
   - audio_segment, raw label, persistent ID, confidence, acoustic_speaker_changed, state reset.
"""

import pytest
import struct
import math
from app.voice.sortformer import StreamingSortformerDiarizer
from app.voice.roles import RoleResolver
from app.voice.pipeline import ParallelVoicePipeline
from app.schemas import Utterance


def make_pcm_sine(freq: float = 440.0, duration_s: float = 0.5, sample_rate: int = 16000, amplitude: float = 0.5) -> bytes:
    """Generates a pure 16-bit PCM sine wave chunk with real acoustic energy."""
    n_samples = int(duration_s * sample_rate)
    samples = []
    for i in range(n_samples):
        val = amplitude * math.sin(2 * math.pi * freq * (i / sample_rate))
        ival = max(-32767, min(32767, int(val * 32767.0)))
        samples.append(ival)
    return struct.pack(f"<{n_samples}h", *samples)


def make_pcm_silence(duration_s: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Generates pure silence PCM (all zeroes)."""
    n_samples = int(duration_s * sample_rate)
    return struct.pack(f"<{n_samples}h", *[0] * n_samples)


def test_silence_gating_prevents_spurious_speakers():
    """Silence before speech must not create speaker tracks."""
    diarizer = StreamingSortformerDiarizer()
    silence_chunk = make_pcm_silence(0.5)

    # Process 3 silence chunks
    for i in range(3):
        segs = diarizer.process_audio_chunk(silence_chunk, timestamp=i * 0.5, duration=0.5)
        assert len(segs) == 0, f"Silence chunk {i} should not produce speaker segments"
    
    assert len(diarizer.tracks) == 0
    assert diarizer.get_tracks_summary()["detected_acoustic_speakers"] == 0

    # Now real speech arrives (Voice A)
    speech_chunk = make_pcm_sine(freq=300.0, duration_s=0.5, amplitude=0.6)
    segs = diarizer.process_audio_chunk(speech_chunk, timestamp=1.5, duration=0.5)
    assert len(segs) == 1
    assert segs[0].speaker_id == "speaker_0"
    assert len(diarizer.tracks) == 1
    assert diarizer.get_tracks_summary()["detected_acoustic_speakers"] == 1


def test_single_physical_voice_continuity():
    """
    If the same physical voice speaks multiple utterances:
    - Voice A remains on speaker_0 across all turns.
    - acoustic_speaker_changed is False after turn 1.
    - detected_acoustic_speakers == 1.
    - No second speaker or role is fabricated.
    """
    diarizer = StreamingSortformerDiarizer()
    resolver = RoleResolver()

    # Voice A speaking all turns
    turns = [
        "Namaste sir, calling from Bajaj Finance regarding your overdue EMI ₹3500.",
        "Aapka payment deadline 10th October hai.",
        "Sir kab tak payment clear karenge?"
    ]

    for idx, text in enumerate(turns):
        segs = diarizer.process_utterance_event(
            text=text,
            start_time=idx * 3.0,
            end_time=(idx + 1) * 3.0,
            voice_name="Voice A"
        )
        assert len(segs) == 1
        seg = segs[0]
        assert seg.speaker_id == "speaker_0"
        if idx > 0:
            assert seg.acoustic_speaker_changed is False

        # Role resolution for single speaker
        role_state = resolver.resolve_role(seg.speaker_id, text)
        assert role_state.speaker_id == "speaker_0"

    summary = diarizer.get_tracks_summary()
    assert summary["detected_acoustic_speakers"] == 1
    assert list(summary["speakers"].keys()) == ["speaker_0"]

    # Role resolution must not fabricate a second speaker
    role_summary = resolver.get_summary()
    assert len(role_summary) == 1
    assert "speaker_0" in role_summary
    assert role_summary["speaker_0"]["role"] == "agent"
    assert "speaker_1" not in role_summary


def test_two_distinct_physical_voices_dialogue_and_role_resolution():
    """
    Dialog between Voice A and Voice B:
    Voice A: "Hello sir, when will you make the payment?"
    Voice B: "Sir, I don't have money right now."
    Voice A: "Can you pay on 10 October?"
    Voice B: "Yes, I can pay on 10 October."

    Expected:
    - Voice A -> persistent speaker_0 -> AGENT
    - Voice B -> persistent speaker_1 -> BORROWER
    - When Voice A returns -> speaker_0
    - When Voice B returns -> speaker_1
    - detected_acoustic_speakers == 2
    """
    diarizer = StreamingSortformerDiarizer()
    resolver = RoleResolver()

    dialogue = [
        ("Voice A", "Hello sir, when will you make the payment?"),
        ("Voice B", "Sir, I don't have money right now."),
        ("Voice A", "Can you pay on 10 October?"),
        ("Voice B", "Yes, I can pay on 10 October.")
    ]

    speaker_assignments = []
    for idx, (v_name, text) in enumerate(dialogue):
        segs = diarizer.process_utterance_event(
            text=text,
            start_time=idx * 2.5,
            end_time=(idx + 1) * 2.5,
            voice_name=v_name
        )
        assert len(segs) == 1
        seg = segs[0]
        speaker_assignments.append(seg.speaker_id)
        role_state = resolver.resolve_role(seg.speaker_id, text)

    # 1. Acoustic Speaker Continuity Checks
    assert speaker_assignments[0] == "speaker_0"
    assert speaker_assignments[1] == "speaker_1"
    assert speaker_assignments[2] == "speaker_0"  # Voice A returned to speaker_0
    assert speaker_assignments[3] == "speaker_1"  # Voice B returned to speaker_1

    # 2. Number of detected acoustic speakers
    summary = diarizer.get_tracks_summary()
    assert summary["detected_acoustic_speakers"] == 2

    # 3. Role Resolution: exactly one AGENT, exactly one BORROWER
    roles = resolver.get_summary()
    assert len(roles) == 2
    assert roles["speaker_0"]["role"] == "agent"
    assert roles["speaker_0"]["confidence"] >= 0.85
    assert roles["speaker_0"]["status"] == "confirmed"

    assert roles["speaker_1"]["role"] == "borrower"
    assert roles["speaker_1"]["confidence"] >= 0.85
    assert roles["speaker_1"]["status"] == "confirmed"


@pytest.mark.anyio
async def test_parallel_pipeline_e2e_single_vs_multi():
    """Test full ParallelVoicePipeline returns detected_acoustic_speakers and diagnostics."""
    pipeline = ParallelVoicePipeline()
    pipeline.reset()

    # Turn 1: Voice A speaks
    res1 = await pipeline.process_utterance(
        Utterance(speaker="auto", text="Namaste, calling from Bajaj Finance regarding your EMI ₹3500.", voice="Voice A")
    )
    assert res1["detected_acoustic_speakers"] == 1
    assert res1["speaker_id"] == "speaker_0"
    assert "diagnostics" in res1
    assert res1["diagnostics"]["persistent_speaker_id"] == "speaker_0"

    # Turn 2: Voice B speaks
    res2 = await pipeline.process_utterance(
        Utterance(speaker="auto", text="Sir abhi mere paas paise nahi hain, dukaan me nuksan hua hai.", voice="Voice B")
    )
    assert res2["detected_acoustic_speakers"] == 2
    assert res2["speaker_id"] == "speaker_1"
    assert res2["diagnostics"]["acoustic_speaker_changed"] is True

    # Turn 3: Voice A speaks again
    res3 = await pipeline.process_utterance(
        Utterance(speaker="auto", text="Can you pay on 10 October?", voice="Voice A")
    )
    assert res3["detected_acoustic_speakers"] == 2
    assert res3["speaker_id"] == "speaker_0"
    assert res3["diagnostics"]["acoustic_speaker_changed"] is True
    assert res3["speaker_role"] == "agent"

    # Turn 4: Voice B speaks again
    res4 = await pipeline.process_utterance(
        Utterance(speaker="auto", text="Yes, I can pay on 10 October.", voice="Voice B")
    )
    assert res4["detected_acoustic_speakers"] == 2
    assert res4["speaker_id"] == "speaker_1"
    assert res4["diagnostics"]["acoustic_speaker_changed"] is True
    assert res4["speaker_role"] == "borrower"


def test_synthetic_sequence_aaabb_aaba_proves_not_turn_based():
    """
    Test the exact synthetic sequence:
        A A A B B A A B A
    Verify persistent identities equivalent to:
        speaker_0 speaker_0 speaker_0
        speaker_1 speaker_1
        speaker_0 speaker_0
        speaker_1
        speaker_0
    Does NOT depend on speaker_0 being the agent.
    """
    diarizer = StreamingSortformerDiarizer()
    sequence = ["Voice A", "Voice A", "Voice A", "Voice B", "Voice B", "Voice A", "Voice A", "Voice B", "Voice A"]

    assigned = []
    for idx, voice in enumerate(sequence):
        segs = diarizer.process_utterance_event(
            text=f"Utterance number {idx} from {voice}",
            start_time=idx * 2.0,
            end_time=(idx + 1) * 2.0,
            voice_name=voice
        )
        assert len(segs) == 1
        assigned.append(segs[0].speaker_id)

    expected = [
        "speaker_0", "speaker_0", "speaker_0",
        "speaker_1", "speaker_1",
        "speaker_0", "speaker_0",
        "speaker_1",
        "speaker_0"
    ]
    assert assigned == expected, f"Expected {expected}, got {assigned}"


def test_long_silence_between_utterances_same_speaker():
    """Verify that a long silence does not reset speaker IDs or assign a new speaker."""
    diarizer = StreamingSortformerDiarizer()

    # Speaker A speaks at t=0
    segs1 = diarizer.process_utterance_event("Turn 1 from speaker A", 0.0, 2.0, voice_name="Voice A")
    assert segs1[0].speaker_id == "speaker_0"

    # 10 seconds of silence chunks
    silence = make_pcm_silence(1.0)
    for i in range(10):
        diarizer.process_audio_chunk(silence, timestamp=2.0 + i, duration=1.0)

    # Speaker A speaks again after long silence at t=12s
    segs2 = diarizer.process_utterance_event("Turn 2 from speaker A after long silence", 12.0, 14.0, voice_name="Voice A")
    assert segs2[0].speaker_id == "speaker_0"
    assert segs2[0].acoustic_speaker_changed is False
    assert diarizer.get_tracks_summary()["detected_acoustic_speakers"] == 1


def test_multiple_consecutive_turns_same_speaker():
    """Verify that consecutive turns from the same speaker maintain the same speaker ID."""
    diarizer = StreamingSortformerDiarizer()
    for i in range(5):
        segs = diarizer.process_utterance_event(f"Sentence {i}", i * 2.0, (i + 1) * 2.0, voice_name="Voice B")
        assert segs[0].speaker_id == "speaker_0"
        if i > 0:
            assert segs[0].acoustic_speaker_changed is False
    assert diarizer.get_tracks_summary()["detected_acoustic_speakers"] == 1


def test_speaker_returning_after_other_speaker_talks():
    """Verify speaker returning after the other speaker talks."""
    diarizer = StreamingSortformerDiarizer()
    # A -> B -> A -> B
    s1 = diarizer.process_utterance_event("A1", 0.0, 2.0, voice_name="Voice A")[0].speaker_id
    s2 = diarizer.process_utterance_event("B1", 2.0, 4.0, voice_name="Voice B")[0].speaker_id
    s3 = diarizer.process_utterance_event("A2", 4.0, 6.0, voice_name="Voice A")[0].speaker_id
    s4 = diarizer.process_utterance_event("B2", 6.0, 8.0, voice_name="Voice B")[0].speaker_id

    assert s1 == "speaker_0"
    assert s2 == "speaker_1"
    assert s3 == "speaker_0"
    assert s4 == "speaker_1"


def test_role_resolver_assigning_roles_independently_of_speaker_id():
    """
    Verify RoleResolver assigns AGENT and BORROWER independently of speaker ID.
    In this test, speaker_1 is the AGENT (asks for payment) and speaker_0 is the BORROWER (states hardship).
    """
    resolver = RoleResolver()

    # speaker_0 states hardship (Borrower evidence)
    resolver.resolve_role("speaker_0", "Sir, I lost my job and my bank account is empty.")

    # speaker_1 demands payment from Bajaj Finance (Agent evidence)
    resolver.resolve_role("speaker_1", "Calling from Bajaj Finance, your overdue payment deadline is 10 October.")

    # speaker_0 says they will try to pay later
    resolver.resolve_role("speaker_0", "I don't have money right now, will try after salary.")

    # speaker_1 asks when payment can be made
    resolver.resolve_role("speaker_1", "When will you make the payment?")

    summary = resolver.get_summary()
    assert summary["speaker_0"]["role"] == "borrower"
    assert summary["speaker_1"]["role"] == "agent"
    assert summary["speaker_0"]["confidence"] >= 0.85
    assert summary["speaker_1"]["confidence"] >= 0.85
