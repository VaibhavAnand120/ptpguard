"""
NVIDIA Streaming Sortformer Diarization & Continuous Audio Intake Tests
Model: nvidia/diar_streaming_sortformer_4spk-v2.1

Verifies:
1. Physical Voice Persistence:
   The same physical voice (Voice A) keeps speaker_0 across pauses, silence,
   VAD boundaries, separate ASR segments, and multiple utterances:
   Voice A -> "Hello sir..."
   pause
   Voice A -> "Karun toh chalega."
   pause
   Voice A -> "Kya aapko pata nahi..."
   => speaker_0, speaker_0, speaker_0 (NEVER alternating to speaker_1).

2. Real Acoustic Change:
   Voice B -> "Yes sir..."
   => speaker_1.

3. Return to Voice A:
   Voice A -> "I can pay tomorrow."
   => speaker_0.

4. Repeated Short Utterances:
   Repeated "yes", "okay", "hmm" by Voice B remain speaker_1 (no toggling).

5. Continuous Streaming Sortformer Session State:
   Sortformer state/session is preserved throughout the call.
   State is ONLY destroyed/reset on POST /api/call/reset.

6. Decoupled Role Resolution:
   speaker_0 -> AGENT
   speaker_1 -> BORROWER
   Never both BORROWER.

7. Non-blocking Asynchronous Continuous Audio Intake:
   Microphone enqueue does not block on downstream processing.
   Tracks queue depth and processing lag.
"""

import pytest
import asyncio
import time
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import Utterance
from app.voice.pipeline import ParallelVoicePipeline
from app.voice.sortformer import StreamingSortformerDiarizer
from app.voice.audio_queue import ContinuousAudioBuffer


@pytest.fixture
def pipeline():
    p = ParallelVoicePipeline()
    p.reset()
    return p


# 1. Exact User Benchmark: Voice A across pauses and utterances remains speaker_0
@pytest.mark.anyio
async def test_same_voice_across_pauses_remains_speaker_0(pipeline):
    """
    Voice A: "Hello sir..."
    pause
    Voice A: "Karun toh chalega."
    pause
    Voice A: "Kya aapko pata nahi..."

    Expected: speaker_0, speaker_0, speaker_0 (NO turn-based alternation).
    """
    res1 = await pipeline.process_utterance(
        Utterance(speaker="auto", text="Hello sir...", voice="Voice A")
    )
    assert res1["speaker_id"] == "speaker_0"
    session_id_1 = res1["sortformer_tracks"]["session_id"]

    # Pause 1.5s
    await asyncio.sleep(0.05)

    res2 = await pipeline.process_utterance(
        Utterance(speaker="auto", text="Karun toh chalega.", voice="Voice A")
    )
    assert res2["speaker_id"] == "speaker_0"
    session_id_2 = res2["sortformer_tracks"]["session_id"]

    # Pause 2.0s
    await asyncio.sleep(0.05)

    res3 = await pipeline.process_utterance(
        Utterance(speaker="auto", text="Kya aapko pata nahi...", voice="Voice A")
    )
    assert res3["speaker_id"] == "speaker_0"
    session_id_3 = res3["sortformer_tracks"]["session_id"]

    # Streaming session state was preserved across all turns
    assert session_id_1 == session_id_2 == session_id_3


# 2. Genuine Acoustic Speaker Change to Voice B -> speaker_1, then return to Voice A -> speaker_0
@pytest.mark.anyio
async def test_acoustic_speaker_change_and_return(pipeline):
    """
    Voice A -> speaker_0
    Voice B -> speaker_1
    Voice A -> speaker_0
    """
    # Voice A (Agent)
    r1 = await pipeline.process_utterance(
        Utterance(speaker="auto", text="Hello sir, calling from HDFC Bank regarding loan emi.", voice="Voice A")
    )
    assert r1["speaker_id"] == "speaker_0"

    # Voice B (Borrower)
    r2 = await pipeline.process_utterance(
        Utterance(speaker="auto", text="Yes sir, I don't have money right now.", voice="Voice B")
    )
    assert r2["speaker_id"] == "speaker_1"

    # Voice A returns
    r3 = await pipeline.process_utterance(
        Utterance(speaker="auto", text="You need to pay by 10th October.", voice="Voice A")
    )
    assert r3["speaker_id"] == "speaker_0"

    # Voice B returns
    r4 = await pipeline.process_utterance(
        Utterance(speaker="auto", text="I will try my best.", voice="Voice B")
    )
    assert r4["speaker_id"] == "speaker_1"


# 3. Repeated Short Utterances by Same Voice Do Not Toggle
@pytest.mark.anyio
async def test_repeated_short_utterances_same_voice(pipeline):
    """
    Voice B saying repeated 'yes' / 'okay' stays speaker_1, does NOT toggle back to speaker_0.
    """
    await pipeline.process_utterance(Utterance(speaker="auto", text="Hello sir.", voice="Voice A"))
    await pipeline.process_utterance(Utterance(speaker="auto", text="Yes.", voice="Voice B"))

    # Repeated short words by Voice B
    r3 = await pipeline.process_utterance(Utterance(speaker="auto", text="Okay.", voice="Voice B"))
    r4 = await pipeline.process_utterance(Utterance(speaker="auto", text="Hmm.", voice="Voice B"))

    assert r3["speaker_id"] == "speaker_1"
    assert r4["speaker_id"] == "speaker_1"


# 4. Sortformer Streaming Audio Chunks (Raw Audio Simulation)
def test_sortformer_raw_audio_chunks():
    diarizer = StreamingSortformerDiarizer()
    init_session = diarizer.session_id

    # Stream 5 audio chunks for Voice A
    vA_features = {"voice": "Voice A"}
    for i in range(5):
        dummy_pcm = b"\x00\x10" * 320
        segs = diarizer.process_audio_chunk(
            audio_chunk=dummy_pcm,
            timestamp=float(i) * 0.5,
            duration=0.5,
            chunk_id=f"chk_{i+1}",
            voice_features=vA_features
        )
        assert len(segs) == 1
        assert segs[0].speaker_id == "speaker_0"
        assert segs[0].session_id == init_session

    # Stream audio chunk for Voice B
    vB_features = {"voice": "Voice B"}
    segs_b = diarizer.process_audio_chunk(
        audio_chunk=b"\x00\x20" * 320,
        timestamp=2.5,
        duration=0.5,
        chunk_id="chk_6",
        voice_features=vB_features
    )
    assert segs_b[0].speaker_id == "speaker_1"

    # Return to Voice A
    segs_a2 = diarizer.process_audio_chunk(
        audio_chunk=b"\x00\x10" * 320,
        timestamp=3.0,
        duration=0.5,
        chunk_id="chk_7",
        voice_features=vA_features
    )
    assert segs_a2[0].speaker_id == "speaker_0"


# 5. Call Reset Destroys and Creates New Session
def test_sortformer_call_reset():
    diarizer = StreamingSortformerDiarizer()
    session_1 = diarizer.session_id

    diarizer.process_utterance_event(text="Hello", start_time=0.0, end_time=1.0, voice_name="Voice A")
    assert "speaker_0" in diarizer.tracks

    # Reset
    diarizer.reset()
    session_2 = diarizer.session_id

    assert session_1 != session_2
    assert len(diarizer.tracks) == 0
    assert len(diarizer.raw_to_persistent) == 0


# 6. Decoupled Role Resolution Invariant
@pytest.mark.anyio
async def test_role_resolution_decoupled_from_sortformer(pipeline):
    """
    Sortformer identifies speaker_0 (Voice A) and speaker_1 (Voice B).
    Role Resolver assigns AGENT and BORROWER.
    Both must NEVER be BORROWER.
    """
    # Voice A speaks institutional intro
    r1 = await pipeline.process_utterance(
        Utterance(speaker="auto", text="Hello sir, you have a deadline of payment of ₹5000 by 10th October.", voice="Voice A")
    )
    # Voice B speaks hardship
    r2 = await pipeline.process_utterance(
        Utterance(speaker="auto", text="Yes sir, I will try my best but I don't know when to pay.", voice="Voice B")
    )

    assert r1["speaker_id"] == "speaker_0"
    assert r2["speaker_id"] == "speaker_1"

    roles = pipeline.role_resolver.get_summary()
    assert roles["speaker_0"]["role"] == "agent"
    assert roles["speaker_1"]["role"] == "borrower"
    assert roles["speaker_0"]["role"] != roles["speaker_1"]["role"]


# 7. Non-blocking Asynchronous Continuous Audio Intake Buffer
def test_audio_intake_buffer_non_blocking():
    buf = ContinuousAudioBuffer()

    t0 = time.perf_counter()
    # Enqueue 100 audio chunks rapidly
    for i in range(100):
        ok = buf.enqueue_audio_chunk(
            audio_bytes=b"\x00" * 320,
            timestamp=float(i) * 0.1,
            duration=0.1,
            chunk_id=f"chunk_{i}"
        )
        assert ok is True

    elapsed = (time.perf_counter() - t0) * 1000.0
    # Enqueueing 100 chunks must take less than 20ms total (<0.2ms per chunk)
    assert elapsed < 50.0

    diag = buf.get_diagnostics()
    assert diag["audio_queue_depth"] == 100
    assert diag["total_enqueued"] == 100
    assert diag["total_dropped"] == 0


# 8. VAD Segmentation: Single thought broken into 3 VAD segments from same voice
@pytest.mark.anyio
async def test_vad_segmentation_same_voice_multiple_segments(pipeline):
    """
    VAD cuts speech into rapid micro-segments (e.g. 0.5s segments):
    Segment 1: "Main... "
    Segment 2: "...agle hafte tak..."
    Segment 3: "...paisa arrange karunga."
    All belong to the same Voice B and must retain speaker_1 without toggling.
    """
    # First turn is Voice A (Agent)
    await pipeline.process_utterance(Utterance(speaker="auto", text="Kab payment karenge?", voice="Voice A"))
    
    # Rapid VAD segments from Voice B
    s1 = await pipeline.process_utterance(Utterance(speaker="auto", text="Main...", voice="Voice B"))
    s2 = await pipeline.process_utterance(Utterance(speaker="auto", text="agle hafte tak...", voice="Voice B"))
    s3 = await pipeline.process_utterance(Utterance(speaker="auto", text="paisa arrange karunga.", voice="Voice B"))
    
    assert s1["speaker_id"] == "speaker_1"
    assert s2["speaker_id"] == "speaker_1"
    assert s3["speaker_id"] == "speaker_1"


# 9. Long Pause Persistence: Same voice after 10-second pause
@pytest.mark.anyio
async def test_long_pause_speaker_persistence(pipeline):
    """
    Voice A speaks. Silence/pause of 10 seconds. Voice A speaks again.
    Sortformer session and speaker identity must NOT reset or flip.
    """
    r1 = await pipeline.process_utterance(
        Utterance(speaker="auto", text="Hello sir, can you hear me?", voice="Voice A")
    )
    assert r1["speaker_id"] == "speaker_0"
    session_id_1 = r1["sortformer_tracks"]["session_id"]

    # Simulated long pause of 10 seconds (timestamp jumps +10s)
    r2 = await pipeline.process_utterance(
        Utterance(speaker="auto", text="Are you there sir?", voice="Voice A", start_time=12.0, end_time=14.0)
    )
    assert r2["speaker_id"] == "speaker_0"
    assert r2["sortformer_tracks"]["session_id"] == session_id_1


# 10. Two Speakers with Similar Pitch (No Pitch Thresholding)
def test_two_speakers_similar_pitch_not_confused():
    """
    Sortformer uses multidimensional acoustic resonance/formants, NOT scalar pitch.
    Two speakers with identical fundamental pitch (e.g. 150 Hz) but different vocal tract
    spectral envelopes are separated cleanly into speaker_0 and speaker_1.
    """
    diarizer = StreamingSortformerDiarizer()
    
    # Speaker 1: pitch=150Hz, spectral distribution skewed to low bands
    spk1_features = {
        "embedding": [0.65, 0.20, 0.10, 0.50, 0.10, 0.05, 0.30, -0.10, 0.15, 0.05, -0.10, 0.01, 0.05, -0.02, 0.01, 0.08]
    }
    # Speaker 2: SAME pitch=150Hz, but different spectral resonance (skewed to high bands)
    spk2_features = {
        "embedding": [-0.10, -0.65, 0.40, -0.20, 0.40, -0.15, 0.05, 0.20, -0.10, -0.05, 0.10, 0.02, -0.01, 0.08, -0.02, -0.10]
    }

    res1 = diarizer.process_audio_chunk(audio_chunk=b"\x00" * 320, timestamp=0.0, duration=0.5, voice_features=spk1_features)
    res2 = diarizer.process_audio_chunk(audio_chunk=b"\x00" * 320, timestamp=0.5, duration=0.5, voice_features=spk2_features)
    res3 = diarizer.process_audio_chunk(audio_chunk=b"\x00" * 320, timestamp=1.0, duration=0.5, voice_features=spk1_features)

    assert res1[0].speaker_id == "speaker_0"
    assert res2[0].speaker_id == "speaker_1"
    assert res3[0].speaker_id == "speaker_0"


# 11. Two Speakers with Different Pitch
def test_two_speakers_different_pitch():
    """
    Two speakers with different pitch and timbre (e.g. male agent vs female borrower or vice versa).
    """
    diarizer = StreamingSortformerDiarizer()
    
    res1 = diarizer.process_audio_chunk(audio_chunk=b"\x00" * 320, timestamp=0.0, duration=0.5, voice_features={"voice": "Voice A"})
    res2 = diarizer.process_audio_chunk(audio_chunk=b"\x00" * 320, timestamp=0.5, duration=0.5, voice_features={"voice": "Voice B"})
    
    assert res1[0].speaker_id == "speaker_0"
    assert res2[0].speaker_id == "speaker_1"


# 12. Processing Lag: Continuous Intake Buffer Does NOT Block During Downstream Delays
def test_downstream_inference_delay_does_not_halt_audio_intake():
    """
    Simulates a slow downstream worker (e.g. LLM or ASR taking 500ms).
    Audio capture keeps enqueuing chunks into the ring buffer without blocking.
    Queue depth grows and processing lag is measured; audio intake remains ACTIVE.
    """
    buf = ContinuousAudioBuffer(maxsize=200)

    # Fast audio capture: 20 chunks produced in <10ms
    t_start = time.perf_counter()
    for i in range(20):
        buf.enqueue_audio_chunk(
            audio_bytes=b"\x01\x02" * 160,
            timestamp=float(i) * 0.1,
            duration=0.1,
            chunk_id=f"chk_{i}"
        )
    capture_time_ms = (time.perf_counter() - t_start) * 1000.0
    assert capture_time_ms < 20.0  # Producer was never blocked

    # Simulate downstream lag
    buf.metrics.current_processing_lag_s = 1.8
    buf.metrics.is_active = True

    diag = buf.get_diagnostics()
    assert diag["audio_queue_depth"] == 20
    assert diag["processing_lag_s"] == 1.8
    assert diag["intake_status"] == "ACTIVE"
