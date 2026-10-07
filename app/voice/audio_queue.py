"""
Continuous Audio Intake & Asynchronous Producer-Consumer Buffer
Decouples live microphone intake from downstream inference workers.

CRITICAL PRIORITY:
CONTINUOUS AUDIO INTAKE > DOWNSTREAM PROCESSING SPEED

Architecture:
MICROPHONE (Live Web Audio / Telephony Stream)
    ↓
Continuous Audio Queue (Bounded ring buffer / queue)
    ↓ (Independent asynchronous worker)
    ├── VAD Worker
    ├── Sortformer Worker (nvidia/diar_streaming_sortformer_4spk-v2.1)
    └── Streaming ASR Worker
             ↓
       Timestamp Alignment Engine
             ↓
       Speaker Identity (speaker_0 / speaker_1)
             ↓
       Role Resolver (AGENT / BORROWER)
             ↓
       Semantic Analysis & Credibility Engine

Audio capture NEVER waits or blocks on:
- ASR
- Sortformer
- LLM / Gemini / Ollama
- Role resolution
- UI socket broadcasts
- Database writes
"""

import time
import asyncio
import logging
from typing import Optional, Dict, Any, List, Callable, Coroutine
from dataclasses import dataclass, field

logger = logging.getLogger("ptpguard.audio_queue")
logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass
class AudioChunkItem:
    chunk_id: str
    audio_bytes: bytes
    timestamp: float                  # Client-side / stream start timestamp (seconds)
    duration: float                   # Chunk duration (seconds)
    client_capture_time: float = 0.0  # Unix timestamp when microphone captured the chunk
    enqueued_at: float = field(default_factory=time.time)
    voice_features: Optional[Dict[str, Any]] = None
    voice_name: Optional[str] = None
    transcript_text: Optional[str] = None   # Text if sent alongside or from speech recognition
    speaker_hint: Optional[str] = None


@dataclass
class IntakeMetrics:
    is_active: bool = True
    total_enqueued: int = 0
    total_processed: int = 0
    total_dropped: int = 0
    current_queue_depth: int = 0
    max_queue_depth: int = 500
    current_processing_lag_s: float = 0.0
    audio_capture_latency_ms: float = 0.0
    last_chunk_id: str = ""
    last_processed_at: float = 0.0


class ContinuousAudioBuffer:
    """
    Asynchronous bounded ring buffer & producer-consumer orchestrator.
    Guarantees continuous microphone intake without blocking.
    """

    MAX_QUEUE_CAPACITY = 500

    def __init__(
        self,
        worker_callback: Optional[Callable[[AudioChunkItem], Coroutine[Any, Any, None]]] = None,
        maxsize: int = MAX_QUEUE_CAPACITY
    ):
        self.max_capacity = maxsize
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.worker_callback = worker_callback
        self.metrics = IntakeMetrics(max_queue_depth=maxsize)
        self._consumer_task: Optional[asyncio.Task] = None
        self._is_running: bool = False

    def start(self):
        """Starts the background consumer worker loop."""
        if self._is_running and self._consumer_task and not self._consumer_task.done():
            return
        self._is_running = True
        self._consumer_task = asyncio.create_task(self._worker_loop())
        logger.info("[AUDIO BUFFER] Continuous intake worker loop started")

    def stop(self):
        """Stops the background consumer worker."""
        self._is_running = False
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
        logger.info("[AUDIO BUFFER] Continuous intake worker loop stopped")

    def reset(self):
        """Clears queue and resets metrics for a new call."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except Exception:
                break
        self.metrics.total_enqueued = 0
        self.metrics.total_processed = 0
        self.metrics.total_dropped = 0
        self.metrics.current_queue_depth = 0
        self.metrics.current_processing_lag_s = 0.0
        self.metrics.audio_capture_latency_ms = 0.0
        self.metrics.last_chunk_id = ""
        logger.info("[AUDIO BUFFER] Queue cleared and metrics reset for new call")

    def enqueue_audio_chunk(
        self,
        audio_bytes: bytes,
        timestamp: float,
        duration: float,
        chunk_id: Optional[str] = None,
        client_capture_time: Optional[float] = None,
        voice_features: Optional[Dict[str, Any]] = None,
        voice_name: Optional[str] = None,
        transcript_text: Optional[str] = None,
        speaker_hint: Optional[str] = None
    ) -> bool:
        """
        NON-BLOCKING PRODUCER:
        Accepts incoming audio chunk from microphone and places it into the buffer immediately.
        Takes < 0.05ms. NEVER waits for Sortformer, ASR, LLM, or semantic analysis.
        """
        now = time.time()
        cid = chunk_id or f"chunk_{self.metrics.total_enqueued + 1}"
        cap_time = client_capture_time or now

        item = AudioChunkItem(
            chunk_id=cid,
            audio_bytes=audio_bytes,
            timestamp=timestamp,
            duration=duration,
            client_capture_time=cap_time,
            enqueued_at=now,
            voice_features=voice_features,
            voice_name=voice_name,
            transcript_text=transcript_text,
            speaker_hint=speaker_hint
        )

        # Compute instant ingestion latency
        if client_capture_time and client_capture_time > 0:
            self.metrics.audio_capture_latency_ms = max(0.1, round((now - client_capture_time) * 1000.0, 2))

        # Check queue capacity for controlled backpressure
        if self.queue.full():
            # Discard oldest chunk to ensure microphone capture never stops
            try:
                dropped = self.queue.get_nowait()
                self.queue.task_done()
                self.metrics.total_dropped += 1
                logger.warning(
                    f"[AUDIO BACKPRESSURE] Dropped oldest chunk '{dropped.chunk_id}' "
                    f"to prevent blocking live microphone intake (capacity: {self.MAX_QUEUE_CAPACITY})"
                )
            except asyncio.QueueEmpty:
                pass

        try:
            self.queue.put_nowait(item)
            self.metrics.total_enqueued += 1
            self.metrics.current_queue_depth = self.queue.qsize()
            self.metrics.last_chunk_id = cid
            return True
        except asyncio.QueueFull:
            self.metrics.total_dropped += 1
            logger.error(f"[AUDIO BUFFER OVERFLOW] Failed to enqueue chunk '{cid}'")
            return False

    async def _worker_loop(self):
        """
        INDEPENDENT CONSUMER WORKER:
        Drains the audio queue and forwards items to the processing callback.
        If processing lags behind live intake, queue depth grows while microphone keeps capturing.
        """
        while self._is_running:
            try:
                item: AudioChunkItem = await self.queue.get()
                now = time.time()
                self.metrics.current_queue_depth = self.queue.qsize()

                # Calculate processing lag (difference between now and when chunk arrived)
                lag_s = max(0.0, now - item.enqueued_at)
                self.metrics.current_processing_lag_s = round(lag_s, 2)

                if lag_s > 1.0:
                    logger.warning(
                        f"[PROCESSING LAG] Processing lag behind live audio: {lag_s:.2f}s "
                        f"(queue depth: {self.metrics.current_queue_depth}). Audio capture continues uninterrupted."
                    )

                # Process item through downstream pipeline
                if self.worker_callback:
                    try:
                        await self.worker_callback(item)
                    except Exception as e:
                        logger.error(f"[WORKER ERROR] Error processing audio chunk '{item.chunk_id}': {e}", exc_info=True)

                self.metrics.total_processed += 1
                self.metrics.last_processed_at = now
                self.queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[AUDIO CONSUMER LOOP ERROR] {e}", exc_info=True)
                await asyncio.sleep(0.01)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns diagnostics for real-time telemetry and UI."""
        return {
            "intake_status": "ACTIVE" if self.metrics.is_active else "IDLE",
            "audio_queue_depth": self.metrics.current_queue_depth,
            "max_capacity": self.metrics.max_queue_depth,
            "processing_lag_s": self.metrics.current_processing_lag_s,
            "audio_capture_latency_ms": self.metrics.audio_capture_latency_ms,
            "total_enqueued": self.metrics.total_enqueued,
            "total_processed": self.metrics.total_processed,
            "total_dropped": self.metrics.total_dropped,
            "last_chunk_id": self.metrics.last_chunk_id,
        }
