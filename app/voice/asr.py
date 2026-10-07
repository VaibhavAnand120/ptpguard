"""
Streaming ASR Worker
Runs in parallel with Diarization on the live audio stream.
Emits timestamped transcript segments independently.
"""

import asyncio
import time
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass


@dataclass
class TranscriptSegment:
    text: str
    start: float
    end: float
    confidence: float = 0.95
    is_final: bool = True


class StreamingASR:
    """
    Streaming ASR engine interface.
    Runs concurrently with the Sortformer diarizer without blocking.
    """

    def __init__(self, endpoint_url: Optional[str] = None):
        self.endpoint_url = endpoint_url
        self.current_time: float = 0.0

    def reset(self):
        self.current_time = 0.0

    async def transcribe_chunk(
        self,
        audio_chunk: bytes,
        start_time: float,
        duration: float
    ) -> Optional[TranscriptSegment]:
        """
        Asynchronously transcribes an incoming audio chunk.
        In a production telephony environment, streams to Whisper/Conformer endpoint.
        """
        self.current_time = start_time + duration
        # Minimal processing latency simulation (e.g. 15-25ms)
        await asyncio.sleep(0.01)
        return None

    def segment_from_text(
        self,
        text: str,
        start_time: Optional[float] = None,
        duration: Optional[float] = None
    ) -> TranscriptSegment:
        """
        Constructs a timestamped transcript segment from live text (e.g. Web Speech API).
        Estimates timing based on utterance word count if start_time not explicit.
        """
        clean_text = text.strip()
        word_count = len(clean_text.split())
        est_duration = duration if duration is not None else max(1.0, word_count * 0.35)

        start = start_time if start_time is not None else self.current_time
        end = start + est_duration
        self.current_time = end

        return TranscriptSegment(
            text=clean_text,
            start=round(start, 2),
            end=round(end, 2),
            confidence=0.96,
            is_final=True
        )
