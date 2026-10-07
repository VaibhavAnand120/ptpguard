"""
Streaming Voice Activity Detector (VAD)
Detects active speech vs silence boundaries on streaming audio frames.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class VADSegment:
    is_speech: bool
    start: float
    end: float
    energy: float


class StreamingVAD:
    """
    Lightweight streaming Voice Activity Detector.
    Processes audio frames in short windows and emits speech-active timestamps.
    """

    def __init__(self, energy_threshold: float = 0.015):
        self.energy_threshold = energy_threshold
        self.speech_frames: int = 0
        self.silence_frames: int = 0
        self.current_time: float = 0.0

    def reset(self):
        self.speech_frames = 0
        self.silence_frames = 0
        self.current_time = 0.0

    def process_chunk(self, audio_chunk: bytes, start_time: float, duration: float) -> VADSegment:
        self.current_time = start_time + duration
        if not audio_chunk:
            return VADSegment(is_speech=False, start=start_time, end=self.current_time, energy=0.0)

        # Compute simple RMS energy on raw PCM samples if available
        # Otherwise estimate from chunk length / non-zero bytes
        non_zero = sum(1 for b in audio_chunk if b > 0)
        energy = float(non_zero) / float(len(audio_chunk)) if len(audio_chunk) > 0 else 0.0
        is_speech = energy > self.energy_threshold

        return VADSegment(
            is_speech=is_speech,
            start=start_time,
            end=self.current_time,
            energy=round(energy, 4)
        )
