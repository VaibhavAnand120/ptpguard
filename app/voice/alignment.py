"""
Timestamp Alignment and Fusion Engine
Combines parallel ASR transcript timestamps with Sortformer speaker timestamps
to produce the canonical conversation representation:
{
    "speaker_id": "speaker_1",
    "text": "Salary nahi aayi hai",
    "start": 15.9,
    "end": 18.4
}
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .sortformer import SpeakerSegment
from .asr import TranscriptSegment


@dataclass
class AlignedTurn:
    speaker_id: str      # "speaker_0", "speaker_1", "speaker_2", "speaker_3"
    text: str            # Clean transcript
    start: float         # Start in seconds
    end: float           # End in seconds
    overlap_ratio: float # Temporal overlap confidence
    timestamp_str: str   # Formatted MM:SS


class TimestampAlignmentEngine:
    """
    Fuses ASR transcript segments and Sortformer speaker segments
    based on temporal intersection over time intervals.
    Does NOT fall back to previous speaker as an automatic default.
    """

    def __init__(self):
        self.recent_speaker_segments: List[SpeakerSegment] = []

    def reset(self):
        self.recent_speaker_segments.clear()

    def register_speaker_segment(self, segment: SpeakerSegment):
        """Append a speaker segment from Sortformer worker."""
        self.recent_speaker_segments.append(segment)
        if len(self.recent_speaker_segments) > 100:
            self.recent_speaker_segments = self.recent_speaker_segments[-60:]

    def align(self, asr_seg: TranscriptSegment) -> AlignedTurn:
        """
        Align an ASR segment with the most temporally overlapping Sortformer speaker segment.
        """
        best_spk = None
        max_overlap = 0.0

        asr_start = asr_seg.start
        asr_end = asr_seg.end
        asr_duration = max(0.01, asr_end - asr_start)

        for spk_seg in self.recent_speaker_segments:
            # Overlap interval: [max(start1, start2), min(end1, end2)]
            overlap_start = max(asr_start, spk_seg.start)
            overlap_end = min(asr_end, spk_seg.end)
            overlap_len = max(0.0, overlap_end - overlap_start)

            if overlap_len > max_overlap:
                max_overlap = overlap_len
                best_spk = spk_seg.speaker_id

        # If direct overlap exists:
        if best_spk is not None and max_overlap > 0:
            overlap_ratio = min(1.0, max_overlap / asr_duration)
        elif self.recent_speaker_segments:
            # Find the segment with the smallest temporal gap to the ASR segment
            closest_seg = min(
                self.recent_speaker_segments,
                key=lambda s: min(abs(s.start - asr_end), abs(s.end - asr_start))
            )
            best_spk = closest_seg.speaker_id
            overlap_ratio = 0.50
        else:
            best_spk = "speaker_0"
            overlap_ratio = 0.30

        mins = int(asr_start // 60)
        secs = int(asr_start % 60)
        time_str = f"{mins:02d}:{secs:02d}"

        return AlignedTurn(
            speaker_id=best_spk,
            text=asr_seg.text,
            start=asr_start,
            end=asr_end,
            overlap_ratio=round(overlap_ratio, 2),
            timestamp_str=time_str
        )
