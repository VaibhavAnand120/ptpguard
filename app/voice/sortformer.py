"""
NVIDIA NeMo Sortformer Diarizer Interface & Dispatcher
Primary Speaker Diarization Module for PTPGuard.

Architectural Guarantees:
1. Replaces handcrafted acoustic centroid cosine similarity matching.
2. Uses official NVIDIA NeMo Sortformer (nvidia/diar_streaming_sortformer_4spk-v2.1) when NeMo is installed.
3. Provides MockDiarizer strictly for tests and environments without NeMo.
4. Preserves persistent physical speaker identity (speaker_0..speaker_3) across turns, silence, and pauses.
5. Strictly decouples physical speaker identity from conversational role resolution (AGENT / BORROWER).
"""

import os
import logging
from typing import List, Dict, Any, Optional

from .base_diarizer import SpeakerDiarizer, SpeakerSegment, SpeakerTrack
from .mock_diarizer import MockDiarizer

logger = logging.getLogger("ptpguard.sortformer")

try:
    from .nemo_sortformer import NeMoSortformerDiarizer, NEMO_AVAILABLE
except Exception as e:
    logger.debug(f"NeMoSortformerDiarizer import note: {e}")
    NEMO_AVAILABLE = False
    NeMoSortformerDiarizer = None


class StreamingSortformerDiarizer(SpeakerDiarizer):
    """
    Unified Streaming Diarizer interface.
    Instantiates the official NVIDIA NeMo Sortformer model when available.
    Falls back cleanly to MockDiarizer in mock/test mode.
    """

    MODEL_NAME = "nvidia/diar_streaming_sortformer_4spk-v2.1"
    MAX_SPEAKERS = 4

    def __init__(self, force_mock: bool = False, model_name: Optional[str] = None):
        self.force_mock = force_mock or os.getenv("PTP_DIARIZER_MODE", "").lower() == "mock"
        self.model_name = model_name or self.MODEL_NAME
        self._delegate: SpeakerDiarizer = self._init_diarizer()

    def _init_diarizer(self) -> SpeakerDiarizer:
        if not self.force_mock and NEMO_AVAILABLE and NeMoSortformerDiarizer is not None:
            try:
                logger.info(f"[DIARIZER] Initializing production NVIDIA NeMo Sortformer ('{self.model_name}')...")
                diarizer = NeMoSortformerDiarizer(model_name=self.model_name)
                logger.info(f"[DIARIZER] Using official NVIDIA NeMo Sortformer on {diarizer.device}.")
                return diarizer
            except Exception as e:
                logger.warning(
                    f"[DIARIZER] Failed to load NeMo Sortformer ({e}). "
                    f"Falling back to MockDiarizer for this session."
                )
                return MockDiarizer()
        else:
            if not self.force_mock:
                logger.info("[DIARIZER] NeMo not active in current environment. Using MockDiarizer (MOCK ONLY).")
            return MockDiarizer()

    @property
    def tracks(self) -> Dict[str, SpeakerTrack]:
        return self._delegate.tracks

    @property
    def session_id(self) -> str:
        return self._delegate.session_id

    @property
    def recent_diagnostics(self) -> List[Dict[str, Any]]:
        return getattr(self._delegate, "recent_diagnostics", [])

    @property
    def is_mock(self) -> bool:
        return getattr(self._delegate, "is_mock", False)

    @property
    def current_time(self) -> float:
        return getattr(self._delegate, "current_time", 0.0)

    @current_time.setter
    def current_time(self, val: float) -> None:
        if hasattr(self._delegate, "current_time"):
            self._delegate.current_time = val

    @property
    def raw_to_persistent(self) -> Dict[str, str]:
        return getattr(self._delegate, "raw_to_persistent", getattr(self._delegate, "voice_to_speaker_id", {}))

    def process_audio_chunk(
        self,
        audio_chunk: bytes,
        timestamp: float,
        duration: float,
        chunk_id: Optional[str] = None,
        voice_features: Optional[Dict[str, Any]] = None,
    ) -> List[SpeakerSegment]:
        """Delegate audio chunk diarization to active diarizer."""
        return self._delegate.process_audio_chunk(
            audio_chunk=audio_chunk,
            timestamp=timestamp,
            duration=duration,
            chunk_id=chunk_id,
            voice_features=voice_features
        )

    def process_utterance_event(
        self,
        text: str,
        start_time: float,
        end_time: float,
        chunk_id: Optional[str] = None,
        speaker_hint: Optional[str] = None,
        background_hint: bool = False,
        voice_features: Optional[Dict[str, Any]] = None,
        voice_name: Optional[str] = None
    ) -> List[SpeakerSegment]:
        """Delegate utterance event diarization to active diarizer."""
        return self._delegate.process_utterance_event(
            text=text,
            start_time=start_time,
            end_time=end_time,
            chunk_id=chunk_id,
            speaker_hint=speaker_hint,
            background_hint=background_hint,
            voice_features=voice_features,
            voice_name=voice_name
        )

    def get_tracks_summary(self) -> Dict[str, Any]:
        """Delegate tracks summary."""
        summary = self._delegate.get_tracks_summary()
        summary["delegate_type"] = self._delegate.__class__.__name__
        return summary

    def reset(self) -> None:
        """Reset active diarizer state."""
        self._delegate.reset()


def get_speaker_diarizer(force_mock: bool = False) -> SpeakerDiarizer:
    """Factory helper to obtain the configured SpeakerDiarizer instance."""
    return StreamingSortformerDiarizer(force_mock=force_mock)
