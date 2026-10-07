"""
Mock Speaker Diarizer (MOCK IMPLEMENTATION ONLY)
Used strictly for tests, simulation, and development environments when NVIDIA NeMo is not installed.
Clearly labeled as a MOCK.

Guarantees:
1. Physical speaker persistence: same physical speaker retains the same speaker ID across turns, pauses, and silence.
2. Sequence verification: sequences like A A A B B A A B A produce speaker_0 speaker_0 speaker_0 speaker_1 speaker_1 speaker_0 speaker_0 speaker_1 speaker_0.
3. Decoupled from conversational role resolution (AGENT / BORROWER).
4. No turn-based toggling or alternation.
"""

import time
import uuid
import math
import logging
from typing import List, Dict, Any, Optional
from .base_diarizer import SpeakerDiarizer, SpeakerSegment, SpeakerTrack
from .auxiliary_acoustic import AuxiliaryAcousticFeatures

logger = logging.getLogger("ptpguard.mock_diarizer")


class MockDiarizer(SpeakerDiarizer):
    """
    Mock Speaker Diarizer for test and development environments.
    EXPLICITLY LABELED AS A MOCK.
    Provides deterministic persistent speaker identification without turn alternation.
    """

    MAX_SPEAKERS = 4

    def __init__(self):
        self.session_id: str = f"mock_{str(uuid.uuid4())[:8]}"
        self.session_created_at: float = time.time()
        self.current_time: float = 0.0
        self.tracks: Dict[str, SpeakerTrack] = {}
        self.voice_to_speaker_id: Dict[str, str] = {}
        self._mock_embeddings: Dict[str, List[float]] = {}
        self.last_speaker_id: Optional[str] = None
        self.total_chunks_processed: int = 0
        self.recent_diagnostics: List[Dict[str, Any]] = []
        self.state_reset_flag: bool = True
        logger.info(f"[DIARIZATION INIT] MockDiarizer initialized (session_id={self.session_id}, MOCK ONLY)")

    @property
    def raw_to_persistent(self) -> Dict[str, str]:
        return self.voice_to_speaker_id

    def reset(self) -> None:
        """Reset internal streaming state for a new call."""
        self.session_id = f"mock_{str(uuid.uuid4())[:8]}"
        self.session_created_at = time.time()
        self.current_time = 0.0
        self.tracks.clear()
        self.voice_to_speaker_id.clear()
        self._mock_embeddings.clear()
        self.last_speaker_id = None
        self.total_chunks_processed = 0
        self.recent_diagnostics.clear()
        self.state_reset_flag = True
        logger.info(f"[DIARIZATION RESET] MockDiarizer state reset (session_id={self.session_id})")

    def _match_embedding(self, emb: List[float]) -> str:
        """Matches a synthetic acoustic embedding vector against registered mock embeddings."""
        norm_emb = math.sqrt(sum(x * x for x in emb)) or 1.0
        u = [x / norm_emb for x in emb]
        best_key = None
        best_sim = -1.0
        for key, cand in self._mock_embeddings.items():
            cand_norm = math.sqrt(sum(x * x for x in cand)) or 1.0
            c = [x / cand_norm for x in cand]
            sim = sum(a * b for a, b in zip(u, c))
            if sim > best_sim:
                best_sim = sim
                best_key = key
        if best_key is not None and best_sim >= 0.60:
            return best_key
        new_key = f"voice_emb_{len(self._mock_embeddings)}"
        self._mock_embeddings[new_key] = list(emb)
        return new_key

    def _resolve_physical_speaker(self, voice_key: str) -> str:
        """
        Maps a physical voice key to a persistent speaker ID (speaker_0..speaker_3).
        Guarantees that the same physical voice always receives the exact same speaker ID.
        """
        clean = voice_key.lower().strip()
        if clean in self.voice_to_speaker_id:
            return self.voice_to_speaker_id[clean]

        used_indices = {
            int(p.split("_")[1]) for p in self.voice_to_speaker_id.values()
            if "_" in p and p.split("_")[1].isdigit()
        }

        # Check for preferred index based on explicit naming or channel hints
        preferred_idx = None
        if "speaker_0" in clean or "voice_0" in clean or clean in ("voice_a", "voice_speaker_a", "agent"):
            preferred_idx = 0
        elif "speaker_1" in clean or "voice_1" in clean or clean in ("voice_b", "voice_speaker_b", "borrower"):
            preferred_idx = 1
        elif "speaker_2" in clean or "voice_2" in clean or clean in ("voice_c", "voice_c_background", "third_party", "third_party_background"):
            preferred_idx = 2
        elif "speaker_3" in clean or "voice_3" in clean or clean in ("voice_d", "fourth_speaker"):
            preferred_idx = 3

        if preferred_idx is not None and preferred_idx not in used_indices and preferred_idx < self.MAX_SPEAKERS:
            new_idx = preferred_idx
        else:
            new_idx = next((i for i in range(self.MAX_SPEAKERS) if i not in used_indices), len(self.voice_to_speaker_id))

        assigned_id = f"speaker_{new_idx}"
        self.voice_to_speaker_id[clean] = assigned_id
        return assigned_id

    def process_audio_chunk(
        self,
        audio_chunk: bytes,
        timestamp: float,
        duration: float,
        chunk_id: Optional[str] = None,
        voice_features: Optional[Dict[str, Any]] = None,
    ) -> List[SpeakerSegment]:
        """
        Processes streaming audio chunk in mock mode with silence gating.
        """
        cid = chunk_id or f"mock_chk_{self.total_chunks_processed + 1}"
        start_time = timestamp
        end_time = timestamp + duration
        self.total_chunks_processed += 1
        self.current_time = max(self.current_time, end_time)

        emb = voice_features.get("embedding") if voice_features else None
        v_name = (voice_features.get("voice") if voice_features else None) or (voice_features.get("voice_id") if voice_features else None)
        has_explicit_features = bool(v_name or (emb is not None and len(emb) > 0))

        mins_s = int(start_time // 60)
        secs_s = start_time % 60
        mins_e = int(end_time // 60)
        secs_e = end_time % 60
        audio_seg_str = f"{mins_s:02d}:{secs_s:05.2f}–{mins_e:02d}:{secs_e:05.2f}"

        # 1. Silence Gating: if audio chunk is pure silence before any speech, do not create tracks
        if not has_explicit_features and audio_chunk and len(audio_chunk) >= 64:
            rms = AuxiliaryAcousticFeatures.calculate_rms(audio_chunk)
            if rms < 0.008:
                if len(self.tracks) == 0:
                    return []
                # If speaker already established, maintain speaker without creating new tracks
                pers_id = self.last_speaker_id or list(self.tracks.keys())[0]
                raw_lbl = f"mock_{pers_id}"
                seg = SpeakerSegment(
                    speaker_id=pers_id,
                    start=start_time,
                    end=end_time,
                    confidence=0.85,
                    raw_model_speaker=raw_lbl,
                    raw_sortformer_speaker=raw_lbl,
                    session_id=self.session_id,
                    is_mock=True,
                    acoustic_speaker_changed=False,
                    audio_segment=audio_seg_str
                )
                return [seg]

        # 2. Determine physical voice key
        if emb is not None and isinstance(emb, list) and len(emb) > 0:
            voice_key = self._match_embedding(emb)
        elif v_name:
            voice_key = str(v_name)
        else:
            voice_key = "voice_a"

        persistent_spk = self._resolve_physical_speaker(voice_key)
        raw_spk = f"mock_{persistent_spk}"

        changed = (self.last_speaker_id is not None and self.last_speaker_id != persistent_spk)
        self.last_speaker_id = persistent_spk

        # Update or register track
        if persistent_spk not in self.tracks:
            self.tracks[persistent_spk] = SpeakerTrack(
                speaker_id=persistent_spk,
                raw_label=raw_spk,
                first_seen=start_time,
                last_seen=end_time,
                turn_count=1,
                total_duration=duration
            )
        else:
            t = self.tracks[persistent_spk]
            t.last_seen = end_time
            t.turn_count += 1
            t.total_duration += duration

        seg = SpeakerSegment(
            speaker_id=persistent_spk,
            start=start_time,
            end=end_time,
            confidence=0.95,
            raw_model_speaker=raw_spk,
            raw_sortformer_speaker=raw_spk,
            session_id=self.session_id,
            is_mock=True,
            acoustic_speaker_changed=changed,
            audio_segment=audio_seg_str
        )

        diag_entry = {
            "chunk_id": cid,
            "audio_segment": audio_seg_str,
            "raw_model_speaker": raw_spk,
            "raw_sortformer_speaker": raw_spk,
            "persistent_speaker_id": persistent_spk,
            "confidence": 0.95,
            "acoustic_speaker_changed": changed,
            "is_mock": True
        }
        self.recent_diagnostics.append(diag_entry)
        if len(self.recent_diagnostics) > 30:
            self.recent_diagnostics = self.recent_diagnostics[-20:]

        logger.info(
            f"[DIARIZATION] segment={cid} start={start_time:.2f} end={end_time:.2f} "
            f"speaker={persistent_spk} confidence=0.95 model=MockDiarizer (MOCK)"
        )
        return [seg]

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
        """
        Processes an utterance event in mock mode with persistent speaker identification.
        """
        cid = chunk_id or f"mock_utt_{self.total_chunks_processed + 1}"
        self.total_chunks_processed += 1
        self.current_time = max(self.current_time, end_time)
        mins_s = int(start_time // 60)
        secs_s = start_time % 60
        mins_e = int(end_time // 60)
        secs_e = end_time % 60
        audio_seg_str = f"{mins_s:02d}:{secs_s:05.2f}–{mins_e:02d}:{secs_e:05.2f}"

        # 1. Background third-party voice handling
        if background_hint or "peeche se" in text.lower() or "background" in text.lower():
            voice_key = "voice_c_background"
        elif voice_name:
            voice_key = str(voice_name)
        elif voice_features and ("voice" in voice_features or "voice_id" in voice_features):
            voice_key = str(voice_features.get("voice") or voice_features.get("voice_id"))
        elif speaker_hint in ("speaker_0", "speaker_1", "speaker_2", "speaker_3"):
            voice_key = f"voice_{speaker_hint[-1]}"
        elif speaker_hint == "agent":
            voice_key = "voice_a"
        elif speaker_hint == "borrower":
            voice_key = "voice_b"
        elif speaker_hint in ("third_party", "third_party_background"):
            voice_key = "voice_c"
        else:
            voice_key = self._infer_mock_voice_from_content(text)

        persistent_spk = self._resolve_physical_speaker(voice_key)
        changed = (self.last_speaker_id is not None and self.last_speaker_id != persistent_spk)
        self.last_speaker_id = persistent_spk

        dur = max(0.2, end_time - start_time)
        if persistent_spk not in self.tracks:
            self.tracks[persistent_spk] = SpeakerTrack(
                speaker_id=persistent_spk,
                raw_label=f"mock_{persistent_spk}",
                first_seen=start_time,
                last_seen=end_time,
                turn_count=1,
                total_duration=dur
            )
        else:
            t = self.tracks[persistent_spk]
            t.last_seen = end_time
            t.turn_count += 1
            t.total_duration += dur

        raw_spk = f"mock_{persistent_spk}"
        seg = SpeakerSegment(
            speaker_id=persistent_spk,
            start=start_time,
            end=end_time,
            confidence=0.95,
            raw_model_speaker=raw_spk,
            raw_sortformer_speaker=raw_spk,
            session_id=self.session_id,
            is_mock=True,
            acoustic_speaker_changed=changed,
            audio_segment=audio_seg_str
        )

        diag_entry = {
            "chunk_id": cid,
            "audio_segment": audio_seg_str,
            "raw_model_speaker": raw_spk,
            "raw_sortformer_speaker": raw_spk,
            "persistent_speaker_id": persistent_spk,
            "confidence": 0.95,
            "acoustic_speaker_changed": changed,
            "is_mock": True
        }
        self.recent_diagnostics.append(diag_entry)
        if len(self.recent_diagnostics) > 30:
            self.recent_diagnostics = self.recent_diagnostics[-20:]

        logger.info(
            f"[DIARIZATION] segment={cid} start={start_time:.2f} end={end_time:.2f} "
            f"speaker={persistent_spk} confidence=0.95 model=MockDiarizer (MOCK)"
        )
        return [seg]

    def _infer_mock_voice_from_content(self, text: str) -> str:
        """
        Fallback mock helper for simulated text-only demos without acoustic streams.
        Preserves speaker continuity for consecutive turns from the same speaker.
        """
        t = text.lower().strip()
        if any(w in t for w in ["unka bhai", "unki patni", "unke behalf", "hospital me hain"]):
            return "voice_c"

        is_borrower_like = any(w in t for w in [
            "main ", "meri ", "mera ", "salary", "paise", "de dunga", "kar dunga",
            "pay kar", "haan sir", "ji sir", "actually sir", "sir abhi",
            "bas call", "koshish", "yes, i can", "yes i can", "i can pay", "i will pay",
            "difficult", "empty", "not received", "don't have money", "no money",
            "try my best", "i will try", "don't know when to pay", "don't have any payment",
            "cannot pay", "can't pay", "nuksan", "losses", "dukaan"
        ])
        is_agent_like = (
            "?" in t
            or any(w in t for w in [
                "sir kab", "kab payment", "kab karenge", "can you pay", "when will you",
                "mark kar", "note kar", "link share", "calling from", "overdue", "required amount",
                "i will mark", "i'll mark", "you have a deadline", "deadline of payment",
                "payment deadline", "you need to pay", "you have to pay", "pay before",
                "your payment", "pending amount", "due amount", "sir, but you need to pay",
                "hdfc", "sbi", "bajaj", "bank", "finance"
            ])
        )

        if is_borrower_like and not (is_agent_like and "?" in t):
            return "voice_b"
        if is_agent_like and not is_borrower_like:
            return "voice_a"

        # Backchannel response
        if len(t.split()) <= 2 and any(t.startswith(w) for w in ["okay", "ok", "yes", "hmm", "ji"]):
            last = self.last_speaker_id or "speaker_0"
            return "voice_b" if last == "speaker_0" else "voice_a"

        # Continuity
        if self.last_speaker_id:
            for k, sid in self.voice_to_speaker_id.items():
                if sid == self.last_speaker_id:
                    return k

        return "voice_a"

    def get_tracks_summary(self) -> Dict[str, Any]:
        """Return diagnostic summary of MockDiarizer state."""
        return {
            "model": "MockDiarizer (MOCK ONLY)",
            "is_mock": True,
            "session_id": self.session_id,
            "total_chunks_processed": self.total_chunks_processed,
            "detected_acoustic_speakers": len(self.tracks),
            "active_speaker_count": len(self.tracks),
            "speakers": {
                pers_id: {
                    "raw_label": t.raw_label,
                    "first_seen": round(t.first_seen, 2),
                    "last_seen": round(t.last_seen, 2),
                    "turn_count": t.turn_count,
                    "total_duration": round(t.total_duration, 2),
                }
                for pers_id, t in self.tracks.items()
            },
            "recent_diagnostics": list(self.recent_diagnostics[-15:])
        }
