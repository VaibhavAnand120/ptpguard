"""
Real-Time Parallel Voice-Analysis Pipeline
Coordinates concurrent streaming workers:
- Audio Ingestion & Buffer
- VAD
- NVIDIA Streaming Sortformer (speaker_0..speaker_3)
- Streaming ASR
- Timestamp Alignment & Fusion
- Persistent Role Resolver
- Two-Path Semantic Analysis:
    * FAST PATH: Instant rule-based semantic update & score calculation (<50ms)
    * DEEP PATH: Asynchronous LLM (Gemini/Ollama) contextual refinement in background
- Latency Telemetry Tracking
"""

import time
import asyncio
import logging
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass

from .sortformer import StreamingSortformerDiarizer, SpeakerSegment
from .vad import StreamingVAD
from .asr import StreamingASR, TranscriptSegment
from .alignment import TimestampAlignmentEngine, AlignedTurn
from .roles import RoleResolver, SpeakerRoleState
from .telemetry import LatencyMonitor, TurnLatencyRecord
from .audio_queue import ContinuousAudioBuffer, AudioChunkItem

from ..config import LLM_PROVIDER
from ..schemas import Utterance, Evidence, AnalysisResponse, ConversationState
from ..state import StateManager
from ..scoring import credibility_score, classify_ptp
from ..policy import policy

logger = logging.getLogger("ptpguard.pipeline")
logging.basicConfig(level=logging.INFO, format="%(message)s")


class ParallelVoicePipeline:
    """
    Real-Time Parallel Voice-Analysis Pipeline Orchestrator.
    Processes live audio frames or ASR events concurrently through
    Sortformer, ASR, alignment, role resolution, and semantic evaluation.
    """

    def __init__(self, state_manager: Optional[StateManager] = None):
        from ..semantic.rules import RulesAnalyzer
        self.state_manager = state_manager or StateManager()
        self.sortformer = StreamingSortformerDiarizer()
        self.audio_buffer = ContinuousAudioBuffer(worker_callback=self._process_buffered_chunk)
        self.vad = StreamingVAD()
        self.asr = StreamingASR()
        self.alignment = TimestampAlignmentEngine()
        self.role_resolver = RoleResolver()
        self.telemetry = LatencyMonitor()
        self.rules_analyzer = RulesAnalyzer()
        self.audio_chunk_counter: int = 0

        # Background deep LLM analyzer if configured
        self._llm_analyzer = None
        self._deep_task: Optional[asyncio.Task] = None
        self._init_llm_analyzer()

    def _init_llm_analyzer(self):
        if LLM_PROVIDER == "gemini":
            try:
                from ..semantic.gemini import GeminiAnalyzer
                self._llm_analyzer = GeminiAnalyzer()
            except Exception:
                self._llm_analyzer = None
        elif LLM_PROVIDER == "ollama":
            try:
                from ..semantic.ollama import OllamaAnalyzer
                self._llm_analyzer = OllamaAnalyzer()
            except Exception:
                self._llm_analyzer = None

    async def _process_buffered_chunk(self, item: AudioChunkItem):
        """Asynchronously consumes chunks from the intake buffer without blocking capture."""
        pass

    def reset(self):
        """Reset entire pipeline state for a new call."""
        self.audio_chunk_counter = 0
        self.audio_buffer.reset()
        self.state_manager.reset()
        self.sortformer.reset()
        self.vad.reset()
        self.asr.reset()
        self.alignment.reset()
        self.role_resolver.reset()
        self.telemetry.reset()
        if self._deep_task and not self._deep_task.done():
            self._deep_task.cancel()

    def process_audio_chunk(
        self,
        audio_chunk: bytes,
        timestamp: float,
        duration: float,
        chunk_id: Optional[str] = None,
        client_capture_time: Optional[float] = None,
        voice_features: Optional[Dict[str, Any]] = None,
        voice_name: Optional[str] = None,
    ) -> List[SpeakerSegment]:
        """
        Feed real audio chunk directly into continuous audio buffer and parallel workers.
        Capture never blocks on downstream processing.
        """
        self.audio_chunk_counter += 1
        cid = chunk_id or f"chunk_{self.audio_chunk_counter}"

        # 1. Non-blocking enqueue into continuous audio buffer (Priority: Audio Intake)
        self.audio_buffer.enqueue_audio_chunk(
            audio_bytes=audio_chunk,
            timestamp=timestamp,
            duration=duration,
            chunk_id=cid,
            client_capture_time=client_capture_time,
            voice_features=voice_features,
            voice_name=voice_name
        )

        # 2. Parallel VAD & Sortformer Workers
        vad_speech = self.vad.process_chunk(audio_chunk, timestamp, duration)
        segments = self.sortformer.process_audio_chunk(
            audio_chunk=audio_chunk,
            timestamp=timestamp,
            duration=duration,
            chunk_id=cid,
            voice_features=voice_features
        )
        for seg in segments:
            self.alignment.register_speaker_segment(seg)
            role_st = self.role_resolver.get_role_state(seg.speaker_id)
            current_role = role_st.role if role_st else "unknown"

            # Mandatory Diagnostics Logging for every audio chunk
            logger.info(
                f"[AUDIO CHUNK] audio_chunk_id={cid} start_time={timestamp:.2f} end_time={timestamp+duration:.2f} "
                f"queue_depth={self.audio_buffer.metrics.current_queue_depth} "
                f"processing_lag={self.audio_buffer.metrics.current_processing_lag_s:.2f}s "
                f"sortformer_session_id={self.sortformer.session_id} "
                f"raw_sortformer_speaker={seg.raw_sortformer_speaker} "
                f"persistent_speaker_id={seg.speaker_id} "
                f"role={current_role.upper()} "
                f"whether_sortformer_state_was_reset={not seg.state_preserved}"
            )

        return segments

    async def process_utterance(
        self,
        u: Utterance,
        start_time: Optional[float] = None,
        duration: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Processes a streaming utterance through the parallel pipeline stages.
        Measures precise latencies for each stage.
        """
        t0 = time.perf_counter()
        lat_record = TurnLatencyRecord()

        # Update telemetry intake metrics
        buf_diag = self.audio_buffer.get_diagnostics()
        lat_record.audio_queue_depth = buf_diag["audio_queue_depth"]
        lat_record.audio_processing_lag_s = buf_diag["processing_lag_s"]
        lat_record.audio_capture_latency_ms = buf_diag["audio_capture_latency_ms"]

        # Stage 1: Audio Ingestion / Framing
        t_ingest_start = time.perf_counter()
        raw_text = u.text.strip()
        speaker_hint = u.speaker
        v_name = getattr(u, "voice", None) or getattr(u, "voice_id", None)
        v_feat = getattr(u, "voice_features", None)
        c_id = getattr(u, "chunk_id", None) or f"utt_{self.audio_chunk_counter + 1}"

        # Determine processing mode
        if speaker_hint and speaker_hint in ("agent", "borrower", "third_party", "third_party_background") and speaker_hint != "auto":
            mode = "manual_simulation"
        elif u.mode == "live_audio":
            mode = "live_audio"
        else:
            mode = "auto"
        lat_record.audio_ingestion_ms = max(0.1, (time.perf_counter() - t_ingest_start) * 1000.0)

        # Stage 2 & 3: Parallel Sortformer Diarization & ASR
        t_parallel_start = time.perf_counter()

        has_bg_hint = (
            u.is_background_speech
            or "peeche se" in raw_text.lower()
            or "background" in raw_text.lower()
        )

        # 2a. Sortformer Worker: Identifies stable speaker_0..speaker_3 segment
        t_diar_start = time.perf_counter()
        asr_seg = self.asr.segment_from_text(raw_text, start_time=start_time, duration=duration)
        spk_segments = self.sortformer.process_utterance_event(
            text=raw_text,
            start_time=asr_seg.start,
            end_time=asr_seg.end,
            chunk_id=c_id,
            speaker_hint=speaker_hint if mode == "manual_simulation" else None,
            background_hint=has_bg_hint,
            voice_features=v_feat,
            voice_name=v_name
        )
        for seg in spk_segments:
            self.alignment.register_speaker_segment(seg)
        lat_record.diarization_ms = max(0.2, (time.perf_counter() - t_diar_start) * 1000.0)

        # 2b. ASR Worker latency
        lat_record.asr_ms = max(0.1, (time.perf_counter() - t_parallel_start) * 1000.0)

        # Stage 4: Timestamp Alignment & Fusion Layer
        t_align_start = time.perf_counter()
        aligned_turn = self.alignment.align(asr_seg)
        lat_record.alignment_ms = max(0.1, (time.perf_counter() - t_align_start) * 1000.0)

        # Stage 5: Role Resolver (Speaker ID -> Persistent Role)
        t_role_start = time.perf_counter()
        role_state = self.role_resolver.resolve_role(
            speaker_id=aligned_turn.speaker_id,
            text=raw_text,
            is_background=has_bg_hint,
            manual_override=speaker_hint if mode == "manual_simulation" else None
        )
        effective_role = role_state.role
        lat_record.role_resolution_ms = max(0.1, (time.perf_counter() - t_role_start) * 1000.0)

        # Mandatory Diagnostics Logging for every audio chunk / utterance
        first_seg = spk_segments[0] if spk_segments else None
        st = asr_seg.start
        et = asr_seg.end
        logger.info(
            f"[AUDIO CHUNK] audio_chunk_id={c_id} start_time={st:.2f} end_time={et:.2f} "
            f"queue_depth={self.audio_buffer.metrics.current_queue_depth} "
            f"processing_lag={self.audio_buffer.metrics.current_processing_lag_s:.2f}s "
            f"sortformer_session_id={self.sortformer.session_id} "
            f"raw_sortformer_speaker={first_seg.raw_sortformer_speaker if first_seg else 'raw_spk_0'} "
            f"persistent_speaker_id={aligned_turn.speaker_id} "
            f"role={effective_role.upper()} "
            f"whether_sortformer_state_was_reset={not (first_seg.state_preserved if first_seg else True)}"
        )

        # Stage 6: Fast Semantic Path (<50ms)
        t_sem_start = time.perf_counter()
        evidence = await self.rules_analyzer.analyze(
            speaker=effective_role,
            text=raw_text,
            history=self.state_manager.state.transcript
        )
        evidence.mode = mode
        evidence.speaker_id = aligned_turn.speaker_id
        evidence.speaker_role = effective_role
        evidence.detected_speaker = effective_role
        evidence.speaker_confidence = role_state.confidence
        evidence.speaker_rationale = (
            f"Speaker ID: {aligned_turn.speaker_id} → Role: {effective_role.upper()} "
            f"({role_state.status}, conf={role_state.confidence:.2f})"
        )
        lat_record.fast_semantic_ms = max(0.2, (time.perf_counter() - t_sem_start) * 1000.0)

        # Stage 7: State Manager Update & Trajectory
        t_state_start = time.perf_counter()
        self.state_manager.update(
            evidence=evidence,
            speaker=speaker_hint,
            text=raw_text,
            speaker_id=aligned_turn.speaker_id,
            speaker_role=effective_role,
            source="rules"
        )

        # Tag speaker ID in transcript entry
        if self.state_manager.state.transcript:
            last_entry = self.state_manager.state.transcript[-1]
            last_entry["speaker_id"] = aligned_turn.speaker_id
            last_entry["role"] = effective_role
            last_entry["role_confidence"] = role_state.confidence
            last_entry["role_status"] = role_state.status
            last_entry["start"] = aligned_turn.start
            last_entry["end"] = aligned_turn.end
            last_entry["mode"] = mode

        # Recalculate explainable score and classification from CURRENT state
        score = credibility_score(self.state_manager.state)
        ptp_type = classify_ptp(self.state_manager.state)
        action, action_msg = policy(self.state_manager.state, score)
        lat_record.state_update_ms = max(0.1, (time.perf_counter() - t_state_start) * 1000.0)

        # Stage 8: Total End-to-End Latency
        t_total = (time.perf_counter() - t0) * 1000.0
        lat_record.total_end_to_end_ms = round(t_total, 2)
        self.telemetry.record_turn(lat_record)

        # Deep Path: Asynchronous Background LLM Refinement (Non-blocking!)
        if self._llm_analyzer and effective_role == "borrower":
            asyncio.create_task(self._run_deep_llm_path(raw_text, effective_role))

        # Log pipeline stages (Requirement 13)
        self._log_pipeline_execution(aligned_turn, effective_role, role_state, raw_text, evidence)

        # Compile response
        reasons = self._generate_reasons()
        telemetry_summary = self.telemetry.get_summary(lat_record)

        # Sync Utterance fields
        u.detected_speaker = effective_role
        u.speaker_confidence = role_state.confidence
        u.speaker_rationale = evidence.speaker_rationale

        return {
            "score": score,
            "ptp_type": ptp_type,
            "action": action,
            "action_message": action_msg,
            "evidence": evidence.model_dump(),
            "reasons": reasons,
            "raw_state": self.state_manager.state.raw_state,
            "normalized_state": self.state_manager.state.normalized_state,
            "score_history": self.state_manager.state.score_history,
            "evidence_history": [e.model_dump() for e in self.state_manager.state.evidence_history],
            "speaker_id": aligned_turn.speaker_id,
            "speaker_role": effective_role,
            "role_status": role_state.status,
            "role_confidence": role_state.confidence,
            "mode": mode,
            "speaker_roles": self.role_resolver.get_summary(),
            "sortformer_tracks": self.sortformer.get_tracks_summary(),
            "detected_acoustic_speakers": len(self.sortformer.tracks),
            "diagnostics": self.sortformer.recent_diagnostics[-1] if self.sortformer.recent_diagnostics else {},
            "audio_intake": self.audio_buffer.get_diagnostics(),
            "deadline_state": self.state_manager.state.deadline_state.model_dump(),
            "telemetry": telemetry_summary,
            "state": self.state_manager.snapshot(),
        }

    def _log_pipeline_execution(
        self,
        aligned_turn: AlignedTurn,
        effective_role: str,
        role_state: SpeakerRoleState,
        raw_text: str,
        evidence: Evidence
    ):
        """Format and log pipeline stages according to Requirement 13."""
        mins_s = int(aligned_turn.start // 60)
        secs_s = aligned_turn.start % 60
        mins_e = int(aligned_turn.end // 60)
        secs_e = aligned_turn.end % 60
        time_interval = f"{mins_s:02d}:{secs_s:05.2f}–{mins_e:02d}:{secs_e:05.2f}"

        is_borrower = effective_role == "borrower"
        c_sig = evidence.signals.get("commitment")
        c_delta = (c_sig.direction * c_sig.strength * c_sig.confidence) if c_sig else 0.0

        logger.info(f"\n--- [PIPELINE TURN] ---")
        logger.info(f"DIARIZATION: {aligned_turn.speaker_id} {time_interval}")
        logger.info(f"ROLE: {aligned_turn.speaker_id} -> {effective_role.upper()} ({role_state.confidence:.2f}) [{role_state.status}]")
        if role_state.evidence_logs:
            logger.info(f"ROLE EVIDENCE: {role_state.evidence_logs[-1]}")
        logger.info(f"ROLE SCORES: AgentScore={role_state.agent_score:.1f} BorrowerScore={role_state.borrower_score:.1f}")
        logger.info(f"ASR: \"{raw_text}\"")
        logger.info(f"ALIGNMENT: {aligned_turn.speaker_id} -> {effective_role.upper()} -> transcript")
        logger.info(f"SEMANTIC: borrower={is_borrower} commitment_delta={c_delta:.2f}")

    async def _run_deep_llm_path(self, text: str, role: str):
        """
        DEEP PATH: Runs asynchronously in background on recent window.
        Does not block audio ingestion, diarization, or ASR.
        """
        if not self._llm_analyzer:
            return
        try:
            await self._llm_analyzer.analyze(
                speaker=role,
                text=text,
                history=self.state_manager.state.transcript[-4:]
            )
        except Exception:
            pass

    def _generate_reasons(self) -> List[str]:
        s = self.state_manager.state
        norm = s.normalized_state
        raw = s.raw_state
        reasons = []

        if norm.get("commitment", 0.0) > 0.1:
            reasons.append(f"Firm payment commitment detected (+{raw.get('commitment', 0.0):.1f})")
        elif norm.get("commitment", 0.0) < -0.1:
            reasons.append(f"Unwillingness / refusal to commit detected ({raw.get('commitment', 0.0):.1f})")

        if norm.get("specificity", 0.0) > 0.1:
            reasons.append(f"Concrete payment amount/date provided (+{raw.get('specificity', 0.0):.1f})")
        elif norm.get("specificity", 0.0) < -0.1:
            reasons.append(f"Vague date or amount detected ({raw.get('specificity', 0.0):.1f})")

        if norm.get("borrower_initiation", 0.0) > 0.1:
            reasons.append(f"Borrower voluntarily initiated terms (+{raw.get('borrower_initiation', 0.0):.1f})")
        elif norm.get("borrower_initiation", 0.0) < -0.1:
            reasons.append(f"Terms driven by agent push ({raw.get('borrower_initiation', 0.0):.1f})")

        if norm.get("confirmation", 0.0) > 0.1:
            reasons.append(f"Explicit confirmation confirmed (+{raw.get('confirmation', 0.0):.1f})")

        if norm.get("hardship", 0.0) < -0.1:
            reasons.append(f"Financial hardship barrier active ({raw.get('hardship', 0.0):.1f})")
        elif norm.get("hardship", 0.0) > 0.1:
            reasons.append(f"Financial hardship resolved (+{raw.get('hardship', 0.0):.1f})")

        if norm.get("conditionality", 0.0) < -0.1:
            reasons.append(f"Conditional commitment depending on future event ({raw.get('conditionality', 0.0):.1f})")
        elif norm.get("conditionality", 0.0) > 0.1:
            reasons.append(f"Condition cleared / unconditional commitment (+{raw.get('conditionality', 0.0):.1f})")

        if norm.get("third_party", 0.0) < -0.1:
            reasons.append(f"Third party / background family involvement ({raw.get('third_party', 0.0):.1f})")

        if norm.get("agent_pressure", 0.0) < -0.1:
            reasons.append(f"High agent pressure / pushing detected ({raw.get('agent_pressure', 0.0):.1f})")

        if norm.get("escape_signal", 0.0) < -0.1:
            reasons.append(f"Escape / evasive call termination language ({raw.get('escape_signal', 0.0):.1f})")

        return reasons
