"""
Latency Monitoring and Telemetry
Instruments every stage in the real-time pipeline:
- audio_ingestion_latency
- diarization_latency
- ASR_latency
- alignment_latency
- role_resolution_latency
- fast_semantic_latency
- LLM_latency
- state_update_latency
- total_end_to_end_latency

Calculates rolling P50, P95, P99 percentiles to monitor whether the
few-hundred-millisecond target is being achieved.
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class TurnLatencyRecord:
    audio_capture_latency_ms: float = 0.0
    audio_queue_depth: int = 0
    audio_processing_lag_s: float = 0.0
    audio_ingestion_ms: float = 0.0
    diarization_ms: float = 0.0
    asr_ms: float = 0.0
    alignment_ms: float = 0.0
    role_resolution_ms: float = 0.0
    fast_semantic_ms: float = 0.0
    llm_ms: float = 0.0
    state_update_ms: float = 0.0
    total_end_to_end_ms: float = 0.0


class LatencyMonitor:
    """
    Rolling latency tracker calculating P50, P95, P99 metrics across turns.
    """

    def __init__(self, max_history: int = 200):
        self.max_history = max_history
        self.records: List[TurnLatencyRecord] = []

    def reset(self):
        self.records.clear()

    def record_turn(self, record: TurnLatencyRecord):
        self.records.append(record)
        if len(self.records) > self.max_history:
            self.records = self.records[-self.max_history:]

    def _percentile(self, values: List[float], p: float) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        k = (len(sorted_vals) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(sorted_vals) - 1)
        d0 = sorted_vals[f] * (c - k)
        d1 = sorted_vals[c] * (k - f)
        return round(d0 + d1, 2)

    def get_summary(self, latest: Optional[TurnLatencyRecord] = None) -> Dict[str, Any]:
        """
        Returns full latency telemetry including current turn and rolling P50, P95, P99.
        """
        totals = [r.total_end_to_end_ms for r in self.records] if self.records else [0.0]
        cur = latest or (self.records[-1] if self.records else TurnLatencyRecord())

        return {
            "current_ms": {
                "audio_capture": round(cur.audio_capture_latency_ms, 2),
                "audio_queue_depth": cur.audio_queue_depth,
                "audio_processing_lag_s": round(cur.audio_processing_lag_s, 2),
                "audio_ingestion": round(cur.audio_ingestion_ms, 2),
                "diarization": round(cur.diarization_ms, 2),
                "asr": round(cur.asr_ms, 2),
                "alignment": round(cur.alignment_ms, 2),
                "role_resolution": round(cur.role_resolution_ms, 2),
                "fast_semantic": round(cur.fast_semantic_ms, 2),
                "llm": round(cur.llm_ms, 2),
                "state_update": round(cur.state_update_ms, 2),
                "total_end_to_end": round(cur.total_end_to_end_ms, 2),
            },
            "p50_ms": self._percentile(totals, 50),
            "p95_ms": self._percentile(totals, 95),
            "p99_ms": self._percentile(totals, 99),
            "sample_count": len(self.records),
        }
