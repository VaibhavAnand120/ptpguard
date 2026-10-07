"""
Auxiliary Acoustic Feature Extractor
Captures acoustic energy, spectral centroid, and zero-crossing rate from 16kHz PCM audio.

IMPORTANT ARCHITECTURAL NOTICE:
These features are strictly auxiliary telemetry and signal analysis utilities.
They are NOT the speaker diarizer.
Primary speaker diarization is performed exclusively by NVIDIA NeMo Sortformer (or MockDiarizer in mock/test mode).
"""

import math
import struct
from typing import List, Dict, Any, Optional


class AuxiliaryAcousticFeatures:
    """
    Auxiliary Acoustic Feature Extraction for Audio Telemetry & VAD Signal Monitoring.
    NOT a speaker diarizer.
    """

    @staticmethod
    def extract_pcm_features(pcm_bytes: bytes, dim: int = 16) -> List[float]:
        """
        Extracts 16-dimensional acoustic energy & spectral distribution features:
        - Zero-crossing rate (ZCR)
        - RMS Energy
        - 8-band subband energy filter bank
        - Spectral centroid proxy
        """
        num_samples = len(pcm_bytes) // 2
        if num_samples < 32:
            return [0.0] * dim

        try:
            samples = struct.unpack(f"<{num_samples}h", pcm_bytes[:num_samples * 2])
        except Exception:
            samples = [0] * num_samples

        float_samples = [s / 32768.0 for s in samples]

        # 1. Zero crossing rate
        zcr = sum(1 for i in range(1, len(float_samples)) if (float_samples[i] >= 0) != (float_samples[i - 1] >= 0))
        zcr_norm = float(zcr) / max(1, len(float_samples))

        # 2. RMS Energy
        energy = math.sqrt(sum(s * s for s in float_samples) / max(1, len(float_samples)))

        # 3. Frequency Subband Energy Estimation (simplified 8-band bank)
        subband_energies = [0.0] * 8
        band_len = max(4, len(float_samples) // 8)
        for b in range(8):
            chunk_slice = float_samples[b * band_len : (b + 1) * band_len]
            if chunk_slice:
                subband_energies[b] = math.sqrt(sum(s * s for s in chunk_slice) / len(chunk_slice))

        # 4. Spectral Centroid Proxy (weighted center of energy)
        weighted_sum = sum((i + 1) * subband_energies[i] for i in range(8))
        total_energy = sum(subband_energies) or 1e-6
        centroid = (weighted_sum / total_energy) / 8.0

        vec = [
            zcr_norm,
            energy,
            centroid,
            subband_energies[0],
            subband_energies[1],
            subband_energies[2],
            subband_energies[3],
            subband_energies[4],
            subband_energies[5],
            subband_energies[6],
            subband_energies[7],
            abs(subband_energies[0] - subband_energies[7]),
            abs(subband_energies[2] - subband_energies[5]),
            (zcr_norm * energy),
            (centroid * energy),
            0.5
        ]

        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    @staticmethod
    def calculate_rms(pcm_bytes: bytes) -> float:
        """Calculates RMS energy of 16-bit PCM bytes."""
        num_samples = len(pcm_bytes) // 2
        if num_samples == 0:
            return 0.0
        try:
            samples = struct.unpack(f"<{num_samples}h", pcm_bytes[:num_samples * 2])
            return math.sqrt(sum((s / 32768.0) ** 2 for s in samples) / max(1, num_samples))
        except Exception:
            return 0.0
