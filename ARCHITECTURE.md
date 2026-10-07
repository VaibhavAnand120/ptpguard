# PTPGuard Architecture — Real-Time Streaming Parallel Voice Pipeline

## 1. Parallel Streaming Architecture

PTPGuard operates as an ultra-low-latency real-time voice intelligence engine during collections calls:

```text
                    LIVE AUDIO
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
             VAD              AUDIO BUFFER
              │                   │
              │          ┌────────┴────────┐
              │          ▼                 ▼
              │     SORTFORMER            ASR
              │   (4-Spk Stream)     (Stream Chunk)
              │          │                 │
              │          ▼                 ▼
              │     SPEAKER ID         TRANSCRIPT
              │    (speaker_0..3)    ([t_start, t_end])
              │          │                 │
              └──────────┴─────────────────┘
                         │
                         ▼
                  TIME ALIGNMENT
                         │
                         ▼
              SPEAKER + TRANSCRIPT
                         │
                ┌────────┴────────┐
                ▼                 ▼
          ROLE RESOLVER     SEMANTIC ANALYZER
       (Persistent State)   ┌─────┴─────┐
                │           ▼           ▼
                │       FAST PATH   DEEP PATH
                │        (Rules)   (Async LLM)
                │        < 50ms      (Window)
                └────────┬──────────────┘
                         ▼
                  EVIDENCE STATE
             (10 Continuous Dimensions)
                         │
                    ┌────┴────┐
                    ▼         ▼
                  SCORE    CLASSIFICATION
                (0 - 100)   (PTP Class)
                    │         │
                    └────┬────┘
                         ▼
                   POLICY ENGINE
               (Safety / Privacy)
                         │
                         ▼
                 REAL-TIME TELEMETRY
               (P50, P95, P99 Latency)
```

## 2. Decoupled Speaker Identity vs Role Resolution

### Component Taxonomy & Responsibilities
- **NVIDIA NeMo Sortformer** = **Speaker Diarization**:
  Identifies *WHO* is physically speaking (`speaker_0`, `speaker_1`, `speaker_2`, `speaker_3`).
  Model: `nvidia/diar_streaming_sortformer_4spk-v2.1`.
  Never infers identity from conversational turn order, turn alternation, or keyword matching.
- **RoleResolver** = **Conversational Role Classification**:
  Answers *Is this speaker the AGENT or BORROWER?*
  Accumulates multi-turn linguistic and context evidence independently of speaker IDs.
- **Semantic LLM** = **Meaning / PTP Extraction**:
  Analyzes dialogue semantics, extracting conditionality, commitments, amounts, and dates.
- **ML Risk Model** = **PTP Keep Probability**:
  Evaluates 10 signed continuous dimension states with soft saturation to predict payment fulfillment odds.
- **Policy Engine** = **Safe Business Action**:
  Executes hardship safety gates, borrower privacy protections, and counterfactual action recommendations.

### Diarizer Class Hierarchy & Fallback Architecture
```text
                    SpeakerDiarizer (ABC)
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   NeMoSortformerDiarizer                MockDiarizer
   - Official NVIDIA NeMo checkpoint    - Deterministic physical mapping
   - CUDA GPU / CPU tensor execution    - Strictly for tests & dev fallback
   - True streaming forward pass        - Clearly labeled as MOCK
```

- **AuxiliaryAcousticFeatures**:
  Captures RMS energy, spectral centroid, zero-crossing rate, and subband energies from 16kHz PCM bytes.
  **Strictly auxiliary telemetry and VAD signal monitoring** — NOT the primary diarizer.
  Centroid cosine similarity is explicitly eliminated as the primary diarizer.

### Role Resolver: Decoupled & Persistent
- Resolves speaker roles separately from speaker IDs:
  `AGENT`, `BORROWER`, `THIRD_PARTY`, `UNKNOWN`.
- Tracks persistent role state:
  ```json
  {
    "speaker_0": {"role": "agent", "confidence": 0.98, "status": "confirmed"},
    "speaker_1": {"role": "borrower", "confidence": 0.94, "status": "confirmed"}
  }
  ```
- **Provisional $\to$ Confirmed**:
  - Confidence $< 0.85 \implies$ provisional.
  - Confidence $\ge 0.85 \implies$ confirmed.
- **Inertia**: Once a role is confirmed, subsequent utterances cannot easily flip it. For example, a borrower confirming an amount or date does not flip into an agent.

## 3. Parallel Execution & Two-Path Semantic Analysis

- **Parallel Ingestion**: ASR and Sortformer run concurrently on the incoming audio buffer without waiting on each other.
- **Timestamp Alignment**: Fuses ASR segment timestamps with Sortformer speaker intervals via temporal intersection over time ($IoU$).
- **Fast Path (< 50ms)**:
  Rule-based lightweight directional semantic extraction immediately updates the continuous signed state, calculates the credibility score, evaluates policy rules, and pushes updates to the client.
- **Deep Path (Async Background)**:
  Asynchronously invokes Gemini / Ollama on recent conversation windows to refine nuanced context without blocking live audio or real-time responses.

## 4. Entity Separation: Agent Proposed vs Borrower Stated Terms

To prevent false commitment attribution:
- `agent_proposed_date` and `agent_proposed_amount`:
  Recorded when the collections agent asks questions or unilaterally marks a commitment (e.g., *"I will mark ₹5,000 on 10 October"* or *"Can you pay before 10 October?"*).
  These terms do **NOT** create borrower commitment, borrower specificity, or borrower initiation!
- `borrower_stated_date` and `borrower_stated_amount`:
  Recorded when the borrower independently or explicitly commits to terms (e.g., *"Yes, I can pay before 10 October"*).
  These terms drive borrower commitment, specificity, and confirmation.
- **Agent Push Detection**:
  If the agent unilaterally proposed an amount or date and the borrower only offered passive assent (*"Okay"*, *"ji"*) without independent commitment, the classification becomes `AGENT_RECORDED_OR_PUSHED`.

## 5. Live Audio vs Text Simulation Modes

1. **Live Audio / Auto-Diarization (`mode="live_audio"`)**:
   - Audio ingested via microphone or audio streaming chunks.
   - NVIDIA Streaming Sortformer clusters acoustic frame embeddings into stable tracks (`speaker_0`..`speaker_3`).
   - RoleResolver resolves persistent roles automatically.
   - Tagged in UI as: `[SPEAKER_X | ROLE | 96% AUTO RESOLVED]`.
2. **Text Simulation (`mode="manual_simulation"`)**:
   - Used for scripted scenario validation without acoustic data.
   - User explicitly selects Agent / Borrower / Third Party.
   - The selected role is an explicit simulation input.
   - Tagged in UI as: `[SPEAKER_X | ROLE | MANUAL SIMULATION]`.

## 6. Elimination of Fallback-to-Previous-Speaker Bug

- Previous speaker is never used as an automatic default when an utterance lacks acoustic match or strong regex.
- Conversational turn structure models interlocutor exchange: when one speaker asks a question or proposes terms, the subsequent response or backchannel belongs to the other party.
- If an utterance is ambiguous and no acoustic data is present, role confidence remains provisional with explicit uncertainty (`UNKNOWN` / provisional), preventing artificial borrower locks.

## 7. Latency Telemetry & Monitoring

Every pipeline stage is instrumented with sub-millisecond precision:
- `audio_ingestion_latency`
- `diarization_latency` (Sortformer)
- `ASR_latency`
- `alignment_latency`
- `role_resolution_latency`
- `fast_semantic_latency`
- `LLM_latency`
- `state_update_latency`
- `total_end_to_end_latency`

Calculates rolling **P50, P95, and P99** percentiles to ensure real-time budget compliance.
