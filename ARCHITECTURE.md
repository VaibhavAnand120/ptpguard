# PTPGuard Architecture Notes

## Runtime path

1. Browser/telephony produces speech.
2. ASR produces an utterance.
3. **Speaker Diarizer & Role Classifier** separates:
   - **Lender / Agent**: Institutional introductions, overdue inquiries, payment requests, PTP confirmations.
   - **Borrower**: First-person financial situation, hardship, payment commitments.
   - **Third Party (Direct)**: Spouse, relative, representative directly on the line.
   - **Third Party (Background / Family)**: Spouse / family member speaking or coaching in the background (e.g. wife prompting *"bol do paise nahi hai"*, *"phone kaat do"*).
4. Semantic provider extracts observable evidence.
5. StateManager merges evidence without repeatedly counting the same feature.
6. Credibility engine calculates an explainable 0–100 score.
7. PTP classifier determines the current PTP class (including `THIRD_PARTY_BACKGROUND_INTERFERENCE`).
8. Policy engine applies safety & privacy overrides (`VERIFY_BORROWER_PRIVACY`).
9. WebSocket sends the updated state to the dashboard.

## Provider abstraction

`SemanticAnalyzer` is the interface.

Implementations:

- `RulesAnalyzer` (with built-in `SpeakerDiarizer`)
- `GeminiAnalyzer` (with LLM diarization + fallback)
- `OllamaAnalyzer` (with LLM diarization + fallback)

This means the rest of the application does not care which LLM is being used.

## Speaker Diarization & Background Speech Separation

- Detects explicit background annotations (`(peeche se: ...)`, `[wife in background]`, `*family member*`).
- Detects side-coaching imperatives directed at the borrower (`"bol do salary nahi aayi"`, `"phone kaat do"`, `"mat do abhi paise"`).
- Differentiates lender inquiries from borrower responses based on lexical signals and conversational turn-taking.
- Resolves multi-speaker utterances where background family speech occurs alongside borrower speech.

## Production replacement points

### ASR
Replace browser SpeechRecognition with a streaming ASR adapter (e.g., streaming Whisper-compatible service).

### Telephony Multi-Track Diarization
In multi-channel telephony (SIP/VoIP with agent on channel 1, borrower on channel 2), bind channel 1 directly to `agent`, and run acoustic speaker separation (e.g. PyAnnote / NeMo) on channel 2 to isolate primary borrower voice from background ambient family voices.

### Historical data
Add account/payment connectors.

### Predictive model
Use stored semantic features + PTP outcomes to train a calibrated keep-probability model.

### Intervention optimization
A/B test recommendations and measure incremental recovery.
