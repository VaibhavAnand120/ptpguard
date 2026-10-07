# PTPGuard — Real-Time PTP Credibility Engine

PTPGuard is a hackathon-ready prototype for **real-time Promise-to-Pay (PTP) credibility analysis** during collections calls.

It is designed as a third listener:

```text
Agent <──────────────> Borrower
              │
              ▼
          PTPGuard
              │
      ┌───────┼────────┐
      ▼       ▼        ▼
     ASR    Semantic  History
      │       │        │
      └───────┼────────┘
              ▼
       Evidence State
              ▼
      Credibility Score
              ▼
       PTP Classification
              ▼
       Safety / Policy
              ▼
       Agent Recommendation
```

## Important design choice

The prototype does **not** claim that `68/100` means a 68% probability of payment.

The current score is an **explainable PTP Credibility Score** generated from semantic evidence. A future production version can train a calibrated PTP-keep probability model once historical PTP/payment outcomes are available.

## What is included

- FastAPI backend
- WebSocket live analysis
- React-free single-page dashboard served by FastAPI
- Real-time automatic speaker diarization (Lender vs Borrower vs 3rd Party vs Background Family Voice)
- Browser SpeechRecognition with live auto-detection
- Gemini API semantic analyzer
- Ollama semantic analyzer
- Rule-based fallback semantic analyzer with diarization
- Stateful PTP evidence tracker
- Explainable credibility scoring
- PTP type classification (including background family interference)
- Hardship safety override & borrower privacy protection
- Agent-pushed / third-party / background coaching detection
- Seven interactive demo scenarios
- Unit tests
- Docker support

## PTP classes

- `GENUINE_FEASIBLE`
- `GENUINE_NOT_FEASIBLE`
- `ESCAPE_PROMISE`
- `THIRD_PARTY_PROMISE`
- `THIRD_PARTY_BACKGROUND_INTERFERENCE`
- `AGENT_RECORDED_OR_PUSHED`
- `UNCONFIRMED`

## Quick start

### 1. Create environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Configure

Copy:

```bash
copy .env.example .env
```

On Linux/macOS:

```bash
cp .env.example .env
```

Choose one:

```env
LLM_PROVIDER=rules
```

This works with no API key.

Or Gemini:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=YOUR_KEY
GEMINI_MODEL=gemini-2.5-flash
```

Or Ollama:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
```

### 4. Run

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Demo

The dashboard has:

- Demo scenario buttons
- Live transcript
- Speaker role
- PTP score
- PTP type
- evidence chips
- explanation
- recommended action
- state JSON

Try:

1. Genuine feasible
2. Genuine hardship
3. Wife in Background (Third-party background voice & coaching)
4. Auto-Diarize Live Call (Multi-turn automatic role separation)
5. Third-party promise (Direct speaker)
6. Agent-pushed PTP
7. Escape promise

## Live microphone

The dashboard uses the browser's `SpeechRecognition` API when supported.

Recommended browsers:

- Chrome
- Edge

Click **Start microphone**, speak, and the browser sends finalized utterances to the backend.

For an actual production telephony deployment, replace browser SpeechRecognition with a streaming telephony/ASR adapter such as a streaming Whisper-compatible service.

## API

### Health

```http
GET /api/health
```

### Reset call

```http
POST /api/call/reset
```

### Analyze one utterance

```http
POST /api/call/utterance
Content-Type: application/json

{
  "speaker": "borrower",
  "text": "salary 10 tareekh ko aayegi, uske baad 2000 de dunga"
}
```

### WebSocket

```text
ws://127.0.0.1:8000/ws/live
```

Send:

```json
{
  "type": "utterance",
  "speaker": "borrower",
  "text": "salary 10 tareekh ko aayegi"
}
```

## Architecture

```text
Browser / Telephony
        |
        v
 Speech Recognition / ASR
        |
        v
 Speaker Role
        |
        v
 Semantic Analyzer
   /       |       \
Gemini   Ollama   Rules
        |
        v
Evidence Extractor
        |
        v
Conversation State
        |
        v
Credibility Engine
        |
        +------> PTP Type
        |
        +------> Safety Policy
        |
        v
Agent Recommendation
```

## Scoring model

The scoring model uses a **dynamic, reversible numeric evidence state** across 10 dimensions rather than irreversible boolean flags or permanent point additions.

### 10 Evidence Dimensions (Consistent Sign Convention)
All dimensions follow one universal rule:
- **POSITIVE (+)** = evidence supporting credible, firm, feasible PTP
- **NEGATIVE (-)** = evidence reducing PTP credibility (conditionality, distress, evasion, coercion)

| Dimension | Positive (+) | Negative (-) | Weight |
| :--- | :--- | :--- | :--- |
| `commitment` | Firm commitment | Refusal / weak commitment | 0.16 |
| `specificity` | Clear concrete amount/date | Vague commitment / timeline | 0.18 |
| `borrower_initiation` | Borrower voluntarily proposes PTP | Agent driving terms / pushing | 0.14 |
| `confirmation` | Explicit confirmation | Avoidance / non-confirmation | 0.16 |
| `feasibility` | Clear ability to pay / salary credited | Inability to pay / financial distress | 0.12 |
| `conditionality` | Unconditional / condition cleared | Conditional ("if salary comes") | 0.10 |
| `hardship` | Feasible / barrier resolved | Genuine financial hardship | 0.06 |
| `agent_pressure` | Borrower-originated / low pressure | Strong agent pressure / coercion | 0.04 |
| `third_party` | Borrower personal responsibility | Third-party promise / interference | 0.02 |
| `escape_signal` | Firm, engaged commitment | Avoidance / "end the call" evasion | 0.02 |

### State Evolution & Soft Saturation
1. **Signed Evidence Deltas**:
   $$\Delta = \text{direction} \times \text{strength} \times \text{confidence}$$
2. **Dynamic Accumulation & Reversibility**:
   $$\text{raw\_state}_{new} = \text{raw\_state}_{old} + \Delta$$
   New evidence can strengthen, weaken, or completely reverse prior evidence (e.g. conditional salary barrier resolved).
3. **Soft Saturation (Bounded Influence)**:
   $$\text{normalized\_state}_i = \tanh\left(\frac{\text{raw\_state}_i}{\text{SCALE}}\right) \in (-1.0, +1.0)$$
   (where $\text{SCALE} = 2.5$). Diminishing returns prevent repeated identical sentences from growing without bounds.
4. **Dimension Contribution**:
   $$\text{contribution}_i = \frac{\text{normalized\_state}_i + 1}{2} \in [0.0, 1.0]$$
5. **Explainable Credibility Score**:
   $$\text{Score} = \text{round}\left(\sum_i \text{weight}_i \times \text{contribution}_i \times 100\right) \in [0, 100]$$

## Why the LLM does not directly produce the score

The LLM is an **objective semantic observer**, not an uninterpretable score predictor.

For example, when the borrower says:
```text
"Salary aa gayi hai already, 10 ko ₹2000 payment confirm hai."
```

The semantic provider produces directional observations:
```json
{
  "signals": {
    "commitment": {"direction": 1, "strength": 2, "confidence": 0.92},
    "specificity": {"direction": 1, "strength": 3, "confidence": 0.96},
    "confirmation": {"direction": 1, "strength": 2, "confidence": 0.94},
    "conditionality": {"direction": 1, "strength": 2, "confidence": 0.90},
    "feasibility": {"direction": 1, "strength": 2, "confidence": 0.92}
  },
  "amount": 2000.0,
  "date": "10th"
}
```

The state manager updates the continuous signed state, applies soft saturation, and the scoring engine deterministically aggregates the score. This makes every turn explainable, auditable, and fully reversible.

## Safety policy

A low score does not automatically mean harsher treatment.

If hardship is detected, the policy engine can return:

```text
HUMAN_OR_RESTRUCTURE
```

instead of aggressive follow-up.

If an agent appears to have supplied the PTP amount/date without clear borrower confirmation:

```text
CONFIRM_PTP
```

## Future production model

Once historical data is available:

```text
Conversation
+
Semantic evidence
+
Account history
+
Payment outcomes
        |
        v
     XGBoost
        |
        v
Calibrated P(PTP kept)
```

The current state schema is intentionally designed so those features can later be used directly for ML.

## Project structure

```text
ptpguard/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   ├── state.py
│   ├── scoring.py
│   ├── policy.py
│   ├── demo.py
│   └── semantic/
│       ├── base.py
│       ├── gemini.py
│       ├── ollama.py
│       └── rules.py
├── static/
│   └── index.html
├── tests/
│   ├── test_scoring.py
│   └── test_policy.py
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Disclaimer

This prototype is a decision-support demonstration. It should not be used to make adverse collection decisions without appropriate business, legal, privacy, fairness and human-review controls.
