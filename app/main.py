from pathlib import Path
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from .config import LLM_PROVIDER
from .schemas import Utterance, AnalysisResponse
from .state import StateManager
from .scoring import credibility_score, classify_ptp
from .policy import policy
from .semantic.rules import RulesAnalyzer
from .semantic.gemini import GeminiAnalyzer
from .semantic.ollama import OllamaAnalyzer
from .demo import SCENARIOS

app = FastAPI(title="PTPGuard", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

state_manager = StateManager()

def make_analyzer():
    if LLM_PROVIDER == "gemini":
        return GeminiAnalyzer()
    if LLM_PROVIDER == "ollama":
        return OllamaAnalyzer()
    return RulesAnalyzer()

analyzer = make_analyzer()

def reasons_from_state():
    s = state_manager.state
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
        reasons.append("Borrower voluntarily proposed terms")
    elif norm.get("borrower_initiation", 0.0) < -0.1:
        reasons.append("Terms appear to be driven by agent")

    if norm.get("confirmation", 0.0) > 0.1:
        reasons.append("Explicit borrower confirmation recorded")

    if norm.get("conditionality", 0.0) < -0.1:
        reasons.append(f"Conditional promise detected ({raw.get('conditionality', 0.0):.1f})")
    elif norm.get("conditionality", 0.0) > 0.1:
        reasons.append("Commitment is unconditional / condition cleared")

    if norm.get("feasibility", 0.0) > 0.1:
        reasons.append("Evidence of payment capacity / funds available")
    elif norm.get("feasibility", 0.0) < -0.1:
        reasons.append("Inability to pay / financial constraints")

    if norm.get("hardship", 0.0) < -0.1:
        reasons.append(f"Potential hardship indicators present ({raw.get('hardship', 0.0):.1f})")

    if norm.get("agent_pressure", 0.0) < -0.1:
        reasons.append("Agent pressure detected")

    if s.third_party_background:
        reasons.append("Third person speaking in background (spouse/family member)")
    elif norm.get("third_party", 0.0) < -0.1:
        reasons.append("Third party appears to be making the commitment")

    if norm.get("escape_signal", 0.0) < -0.1:
        reasons.append("Evasive language / call avoidance detected")

    return reasons

async def process_utterance(u: Utterance):
    evidence = await analyzer.analyze(
        speaker=u.speaker,
        text=u.text,
        history=state_manager.state.transcript
    )

    if evidence.detected_speaker:
        u.detected_speaker = evidence.detected_speaker
        u.speaker_confidence = evidence.speaker_confidence
        u.speaker_rationale = evidence.speaker_rationale
        u.is_background_speech = evidence.third_party_background
        u.background_speaker_info = evidence.background_details

    state_manager.update(evidence, u.speaker, u.text)

    score = credibility_score(state_manager.state)
    ptp_type = classify_ptp(state_manager.state)
    action, message = policy(state_manager.state, score)

    return AnalysisResponse(
        score=score,
        ptp_type=ptp_type,
        action=action,
        action_message=message,
        evidence=evidence,
        reasons=reasons_from_state(),
        raw_state=state_manager.state.raw_state,
        normalized_state=state_manager.state.normalized_state,
        score_history=state_manager.state.score_history,
        evidence_history=state_manager.state.evidence_history,
        state=state_manager.snapshot(),
    )

@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent.parent / "static" / "index.html")

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "provider": LLM_PROVIDER,
        "service": "PTPGuard",
        "architecture": "continuous_numeric_state",
        "diarization": "active"
    }

@app.post("/api/call/reset")
async def reset():
    state_manager.reset()
    return {"status": "reset"}

@app.post("/api/call/utterance")
async def utterance(u: Utterance):
    return await process_utterance(u)

@app.get("/api/demo/{name}")
async def demo(name: str):
    if name not in SCENARIOS:
        return {"error": "Unknown scenario", "available": list(SCENARIOS)}
    state_manager.reset()
    outputs = []
    for speaker, text in SCENARIOS[name]:
        outputs.append((await process_utterance(
            Utterance(speaker=speaker, text=text)
        )).model_dump())
    return {"scenario": name, "steps": outputs}

@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            message = await ws.receive_json()

            if message.get("type") == "reset":
                state_manager.reset()
                await ws.send_json({"type": "reset", "state": state_manager.snapshot()})
                continue

            if message.get("type") == "utterance":
                u = Utterance(
                    speaker=message.get("speaker", "auto"),
                    text=message.get("text", "")
                )
                if not u.text.strip():
                    continue
                result = await process_utterance(u)
                await ws.send_json({
                    "type": "analysis",
                    "result": result.model_dump()
                })
    except WebSocketDisconnect:
        pass
