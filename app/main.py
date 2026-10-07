from pathlib import Path
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from .config import LLM_PROVIDER
from .schemas import Utterance, AnalysisResponse
from .state import StateManager
from .voice.pipeline import ParallelVoicePipeline
from .demo import SCENARIOS

app = FastAPI(title="PTPGuard", version="0.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

state_manager = StateManager()
voice_pipeline = ParallelVoicePipeline(state_manager=state_manager)


async def process_utterance(u: Utterance) -> AnalysisResponse:
    res_dict = await voice_pipeline.process_utterance(u)
    return AnalysisResponse(**res_dict)


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent.parent / "static" / "index.html")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "provider": LLM_PROVIDER,
        "service": "PTPGuard",
        "architecture": "streaming_parallel_voice_analysis",
        "diarization_model": "nvidia/diar_streaming_sortformer_4spk-v2.1",
        "parallel_stages": [
            "vad",
            "sortformer_streaming",
            "asr_streaming",
            "timestamp_alignment",
            "persistent_role_resolver",
            "fast_rules_semantic",
            "async_deep_llm"
        ],
        "telemetry": voice_pipeline.telemetry.get_summary()
    }


@app.post("/api/call/reset")
async def reset():
    voice_pipeline.reset()
    return {"status": "reset", "state": state_manager.snapshot()}


@app.post("/api/call/utterance")
async def utterance(u: Utterance):
    return await process_utterance(u)


@app.get("/api/demo/{name}")
async def demo(name: str):
    if name not in SCENARIOS:
        return {"error": "Unknown scenario", "available": list(SCENARIOS)}
    voice_pipeline.reset()
    outputs = []
    for speaker, text in SCENARIOS[name]:
        res = await process_utterance(Utterance(speaker=speaker, text=text))
        outputs.append(res.model_dump())
    return {"scenario": name, "steps": outputs}


@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            message = await ws.receive_json()

            if message.get("type") == "reset":
                voice_pipeline.reset()
                await ws.send_json({"type": "reset", "state": state_manager.snapshot()})
                continue

            if message.get("type") == "audio_chunk":
                import base64
                audio_b64 = message.get("audio", "")
                ts = float(message.get("timestamp", 0.0))
                dur = float(message.get("duration", 0.5))
                client_time = float(message.get("client_capture_time", 0.0))
                cid = message.get("chunk_id", None)
                v_name = message.get("voice_name") or message.get("voice") or message.get("voice_id")
                v_feat = message.get("voice_features", None)
                audio_bytes = base64.b64decode(audio_b64) if audio_b64 else b""
                segs = voice_pipeline.process_audio_chunk(
                    audio_chunk=audio_bytes,
                    timestamp=ts,
                    duration=dur,
                    chunk_id=cid,
                    client_capture_time=client_time,
                    voice_features=v_feat,
                    voice_name=v_name
                )
                await ws.send_json({
                    "type": "diarization_chunk",
                    "segments": [s.__dict__ for s in segs],
                    "audio_intake": voice_pipeline.audio_buffer.get_diagnostics(),
                    "sortformer": voice_pipeline.sortformer.get_tracks_summary()
                })
                continue

            if message.get("type") == "utterance":
                u = Utterance(
                    speaker=message.get("speaker", "auto"),
                    text=message.get("text", ""),
                    mode=message.get("mode", "auto"),
                    voice=message.get("voice") or message.get("voice_id"),
                    voice_id=message.get("voice_id") or message.get("voice"),
                    voice_features=message.get("voice_features"),
                    chunk_id=message.get("chunk_id")
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
