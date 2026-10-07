import asyncio
import json
from fastapi.testclient import TestClient
from app.main import app

def test_websocket_stream():
    client = TestClient(app)
    with client.websocket_connect("/ws/live") as ws:
        # Turn 1: Voice A (Agent)
        ws.send_json({"type": "utterance", "speaker": "auto", "text": "Sir payment kab kar paoge?", "voice": "Voice A"})
        data1 = ws.receive_json()
        print("WS Turn 1:", data1["result"]["speaker_id"], "->", data1["result"]["speaker_role"])
        assert data1["result"]["speaker_id"] == "speaker_0"
        assert data1["result"]["speaker_role"] == "agent"

        # Turn 2: Voice B (Borrower)
        ws.send_json({"type": "utterance", "speaker": "auto", "text": "Salary 7 ko aa jayegi, 8 ko payment kar dunga.", "voice": "Voice B"})
        data2 = ws.receive_json()
        print("WS Turn 2:", data2["result"]["speaker_id"], "->", data2["result"]["speaker_role"])
        assert data2["result"]["speaker_id"] == "speaker_1"
        assert data2["result"]["speaker_role"] == "borrower"

        # Turn 3: Voice A returns (Agent)
        ws.send_json({"type": "utterance", "speaker": "auto", "text": "Can you pay on 8 October?", "voice": "Voice A"})
        data3 = ws.receive_json()
        print("WS Turn 3:", data3["result"]["speaker_id"], "->", data3["result"]["speaker_role"])
        assert data3["result"]["speaker_id"] == "speaker_0"
        assert data3["result"]["speaker_role"] == "agent"

        # Turn 4: Voice B returns (Borrower)
        ws.send_json({"type": "utterance", "speaker": "auto", "text": "Yes sir.", "voice": "Voice B"})
        data4 = ws.receive_json()
        print("WS Turn 4:", data4["result"]["speaker_id"], "->", data4["result"]["speaker_role"])
        assert data4["result"]["speaker_id"] == "speaker_1"
        assert data4["result"]["speaker_role"] == "borrower"

        print("SUCCESS: WebSocket streaming works perfectly!")

if __name__ == "__main__":
    test_websocket_stream()
