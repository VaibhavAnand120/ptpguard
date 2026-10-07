import anyio
from fastapi.testclient import TestClient
from app.main import app


def test_api_health():
    client = TestClient(app)
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_demo_background_family():
    client = TestClient(app)
    res = client.get("/api/demo/background_family")
    assert res.status_code == 200
    data = res.json()
    assert data["scenario"] == "background_family"
    assert len(data["steps"]) == 7

    # The final step should reflect background interference / coaching
    final_step = data["steps"][-1]
    assert final_step["evidence"]["third_party_background"] is True or final_step["state"]["third_party_background"] is True
    assert final_step["ptp_type"] == "THIRD_PARTY_BACKGROUND_INTERFERENCE"
    assert final_step["action"] == "VERIFY_BORROWER_PRIVACY"


def test_demo_auto_diarize_live():
    client = TestClient(app)
    res = client.get("/api/demo/auto_diarize_live")
    assert res.status_code == 200
    data = res.json()
    assert data["scenario"] == "auto_diarize_live"
    assert len(data["steps"]) == 4

    # Verify that different speakers were identified in the transcript
    transcript = data["steps"][-1]["state"]["transcript"]
    speakers_in_transcript = [t["speaker"] for t in transcript]
    assert "agent" in speakers_in_transcript
    assert "borrower" in speakers_in_transcript
    assert "third_party_background" in speakers_in_transcript
