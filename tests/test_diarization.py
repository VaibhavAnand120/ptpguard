import anyio
from app.semantic.diarization import SpeakerDiarizer
from app.schemas import ConversationState
from app.semantic.rules import RulesAnalyzer
from app.scoring import credibility_score, classify_ptp
from app.policy import policy


def test_detect_lender_agent():
    text = "Namaste Sir, main Bajaj Finance se bol raha hoon. Aapka overdue EMI ₹3,500 pending hai, kab pay karenge?"
    res = SpeakerDiarizer.detect(text, hint_speaker="auto")
    assert res.primary_speaker == "agent"
    assert res.confidence >= 0.8


def test_detect_lender_questions():
    # Inquiries asked by lender
    for line in [
        "Sir aap payment kab kar paoge?",
        "2000 on 10th confirmed?",
        "15 ke baad payment kar paoge?",
        "Sir pending amount kab pay karenge?",
        "Kab karenge?",
        "Sir ₹5000 10th ko kar denge, correct?",
    ]:
        res = SpeakerDiarizer.detect(line, hint_speaker="auto")
        assert res.primary_speaker == "agent", f"Failed on lender query: {line}"


def test_detect_borrower():
    text = "Sir meri salary 10 tareekh ko aayegi, uske baad main ₹2,000 pay kar dunga."
    res = SpeakerDiarizer.detect(text, hint_speaker="auto")
    assert res.primary_speaker == "borrower"
    assert res.confidence >= 0.7


def test_detect_borrower_commitments():
    # Promises and hardship stated by borrower
    for line in [
        "10 tareekh ko salary aayegi.",
        "Main 10 tareekh ko 2000 pay kar dunga.",
        "Sir abhi paise nahi hain.",
        "Salary 15 ko aayegi, uske baad kar dunga.",
        "Haan sir, salary ke baad kar dunga.",
        "Haan haan sir kar dunga.",
        "Bas call rakhiye sir, kar dunga.",
        "Haan sir, confirm hai.",
    ]:
        res = SpeakerDiarizer.detect(line, hint_speaker="auto")
        assert res.primary_speaker == "borrower", f"Failed on borrower turn: {line}"


def test_multi_turn_not_reversed():
    conversation = [
        ("Sir kab payment kar paoge?", "agent"),
        ("10 tareekh ko salary aayegi.", "borrower"),
        ("Main 10 tareekh ko 2000 pay kar dunga.", "borrower"),
        ("2000 on 10th confirmed?", "agent"),
        ("Haan sir, confirm hai.", "borrower"),
    ]
    history = []
    for line, expected_speaker in conversation:
        res = SpeakerDiarizer.detect(line, hint_speaker="auto", history=history)
        assert res.primary_speaker == expected_speaker, f"Expected {expected_speaker} but got {res.primary_speaker} for: {line}"
        history.append({"speaker": res.primary_speaker})


def test_detect_direct_third_party():
    text = "Main unka bhai bol raha hoon, wo abhi hospital me hain."
    res = SpeakerDiarizer.detect(text, hint_speaker="auto")
    assert res.primary_speaker == "third_party"


def test_detect_background_wife_coaching():
    text = "(Peeche se patni: Arey bolo paise nahi hai abhi, phone kaato unka!)"
    res = SpeakerDiarizer.detect(text, hint_speaker="auto")
    assert res.primary_speaker == "third_party_background"
    assert res.has_background_speech is True
    assert res.background_coaching is True
    assert res.background_speaker_type == "wife"


def test_detect_background_coaching_phrase():
    text = "Bol do abhi salary nahi aayi hai, baad me call karein"
    res = SpeakerDiarizer.detect(text, hint_speaker="auto")
    assert res.primary_speaker == "third_party_background"
    assert res.background_coaching is True


def test_embedded_background_speech_in_borrower():
    text = "Main 10 ko dunga (peeche se: mat do paise) dekhta hoon."
    res = SpeakerDiarizer.detect(text, hint_speaker="borrower")
    assert res.has_background_speech is True
    assert res.background_coaching is True


def test_rules_analyzer_background_inference():
    async def run():
        analyzer = RulesAnalyzer()
        evidence = await analyzer.analyze(
            speaker="auto",
            text="(Peeche se patni: bolo paise nahi hai abhi, phone kaat do!)"
        )
        assert evidence.third_party_background is True
        assert evidence.background_coaching is True
        assert evidence.detected_speaker == "third_party_background"

    anyio.run(run)


def test_policy_background_override():
    s = ConversationState(
        ptp_detected=True,
        third_party_background=True,
        background_coaching=True
    )
    action, msg = policy(s, score=40)
    assert action == "VERIFY_BORROWER_PRIVACY"
    assert classify_ptp(s) == "THIRD_PARTY_BACKGROUND_INTERFERENCE"
