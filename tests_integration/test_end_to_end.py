"""
Chunk E1 — end-to-end tests across all three members.
Real backend over HTTP + real AI engine + real voice bridge; fake LLM/Whisper/TTS.
"""

import json
import socket

import httpx
import pytest

from conversation import VoiceConversation
from src.backend.client import BackendClient, CallConflictError, PatientConflictError
from src.conversation.call_manager import CallManager
from src.conversation.finalize import build_call_payload
from src.llm.extraction import ExtractionResult


# ── Fakes for the external AI/voice services only ────────────────────────────

SCRIPT = {
    "I have a fever and feel weak.": [{"name": "fever"}, {"name": "weakness"}],
    "Yesterday.": [{"name": "fever", "duration": "yesterday"}],
    "Moderate.": [{"name": "fever", "severity": "moderate"}],
    "Yes, 101 F.": [{"name": "fever", "details": "101 F"}],
    "Since yesterday too.": [{"name": "weakness", "duration": "since yesterday"}],
    "Mild.": [{"name": "weakness", "severity": "mild"}],
    "Yes, I take it.": [{"name": "fever", "medication_adherence": "taking"}],
    "My fever is gone, but I still feel weak.": [
        {"name": "fever", "reported_change": "gone"}, {"name": "weakness"}],
    "About the same.": [{"name": "weakness", "reported_change": "same"}],
    "Yes.": [{"name": "weakness", "medication_adherence": "taking"}],
    "My fever is back.": [{"name": "fever"}],
    "I have a fever.": [{"name": "fever"}],
}


def fake_llm(text, context):
    return ExtractionResult(symptoms=SCRIPT.get(text, []))


def fake_whisper(audio_bytes, suffix):
    return audio_bytes.decode() if audio_bytes else ""


def fake_tts(text):
    return b"MP3:" + text.encode()


# ── Helpers ──────────────────────────────────────────────────────────────────

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def api(url, **kwargs) -> BackendClient:
    return BackendClient(base_url=url, sleep=lambda s: None, **kwargs)


def voice(url, dirs, client=None) -> VoiceConversation:
    manager = CallManager(client=client or api(url), extractor=fake_llm, **dirs)
    return VoiceConversation(manager, fake_whisper, fake_tts)


def talk(convo, *answers):
    turn = None
    for answer in answers:
        turn = convo.handle_audio(answer.encode())
        if turn.done:
            break
    return turn


def patient_id(url, external_id="P001"):
    return api(url).get_patient_by_external(external_id)["id"]


def state(url, pid):
    return {s["name"]: s for s in api(url).get_symptom_state(pid)}


def json_files(folder):
    return sorted(p.name for p in folder.glob("*.json")) if folder.exists() else []


# ── 1. Full journey: Day 1 -> Day 2 (resolved) -> Day 3 (recurrence) ────────

def test_full_patient_journey(backend_url, dirs):
    # Day 1
    convo = voice(backend_url, dirs)
    greeting, audio = convo.start("P001", "Asha")
    assert "How are you feeling today?" in greeting and audio
    turn = talk(convo, "I have a fever and feel weak.", "Yesterday.", "Moderate.",
                "Yes, 101 F.", "Since yesterday too.", "Mild.", "Yes, I take it.")
    assert turn.done is True
    assert convo.end().saved is True

    pid = patient_id(backend_url)
    s = state(backend_url, pid)
    assert (s["fever"]["status"], s["fever"]["latest_trend"]) == ("ongoing", "new")
    assert s["fever"]["latest_severity"] == "moderate"
    assert s["fever"]["latest_duration"] == "yesterday"
    assert s["weakness"]["latest_severity"] == "mild"

    # Day 2
    convo = voice(backend_url, dirs)
    greeting, _ = convo.start("P001", "Asha")
    assert "fever and weakness" in greeting
    turn = talk(convo, "My fever is gone, but I still feel weak.", "About the same.", "Yes.")
    assert turn.done is True and "fever has gone" in turn.assistant_text
    assert convo.end().saved is True

    s = state(backend_url, pid)
    assert (s["fever"]["status"], s["fever"]["latest_trend"]) == ("resolved", "resolved")
    assert (s["weakness"]["status"], s["weakness"]["latest_trend"]) == ("ongoing", "stable")
    assert s["fever"]["latest_severity"] == "moderate"          # history kept

    # Day 3: fever comes back
    convo = voice(backend_url, dirs)
    greeting, _ = convo.start("P001", "Asha")
    assert "weakness" in greeting and "fever" not in greeting
    turn = talk(convo, "My fever is back.")
    assert turn.assistant_text == "When did your fever come back?"
    assert convo.end().saved is True

    s = state(backend_url, pid)
    assert s["fever"]["status"] == "ongoing"
    assert s["fever"]["recurred"] is True
    assert s["fever"]["times_reported"] == 3

    # Every observation preserved, in order
    fever_rows = [r for r in api(backend_url).get_symptoms(pid) if r["name"] == "fever"]
    assert [r["status"] for r in fever_rows] == ["ongoing", "resolved", "ongoing"]
    assert [r["trend"] for r in fever_rows] == ["new", "resolved", "new"]

    calls = api(backend_url).get_calls(pid)
    assert len(calls) == 3
    assert all(c["status"] == "completed" for c in calls)
    assert len(httpx.get(f"{backend_url}/patients").json()) == 1


# ── 2. Duplicate prevention ──────────────────────────────────────────────────

def test_duplicate_patient_and_call_prevention(backend_url, dirs):
    client = api(backend_url)
    assert client.create_or_get_patient("P001", "Asha")["created"] is True
    assert client.create_or_get_patient("P001", "Asha")["created"] is False
    with pytest.raises(PatientConflictError):
        client.create_or_get_patient("P001", "Someone Else")
    assert len(httpx.get(f"{backend_url}/patients").json()) == 1

    manager = CallManager(client=client, extractor=fake_llm, **dirs)
    session, _ = manager.start_call("P001", "Asha")
    manager.handle_patient_message(session, "I have a fever.")
    first = manager.end_call(session)
    assert first.saved is True and first.duplicate is False

    payload = build_call_payload(session)                 # identical rebuild
    again = client.save_call(session.patient_id, payload)
    assert again.duplicate is True and again.call_id == first.call_id

    with pytest.raises(CallConflictError):
        client.save_call(session.patient_id, {**payload, "transcript": "changed"})

    assert len(client.get_calls(session.patient_id)) == 1


# ── 3. Dropped call ──────────────────────────────────────────────────────────

def test_dropped_call_saved_as_interrupted(backend_url, dirs):
    convo = voice(backend_url, dirs)
    convo.start("P001", "Asha")
    talk(convo, "I have a fever and feel weak.")
    assert convo.end(interrupted=True).saved is True

    call = api(backend_url).get_calls(patient_id(backend_url))[0]
    assert call["status"] == "interrupted"
    assert call["summary"].startswith("Call ended unexpectedly")
    assert "Patient: I have a fever and feel weak." in call["transcript"]


# ── 4. App crash mid-call, then restart ──────────────────────────────────────

def test_crash_recovered_once_on_restart(backend_url, dirs):
    convo = voice(backend_url, dirs)
    convo.start("P001", "Asha")
    talk(convo, "I have a fever and feel weak.")
    del convo                                              # crash: end() never runs

    results = voice(backend_url, dirs).startup()
    assert [r.saved for r in results] == [True]
    calls = api(backend_url).get_calls(patient_id(backend_url))
    assert len(calls) == 1 and calls[0]["status"] == "interrupted"

    assert voice(backend_url, dirs).startup() == []        # nothing left
    assert len(api(backend_url).get_calls(patient_id(backend_url))) == 1
    assert json_files(dirs["drafts_dir"]) == []


# ── 5. Backend down when the call ends ───────────────────────────────────────

def test_backend_down_at_end_saved_exactly_once_later(backend_url, dirs):
    convo = voice(backend_url, dirs)
    convo.start("P001", "Asha")
    talk(convo, "I have a fever.")

    convo.manager.client = api(f"http://127.0.0.1:{free_port()}", max_retries=0, timeout=2)
    result = convo.end()
    assert result.saved is False                           # never claims success
    assert result.local_path and json_files(dirs["pending_dir"])
    assert api(backend_url).get_calls(patient_id(backend_url)) == []

    results = voice(backend_url, dirs).startup()
    assert [r.saved for r in results] == [True]
    assert len(api(backend_url).get_calls(patient_id(backend_url))) == 1
    assert json_files(dirs["pending_dir"]) == []
    assert voice(backend_url, dirs).startup() == []


# ── 6. Save succeeded but the response was lost ──────────────────────────────

def test_lost_response_retry_gives_duplicate(backend_url, dirs):
    client = api(backend_url)
    manager = CallManager(client=client, extractor=fake_llm, **dirs)
    session, _ = manager.start_call("P001", "Asha")
    manager.handle_patient_message(session, "I have a fever.")
    session.end()

    payload = build_call_payload(session)
    client.save_call(session.patient_id, payload)          # backend saved it...
    pending = dirs["pending_dir"] / f"{session.client_call_id}.json"
    pending.parent.mkdir(parents=True, exist_ok=True)      # ...but the app never saw the reply
    pending.write_text(json.dumps({"patient_id": session.patient_id, "payload": payload,
                                   "attempts": 0, "last_error": None}))

    results = manager.startup()
    saved = [r for r in results if r.saved]
    assert len(saved) == 1 and saved[0].duplicate is True
    assert len(client.get_calls(session.patient_id)) == 1


# ── 7. Old-style data still works ────────────────────────────────────────────

def test_legacy_requests_still_accepted(backend_url):
    pid = httpx.post(f"{backend_url}/patients", json={"name": "Legacy"}).json()["patient_id"]
    for _ in range(2):
        r = httpx.post(f"{backend_url}/patients/{pid}/calls",
                       json={"transcript": "t", "symptoms": [{"name": "cough", "severity": "mild"}]})
        assert r.status_code == 200

    client = api(backend_url)
    assert client.get_patient(pid)["external_id"] is None
    assert len(client.get_calls(pid)) == 2                 # no ID -> no deduplication
    cough = state(backend_url, pid)["cough"]
    assert (cough["status"], cough["times_reported"]) == ("ongoing", 2)
    assert client.get_context(pid)["symptom_state"][0]["name"] == "cough"


# ── 8. History longer than five calls ────────────────────────────────────────

def test_all_history_used_beyond_five_calls(backend_url, dirs):
    client = api(backend_url)
    pid = client.create_or_get_patient("P001", "Asha")["patient_id"]
    client.save_call(pid, {"client_call_id": "old-1", "transcript": "t",
                           "symptoms": [{"name": "headache", "trend": "new"}]})
    for i in range(6):
        client.save_call(pid, {"client_call_id": f"c-{i}", "transcript": "t",
                               "symptoms": [{"name": "fatigue"}]})

    context = client.get_context(pid)
    assert len(context["recent_history"]) == 5
    assert {s["name"] for s in context["symptom_state"]} == {"headache", "fatigue"}

    greeting, _ = voice(backend_url, dirs).start("P001", "Asha")
    assert "fatigue and headache" in greeting