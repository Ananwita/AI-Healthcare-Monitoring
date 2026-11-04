"""Chunk B7: summary, payload, saving, pending retry and interrupted-call recovery.
Uses an in-memory fake backend with the same idempotency rules as Chunk A4."""

import json

import pytest

from src.backend.client import (
    BackendError,
    BackendUnavailableError,
    CallConflictError,
    SaveResult,
)
from src.conversation.finalize import (
    FAILED_SUBDIR,
    build_call_payload,
    build_summary,
    finalize_call,
    recover_interrupted_calls,
    retry_pending_calls,
    save_draft,
)
from src.conversation.session import CallSession
from src.llm.extraction import ExtractionResult


# ── Fakes ────────────────────────────────────────────────────────────────────

class FakeBackend:
    """save_call() with A4 rules: same id + same payload -> duplicate; different -> 409."""

    def __init__(self):
        self.calls = {}          # (patient_id, client_call_id) -> (call_id, payload)
        self.down = False
        self.reject_with = None  # e.g. BackendError("bad", 422)
        self.attempts = 0

    def save_call(self, patient_id, payload):
        self.attempts += 1
        if self.down:
            raise BackendUnavailableError("backend down")
        if self.reject_with:
            raise self.reject_with
        key = (patient_id, payload["client_call_id"])
        if key in self.calls:
            call_id, stored = self.calls[key]
            if stored != payload:
                raise CallConflictError("conflict", status_code=409, detail={"call_id": call_id})
            return SaveResult(call_id, patient_id, key[1], payload["status"], len(payload["symptoms"]), True)
        call_id = len(self.calls) + 1
        self.calls[key] = (call_id, json.loads(json.dumps(payload)))
        return SaveResult(call_id, patient_id, key[1], payload["status"], len(payload["symptoms"]), False)


@pytest.fixture
def dirs(tmp_path):
    return {"pending_dir": tmp_path / "pending", "drafts_dir": tmp_path / "drafts"}


def say(session, text, symptoms=()):
    session.process_patient_message(
        text, lambda t, c: ExtractionResult(symptoms=list(symptoms))
    )


def day1_session():
    s = CallSession(patient_id=1, external_id="P001")
    s.add_assistant_message("How are you feeling today?")
    say(s, "I have a fever and feel weak.", [{"name": "fever"}, {"name": "weakness"}])
    s.add_assistant_message("When did your fever start?", "fever", "duration")
    say(s, "Yesterday.", [{"name": "fever", "duration": "yesterday"}])
    s.add_assistant_message("Have you measured your temperature?", "fever", "details")
    say(s, "Yes, 101 F.", [{"name": "fever", "details": "101 F"}])
    return s


def day2_session():
    s = CallSession(patient_id=1, previous_symptoms=[
        {"name": "fever", "status": "ongoing"}, {"name": "weakness", "status": "ongoing"},
        {"name": "headache", "status": "ongoing"}, {"name": "cough", "status": "resolved"},
    ])
    say(s, "My fever is gone, I still feel weak, and my cough is back.",
        [{"name": "fever", "reported_change": "gone"},
         {"name": "weakness", "severity": "mild"},
         {"name": "cough"}])
    return s


def files(directory):
    return sorted(p.name for p in directory.glob("*.json")) if directory.exists() else []


# ── Summary ──────────────────────────────────────────────────────────────────

def test_summary_day1():
    assert build_summary(day1_session()) == (
        "Fever: new (duration: yesterday; 101 F). Weakness: new."
    )


def test_summary_day2_resolution_recurrence_and_not_discussed():
    summary = build_summary(day2_session())
    assert "Fever: resolved." in summary
    assert "Weakness: stable (mild)." in summary
    assert "Cough: returned after being resolved." in summary
    assert "Not discussed in this call (status unchanged): headache." in summary


def test_summary_interrupted_and_empty():
    s = CallSession(patient_id=1)
    assert build_summary(s, "interrupted") == (
        "Call ended unexpectedly; information is partial. No symptoms reported."
    )


def test_summary_includes_medication():
    s = CallSession(patient_id=1)
    say(s, "x", [{"name": "cough", "medication_adherence": "not_taking"}])
    assert "medication: not taking" in build_summary(s)


# ── Payload ──────────────────────────────────────────────────────────────────

def test_payload_matches_backend_schema():
    payload = build_call_payload(day2_session())
    assert set(payload) == {"client_call_id", "status", "transcript", "summary", "symptoms"}
    assert payload["status"] == "completed"
    symptoms = {s["name"]: s for s in payload["symptoms"]}
    assert set(symptoms["fever"]) == {
        "name", "severity", "duration", "medication_adherence", "status", "trend"
    }
    assert (symptoms["fever"]["status"], symptoms["fever"]["trend"]) == ("resolved", "resolved")
    assert (symptoms["weakness"]["status"], symptoms["weakness"]["trend"]) == ("ongoing", "stable")
    assert (symptoms["cough"]["status"], symptoms["cough"]["trend"]) == ("ongoing", "new")
    assert "headache" not in symptoms          # unmentioned -> no new observation


def test_payload_is_deterministic():
    s = day1_session()
    assert build_call_payload(s) == build_call_payload(CallSession.from_dict(s.to_dict()))


def test_payload_invalid_status():
    with pytest.raises(ValueError):
        build_call_payload(day1_session(), "dropped")


# ── finalize_call ────────────────────────────────────────────────────────────

def test_finalize_success_cleans_up(dirs):
    backend, s = FakeBackend(), day1_session()
    save_draft(s, dirs["drafts_dir"])
    result = finalize_call(s, backend, **dirs)

    assert result.saved is True and result.call_id == 1 and result.duplicate is False
    assert s.ended is True
    assert files(dirs["pending_dir"]) == [] and files(dirs["drafts_dir"]) == []
    assert len(backend.calls) == 1


def test_finalize_backend_down_keeps_call_and_never_claims_saved(dirs):
    backend, s = FakeBackend(), day1_session()
    backend.down = True
    result = finalize_call(s, backend, **dirs)

    assert result.saved is False and result.failed is False
    assert "kept" in result.message
    stored = json.loads((dirs["pending_dir"] / f"{s.client_call_id}.json").read_text())
    assert stored["attempts"] == 1
    assert stored["payload"] == build_call_payload(s)


def test_retry_after_backend_returns(dirs):
    backend, s = FakeBackend(), day1_session()
    backend.down = True
    finalize_call(s, backend, **dirs)
    backend.down = False

    results = retry_pending_calls(backend, dirs["pending_dir"])
    assert [r.saved for r in results] == [True]
    assert files(dirs["pending_dir"]) == []
    assert len(backend.calls) == 1


def test_retry_resends_exact_stored_payload(dirs):
    backend, s = FakeBackend(), day1_session()
    backend.down = True
    finalize_call(s, backend, **dirs)
    path = dirs["pending_dir"] / f"{s.client_call_id}.json"
    stored_payload = json.loads(path.read_text())["payload"]

    backend.down = False
    retry_pending_calls(backend, dirs["pending_dir"])
    assert backend.calls[(1, s.client_call_id)][1] == stored_payload


def test_lost_response_then_retry_gives_duplicate_not_second_call(dirs):
    # Backend saved the call, but the app crashed before deleting the pending file
    backend, s = FakeBackend(), day1_session()
    payload = build_call_payload(s)
    backend.save_call(1, payload)                        # first (lost) success
    pending = dirs["pending_dir"] / f"{s.client_call_id}.json"
    pending.parent.mkdir(parents=True)
    pending.write_text(json.dumps({"patient_id": 1, "payload": payload, "attempts": 0}))

    results = retry_pending_calls(backend, dirs["pending_dir"])
    assert results[0].saved is True and results[0].duplicate is True
    assert len(backend.calls) == 1
    assert files(dirs["pending_dir"]) == []


@pytest.mark.parametrize("error", [
    CallConflictError("conflict", status_code=409, detail={"call_id": 1}),
    BackendError("invalid", status_code=422),
    BackendError("unknown patient", status_code=404),
])
def test_rejected_call_moved_to_failed_and_not_retried(dirs, error):
    backend, s = FakeBackend(), day1_session()
    backend.reject_with = error
    result = finalize_call(s, backend, **dirs)

    assert result.saved is False and result.failed is True
    failed_file = dirs["pending_dir"] / FAILED_SUBDIR / f"{s.client_call_id}.json"
    assert failed_file.exists()
    assert json.loads(failed_file.read_text())["last_error"]
    assert files(dirs["pending_dir"]) == []

    attempts = backend.attempts
    retry_pending_calls(backend, dirs["pending_dir"])
    assert backend.attempts == attempts                  # never resent automatically


def test_unexpected_error_keeps_call(dirs):
    class Broken:
        def save_call(self, *a):
            raise RuntimeError("bug")
    s = day1_session()
    result = finalize_call(s, Broken(), **dirs)
    assert result.saved is False and result.failed is False
    assert files(dirs["pending_dir"]) == [f"{s.client_call_id}.json"]


def test_unreadable_pending_file_moved_to_failed(dirs):
    dirs["pending_dir"].mkdir(parents=True)
    (dirs["pending_dir"] / "broken.json").write_text("{not json")
    results = retry_pending_calls(FakeBackend(), dirs["pending_dir"])
    assert results[0].failed is True
    assert (dirs["pending_dir"] / FAILED_SUBDIR / "broken.json").exists()


def test_retry_with_no_pending_folder(dirs):
    assert retry_pending_calls(FakeBackend(), dirs["pending_dir"]) == []


# ── Interrupted calls ────────────────────────────────────────────────────────

def test_draft_recovered_as_interrupted(dirs):
    backend, s = FakeBackend(), day1_session()
    save_draft(s, dirs["drafts_dir"])                    # app crashes here

    results = recover_interrupted_calls(backend, **dirs)
    assert [r.saved for r in results] == [True]
    _, stored = backend.calls[(1, s.client_call_id)]
    assert stored["status"] == "interrupted"
    assert stored["summary"].startswith("Call ended unexpectedly")
    assert "Patient: Yes, 101 F." in stored["transcript"]
    assert files(dirs["drafts_dir"]) == [] and files(dirs["pending_dir"]) == []


def test_draft_with_only_greeting_is_discarded(dirs):
    backend = FakeBackend()
    s = CallSession(patient_id=1)
    s.add_assistant_message("How are you feeling today?")
    save_draft(s, dirs["drafts_dir"])

    assert recover_interrupted_calls(backend, **dirs) == []
    assert backend.calls == {} and files(dirs["drafts_dir"]) == []


def test_crash_after_pending_written_before_draft_deleted(dirs):
    # Both files exist: the call was already finalized as "completed"
    backend, s = FakeBackend(), day1_session()
    save_draft(s, dirs["drafts_dir"])
    backend.down = True
    finalize_call(s, backend, **dirs)                    # writes pending
    save_draft(CallSession.from_dict({**s.to_dict(), "ended": False}), dirs["drafts_dir"])
    backend.down = False

    assert recover_interrupted_calls(backend, **dirs) == []      # draft just removed
    retry_pending_calls(backend, dirs["pending_dir"])
    _, stored = backend.calls[(1, s.client_call_id)]
    assert stored["status"] == "completed"
    assert len(backend.calls) == 1


def test_recovery_while_backend_down_keeps_call(dirs):
    backend, s = FakeBackend(), day1_session()
    save_draft(s, dirs["drafts_dir"])
    backend.down = True
    results = recover_interrupted_calls(backend, **dirs)
    assert results[0].saved is False
    assert files(dirs["drafts_dir"]) == []
    assert files(dirs["pending_dir"]) == [f"{s.client_call_id}.json"]

    backend.down = False
    retry_pending_calls(backend, dirs["pending_dir"])
    assert backend.calls[(1, s.client_call_id)][1]["status"] == "interrupted"


def test_recovery_twice_never_duplicates(dirs):
    backend, s = FakeBackend(), day1_session()
    save_draft(s, dirs["drafts_dir"])
    recover_interrupted_calls(backend, **dirs)
    recover_interrupted_calls(backend, **dirs)
    retry_pending_calls(backend, dirs["pending_dir"])
    assert len(backend.calls) == 1


def test_unreadable_draft_moved_aside(dirs):
    dirs["drafts_dir"].mkdir(parents=True)
    (dirs["drafts_dir"] / "bad.json").write_text("{oops")
    assert recover_interrupted_calls(FakeBackend(), **dirs) == []
    assert (dirs["drafts_dir"] / FAILED_SUBDIR / "bad.json").exists()


def test_no_drafts_folder(dirs):
    assert recover_interrupted_calls(FakeBackend(), **dirs) == []