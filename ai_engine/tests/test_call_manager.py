"""Chunk B8: CallManager end-to-end over multiple calls (fake backend + fake LLM)."""

import json

import pytest

from src.backend.client import BackendUnavailableError, CallConflictError, SaveResult
from src.conversation.call_manager import NOT_HEARD_MESSAGE, CallManager
from src.llm.extraction import ExtractionResult


# ── Fakes ────────────────────────────────────────────────────────────────────

class FakeBackend:
    """Patients + calls in memory, A3/A4 idempotency, A5-style symptom state."""

    def __init__(self):
        self.patients = {}       # external_id -> patient_id
        self.calls = {}          # (patient_id, client_call_id) -> payload
        self.order = []          # call keys in save order
        self.down = False

    def _check(self):
        if self.down:
            raise BackendUnavailableError("backend down")

    def create_or_get_patient(self, external_id, name, **kwargs):
        self._check()
        if external_id in self.patients:
            return {"patient_id": self.patients[external_id], "created": False}
        self.patients[external_id] = len(self.patients) + 1
        return {"patient_id": self.patients[external_id], "created": True}

    def get_symptom_state(self, patient_id):
        self._check()
        state = {}
        for key in self.order:
            if key[0] != patient_id:
                continue
            for s in self.calls[key]["symptoms"]:
                entry = state.setdefault(s["name"], {"name": s["name"]})
                entry["status"] = s["status"]
                entry["latest_trend"] = s["trend"]
                for field in ("severity", "duration"):
                    if s[field]:
                        entry[f"latest_{field}"] = s[field]
        return list(state.values())

    def save_call(self, patient_id, payload):
        self._check()
        key = (patient_id, payload["client_call_id"])
        if key in self.calls:
            if self.calls[key] != payload:
                raise CallConflictError("conflict", status_code=409)
            return SaveResult(1, patient_id, key[1], payload["status"], 0, True)
        self.calls[key] = json.loads(json.dumps(payload))
        self.order.append(key)
        return SaveResult(len(self.order), patient_id, key[1], payload["status"],
                          len(payload["symptoms"]), False)

    def state(self, patient_id=1):
        return {s["name"]: s for s in self.get_symptom_state(patient_id)}


class ScriptedLLM:
    """Maps patient text -> extracted symptoms. Unknown text -> nothing found."""

    def __init__(self, script, fail_on=()):
        self.script = script
        self.fail_on = set(fail_on)

    def __call__(self, text, context):
        if text in self.fail_on:
            return None
        return ExtractionResult(symptoms=self.script.get(text, []))


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
}


@pytest.fixture
def setup(tmp_path):
    backend = FakeBackend()

    def make_manager(llm=None):
        return CallManager(
            client=backend, extractor=llm or ScriptedLLM(SCRIPT),
            pending_dir=tmp_path / "pending", drafts_dir=tmp_path / "drafts",
        )
    return backend, make_manager, tmp_path


def run_call(manager, answers, external_id="P001", name="Asha"):
    session, greeting = manager.start_call(external_id, name)
    questions = [greeting]
    for answer in answers:
        reply = manager.handle_patient_message(session, answer)
        questions.append(reply.text)
        if reply.done:
            break
    return session, questions, manager.end_call(session)


def json_files(path):
    return sorted(p.name for p in path.glob("*.json")) if path.exists() else []


# ── Day 1 -> Day 2 -> Day 3 across separate calls ────────────────────────────

def test_full_patient_journey(setup):
    backend, make_manager, _ = setup

    # Day 1
    _, q1, r1 = run_call(make_manager(), [
        "I have a fever and feel weak.", "Yesterday.", "Moderate.", "Yes, 101 F.",
        "Since yesterday too.", "Mild.", "Yes, I take it.",
    ])
    assert q1[0].startswith("Hello, I'm your health monitoring assistant")
    assert q1[1] == "When did your fever start?"
    assert q1[2] == "How bad is your fever: mild, moderate, or severe?"
    assert "temperature" in q1[3]
    assert q1[4] == "When did your weakness start?"
    assert "medicine" in q1[6]
    assert q1[-1].startswith("Thank you for the update.")
    assert r1.saved is True
    state = backend.state()
    assert (state["fever"]["status"], state["fever"]["latest_trend"]) == ("ongoing", "new")

    # Day 2: same patient, history used, fever resolved, weakness ongoing
    _, q2, r2 = run_call(make_manager(), [
        "My fever is gone, but I still feel weak.", "About the same.", "Yes.",
    ])
    assert "Last time, you mentioned fever and weakness" in q2[0]
    assert "weakness better, worse, or about the same" in q2[1]
    assert "fever has gone" in q2[-1]
    assert r2.saved is True
    state = backend.state()
    assert state["fever"]["status"] == "resolved"
    assert (state["weakness"]["status"], state["weakness"]["latest_trend"]) == ("ongoing", "stable")
    assert state["fever"]["latest_duration"] == "yesterday"     # history preserved

    # Day 3: fever comes back -> recurrence
    session, q3, r3 = run_call(make_manager(), ["My fever is back."])
    assert "fever" not in q3[0]                                  # only ongoing mentioned
    assert q3[1] == "When did your fever come back?"
    assert r3.saved is True
    assert backend.state()["fever"]["status"] == "ongoing"
    saved = backend.calls[(1, session.client_call_id)]
    assert "Fever: returned after being resolved" in saved["summary"]

    assert len(backend.patients) == 1 and len(backend.calls) == 3


# ── Turn handling ────────────────────────────────────────────────────────────

def test_empty_answer_not_recorded_and_question_repeated(setup):
    _, make_manager, _ = setup
    manager = make_manager()
    session, _ = manager.start_call("P001", "Asha")
    manager.handle_patient_message(session, "I have a fever and feel weak.")
    reply = manager.handle_patient_message(session, "   ")
    assert reply.text == f"{NOT_HEARD_MESSAGE} When did your fever start?"
    assert reply.done is False
    assert session.last_question_symptom == "fever"         # still linked
    assert len(session.turns) == 3


def test_extraction_failure_is_flagged_and_call_continues(setup):
    _, make_manager, _ = setup
    manager = make_manager(ScriptedLLM(SCRIPT, fail_on={"Yesterday."}))
    session, _ = manager.start_call("P001", "Asha")
    manager.handle_patient_message(session, "I have a fever and feel weak.")
    reply = manager.handle_patient_message(session, "Yesterday.")
    assert reply.extraction_failed is True
    assert reply.text == "How bad is your fever: mild, moderate, or severe?"   # moves on, no repeat


def test_draft_saved_after_every_turn(setup):
    _, make_manager, tmp = setup
    manager = make_manager()
    session, _ = manager.start_call("P001", "Asha")
    manager.handle_patient_message(session, "I have a fever and feel weak.")
    draft = json.loads((tmp / "drafts" / f"{session.client_call_id}.json").read_text())
    assert len(draft["turns"]) == 3


# ── Ending ───────────────────────────────────────────────────────────────────

def test_call_without_patient_speech_not_saved(setup):
    backend, make_manager, tmp = setup
    manager = make_manager()
    session, _ = manager.start_call("P001", "Asha")
    result = manager.end_call(session)
    assert result.saved is False
    assert backend.calls == {}
    assert json_files(tmp / "drafts") == [] and json_files(tmp / "pending") == []


def test_interrupted_end(setup):
    backend, make_manager, _ = setup
    manager = make_manager()
    session, _ = manager.start_call("P001", "Asha")
    manager.handle_patient_message(session, "I have a fever and feel weak.")
    result = manager.end_call(session, interrupted=True)
    assert result.saved is True
    assert backend.calls[(1, session.client_call_id)]["status"] == "interrupted"


def test_backend_down_at_end_then_saved_on_next_startup(setup):
    backend, make_manager, tmp = setup
    manager = make_manager()
    session, _ = manager.start_call("P001", "Asha")
    manager.handle_patient_message(session, "I have a fever and feel weak.")
    backend.down = True
    result = manager.end_call(session)
    assert result.saved is False and "kept" in result.message
    assert json_files(tmp / "pending") == [f"{session.client_call_id}.json"]

    backend.down = False
    results = make_manager().startup()
    assert [r.saved for r in results] == [True]
    assert len(backend.calls) == 1 and json_files(tmp / "pending") == []


# ── Startup recovery ─────────────────────────────────────────────────────────

def test_startup_recovers_crashed_call_as_interrupted(setup):
    backend, make_manager, tmp = setup
    manager = make_manager()
    session, _ = manager.start_call("P001", "Asha")
    manager.handle_patient_message(session, "I have a fever and feel weak.")
    # app crashes here: end_call never runs

    results = make_manager().startup()
    assert [r.saved for r in results] == [True]
    assert backend.calls[(1, session.client_call_id)]["status"] == "interrupted"
    assert json_files(tmp / "drafts") == []

    assert make_manager().startup() == []          # nothing left, no duplicates
    assert len(backend.calls) == 1


def test_startup_with_nothing_to_do(setup):
    _, make_manager, _ = setup
    assert make_manager().startup() == []


# ── Start failures ───────────────────────────────────────────────────────────

def test_backend_down_at_start_raises_and_creates_nothing(setup):
    backend, make_manager, tmp = setup
    backend.down = True
    with pytest.raises(BackendUnavailableError):
        make_manager().start_call("P001", "Asha")
    assert json_files(tmp / "drafts") == []


def test_same_patient_not_duplicated(setup):
    backend, make_manager, _ = setup
    manager = make_manager()
    a, _ = manager.start_call("P001", "Asha")
    b, _ = manager.start_call("P001", "Asha")
    assert a.patient_id == b.patient_id
    assert a.client_call_id != b.client_call_id
    assert len(backend.patients) == 1