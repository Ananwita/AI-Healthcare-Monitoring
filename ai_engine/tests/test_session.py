"""Chunk B4: CallSession memory, merging, context and persistence (fake extractor)."""

import json

import pytest

from src.conversation.session import CallSession
from src.llm.extraction import ExtractionResult


class FakeExtractor:
    """Returns queued results; records (text, context) of every call."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, text, context):
        self.calls.append((text, context))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        if result is None:
            return None
        return ExtractionResult(symptoms=result)


DAY2_HISTORY = [
    {"name": "fever", "status": "ongoing", "latest_severity": "moderate",
     "latest_duration": "since yesterday", "times_reported": 1},
    {"name": "weakness", "status": "ongoing"},
    {"name": "cough", "status": "resolved"},
]


def day1_session():
    return CallSession(patient_id=1, external_id="P001")


# ── Creation ─────────────────────────────────────────────────────────────────

def test_new_session_has_unique_client_call_id():
    a, b = day1_session(), day1_session()
    assert a.client_call_id != b.client_call_id
    assert len(a.client_call_id) == 36
    assert a.started_at
    assert a.turns == [] and a.symptoms == {}


def test_previous_symptoms_accept_backend_dicts():
    session = CallSession(patient_id=1, previous_symptoms=DAY2_HISTORY)
    assert session.previous_state("Fever").latest_duration == "since yesterday"
    assert session.previous_state("headache") is None


# ── Day 1 conversation continuity ────────────────────────────────────────────

def test_day1_flow_short_answer_uses_context_and_merges():
    session = day1_session()
    fake = FakeExtractor(
        [{"name": "fever", "evidence": "I have a fever"},
         {"name": "weakness", "evidence": "feel weak"}],
        [{"name": "fever", "duration": "yesterday", "evidence": "Yesterday"}],
        [{"name": "fever", "details": "temperature 101 F", "evidence": "101 F"}],
    )

    session.add_assistant_message("How are you feeling today?")
    session.process_patient_message("I have a fever and feel weak.", fake)
    session.add_assistant_message("When did your fever start?", symptom="fever", field="duration")
    session.process_patient_message("Yesterday.", fake)
    session.add_assistant_message("Have you measured your temperature?", symptom="fever", field="details")
    session.process_patient_message("Yes, 101 F.", fake)

    # Context for "Yesterday." links it to the fever duration question
    text, ctx = fake.calls[1]
    assert text == "Yesterday."
    assert ctx.last_question_symptom == "fever"
    assert ctx.last_question_field == "duration"
    assert [t.text for t in ctx.conversation] == [
        "How are you feeling today?", "I have a fever and feel weak.", "When did your fever start?",
    ]

    # One record per symptom, all answers combined
    assert list(session.symptoms) == ["fever", "weakness"]
    fever = session.symptoms["fever"]
    assert fever.duration == "yesterday"
    assert fever.details == "temperature 101 F"
    assert fever.evidence == ["I have a fever", "Yesterday", "101 F"]
    assert fever.first_turn == 1 and fever.last_turn == 5
    assert session.symptoms["weakness"].duration is None   # not copied from fever


def test_last_question_cleared_after_patient_reply():
    session = day1_session()
    fake = FakeExtractor([], [])
    session.add_assistant_message("When did your fever start?", symptom="fever", field="duration")
    session.process_patient_message("Yesterday.", fake)
    session.process_patient_message("Also my head hurts.", fake)
    second_ctx = fake.calls[1][1]
    assert second_ctx.last_question is None
    assert second_ctx.last_question_symptom is None


# ── Merge rules ──────────────────────────────────────────────────────────────

def test_empty_value_never_erases_known_value():
    session = day1_session()
    fake = FakeExtractor(
        [{"name": "fever", "severity": "mild", "duration": "2 days"}],
        [{"name": "fever"}],
    )
    session.process_patient_message("Mild fever for 2 days.", fake)
    session.process_patient_message("The fever, yes.", fake)
    assert session.symptoms["fever"].severity == "mild"
    assert session.symptoms["fever"].duration == "2 days"


def test_newer_value_replaces_older():
    session = day1_session()
    fake = FakeExtractor(
        [{"name": "fever", "severity": "mild"}],
        [{"name": "fever", "severity": "severe"}],
    )
    session.process_patient_message("a", fake)
    session.process_patient_message("b", fake)
    assert session.symptoms["fever"].severity == "severe"


def test_reported_change_keeps_latest_explicit_value():
    session = day1_session()
    fake = FakeExtractor(
        [{"name": "fever", "reported_change": "better"}],
        [{"name": "fever"}],
        [{"name": "fever", "reported_change": "gone"}],
    )
    for text in ("a", "b", "c"):
        session.process_patient_message(text, fake)
    assert session.symptoms["fever"].reported_change == "gone"


def test_details_combined_without_duplicates():
    session = day1_session()
    fake = FakeExtractor(
        [{"name": "fever", "details": "temperature 101 F"}],
        [{"name": "fever", "details": "Temperature 101 F"}],
        [{"name": "fever", "details": "with chills"}],
    )
    for text in ("a", "b", "c"):
        session.process_patient_message(text, fake)
    assert session.symptoms["fever"].details == "temperature 101 F; with chills"


def test_same_symptom_twice_in_one_reply_stored_once():
    session = day1_session()
    fake = FakeExtractor([{"name": "fever", "evidence": "x"}, {"name": "Fever", "severity": "mild"}])
    session.process_patient_message("fever ... fever", fake)
    assert list(session.symptoms) == ["fever"]
    assert session.symptoms["fever"].severity == "mild"


# ── Robustness ───────────────────────────────────────────────────────────────

def test_extraction_failure_keeps_turn_and_counts():
    session = day1_session()
    fake = FakeExtractor(None)
    result = session.process_patient_message("I have a fever.", fake)
    assert result.extraction_failed is True
    assert session.extraction_failures == 1
    assert session.transcript() == "Patient: I have a fever."
    assert session.symptoms == {}


def test_extractor_exception_does_not_crash_call():
    session = day1_session()
    fake = FakeExtractor(RuntimeError("Groq down"))
    result = session.process_patient_message("I have a fever.", fake)
    assert result.extraction_failed is True
    assert len(session.turns) == 1


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_message_ignored(text):
    session = day1_session()
    fake = FakeExtractor()
    result = session.process_patient_message(text, fake)
    assert result.ignored is True
    assert session.turns == [] and fake.calls == []


def test_unknown_question_field_rejected():
    with pytest.raises(ValueError):
        day1_session().add_assistant_message("?", symptom="fever", field="temperature")


def test_ended_session_rejects_messages():
    session = day1_session()
    session.end()
    with pytest.raises(RuntimeError):
        session.add_assistant_message("Hello")
    with pytest.raises(RuntimeError):
        session.process_patient_message("Hi", FakeExtractor())


# ── Queries for question generation ──────────────────────────────────────────

def test_was_asked():
    session = day1_session()
    session.add_assistant_message("When did it start?", symptom="Fever", field="duration")
    assert session.was_asked("fever", "duration")
    assert not session.was_asked("fever", "severity")
    assert not session.was_asked("weakness", "duration")


def test_known_value_prefers_this_call_then_history():
    session = CallSession(patient_id=1, previous_symptoms=DAY2_HISTORY)
    assert session.known_value("fever", "duration") == "since yesterday"     # history
    assert session.known_value("weakness", "severity") is None
    assert session.known_value("headache", "duration") is None

    session.process_patient_message("x", FakeExtractor([{"name": "fever", "duration": "3 days"}]))
    assert session.known_value("fever", "duration") == "3 days"              # this call


def test_unmentioned_previous_symptoms():
    session = CallSession(patient_id=1, previous_symptoms=DAY2_HISTORY)
    assert session.unmentioned_previous_symptoms() == ["fever", "weakness"]
    session.process_patient_message("x", FakeExtractor([{"name": "fever", "reported_change": "gone"}]))
    assert session.unmentioned_previous_symptoms() == ["weakness"]


# ── Trend/status (B3) ────────────────────────────────────────────────────────

def test_day2_updates_resolution_and_stable():
    session = CallSession(patient_id=1, previous_symptoms=DAY2_HISTORY)
    session.add_assistant_message("Yesterday you mentioned a fever and weakness. How are you today?")
    session.process_patient_message(
        "My fever is gone, but I still feel weak.",
        FakeExtractor([
            {"name": "fever", "reported_change": "gone", "evidence": "My fever is gone"},
            {"name": "weakness", "evidence": "I still feel weak"},
        ]),
    )
    updates = {u.name: (u.trend, u.status) for u in session.symptom_updates()}
    assert updates == {"fever": ("resolved", "resolved"), "weakness": ("stable", "ongoing")}


def test_recurrence_and_new_symptom_updates():
    session = CallSession(patient_id=1, previous_symptoms=DAY2_HISTORY)
    session.process_patient_message("x", FakeExtractor([{"name": "cough"}, {"name": "headache"}]))
    updates = {u.name: u for u in session.symptom_updates()}
    assert (updates["cough"].trend, updates["cough"].recurred) == ("new", True)
    assert (updates["headache"].trend, updates["headache"].recurred) == ("new", False)


def test_unmentioned_symptoms_get_no_update():
    session = CallSession(patient_id=1, previous_symptoms=DAY2_HISTORY)
    session.process_patient_message("x", FakeExtractor([{"name": "fever"}]))
    assert [u.name for u in session.symptom_updates()] == ["fever"]


# ── Transcript and persistence ───────────────────────────────────────────────

def test_transcript_format():
    session = day1_session()
    session.add_assistant_message("How are you?")
    session.process_patient_message("Tired.", FakeExtractor([]))
    assert session.transcript() == "Assistant: How are you?\nPatient: Tired."


def test_to_dict_is_json_and_round_trips():
    session = CallSession(patient_id=1, external_id="P001", previous_symptoms=DAY2_HISTORY)
    session.add_assistant_message("When did it start?", symptom="fever", field="duration")
    session.process_patient_message("Yesterday.", FakeExtractor([{"name": "fever", "duration": "yesterday"}]))

    data = json.loads(json.dumps(session.to_dict()))     # must be JSON-safe
    restored = CallSession.from_dict(data)

    assert restored.to_dict() == session.to_dict()
    assert restored.client_call_id == session.client_call_id
    assert restored.symptoms["fever"].duration == "yesterday"
    assert restored.was_asked("fever", "duration")


def test_restored_session_continues_normally():
    session = day1_session()
    session.process_patient_message("Fever.", FakeExtractor([{"name": "fever"}]))
    restored = CallSession.from_dict(session.to_dict())
    restored.add_assistant_message("When did it start?", symptom="fever", field="duration")
    restored.process_patient_message("Today.", FakeExtractor([{"name": "fever", "duration": "today"}]))
    assert restored.symptoms["fever"].duration == "today"
    assert len(restored.turns) == 3