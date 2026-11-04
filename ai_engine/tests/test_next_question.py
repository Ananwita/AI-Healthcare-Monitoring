"""Chunk B5: slot-based adaptive questions over a whole call (fake extractor)."""

from src.conversation.adaptive_questions import (
    MAX_QUESTIONS_PER_CALL,
    SAFETY_LINE,
    closing_message,
    generate_question,
    next_question,
    opening_message,
)
from src.conversation.session import CallSession
from src.llm.extraction import ExtractionResult


def fake(symptoms):
    """Extractor that returns the given symptoms once."""
    return lambda text, context: ExtractionResult(symptoms=symptoms)


def ask(session, planned):
    session.add_assistant_message(planned.text, planned.symptom, planned.field)


def say(session, text, symptoms=()):
    session.process_patient_message(text, fake(list(symptoms)))


def slot(planned):
    return None if planned is None else (planned.symptom, planned.field)


DAY2_HISTORY = [
    {"name": "fever", "status": "ongoing", "latest_duration": "since yesterday"},
    {"name": "weakness", "status": "ongoing"},
]


# ── Opening ──────────────────────────────────────────────────────────────────

def test_opening_without_history():
    opening = opening_message(CallSession(patient_id=1))
    assert "How are you feeling today?" in opening.text
    assert (opening.symptom, opening.field) == (None, None)


def test_opening_mentions_ongoing_history_only():
    history = DAY2_HISTORY + [{"name": "cough", "status": "resolved"}]
    opening = opening_message(CallSession(patient_id=1, previous_symptoms=history))
    assert "fever and weakness" in opening.text
    assert "cough" not in opening.text
    assert (opening.symptom, opening.field) == (None, None)


def test_opening_single_symptom_wording():
    session = CallSession(patient_id=1, previous_symptoms=[{"name": "breathing_difficulty"}])
    assert "difficulty breathing" in opening_message(session).text
    assert "how is it now" in opening_message(session).text


# ── Day 1 full flow ──────────────────────────────────────────────────────────

def test_day1_flow_asks_each_missing_detail_once():
    session = CallSession(patient_id=1)
    ask(session, opening_message(session))
    say(session, "I have a fever and feel weak.", [{"name": "fever"}, {"name": "weakness"}])

    asked = []
    answers = {
        ("fever", "duration"): [{"name": "fever", "duration": "yesterday"}],
        ("fever", "severity"): [],                                  # "not sure" -> no info
        ("fever", "details"): [{"name": "fever", "details": "101 F"}],
        ("weakness", "duration"): [{"name": "weakness", "duration": "2 days"}],
        ("weakness", "severity"): [{"name": "weakness", "severity": "mild"}],
        ("fever", "medication_adherence"): [{"name": "fever", "medication_adherence": "taking"}],
    }
    while (planned := next_question(session)) is not None:
        asked.append(slot(planned))
        ask(session, planned)
        say(session, "answer", answers[slot(planned)])

    assert asked == [
        ("fever", "duration"), ("fever", "severity"), ("fever", "details"),
        ("weakness", "duration"), ("weakness", "severity"),
        ("fever", "medication_adherence"),
    ]
    assert len(asked) == len(set(asked))              # nothing repeated


def test_information_given_upfront_is_not_asked():
    session = CallSession(patient_id=1)
    say(session, "Moderate fever since yesterday, 101.",
        [{"name": "fever", "duration": "since yesterday", "severity": "moderate", "details": "101 F"}])
    assert slot(next_question(session)) == ("fever", "medication_adherence")


def test_unanswered_question_not_repeated():
    session = CallSession(patient_id=1)
    say(session, "Fever.", [{"name": "fever"}])
    first = next_question(session)
    ask(session, first)
    say(session, "I don't know.")                     # nothing extracted
    assert slot(next_question(session)) != slot(first)


def test_symptom_without_detail_question_skips_details():
    session = CallSession(patient_id=1)
    say(session, "x", [{"name": "dizziness", "duration": "today", "severity": "mild"}])
    assert slot(next_question(session)) == ("dizziness", "medication_adherence")


# ── Day 2 follow-up ──────────────────────────────────────────────────────────

def test_day2_resolved_fever_and_ongoing_weakness():
    session = CallSession(patient_id=1, previous_symptoms=DAY2_HISTORY)
    ask(session, opening_message(session))
    say(session, "My fever is gone, but I still feel weak.",
        [{"name": "fever", "reported_change": "gone"}, {"name": "weakness"}])

    first = next_question(session)
    assert slot(first) == ("weakness", "reported_change")       # nothing about fever
    assert "better, worse, or about the same" in first.text
    ask(session, first)
    say(session, "About the same.", [{"name": "weakness", "reported_change": "same"}])

    assert slot(next_question(session)) == ("weakness", "medication_adherence")
    ask(session, next_question(session))
    say(session, "Yes.", [{"name": "weakness", "medication_adherence": "taking"}])
    assert next_question(session) is None


def test_known_history_duration_not_asked_for_tracked_symptom():
    session = CallSession(patient_id=1, previous_symptoms=DAY2_HISTORY)
    say(session, "Fever is worse.", [{"name": "fever", "reported_change": "worse"}])
    planned = next_question(session)
    assert slot(planned) == ("fever", "severity")
    assert "worse" in planned.text


def test_unmentioned_previous_symptom_followed_up_not_resolved():
    session = CallSession(patient_id=1, previous_symptoms=DAY2_HISTORY)
    say(session, "The fever is better.", [{"name": "fever", "reported_change": "better"}])
    planned = next_question(session)
    assert slot(planned) == ("weakness", "reported_change")
    assert "Last time, you mentioned weakness" in planned.text
    # weakness is still not marked resolved anywhere
    assert "weakness" not in session.symptoms


def test_follow_up_answer_does_not_trigger_second_change_question():
    session = CallSession(patient_id=1, previous_symptoms=DAY2_HISTORY)
    say(session, "Fever better.", [{"name": "fever", "reported_change": "better"}])
    ask(session, next_question(session))              # asks about weakness
    say(session, "Still there.", [{"name": "weakness"}])   # no change word
    assert slot(next_question(session)) == ("fever", "medication_adherence")


def test_recurred_symptom_asks_when_it_came_back():
    session = CallSession(patient_id=1, previous_symptoms=[{"name": "cough", "status": "resolved"}])
    say(session, "My cough is back.", [{"name": "cough"}])
    planned = next_question(session)
    assert slot(planned) == ("cough", "duration")
    assert "come back" in planned.text


def test_only_resolved_symptoms_means_no_questions():
    session = CallSession(patient_id=1, previous_symptoms=[{"name": "fever", "status": "ongoing"}])
    say(session, "Fever is gone.", [{"name": "fever", "reported_change": "gone"}])
    assert next_question(session) is None


def test_no_symptoms_no_history_no_questions():
    session = CallSession(patient_id=1)
    say(session, "I'm fine.")
    assert next_question(session) is None


# ── Limits and compatibility ─────────────────────────────────────────────────

def test_question_cap_per_call():
    session = CallSession(patient_id=1)
    say(session, "Lots.", [{"name": n} for n in
                           ("fever", "cough", "headache", "pain", "dizziness", "weakness")])
    count = 0
    while (planned := next_question(session)) is not None:
        ask(session, planned)
        say(session, "skip")
        count += 1
    assert count == MAX_QUESTIONS_PER_CALL


def test_planned_question_accepted_by_session():
    session = CallSession(patient_id=1)
    say(session, "x", [{"name": "fever"}])
    planned = next_question(session)
    ask(session, planned)                             # must not raise
    assert session.was_asked(planned.symptom, planned.field)


def test_old_generate_question_unchanged():
    assert generate_question("fever", "new_symptom").question.startswith("Have you measured")


# ── Closing ──────────────────────────────────────────────────────────────────

def test_closing_summarises_trends_without_diagnosis():
    session = CallSession(patient_id=1, previous_symptoms=DAY2_HISTORY + [{"name": "cough", "status": "resolved"}])
    say(session, "x", [{"name": "fever", "reported_change": "gone"},
                       {"name": "weakness", "reported_change": "same"},
                       {"name": "cough"}, {"name": "headache"}])
    text = closing_message(session)
    assert "fever has gone" in text
    assert "weakness is about the same" in text
    assert "cough has come back" in text
    assert "headache (new)" in text
    assert text.endswith(SAFETY_LINE)


def test_closing_without_symptoms():
    text = closing_message(CallSession(patient_id=1))
    assert text.startswith("Thank you for talking with me today.")
    assert SAFETY_LINE in text