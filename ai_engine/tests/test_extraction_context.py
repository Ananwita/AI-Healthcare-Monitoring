"""Chunk B2: context-aware extraction (fake LLM only, no real API calls)."""

import json
from types import SimpleNamespace

import pytest

import src.llm.extraction as extraction
from src.llm.extraction import (
    MAX_CONTEXT_TURNS,
    ConversationTurn,
    ExtractionContext,
    PreviousSymptom,
    build_user_prompt,
    extract_symptoms,
)


# ── Fake Groq client ─────────────────────────────────────────────────────────

class FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]
        )


@pytest.fixture
def fake_llm(monkeypatch):
    def install(*replies):
        completions = FakeCompletions(
            [r if isinstance(r, (str, Exception)) else json.dumps(r) for r in replies]
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        monkeypatch.setattr(extraction, "get_client", lambda: client)
        return completions
    return install


def reply(*symptoms):
    return {"symptoms": list(symptoms)}


def prompt_of(completions, index=0):
    return completions.calls[index]["messages"][1]["content"]


# Day 1, same call: "I have a fever and feel weak." -> "When did your fever start?"
DAY1_CONTEXT = ExtractionContext(
    conversation=[
        ConversationTurn(role="assistant", text="How are you feeling today?"),
        ConversationTurn(role="patient", text="I have a fever and feel weak."),
        ConversationTurn(role="assistant", text="When did your fever start?"),
    ],
    last_question="When did your fever start?",
    last_question_symptom="fever",
)

# Day 2: previous calls recorded fever and weakness
DAY2_CONTEXT = ExtractionContext(
    previous_symptoms=[
        {"name": "fever", "status": "ongoing", "latest_severity": "moderate",
         "latest_duration": "since yesterday", "times_reported": 1},
        {"name": "weakness", "status": "ongoing"},
    ],
    last_question="Yesterday you mentioned a fever and weakness. How are you feeling today?",
)


# ── 1. No context: unchanged behaviour ───────────────────────────────────────

def test_no_context_prompt_has_no_context_section():
    prompt = build_user_prompt("I have a cough.")
    assert "CONTEXT" not in prompt
    assert '{"symptoms": []}\n\nPatient statement: "I have a cough."\n' in prompt


def test_none_and_empty_context_give_identical_prompt():
    text = "I have a cough."
    assert build_user_prompt(text) == build_user_prompt(text, None)
    assert build_user_prompt(text) == build_user_prompt(text, ExtractionContext())


def test_no_context_keeps_b1_behaviour(fake_llm):
    # Without context no grounding guard is applied (exactly as in B1)
    fake_llm(reply({"name": "cough", "evidence": "paraphrased words"}))
    result = extract_symptoms("I have a cough.")
    assert [s.name for s in result.symptoms] == ["cough"]
    assert result.symptoms[0].evidence == "paraphrased words"
    assert result.ungrounded_symptoms == []


def test_extract_symptoms_positional_call_still_works(fake_llm):
    fake_llm(reply({"name": "fever"}))
    assert extract_symptoms("fever").symptoms[0].name == "fever"


# ── 2. Context appears in the prompt ─────────────────────────────────────────

def test_context_prompt_contains_all_parts(fake_llm):
    completions = fake_llm(reply())
    extract_symptoms("Yesterday.", DAY1_CONTEXT)
    prompt = prompt_of(completions)

    assert "CONTEXT" in prompt
    assert "Patient: I have a fever and feel weak." in prompt
    assert 'Last question asked: "When did your fever start?" (about: fever)' in prompt
    assert "Previously recorded symptoms for this patient:\n- none" in prompt
    assert prompt.rstrip().endswith('Patient statement: "Yesterday."')


def test_previous_symptoms_in_prompt():
    prompt = build_user_prompt("Better today.", DAY2_CONTEXT)
    assert "- fever: status ongoing, last severity moderate, last duration since yesterday" in prompt
    assert "- weakness: status ongoing" in prompt
    assert "Last question asked: \"Yesterday you mentioned" in prompt
    assert "(about: no specific symptom)" in prompt


def test_context_rules_in_prompt():
    prompt = build_user_prompt("Yes.", DAY1_CONTEXT)
    assert "Extract ONLY information stated in the NEW patient statement" in prompt
    assert "do not guess" in prompt
    assert "A symptom that is not mentioned is NOT gone" in prompt
    assert "If unclear, ignore it" in prompt
    assert "Never copy severity, duration or details from the context" in prompt


def test_only_recent_turns_included():
    turns = [ConversationTurn(role="patient", text=f"turn {i}") for i in range(12)]
    prompt = build_user_prompt("ok", ExtractionContext(conversation=turns))
    assert "turn 3" not in prompt
    assert f"turn {12 - MAX_CONTEXT_TURNS}" in prompt
    assert "turn 11" in prompt


def test_backend_symptom_state_accepted():
    # Extra keys from /symptom-state are ignored
    p = PreviousSymptom(**{"name": "fever", "status": "resolved", "recurred": False,
                           "latest_trend": "resolved", "last_call_id": 3})
    assert p.status == "resolved"


def test_unsupported_question_symptom_becomes_none():
    ctx = ExtractionContext(last_question="Where?", last_question_symptom="stomach ache")
    assert ctx.last_question_symptom is None
    assert ExtractionContext(last_question_symptom="Fever").last_question_symptom == "fever"


# ── 3. Short answers linked to the last question ─────────────────────────────

def test_short_answer_applied_to_question_symptom(fake_llm):
    fake_llm(reply({"name": "fever", "duration": "since yesterday", "evidence": "Yesterday"}))
    result = extract_symptoms("Yesterday.", DAY1_CONTEXT)
    assert len(result.symptoms) == 1
    fever = result.symptoms[0]
    assert fever.name == "fever"
    assert fever.duration == "since yesterday"
    assert fever.evidence == "Yesterday"


def test_short_answer_kept_for_question_symptom_even_without_evidence(fake_llm):
    fake_llm(reply({"name": "fever", "details": "temperature 101°F"}))
    result = extract_symptoms("Yes, 101.", DAY1_CONTEXT)
    assert result.symptoms[0].name == "fever"
    assert result.symptoms[0].details == "temperature 101°F"
    assert result.symptoms[0].evidence is None


def test_short_answer_not_attached_to_other_symptom(fake_llm):
    # LLM wrongly attaches "Yesterday" to weakness with words not in the answer
    fake_llm(reply({"name": "weakness", "duration": "since yesterday",
                    "evidence": "feel weak since yesterday"}))
    result = extract_symptoms("Yesterday.", DAY1_CONTEXT)
    assert result.symptoms == []
    assert result.ungrounded_symptoms == ["weakness"]


def test_short_answer_without_question_symptom_not_guessed(fake_llm):
    ctx = ExtractionContext(
        previous_symptoms=[{"name": "fever"}, {"name": "cough"}],
        last_question="How long has this been going on?",
    )
    fake_llm(reply({"name": "cough", "duration": "two days", "evidence": "the cough for two days"}))
    result = extract_symptoms("Two days.", ctx)
    assert result.symptoms == []


# ── 4. No duplication of earlier information ─────────────────────────────────

def test_symptoms_from_earlier_turns_not_repeated(fake_llm):
    fake_llm(reply(
        {"name": "fever", "duration": "since yesterday", "evidence": "Yesterday"},
        {"name": "weakness", "evidence": "I feel weak"},     # from earlier turn
    ))
    result = extract_symptoms("Yesterday.", DAY1_CONTEXT)
    assert [s.name for s in result.symptoms] == ["fever"]
    assert result.ungrounded_symptoms == ["weakness"]


def test_duration_copied_from_previous_call_is_cleared(fake_llm):
    fake_llm(reply({"name": "fever", "duration": "since yesterday",
                    "reported_change": "same", "evidence": "fever is still there"}))
    result = extract_symptoms("My fever is still there.", DAY2_CONTEXT)
    fever = result.symptoms[0]
    assert fever.duration is None
    assert fever.reported_change == "same"


def test_new_duration_in_statement_kept(fake_llm):
    fake_llm(reply({"name": "fever", "duration": "three days", "evidence": "fever for three days"}))
    result = extract_symptoms("I have had the fever for three days now.", DAY2_CONTEXT)
    assert result.symptoms[0].duration == "three days"


def test_paraphrased_evidence_mostly_matching_is_kept(fake_llm):
    fake_llm(reply({"name": "weakness", "evidence": "I still feel weak"}))
    result = extract_symptoms("But I still feel quite weak today.", DAY2_CONTEXT)
    assert [s.name for s in result.symptoms] == ["weakness"]


# ── 5. Resolution only when explicit ─────────────────────────────────────────

def test_explicit_resolution_kept(fake_llm):
    fake_llm(reply(
        {"name": "fever", "reported_change": "gone", "evidence": "My fever is gone"},
        {"name": "weakness", "reported_change": "same", "evidence": "I still feel weak"},
    ))
    result = extract_symptoms("My fever is gone, but I still feel weak.", DAY2_CONTEXT)
    changes = {s.name: s.reported_change for s in result.symptoms}
    assert changes == {"fever": "gone", "weakness": "same"}


def test_unmentioned_symptom_not_marked_gone(fake_llm):
    # LLM wrongly claims weakness is gone although the patient only spoke of fever
    fake_llm(reply(
        {"name": "fever", "reported_change": "better", "evidence": "fever is better"},
        {"name": "weakness", "reported_change": "gone", "evidence": "no longer weak"},
    ))
    result = extract_symptoms("The fever is better today.", DAY2_CONTEXT)
    assert [s.name for s in result.symptoms] == ["fever"]
    assert result.ungrounded_symptoms == ["weakness"]


def test_gone_without_support_in_answer_is_cleared(fake_llm):
    # Question was about fever; "Yes." does not say it is gone
    ctx = ExtractionContext(last_question="Do you still have the fever?",
                            last_question_symptom="fever")
    fake_llm(reply({"name": "fever", "reported_change": "gone", "evidence": "the fever went away"}))
    result = extract_symptoms("Yes.", ctx)
    assert result.symptoms[0].name == "fever"
    assert result.symptoms[0].reported_change is None


def test_gone_in_short_answer_to_question_is_kept(fake_llm):
    ctx = ExtractionContext(last_question="Has your fever improved?",
                            last_question_symptom="fever")
    fake_llm(reply({"name": "fever", "reported_change": "gone", "evidence": "it's completely gone"}))
    result = extract_symptoms("Yes, it's completely gone.", ctx)
    assert result.symptoms[0].reported_change == "gone"


# ── 6. Ambiguous / insufficient context ──────────────────────────────────────

def test_ambiguous_reply_with_no_symptoms_returns_empty(fake_llm):
    fake_llm(reply())
    result = extract_symptoms("Same as before, I guess.", DAY2_CONTEXT)
    assert result.symptoms == []


def test_invented_details_without_evidence_removed(fake_llm):
    # No question about cough, no cough in the statement -> invented
    fake_llm(reply({"name": "cough", "severity": "mild", "details": "dry cough"}))
    result = extract_symptoms("I'm okay.", DAY2_CONTEXT)
    assert result.symptoms == []
    assert result.ungrounded_symptoms == ["cough"]


def test_context_guards_do_not_change_input_objects(fake_llm):
    fake_llm(reply({"name": "fever", "duration": "since yesterday", "evidence": "fever still"}))
    before = DAY2_CONTEXT.model_dump()
    extract_symptoms("fever still", DAY2_CONTEXT)
    assert DAY2_CONTEXT.model_dump() == before


# ── 7. Retries still work with context ───────────────────────────────────────

def test_retry_with_context(fake_llm):
    completions = fake_llm("not json", reply({"name": "fever", "evidence": "Yesterday"}))
    result = extract_symptoms("Yesterday.", DAY1_CONTEXT)
    assert result.symptoms[0].name == "fever"
    assert len(completions.calls) == 2
    assert prompt_of(completions, 0) == prompt_of(completions, 1)



# ── 8. Question field (which detail was asked) ───────────────────────────────

def test_question_field_in_prompt():
    ctx = DAY1_CONTEXT.model_copy(update={"last_question_field": "duration"})
    prompt = build_user_prompt("Yesterday.", ctx)
    assert "(about: fever, asking for: duration)" in prompt
    assert "The last question asked about the duration of fever" in prompt
    assert "set duration from the answer" in prompt
    assert "Example (different patient)" in prompt


def test_question_without_field_still_linked_to_symptom():
    prompt = build_user_prompt("Yesterday.", DAY1_CONTEXT)
    assert "The last question asked about fever." in prompt
    assert "asking for:" not in prompt


def test_invalid_question_field_rejected():
    with pytest.raises(Exception):
        ExtractionContext(last_question_field="temperature")


def test_no_example_when_question_not_about_symptom():
    prompt = build_user_prompt("Two days.", DAY2_CONTEXT)
    assert "Example (different patient)" not in prompt
    assert "not about a specific symptom" in prompt