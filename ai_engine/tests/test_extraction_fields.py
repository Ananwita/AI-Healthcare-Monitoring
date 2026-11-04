"""Chunk B1: optional fields, normalisation, request settings, retries (no real API calls)."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import src.llm.extraction as extraction
from src.llm.extraction import (
    ExtractionResult,
    SymptomObservation,
    build_request_kwargs,
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
    """Usage: completions = fake_llm(['{"symptoms": []}', ...])"""
    def install(replies):
        completions = FakeCompletions(replies)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        monkeypatch.setattr(extraction, "get_client", lambda: client)
        return completions
    return install


# ── Schema: optional fields ──────────────────────────────────────────────────

def test_only_name_required():
    obs = SymptomObservation(name="fever")
    assert obs.severity is None
    assert obs.duration is None
    assert obs.medication_adherence is None
    assert obs.reported_change is None
    assert obs.details is None
    assert obs.evidence is None


@pytest.mark.parametrize("empty", [None, "", "  ", "null", "None", "unknown", "N/A", "not mentioned"])
def test_empty_values_become_none(empty):
    obs = SymptomObservation(
        name="fever", severity=empty, duration=empty,
        medication_adherence=empty, reported_change=empty,
        details=empty, evidence=empty,
    )
    assert obs.model_dump(exclude={"name"}) == {
        "severity": None, "duration": None, "medication_adherence": None,
        "reported_change": None, "details": None, "evidence": None,
    }


def test_weakness_supported():
    assert SymptomObservation(name="Weakness").name == "weakness"


def test_hyphenated_name_normalised():
    assert SymptomObservation(name="Breathing-Difficulty").name == "breathing_difficulty"


# ── Schema: medication adherence ─────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("taking", "taking"), ("Missed", "missed"),
    ("not taking", "not_taking"), ("Not-Taking", "not_taking"),
])
def test_adherence_normalised(raw, expected):
    assert SymptomObservation(name="fever", medication_adherence=raw).medication_adherence == expected


def test_invalid_adherence_rejected():
    with pytest.raises(ValidationError):
        SymptomObservation(name="fever", medication_adherence="sometimes")


# ── Schema: reported change ──────────────────────────────────────────────────

@pytest.mark.parametrize("value", ["new", "worse", "better", "same", "gone"])
def test_valid_reported_changes(value):
    assert SymptomObservation(name="fever", reported_change=value.upper()).reported_change == value


def test_invalid_reported_change_rejected():
    with pytest.raises(ValidationError):
        SymptomObservation(name="fever", reported_change="improved")


# ── Schema: text fields ──────────────────────────────────────────────────────

def test_text_fields_trimmed():
    obs = SymptomObservation(
        name="fever", duration=" since yesterday ",
        details=" temperature 101°F ", evidence=" I have a fever ",
    )
    assert obs.duration == "since yesterday"
    assert obs.details == "temperature 101°F"
    assert obs.evidence == "I have a fever"


# ── ExtractionResult: dropping ───────────────────────────────────────────────

def test_missing_or_bad_names_dropped():
    result = ExtractionResult(symptoms=[{"severity": "mild"}, {"name": "fever"}, "junk"])
    assert [s.name for s in result.symptoms] == ["fever"]
    assert len(result.dropped_symptoms) == 2


def test_empty_symptoms_key_missing():
    assert ExtractionResult().symptoms == []


def test_invalid_severity_on_supported_symptom_still_fails():
    with pytest.raises(ValidationError):
        ExtractionResult(symptoms=[{"name": "fever", "severity": "extreme"}])


# ── Request settings ─────────────────────────────────────────────────────────

def test_request_kwargs_gpt_oss():
    kwargs = build_request_kwargs("hello", model="openai/gpt-oss-20b")
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["temperature"] == 0
    assert kwargs["max_tokens"] >= 2048
    assert "reasoning_effort" in kwargs


def test_request_kwargs_non_reasoning_model():
    kwargs = build_request_kwargs("hello", model="llama-3.1-8b-instant")
    assert "reasoning_effort" not in kwargs


# ── extract_symptoms with fake LLM ───────────────────────────────────────────

FULL_REPLY = """{"symptoms": [
  {"name": "fever", "severity": null, "duration": "since yesterday",
   "medication_adherence": "taking", "reported_change": null,
   "details": "temperature 101°F", "evidence": "fever since yesterday, 101"},
  {"name": "weakness", "severity": "mild", "duration": null,
   "medication_adherence": null, "reported_change": null,
   "details": null, "evidence": "a little weak"}
]}"""


def test_extract_full_fields(fake_llm):
    completions = fake_llm([FULL_REPLY])
    result = extract_symptoms("I have had a fever since yesterday, 101. A little weak.")

    assert [s.name for s in result.symptoms] == ["fever", "weakness"]
    fever = result.symptoms[0]
    assert fever.severity is None                 # not invented
    assert fever.duration == "since yesterday"
    assert fever.details == "temperature 101°F"
    assert fever.medication_adherence == "taking"
    assert "since yesterday" in completions.calls[0]["messages"][1]["content"]


def test_extract_gone_symptom(fake_llm):
    fake_llm(['{"symptoms": [{"name": "fever", "reported_change": "gone"}]}'])
    result = extract_symptoms("My fever is gone.")
    assert result.symptoms[0].reported_change == "gone"


def test_extract_drops_unsupported(fake_llm):
    fake_llm(['{"symptoms": [{"name": "stomach ache"}, {"name": "cough"}]}'])
    result = extract_symptoms("Stomach ache and a cough.")
    assert [s.name for s in result.symptoms] == ["cough"]
    assert result.dropped_symptoms == ["stomach ache"]


def test_extract_strips_think_and_fences(fake_llm):
    fake_llm(['<think>hmm</think>```json\n{"symptoms": [{"name": "cough"}]}\n```'])
    assert extract_symptoms("cough")


def test_retry_after_bad_json(fake_llm):
    completions = fake_llm(["not json", '{"symptoms": [{"name": "cough"}]}'])
    result = extract_symptoms("cough")
    assert result.symptoms[0].name == "cough"
    assert len(completions.calls) == 2


def test_retry_after_empty_content(fake_llm):
    completions = fake_llm([None, '{"symptoms": []}'])
    assert extract_symptoms("hello").symptoms == []
    assert len(completions.calls) == 2


def test_retry_after_api_error(fake_llm):
    fake_llm([RuntimeError("network"), '{"symptoms": [{"name": "fever"}]}'])
    assert extract_symptoms("fever").symptoms[0].name == "fever"


def test_returns_none_when_all_retries_fail(fake_llm):
    fake_llm(["bad", "still bad"])
    assert extract_symptoms("fever") is None