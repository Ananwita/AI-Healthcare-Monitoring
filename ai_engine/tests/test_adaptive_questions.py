"""
Unit tests for adaptive question generation.
No API calls, no file I/O — pure logic tests.
"""

import pytest
from src.conversation.adaptive_questions import (
    generate_question,
    QuestionResult,
)


# ── Return type ───────────────────────────────────────────────────────────────

def test_returns_question_result_instance():
    result = generate_question("fatigue", "worsening")
    assert isinstance(result, QuestionResult)


def test_result_has_non_empty_question():
    result = generate_question("fatigue", "worsening")
    assert isinstance(result.question, str)
    assert len(result.question) > 0


def test_result_has_non_empty_strategy():
    result = generate_question("fatigue", "worsening")
    assert isinstance(result.strategy, str)
    assert len(result.strategy) > 0


# ── Symptom-specific templates ────────────────────────────────────────────────

def test_fatigue_new_symptom():
    result = generate_question("fatigue", "new_symptom")
    assert result.trend == "new_symptom"
    assert result.symptom == "fatigue"
    assert "fatigue" in result.question.lower() or "long" in result.question.lower()


def test_fatigue_worsening():
    result = generate_question("fatigue", "worsening")
    assert result.trend == "worsening"
    assert result.symptom == "fatigue"


def test_fatigue_improving():
    result = generate_question("fatigue", "improving")
    assert result.trend == "improving"


def test_fatigue_stable():
    result = generate_question("fatigue", "stable")
    assert result.trend == "stable"


def test_headache_worsening():
    result = generate_question("headache", "worsening")
    assert result.trend == "worsening"
    assert result.symptom == "headache"


def test_fever_new_symptom():
    result = generate_question("fever", "new_symptom")
    assert result.trend == "new_symptom"
    assert result.symptom == "fever"


def test_breathing_difficulty_worsening():
    result = generate_question("breathing_difficulty", "worsening")
    assert result.trend == "worsening"
    assert result.symptom == "breathing_difficulty"


# ── Fallback for unknown symptom ──────────────────────────────────────────────

def test_unknown_symptom_uses_fallback():
    # "nausea" is not in TEMPLATES — should use fallback, not crash
    result = generate_question("nausea", "worsening")
    assert isinstance(result.question, str)
    assert len(result.question) > 0
    assert result.trend == "worsening"


def test_unknown_symptom_all_trends_have_fallback():
    for trend in ["new_symptom", "worsening", "stable", "improving"]:
        result = generate_question("nausea", trend)
        assert isinstance(result.question, str)
        assert len(result.question) > 0


# ── All supported symptoms return a question ──────────────────────────────────

@pytest.mark.parametrize("symptom", [
    "fatigue", "headache", "fever", "cough", "pain",
    "dizziness", "breathing_difficulty", "sleep_problems", "appetite_changes"
])
@pytest.mark.parametrize("trend", [
    "new_symptom", "worsening", "stable", "improving"
])
def test_all_symptom_trend_combinations_return_question(symptom, trend):
    result = generate_question(symptom, trend)
    assert isinstance(result.question, str)
    assert len(result.question) > 0


# ── Different trends produce different questions ──────────────────────────────

def test_different_trends_produce_different_questions():
    results = [
        generate_question("fatigue", trend)
        for trend in ["new_symptom", "worsening", "stable", "improving"]
    ]
    questions = [r.question for r in results]
    # All four questions must be unique
    assert len(set(questions)) == 4


# ── Invalid trend raises ──────────────────────────────────────────────────────

def test_invalid_trend_raises_value_error():
    with pytest.raises(ValueError):
        generate_question("fatigue", "unknown_trend")


# ── Metadata fields ───────────────────────────────────────────────────────────

def test_symptom_stored_on_result():
    result = generate_question("fatigue", "worsening")
    assert result.symptom == "fatigue"


def test_trend_stored_on_result():
    result = generate_question("fatigue", "worsening")
    assert result.trend == "worsening"