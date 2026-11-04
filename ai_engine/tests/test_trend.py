"""
Unit tests for trend detection — no API calls, no file I/O.
Pure logic tests on the detect_trend function.
"""

import pytest
from src.analysis.trend import detect_trend, TrendResult


# ── new_symptom ───────────────────────────────────────────────────────────────

def test_new_symptom_when_no_previous():
    result = detect_trend("fatigue", "mild", previous_severity=None)
    assert result.trend == "new_symptom"
    assert result.previous_severity is None
    assert result.current_severity == "mild"


def test_new_symptom_reason_contains_symptom_name():
    result = detect_trend("fatigue", "mild", previous_severity=None)
    assert "fatigue" in result.reason.lower()


# ── stable ────────────────────────────────────────────────────────────────────

def test_stable_mild_to_mild():
    result = detect_trend("fatigue", "mild", "mild")
    assert result.trend == "stable"


def test_stable_moderate_to_moderate():
    result = detect_trend("headache", "moderate", "moderate")
    assert result.trend == "stable"


def test_stable_severe_to_severe():
    result = detect_trend("pain", "severe", "severe")
    assert result.trend == "stable"


# ── worsening ─────────────────────────────────────────────────────────────────

def test_worsening_mild_to_moderate():
    result = detect_trend("fatigue", "moderate", "mild")
    assert result.trend == "worsening"


def test_worsening_mild_to_severe():
    result = detect_trend("fatigue", "severe", "mild")
    assert result.trend == "worsening"


def test_worsening_moderate_to_severe():
    result = detect_trend("cough", "severe", "moderate")
    assert result.trend == "worsening"


def test_worsening_reason_mentions_both_severities():
    result = detect_trend("fatigue", "moderate", "mild")
    assert "mild" in result.reason
    assert "moderate" in result.reason


# ── improving ─────────────────────────────────────────────────────────────────

def test_improving_moderate_to_mild():
    result = detect_trend("fatigue", "mild", "moderate")
    assert result.trend == "improving"


def test_improving_severe_to_mild():
    result = detect_trend("fatigue", "mild", "severe")
    assert result.trend == "improving"


def test_improving_severe_to_moderate():
    result = detect_trend("headache", "moderate", "severe")
    assert result.trend == "improving"


def test_improving_reason_mentions_both_severities():
    result = detect_trend("fatigue", "mild", "moderate")
    assert "moderate" in result.reason
    assert "mild" in result.reason


# ── edge cases ────────────────────────────────────────────────────────────────

def test_invalid_current_severity_raises():
    with pytest.raises(ValueError):
        detect_trend("fatigue", "extreme", "mild")


def test_invalid_previous_severity_raises():
    with pytest.raises(ValueError):
        detect_trend("fatigue", "mild", "unknown")


def test_result_is_trendresult_instance():
    result = detect_trend("fatigue", "moderate", "mild")
    assert isinstance(result, TrendResult)


def test_severities_stored_on_result():
    result = detect_trend("fatigue", "moderate", "mild")
    assert result.previous_severity == "mild"
    assert result.current_severity == "moderate"