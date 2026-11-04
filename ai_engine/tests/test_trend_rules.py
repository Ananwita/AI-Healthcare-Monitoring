"""Chunk B3: explicit trend/status rules (no API calls)."""

import pytest

from src.analysis.trend import (
    STATUS_VALUES,
    TREND_VALUES,
    determine_update,
    detect_trend,
    unmentioned_ongoing_symptoms,
)
from src.llm.extraction import PreviousSymptom, SymptomObservation


def obs(name="fever", **kwargs):
    return SymptomObservation(name=name, **kwargs)


def prev(status="ongoing", severity=None, name="fever"):
    return {"name": name, "status": status, "latest_severity": severity}


def check(update, trend, status, recurred=False):
    assert (update.trend, update.status, update.recurred) == (trend, status, recurred)
    assert update.reason


# ── No previous record ───────────────────────────────────────────────────────

def test_first_report_is_new():
    check(determine_update(None, obs()), "new", "ongoing")


def test_first_report_with_severity_is_new():
    check(determine_update(None, obs(severity="severe")), "new", "ongoing")


def test_first_report_as_gone_is_resolved():
    check(determine_update(None, obs(reported_change="gone")), "resolved", "resolved")


# ── Explicit resolution ──────────────────────────────────────────────────────

def test_gone_resolves_ongoing_symptom():
    check(determine_update(prev(), obs(reported_change="gone")), "resolved", "resolved")


def test_gone_again_stays_resolved():
    check(determine_update(prev("resolved"), obs(reported_change="gone")), "resolved", "resolved")


# ── Recurrence ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("change", [None, "new", "worse", "better", "same"])
def test_resolved_symptom_reported_again_recurs(change):
    update = determine_update(prev("resolved"), obs(reported_change=change))
    check(update, "new", "ongoing", recurred=True)


# ── Explicit change words ────────────────────────────────────────────────────

@pytest.mark.parametrize("change,trend", [
    ("worse", "worsening"), ("better", "improving"), ("same", "stable"),
])
def test_explicit_change(change, trend):
    check(determine_update(prev(), obs(reported_change=change)), trend, "ongoing")


def test_explicit_change_beats_severity():
    # Patient says "better" though stated severity is higher -> trust the explicit word
    update = determine_update(prev(severity="mild"), obs(reported_change="better", severity="severe"))
    check(update, "improving", "ongoing")


# ── Severity stated by patient ───────────────────────────────────────────────

def test_higher_stated_severity_is_worsening():
    update = determine_update(prev(severity="mild"), obs(severity="moderate"))
    check(update, "worsening", "ongoing")
    assert "previously mild" in update.reason


def test_lower_stated_severity_is_improving():
    check(determine_update(prev(severity="severe"), obs(severity="mild")), "improving", "ongoing")


def test_same_stated_severity_is_stable():
    check(determine_update(prev(severity="mild"), obs(severity="mild")), "stable", "ongoing")


def test_missing_current_severity_is_not_a_change():
    check(determine_update(prev(severity="severe"), obs()), "stable", "ongoing")


def test_missing_previous_severity_is_not_a_change():
    check(determine_update(prev(severity=None), obs(severity="severe")), "stable", "ongoing")


def test_unknown_previous_severity_ignored():
    check(determine_update(prev(severity="extreme"), obs(severity="mild")), "stable", "ongoing")


# ── Mentioned without change ─────────────────────────────────────────────────

def test_mentioned_without_change_is_stable():
    check(determine_update(prev(), obs()), "stable", "ongoing")


def test_called_new_but_already_ongoing_is_stable():
    update = determine_update(prev(), obs(reported_change="new"))
    check(update, "stable", "ongoing")
    assert "already being tracked" in update.reason


def test_missing_previous_status_treated_as_ongoing():
    check(determine_update({"name": "fever"}, obs(reported_change="better")), "improving", "ongoing")


# ── Input formats ────────────────────────────────────────────────────────────

def test_accepts_previous_symptom_object():
    previous = PreviousSymptom(name="fever", status="resolved")
    check(determine_update(previous, obs()), "new", "ongoing", recurred=True)


def test_accepts_full_backend_state_dict():
    backend_entry = {
        "name": "fever", "status": "ongoing", "latest_trend": "new",
        "latest_severity": "mild", "times_reported": 1, "recurred": False,
        "last_call_id": 3,
    }
    check(determine_update(backend_entry, obs(severity="severe")), "worsening", "ongoing")


def test_output_values_match_backend():
    cases = [
        (None, obs()), (None, obs(reported_change="gone")),
        (prev("resolved"), obs()), (prev(), obs(reported_change="worse")),
        (prev(), obs(reported_change="better")), (prev(), obs()),
    ]
    for previous, observation in cases:
        update = determine_update(previous, observation)
        assert update.trend in TREND_VALUES
        assert update.status in STATUS_VALUES


def test_name_is_normalized():
    assert determine_update(None, obs(name="Breathing Difficulty")).name == "breathing_difficulty"


# ── Unmentioned symptoms ─────────────────────────────────────────────────────

def test_unmentioned_ongoing_symptoms():
    previous_states = [
        prev(name="fever"), prev(name="weakness"),
        prev("resolved", name="cough"), {"name": "Headache"},
    ]
    assert unmentioned_ongoing_symptoms(previous_states, ["Fever"]) == ["weakness", "headache"]


def test_unmentioned_with_no_history():
    assert unmentioned_ongoing_symptoms([], ["fever"]) == []
    assert unmentioned_ongoing_symptoms(None, []) == []


# ── Backward compatibility ───────────────────────────────────────────────────

def test_old_detect_trend_unchanged():
    assert detect_trend("fatigue", "moderate", "mild").trend == "worsening"
    assert detect_trend("fatigue", "mild", None).trend == "new_symptom"