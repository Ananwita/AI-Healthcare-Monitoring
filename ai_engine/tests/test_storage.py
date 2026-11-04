"""
Unit tests for storage.py — no API calls, no LLM.
Tests read/write/retrieve logic against a temporary file.
"""

import os
import json
import pytest
from unittest.mock import patch

# Temporary file path used during tests — does not touch real data/
TEMP_FILE = "data/test_history_temp.json"


# Patch HISTORY_FILE to use temp file for all tests in this module
@pytest.fixture(autouse=True)
def use_temp_file(tmp_path):
    temp = str(tmp_path / "test_history.json")
    with patch("src.history.storage.HISTORY_FILE", temp):
        yield temp


from src.history.storage import (
    save_observation,
    get_observations,
    get_previous_observation,
    get_latest_observation,
    clear_patient_history,
)


def test_save_and_retrieve_single_observation():
    save_observation("P001", "fatigue", "mild", call_number=1)
    obs = get_observations("P001", "fatigue")
    assert len(obs) == 1
    assert obs[0]["severity"] == "mild"
    assert obs[0]["call"] == 1


def test_save_multiple_observations_ordered():
    save_observation("P001", "fatigue", "mild", call_number=1)
    save_observation("P001", "fatigue", "moderate", call_number=2)
    obs = get_observations("P001", "fatigue")
    assert len(obs) == 2
    assert obs[0]["severity"] == "mild"
    assert obs[1]["severity"] == "moderate"


def test_get_previous_observation_returns_last_stored():
    """
    get_previous_observation should return the last stored entry.
    It is always called BEFORE the current observation is saved.
    """
    save_observation("P001", "fatigue", "mild", call_number=1)
    prev = get_previous_observation("P001", "fatigue")
    assert prev["severity"] == "mild"
    assert prev["call"] == 1


def test_get_previous_observation_none_if_no_history():
    """No history at all — should return None."""
    prev = get_previous_observation("P001", "fatigue")
    assert prev is None


def test_get_latest_observation():
    save_observation("P001", "fatigue", "mild", call_number=1)
    save_observation("P001", "fatigue", "moderate", call_number=2)
    latest = get_latest_observation("P001", "fatigue")
    assert latest["severity"] == "moderate"


def test_get_latest_observation_none_if_empty():
    latest = get_latest_observation("P001", "fatigue")
    assert latest is None


def test_different_symptoms_stored_independently():
    save_observation("P001", "fatigue", "mild", call_number=1)
    save_observation("P001", "headache", "moderate", call_number=1)
    assert len(get_observations("P001", "fatigue")) == 1
    assert len(get_observations("P001", "headache")) == 1


def test_different_patients_stored_independently():
    save_observation("P001", "fatigue", "mild", call_number=1)
    save_observation("P002", "fatigue", "severe", call_number=1)
    assert get_observations("P001", "fatigue")[0]["severity"] == "mild"
    assert get_observations("P002", "fatigue")[0]["severity"] == "severe"


def test_clear_patient_history():
    save_observation("P001", "fatigue", "mild", call_number=1)
    clear_patient_history("P001")
    assert get_observations("P001", "fatigue") == []


def test_unknown_patient_returns_empty():
    obs = get_observations("UNKNOWN", "fatigue")
    assert obs == []