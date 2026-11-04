import json
import os
from datetime import datetime
from typing import Optional

# Path to the local JSON storage file
HISTORY_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "patient_history.json"
)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_all() -> dict:
    """Load the entire history file. Return empty dict if file doesn't exist."""
    if not os.path.exists(HISTORY_FILE):
        return {}
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)


def _save_all(data: dict) -> None:
    """Write the entire history dict back to disk."""
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── Public API ────────────────────────────────────────────────────────────────

def save_observation(
    patient_id: str,
    symptom: str,
    severity: str,
    call_number: int,
) -> dict:
    """
    Save one symptom observation for a patient.

    Storage structure:
    {
      "P001": {
        "fatigue": [
          {"call": 1, "severity": "mild",     "timestamp": "..."},
          {"call": 2, "severity": "moderate",  "timestamp": "..."}
        ]
      }
    }

    Returns the observation dict that was saved.
    """
    data = _load_all()

    # Initialise patient and symptom keys if not present
    if patient_id not in data:
        data[patient_id] = {}
    if symptom not in data[patient_id]:
        data[patient_id][symptom] = []

    observation = {
        "call": call_number,
        "severity": severity,
        "timestamp": datetime.now().isoformat(),
    }

    data[patient_id][symptom].append(observation)
    _save_all(data)
    return observation


def get_observations(
    patient_id: str,
    symptom: str,
) -> list[dict]:
    """
    Retrieve all stored observations for a patient + symptom, oldest first.

    Returns empty list if patient or symptom not found.
    """
    data = _load_all()
    return data.get(patient_id, {}).get(symptom, [])


def get_previous_observation(
    patient_id: str,
    symptom: str,
) -> Optional[dict]:
    """
    Return the last stored observation for a symptom.
    Call this BEFORE saving the current observation.
    Returns None if no prior observations exist.
    """
    observations = get_observations(patient_id, symptom)
    if len(observations) == 0:
        return None
    return observations[-1]  # last stored = previous, because current not saved yet


def get_latest_observation(
    patient_id: str,
    symptom: str,
) -> Optional[dict]:
    """
    Return the most recent stored observation for a symptom.
    Returns None if no observations exist.
    """
    observations = get_observations(patient_id, symptom)
    if not observations:
        return None
    return observations[-1]


def clear_patient_history(patient_id: str) -> None:
    """
    Delete all history for a patient.
    Useful for resetting demo state between test runs.
    """
    data = _load_all()
    if patient_id in data:
        del data[patient_id]
        _save_all(data)
    print(f"  History cleared for patient '{patient_id}'.")