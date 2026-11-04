import pytest
from pydantic import ValidationError
from src.llm.extraction import ExtractionResult, SymptomObservation


def test_valid_single_symptom():
    data = {"symptoms": [{"name": "fatigue", "severity": "mild"}]}
    result = ExtractionResult(**data)
    assert len(result.symptoms) == 1
    assert result.symptoms[0].name == "fatigue"
    assert result.symptoms[0].severity == "mild"


def test_valid_multiple_symptoms():
    data = {
        "symptoms": [
            {"name": "fatigue", "severity": "moderate"},
            {"name": "headache", "severity": "mild"},
        ]
    }
    result = ExtractionResult(**data)
    assert len(result.symptoms) == 2


def test_empty_symptoms_list():
    data = {"symptoms": []}
    result = ExtractionResult(**data)
    assert result.symptoms == []


def test_unsupported_symptom_dropped_not_failed():
    data = {"symptoms": [
        {"name": "stomachache", "severity": "mild"},
        {"name": "fever", "severity": "moderate"},
    ]}
    result = ExtractionResult(**data)
    assert [s.name for s in result.symptoms] == ["fever"]
    assert result.dropped_symptoms == ["stomachache"]


def test_invalid_symptom_name_rejected_by_observation():
    with pytest.raises(ValidationError):
        SymptomObservation(name="stomachache", severity="mild")


def test_invalid_severity_rejected():
    data = {"symptoms": [{"name": "fatigue", "severity": "extreme"}]}
    with pytest.raises(ValidationError):
        ExtractionResult(**data)


def test_severity_normalised_to_lowercase():
    data = {"symptoms": [{"name": "fatigue", "severity": "Mild"}]}
    result = ExtractionResult(**data)
    assert result.symptoms[0].severity == "mild"


def test_symptom_name_normalised_to_lowercase():
    data = {"symptoms": [{"name": "Fatigue", "severity": "mild"}]}
    result = ExtractionResult(**data)
    assert result.symptoms[0].name == "fatigue"


def test_symptom_name_space_normalised_to_underscore():
    data = {"symptoms": [{"name": "breathing difficulty", "severity": "moderate"}]}
    result = ExtractionResult(**data)
    assert result.symptoms[0].name == "breathing_difficulty"


def test_all_severity_levels_accepted():
    for level in ["mild", "moderate", "severe"]:
        data = {"symptoms": [{"name": "fatigue", "severity": level}]}
        result = ExtractionResult(**data)
        assert result.symptoms[0].severity == level