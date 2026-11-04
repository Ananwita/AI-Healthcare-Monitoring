"""Baseline tests for the existing backend endpoints (before Chunk A2 changes)."""


def save_call(client, patient_id, transcript="I feel tired.", symptoms=None):
    payload = {
        "transcript": transcript,
        "summary": "Test summary.",
        "symptoms": symptoms if symptoms is not None else [
            {"name": "fatigue", "severity": "mild", "duration": "2 days"}
        ],
    }
    return client.post(f"/patients/{patient_id}/calls", json=payload)


# ---------- Basic ----------

def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "running" in response.json()["message"]


# ---------- Patients ----------

def test_create_patient(client):
    response = client.post("/patients", json={"name": "Asha", "age": 30})
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Asha"
    assert isinstance(body["patient_id"], int)


def test_create_patient_requires_name(client):
    response = client.post("/patients", json={"age": 30})
    assert response.status_code == 422


def test_list_patients(client, patient_id):
    response = client.get("/patients")
    assert response.status_code == 200
    patients = response.json()
    assert len(patients) == 1
    assert patients[0]["id"] == patient_id
    assert patients[0]["language"] == "English"


def test_get_patient(client, patient_id):
    response = client.get(f"/patients/{patient_id}")
    assert response.status_code == 200
    assert response.json()["name"] == "Test Patient"


def test_get_patient_not_found(client):
    assert client.get("/patients/999").status_code == 404


# ---------- Calls ----------

def test_save_call(client, patient_id):
    response = save_call(client, patient_id)
    assert response.status_code == 200
    body = response.json()
    assert body["patient_id"] == patient_id
    assert body["symptoms_saved"] == 1


def test_save_call_without_symptoms(client, patient_id):
    response = save_call(client, patient_id, symptoms=[])
    assert response.status_code == 200
    assert response.json()["symptoms_saved"] == 0


def test_save_call_unknown_patient(client):
    assert save_call(client, 999).status_code == 404


def test_get_calls(client, patient_id):
    save_call(client, patient_id, transcript="Call one")
    save_call(client, patient_id, transcript="Call two")

    response = client.get(f"/patients/{patient_id}/calls")
    assert response.status_code == 200
    calls = response.json()
    assert len(calls) == 2
    assert {c["transcript"] for c in calls} == {"Call one", "Call two"}
    assert calls[0]["symptoms"][0]["name"] == "fatigue"
    assert calls[0]["created_at"] is not None


def test_get_calls_unknown_patient(client):
    assert client.get("/patients/999/calls").status_code == 404


# ---------- Symptoms ----------

def test_get_symptoms(client, patient_id):
    save_call(client, patient_id, symptoms=[
        {"name": "fever", "severity": "moderate"},
        {"name": "fatigue", "severity": "mild"},
    ])

    response = client.get(f"/patients/{patient_id}/symptoms")
    assert response.status_code == 200
    symptoms = response.json()
    assert {s["name"] for s in symptoms} == {"fever", "fatigue"}
    assert all(s["call_id"] for s in symptoms)


def test_get_symptoms_unknown_patient(client):
    assert client.get("/patients/999/symptoms").status_code == 404


# ---------- Context ----------

def test_context_structure(client, patient_id):
    save_call(client, patient_id)

    response = client.get(f"/patients/{patient_id}/context")
    assert response.status_code == 200
    body = response.json()
    assert body["patient"]["id"] == patient_id
    assert len(body["recent_history"]) == 1
    assert body["recent_history"][0]["symptoms"][0]["name"] == "fatigue"


def test_context_limited_to_five_calls(client, patient_id):
    for i in range(6):
        save_call(client, patient_id, transcript=f"Call {i}")

    body = client.get(f"/patients/{patient_id}/context").json()
    assert len(body["recent_history"]) == 5


def test_context_unknown_patient(client):
    assert client.get("/patients/999/context").status_code == 404