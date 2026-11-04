"""Chunk A5: all-history symptom state, resolution, recurrence, carry-forward."""


def save(client, patient_id, symptoms, status="completed", transcript="t"):
    response = client.post(
        f"/patients/{patient_id}/calls",
        json={"transcript": transcript, "status": status, "symptoms": symptoms},
    )
    assert response.status_code == 200
    return response.json()["call_id"]


def state(client, patient_id):
    response = client.get(f"/patients/{patient_id}/symptom-state")
    assert response.status_code == 200
    return {s["name"]: s for s in response.json()}


# ---------- Basics ----------

def test_no_calls_gives_empty_state(client, patient_id):
    assert state(client, patient_id) == {}


def test_unknown_patient_returns_404(client):
    assert client.get("/patients/999/symptom-state").status_code == 404


def test_single_observation(client, patient_id):
    call_id = save(client, patient_id, [
        {"name": "fever", "severity": "moderate", "duration": "1 day", "trend": "new"}
    ])
    fever = state(client, patient_id)["fever"]
    assert fever["status"] == "ongoing"
    assert fever["latest_trend"] == "new"
    assert fever["latest_severity"] == "moderate"
    assert fever["latest_duration"] == "1 day"
    assert fever["times_reported"] == 1
    assert fever["last_call_id"] == call_id
    assert fever["recurred"] is False
    assert fever["first_reported_at"] is not None


# ---------- Resolution ----------

def test_explicit_resolution(client, patient_id):
    save(client, patient_id, [{"name": "fever"}, {"name": "weakness"}])
    save(client, patient_id, [
        {"name": "fever", "status": "resolved", "trend": "resolved"},
        {"name": "weakness", "trend": "stable"},
    ])
    s = state(client, patient_id)
    assert s["fever"]["status"] == "resolved"
    assert s["fever"]["latest_trend"] == "resolved"
    assert s["weakness"]["status"] == "ongoing"


def test_resolution_preserves_history(client, patient_id):
    save(client, patient_id, [{"name": "fever", "severity": "moderate"}])
    save(client, patient_id, [{"name": "fever", "status": "resolved"}])
    rows = client.get(f"/patients/{patient_id}/symptoms").json()
    fever_rows = [r for r in rows if r["name"] == "fever"]
    assert [r["status"] for r in fever_rows] == ["ongoing", "resolved"]


def test_unmentioned_symptom_is_not_resolved(client, patient_id):
    first = save(client, patient_id, [{"name": "fever"}, {"name": "fatigue"}])
    save(client, patient_id, [{"name": "fatigue"}])
    fever = state(client, patient_id)["fever"]
    assert fever["status"] == "ongoing"
    assert fever["last_call_id"] == first
    assert fever["times_reported"] == 1


# ---------- Recurrence ----------

def test_recurrence_after_resolution(client, patient_id):
    save(client, patient_id, [{"name": "fever"}])
    save(client, patient_id, [{"name": "fever", "status": "resolved"}])
    save(client, patient_id, [{"name": "fever", "trend": "new"}])
    fever = state(client, patient_id)["fever"]
    assert fever["status"] == "ongoing"
    assert fever["recurred"] is True
    assert fever["times_reported"] == 3


def test_resolved_again_clears_recurred(client, patient_id):
    save(client, patient_id, [{"name": "fever"}])
    save(client, patient_id, [{"name": "fever", "status": "resolved"}])
    save(client, patient_id, [{"name": "fever"}])
    save(client, patient_id, [{"name": "fever", "status": "resolved"}])
    fever = state(client, patient_id)["fever"]
    assert fever["status"] == "resolved"
    assert fever["recurred"] is False


# ---------- Carry-forward ----------

def test_missing_severity_does_not_erase_known_value(client, patient_id):
    first = save(client, patient_id, [
        {"name": "fever", "severity": "moderate", "duration": "1 day"}
    ])
    second = save(client, patient_id, [{"name": "fever", "trend": "stable"}])
    fever = state(client, patient_id)["fever"]
    assert fever["latest_severity"] == "moderate"
    assert fever["latest_severity_call_id"] == first
    assert fever["latest_duration_call_id"] == first
    assert fever["last_call_id"] == second
    assert fever["latest_trend"] == "stable"


def test_new_severity_replaces_old(client, patient_id):
    save(client, patient_id, [{"name": "fever", "severity": "mild"}])
    second = save(client, patient_id, [{"name": "fever", "severity": "severe"}])
    fever = state(client, patient_id)["fever"]
    assert fever["latest_severity"] == "severe"
    assert fever["latest_severity_call_id"] == second


def test_blank_values_are_not_carried(client, patient_id):
    save(client, patient_id, [{"name": "fever", "severity": "mild"}])
    save(client, patient_id, [{"name": "fever", "severity": "  "}])
    assert state(client, patient_id)["fever"]["latest_severity"] == "mild"


# ---------- All history, not just last 5 ----------

def test_state_uses_all_history(client, patient_id):
    save(client, patient_id, [{"name": "fever", "severity": "moderate"}])
    for _ in range(6):
        save(client, patient_id, [{"name": "fatigue"}])

    s = state(client, patient_id)
    assert s["fever"]["status"] == "ongoing"
    assert s["fatigue"]["times_reported"] == 6

    context = client.get(f"/patients/{patient_id}/context").json()
    assert len(context["recent_history"]) == 5
    assert {x["name"] for x in context["symptom_state"]} == {"fever", "fatigue"}


# ---------- Normalization and interrupted calls ----------

def test_symptom_names_normalized(client, patient_id):
    save(client, patient_id, [{"name": "Fever "}])
    save(client, patient_id, [{"name": "fever", "status": "resolved"}])
    save(client, patient_id, [{"name": "Breathing Difficulty"}])
    s = state(client, patient_id)
    assert set(s) == {"fever", "breathing_difficulty"}
    assert s["fever"]["status"] == "resolved"
    assert s["fever"]["times_reported"] == 2


def test_interrupted_calls_count(client, patient_id):
    call_id = save(client, patient_id, [{"name": "cough"}], status="interrupted")
    cough = state(client, patient_id)["cough"]
    assert cough["last_call_id"] == call_id


# ---------- New fields in existing endpoints ----------

def test_calls_endpoint_includes_new_fields(client, patient_id):
    client.post(f"/patients/{patient_id}/calls", json={
        "client_call_id": "c-1", "status": "interrupted",
        "symptoms": [{"name": "fever", "trend": "new"}],
    })
    call = client.get(f"/patients/{patient_id}/calls").json()[0]
    assert call["client_call_id"] == "c-1"
    assert call["status"] == "interrupted"
    assert call["symptoms"][0]["status"] == "ongoing"
    assert call["symptoms"][0]["trend"] == "new"


def test_symptoms_endpoint_includes_new_fields(client, patient_id):
    save(client, patient_id, [{"name": "fever", "status": "resolved", "trend": "resolved"}])
    row = client.get(f"/patients/{patient_id}/symptoms").json()[0]
    assert row["status"] == "resolved"
    assert row["trend"] == "resolved"
    assert row["created_at"] is not None


def test_context_recent_history_includes_status(client, patient_id):
    save(client, patient_id, [{"name": "fever"}], status="interrupted")
    item = client.get(f"/patients/{patient_id}/context").json()["recent_history"][0]
    assert item["status"] == "interrupted"
    assert item["symptoms"][0]["status"] == "ongoing"