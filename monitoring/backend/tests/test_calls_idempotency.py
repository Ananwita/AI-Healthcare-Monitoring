"""Chunk A4: idempotent call submission via client_call_id."""

from sqlalchemy import event
from sqlalchemy.orm import Session as OrmSession

from backend.database import SessionLocal, engine
from backend.models import Call, Symptom


def payload(**overrides):
    base = {
        "client_call_id": "call-001",
        "transcript": "I have a fever.",
        "summary": "Patient reports fever.",
        "symptoms": [{"name": "fever", "severity": "moderate"}],
    }
    base.update(overrides)
    return base


def post_call(client, patient_id, **overrides):
    return client.post(f"/patients/{patient_id}/calls", json=payload(**overrides))


def db_counts():
    with SessionLocal() as db:
        return db.query(Call).count(), db.query(Symptom).count()


# ---------- Normal save ----------

def test_new_call_saved(client, patient_id):
    response = post_call(client, patient_id)
    assert response.status_code == 200
    body = response.json()
    assert body["duplicate"] is False
    assert body["client_call_id"] == "call-001"
    assert body["status"] == "completed"
    assert body["symptoms_saved"] == 1


def test_unknown_patient_returns_404(client):
    assert post_call(client, 999).status_code == 404


# ---------- Duplicate prevention ----------

def test_same_payload_returns_original_call(client, patient_id):
    first = post_call(client, patient_id).json()
    second = post_call(client, patient_id)
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert second.json()["call_id"] == first["call_id"]
    assert db_counts() == (1, 1)


def test_explicit_defaults_match_omitted_defaults(client, patient_id):
    post_call(client, patient_id)
    response = post_call(
        client, patient_id,
        status="completed",
        symptoms=[{"name": "fever", "severity": "moderate",
                   "status": "ongoing", "trend": None}],
    )
    assert response.status_code == 200
    assert response.json()["duplicate"] is True
    assert db_counts() == (1, 1)


def test_client_call_id_whitespace_trimmed(client, patient_id):
    first = post_call(client, patient_id).json()
    second = post_call(client, patient_id, client_call_id="  call-001 ").json()
    assert second["duplicate"] is True
    assert second["call_id"] == first["call_id"]


# ---------- Conflicts ----------

def test_different_transcript_returns_409(client, patient_id):
    first = post_call(client, patient_id).json()
    response = post_call(client, patient_id, transcript="Something else.")
    assert response.status_code == 409
    assert response.json()["detail"]["call_id"] == first["call_id"]
    assert db_counts() == (1, 1)


def test_different_symptoms_returns_409(client, patient_id):
    post_call(client, patient_id)
    response = post_call(client, patient_id, symptoms=[{"name": "cough"}])
    assert response.status_code == 409


def test_different_status_returns_409(client, patient_id):
    post_call(client, patient_id, status="interrupted")
    response = post_call(client, patient_id, status="completed")
    assert response.status_code == 409


def test_conflict_does_not_change_stored_call(client, patient_id):
    first = post_call(client, patient_id).json()
    post_call(client, patient_id, transcript="Changed.")
    calls = client.get(f"/patients/{patient_id}/calls").json()
    assert len(calls) == 1
    assert calls[0]["id"] == first["call_id"]
    assert calls[0]["transcript"] == "I have a fever."


# ---------- Missing IDs (legacy behaviour) ----------

def test_no_client_call_id_always_creates(client, patient_id):
    post_call(client, patient_id, client_call_id=None)
    post_call(client, patient_id, client_call_id=None)
    assert db_counts() == (2, 2)


def test_blank_client_call_id_stored_as_null(client, patient_id):
    body = post_call(client, patient_id, client_call_id="   ").json()
    assert body["client_call_id"] is None
    post_call(client, patient_id, client_call_id="")
    assert db_counts() == (2, 2)


def test_same_id_for_different_patients_allowed(client, patient_id):
    other = client.post("/patients", json={"name": "Other"}).json()["patient_id"]
    a = post_call(client, patient_id).json()
    b = post_call(client, other).json()
    assert a["duplicate"] is False and b["duplicate"] is False
    assert a["call_id"] != b["call_id"]


# ---------- Status / trend storage and validation ----------

def test_interrupted_status_and_symptom_fields_stored(client, patient_id):
    body = post_call(
        client, patient_id,
        status="interrupted",
        symptoms=[{"name": "fever", "status": "resolved", "trend": "resolved"}],
    ).json()
    assert body["status"] == "interrupted"

    with SessionLocal() as db:
        call = db.get(Call, body["call_id"])
        assert call.status == "interrupted"
        assert call.payload_hash and len(call.payload_hash) == 64
        assert call.symptoms[0].status == "resolved"
        assert call.symptoms[0].trend == "resolved"


def test_symptom_defaults(client, patient_id):
    body = post_call(client, patient_id).json()
    with SessionLocal() as db:
        symptom = db.get(Call, body["call_id"]).symptoms[0]
        assert symptom.status == "ongoing"
        assert symptom.trend is None


def test_invalid_call_status_rejected(client, patient_id):
    assert post_call(client, patient_id, status="dropped").status_code == 422


def test_invalid_symptom_status_rejected(client, patient_id):
    response = post_call(client, patient_id, symptoms=[{"name": "fever", "status": "gone"}])
    assert response.status_code == 422


def test_invalid_trend_rejected(client, patient_id):
    response = post_call(client, patient_id, symptoms=[{"name": "fever", "trend": "better"}])
    assert response.status_code == 422


def test_client_call_id_too_long_rejected(client, patient_id):
    assert post_call(client, patient_id, client_call_id="x" * 65).status_code == 422


# ---------- Race condition ----------

def test_concurrent_save_returns_existing_call(client, patient_id):
    """
    Another request stores the same client_call_id between our check and commit.
    Same payload -> API must return duplicate=True, not an error.
    """
    from backend.main import CallCreate, compute_call_payload_hash

    same_hash = compute_call_payload_hash(CallCreate(**payload()))
    state = {"fired": False}

    def insert_competitor(session, flush_context, instances):
        if state["fired"]:
            return
        state["fired"] = True
        with engine.begin() as conn:
            conn.execute(
                Call.__table__.insert().values(
                    patient_id=patient_id,
                    client_call_id="call-001",
                    payload_hash=same_hash,
                    status="completed",
                    transcript="I have a fever.",
                    summary="Patient reports fever.",
                )
            )

    event.listen(OrmSession, "before_flush", insert_competitor)
    try:
        response = post_call(client, patient_id)
    finally:
        event.remove(OrmSession, "before_flush", insert_competitor)

    assert state["fired"] is True
    assert response.status_code == 200
    assert response.json()["duplicate"] is True
    assert db_counts()[0] == 1