"""Chunk A3: idempotent patient creation via external_id."""

from sqlalchemy import event
from sqlalchemy.orm import Session as OrmSession

from backend.database import engine
from backend.models import Patient

BASE = {"external_id": "P001", "name": "Asha", "age": 30, "language": "English"}


def create(client, **overrides):
    return client.post("/patients", json={**BASE, **overrides})


def patient_count(client):
    return len(client.get("/patients").json())


# ---------- Creation ----------

def test_create_with_external_id(client):
    response = create(client)
    assert response.status_code == 200
    body = response.json()
    assert body["created"] is True
    assert body["external_id"] == "P001"
    assert body["name"] == "Asha"


def test_repeat_same_payload_returns_existing(client):
    first = create(client).json()
    second = create(client)
    assert second.status_code == 200
    assert second.json()["created"] is False
    assert second.json()["patient_id"] == first["patient_id"]
    assert patient_count(client) == 1


def test_repeat_with_subset_of_fields_returns_existing(client):
    first = create(client).json()
    response = client.post("/patients", json={"external_id": "P001", "name": "Asha"})
    assert response.status_code == 200
    assert response.json()["patient_id"] == first["patient_id"]
    assert patient_count(client) == 1


def test_external_id_whitespace_is_trimmed(client):
    first = create(client).json()
    second = create(client, external_id="  P001  ").json()
    assert second["patient_id"] == first["patient_id"]
    assert second["created"] is False


# ---------- Conflicts ----------

def test_different_name_returns_409(client):
    first = create(client).json()
    response = create(client, name="Someone Else")
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["conflicting_fields"] == ["name"]
    assert detail["patient_id"] == first["patient_id"]


def test_conflict_lists_all_differing_fields(client):
    create(client)
    response = create(client, name="X", age=99)
    assert response.status_code == 409
    assert response.json()["detail"]["conflicting_fields"] == ["age", "name"]


def test_conflict_does_not_modify_stored_patient(client):
    first = create(client).json()
    create(client, name="Changed", age=99)
    stored = client.get(f"/patients/{first['patient_id']}").json()
    assert stored["name"] == "Asha"
    assert stored["age"] == 30
    assert patient_count(client) == 1


# ---------- Missing IDs (NULL) ----------

def test_blank_external_id_stored_as_null(client):
    body = create(client, external_id="   ").json()
    assert body["external_id"] is None
    assert body["created"] is True


def test_patients_without_external_id_are_not_deduplicated(client):
    client.post("/patients", json={"name": "Asha"})
    client.post("/patients", json={"name": "Asha"})
    create(client, external_id="")
    assert patient_count(client) == 3


def test_external_id_too_long_rejected(client):
    assert create(client, external_id="X" * 65).status_code == 422


# ---------- Lookup ----------

def test_get_by_external_id(client):
    first = create(client).json()
    response = client.get("/patients/by-external/P001")
    assert response.status_code == 200
    assert response.json()["id"] == first["patient_id"]
    assert response.json()["external_id"] == "P001"


def test_get_by_external_id_not_found(client):
    assert client.get("/patients/by-external/NOPE").status_code == 404


def test_list_and_get_include_external_id(client):
    first = create(client).json()
    assert client.get("/patients").json()[0]["external_id"] == "P001"
    assert client.get(f"/patients/{first['patient_id']}").json()["external_id"] == "P001"


def test_context_includes_external_id(client):
    first = create(client).json()
    body = client.get(f"/patients/{first['patient_id']}/context").json()
    assert body["patient"]["external_id"] == "P001"


# ---------- Race condition ----------

def test_concurrent_create_returns_existing_patient(client):
    """
    Simulate another request inserting the same external_id between our
    existence check and our commit. The unique index rejects our insert;
    the API must return the other request's patient instead of an error.
    """
    state = {"fired": False}

    def insert_competitor(session, flush_context, instances):
        if state["fired"]:
            return
        state["fired"] = True
        with engine.begin() as conn:
            conn.execute(
                Patient.__table__.insert().values(
                    external_id="RACE", name="Asha", language="English"
                )
            )

    event.listen(OrmSession, "before_flush", insert_competitor)
    try:
        response = client.post("/patients", json={"external_id": "RACE", "name": "Asha"})
    finally:
        event.remove(OrmSession, "before_flush", insert_competitor)

    assert state["fired"] is True
    assert response.status_code == 200
    assert response.json()["created"] is False
    assert patient_count(client) == 1