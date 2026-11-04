import hashlib
import json
from typing import Literal, Optional

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.database import Base, engine, get_db
from backend.models import Patient, Call, Symptom

# Create database tables automatically (new DBs only; existing DBs use backend/migrate.py)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Healthcare Monitoring API",
    description="Backend for longitudinal patient monitoring",
    version="1.1.0"
)


# ---------- Request Schemas ----------

class PatientCreate(BaseModel):
    external_id: Optional[str] = Field(default=None, max_length=64)
    name: str
    age: Optional[int] = None
    language: str = "English"
    medical_history: Optional[str] = None

    @field_validator("external_id", mode="before")
    @classmethod
    def blank_external_id_is_null(cls, v):
        """Store missing/blank IDs as NULL, never as ''."""
        if v is None:
            return None
        v = str(v).strip()
        return v or None


class SymptomCreate(BaseModel):
    name: str
    severity: Optional[str] = None
    duration: Optional[str] = None
    medication_adherence: Optional[str] = None
    status: Literal["ongoing", "resolved"] = "ongoing"
    trend: Optional[
        Literal["new", "worsening", "stable", "improving", "resolved"]
    ] = None


class CallCreate(BaseModel):
    client_call_id: Optional[str] = Field(default=None, max_length=64)
    status: Literal["completed", "interrupted"] = "completed"
    transcript: Optional[str] = None
    summary: Optional[str] = None
    symptoms: list[SymptomCreate] = []

    @field_validator("client_call_id", mode="before")
    @classmethod
    def blank_client_call_id_is_null(cls, v):
        """Store missing/blank IDs as NULL, never as ''."""
        if v is None:
            return None
        v = str(v).strip()
        return v or None


# ---------- Helpers ----------

PATIENT_COMPARE_FIELDS = ("name", "age", "language", "medical_history")


def get_patient_or_404(db: Session, patient_id: int) -> Patient:
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


def patient_to_dict(patient: Patient) -> dict:
    return {
        "id": patient.id,
        "external_id": patient.external_id,
        "name": patient.name,
        "age": patient.age,
        "language": patient.language,
        "medical_history": patient.medical_history
    }


def patient_create_response(patient: Patient, created: bool) -> dict:
    return {
        "message": (
            "Patient created successfully" if created
            else "Patient already exists"
        ),
        "patient_id": patient.id,
        "external_id": patient.external_id,
        "name": patient.name,
        "created": created
    }


def resolve_existing_patient(existing: Patient, data: PatientCreate) -> dict:
    """
    Same external_id already stored.
    Only fields the client actually sent are compared.
    Any difference -> 409 Conflict (stored record is NOT changed).
    """
    sent_fields = data.model_fields_set & set(PATIENT_COMPARE_FIELDS)
    conflicts = sorted(
        field for field in sent_fields
        if getattr(data, field) != getattr(existing, field)
    )

    if conflicts:
        raise HTTPException(
            status_code=409,
            detail={
                "message": (
                    "A patient with this external_id already exists "
                    "with different data."
                ),
                "external_id": existing.external_id,
                "patient_id": existing.id,
                "conflicting_fields": conflicts
            }
        )

    return patient_create_response(existing, created=False)

def compute_call_payload_hash(data: CallCreate) -> str:
    """
    SHA-256 of the canonical request payload (excluding client_call_id).
    Defaults are included, so omitting a default field and sending it
    explicitly produce the same hash.
    """
    payload = data.model_dump(exclude={"client_call_id"})
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def find_call_by_client_id(db: Session, patient_id: int, client_call_id: str):
    return db.query(Call).filter(
        Call.patient_id == patient_id,
        Call.client_call_id == client_call_id
    ).first()


def call_save_response(call: Call, duplicate: bool) -> dict:
    return {
        "message": (
            "Call already saved (duplicate request ignored)" if duplicate
            else "Call saved successfully"
        ),
        "call_id": call.id,
        "patient_id": call.patient_id,
        "client_call_id": call.client_call_id,
        "status": call.status,
        "symptoms_saved": len(call.symptoms),
        "duplicate": duplicate
    }


def resolve_existing_call(existing: Call, payload_hash: str) -> dict:
    """
    Same patient + client_call_id already stored.
    Same payload -> return the original call (no new rows).
    Different payload -> 409 Conflict (nothing changed).
    """
    if existing.payload_hash != payload_hash:
        raise HTTPException(
            status_code=409,
            detail={
                "message": (
                    "client_call_id was already used for this patient "
                    "with different call data."
                ),
                "client_call_id": existing.client_call_id,
                "call_id": existing.id
            }
        )
    return call_save_response(existing, duplicate=True)


# ---------- Basic Route ----------

@app.get("/")
def home():
    return {
        "message": "AI Healthcare Monitoring API is running"
    }


# ---------- Patient APIs ----------

@app.post("/patients")
def create_patient(
    data: PatientCreate,
    db: Session = Depends(get_db)
):
    # Idempotent path: same external_id never creates a duplicate
    if data.external_id is not None:
        existing = db.query(Patient).filter(
            Patient.external_id == data.external_id
        ).first()
        if existing:
            return resolve_existing_patient(existing, data)

    patient = Patient(**data.model_dump())
    db.add(patient)

    try:
        db.commit()
    except IntegrityError:
        # Another request created the same external_id at the same moment
        db.rollback()
        existing = db.query(Patient).filter(
            Patient.external_id == data.external_id
        ).first()
        if existing is None:
            raise
        return resolve_existing_patient(existing, data)

    db.refresh(patient)
    return patient_create_response(patient, created=True)


@app.get("/patients")
def list_patients(db: Session = Depends(get_db)):
    patients = db.query(Patient).order_by(Patient.id).all()

    return [
        {
            "id": p.id,
            "external_id": p.external_id,
            "name": p.name,
            "age": p.age,
            "language": p.language
        }
        for p in patients
    ]


@app.get("/patients/by-external/{external_id}")
def get_patient_by_external_id(
    external_id: str,
    db: Session = Depends(get_db)
):
    patient = db.query(Patient).filter(
        Patient.external_id == external_id.strip()
    ).first()

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    return patient_to_dict(patient)


@app.get("/patients/{patient_id}")
def get_patient(
    patient_id: int,
    db: Session = Depends(get_db)
):
    patient = get_patient_or_404(db, patient_id)
    return patient_to_dict(patient)


# ---------- Call and Symptom APIs ----------

@app.post("/patients/{patient_id}/calls")
def save_call(
    patient_id: int,
    data: CallCreate,
    db: Session = Depends(get_db)
):
    get_patient_or_404(db, patient_id)
    payload_hash = compute_call_payload_hash(data)

    # Idempotent path: same client_call_id never creates a duplicate
    if data.client_call_id is not None:
        existing = find_call_by_client_id(db, patient_id, data.client_call_id)
        if existing:
            return resolve_existing_call(existing, payload_hash)

    call = Call(
        patient_id=patient_id,
        client_call_id=data.client_call_id,
        payload_hash=payload_hash,
        status=data.status,
        transcript=data.transcript,
        summary=data.summary
    )

    for item in data.symptoms:
        call.symptoms.append(Symptom(**item.model_dump()))

    db.add(call)

    try:
        db.commit()
    except IntegrityError:
        # Same client_call_id saved by another request at the same moment
        db.rollback()
        if data.client_call_id is None:
            raise
        existing = find_call_by_client_id(db, patient_id, data.client_call_id)
        if existing is None:
            raise
        return resolve_existing_call(existing, payload_hash)

    db.refresh(call)
    return call_save_response(call, duplicate=False)


# ---------- History helpers ----------

CARRY_FORWARD_FIELDS = ("severity", "duration", "medication_adherence")


def iso(dt):
    return dt.isoformat() if dt else None


def normalize_symptom_name(name: str) -> str:
    """'Breathing Difficulty ' -> 'breathing_difficulty'"""
    return "_".join((name or "").strip().lower().replace("-", " ").split())


def symptom_to_dict(s: Symptom) -> dict:
    return {
        "name": s.name,
        "severity": s.severity,
        "duration": s.duration,
        "medication_adherence": s.medication_adherence,
        "status": s.status,
        "trend": s.trend
    }


def compute_symptom_state(db: Session, patient_id: int) -> list[dict]:
    """
    Latest state of every symptom across the patient's ENTIRE history.

    Rules:
    - Observations are replayed oldest -> newest.
    - Latest observation decides status/trend.
    - A symptom not mentioned in later calls keeps its previous state
      (never auto-resolved).
    - Missing severity/duration/adherence do not erase earlier known values;
      each latest_* value carries the call_id it came from.
    - recurred = currently ongoing after having been resolved before.
    """
    rows = (
        db.query(Symptom, Call)
        .join(Call, Symptom.call_id == Call.id)
        .filter(Call.patient_id == patient_id)
        .order_by(Call.created_at.asc(), Call.id.asc(), Symptom.id.asc())
        .all()
    )

    states: dict[str, dict] = {}

    for symptom, call in rows:
        key = normalize_symptom_name(symptom.name)
        if not key:
            continue

        state = states.get(key)
        if state is None:
            state = {
                "name": key,
                "status": "ongoing",
                "latest_trend": None,
                "first_reported_at": iso(call.created_at),
                "last_reported_at": None,
                "last_call_id": None,
                "times_reported": 0,
                "recurred": False,
                "_ever_resolved": False,
            }
            for field in CARRY_FORWARD_FIELDS:
                state[f"latest_{field}"] = None
                state[f"latest_{field}_call_id"] = None
            states[key] = state

        status = symptom.status or "ongoing"

        if status == "resolved":
            state["_ever_resolved"] = True
            state["recurred"] = False
        else:
            state["recurred"] = state["_ever_resolved"]

        state["status"] = status
        state["latest_trend"] = symptom.trend
        state["last_reported_at"] = iso(call.created_at)
        state["last_call_id"] = call.id
        state["times_reported"] += 1

        for field in CARRY_FORWARD_FIELDS:
            value = getattr(symptom, field)
            if value is not None and str(value).strip():
                state[f"latest_{field}"] = value
                state[f"latest_{field}_call_id"] = call.id

    result = []
    for state in states.values():
        state.pop("_ever_resolved")
        result.append(state)

    return sorted(result, key=lambda s: s["name"])


# ---------- History APIs ----------

@app.get("/patients/{patient_id}/calls")
def get_calls(
    patient_id: int,
    db: Session = Depends(get_db)
):
    get_patient_or_404(db, patient_id)

    calls = (
        db.query(Call)
        .filter(Call.patient_id == patient_id)
        .order_by(Call.created_at.desc(), Call.id.desc())
        .all()
    )

    return [
        {
            "id": c.id,
            "client_call_id": c.client_call_id,
            "status": c.status,
            "transcript": c.transcript,
            "summary": c.summary,
            "created_at": iso(c.created_at),
            "symptoms": [symptom_to_dict(s) for s in c.symptoms]
        }
        for c in calls
    ]


@app.get("/patients/{patient_id}/symptoms")
def get_symptoms(
    patient_id: int,
    db: Session = Depends(get_db)
):
    get_patient_or_404(db, patient_id)

    symptoms = (
        db.query(Symptom, Call.created_at)
        .join(Call, Symptom.call_id == Call.id)
        .filter(Call.patient_id == patient_id)
        .order_by(Call.created_at.asc(), Call.id.asc(), Symptom.id.asc())
        .all()
    )

    return [
        {
            "call_id": symptom.call_id,
            **symptom_to_dict(symptom),
            "created_at": iso(call_date)
        }
        for symptom, call_date in symptoms
    ]


@app.get("/patients/{patient_id}/symptom-state")
def get_symptom_state(
    patient_id: int,
    db: Session = Depends(get_db)
):
    """Latest state of each symptom, computed from ALL calls."""
    get_patient_or_404(db, patient_id)
    return compute_symptom_state(db, patient_id)


@app.get("/patients/{patient_id}/context")
def get_patient_context(
    patient_id: int,
    db: Session = Depends(get_db)
):
    """
    Context for the AI conversation engine.
    - recent_history: last 5 calls (summaries/transcript context)
    - symptom_state : latest state of every symptom from ALL calls
    """
    patient = get_patient_or_404(db, patient_id)

    calls = (
        db.query(Call)
        .filter(Call.patient_id == patient_id)
        .order_by(Call.created_at.desc(), Call.id.desc())
        .limit(5)
        .all()
    )

    return {
        "patient": patient_to_dict(patient),
        "recent_history": [
            {
                "call_id": c.id,
                "status": c.status,
                "created_at": iso(c.created_at),
                "summary": c.summary,
                "symptoms": [symptom_to_dict(s) for s in c.symptoms]
            }
            for c in reversed(calls)
        ],
        "symptom_state": compute_symptom_state(db, patient_id)
    }