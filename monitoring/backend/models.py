from sqlalchemy import (
    Column, Integer, String, Text, DateTime,
    ForeignKey, Index
)
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

from backend.database import Base


# Allowed values (validated in the API layer; SQLite cannot add CHECK via ALTER)
CALL_STATUSES = ("completed", "interrupted")
SYMPTOM_STATUSES = ("ongoing", "resolved")
SYMPTOM_TRENDS = ("new", "worsening", "stable", "improving", "resolved")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)

    # NULL when not provided. Unique index allows many NULLs,
    # but each real external ID can exist only once.
    external_id = Column(String(64), nullable=True)

    name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=True)
    language = Column(String(50), default="English")
    medical_history = Column(Text, nullable=True)
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc)
    )

    calls = relationship(
        "Call",
        back_populates="patient",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_patients_external_id", "external_id", unique=True),
    )


class Call(Base):
    __tablename__ = "calls"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(
        Integer,
        ForeignKey("patients.id"),
        nullable=False
    )

    # Idempotency key from the client. NULL for legacy calls.
    client_call_id = Column(String(64), nullable=True)
    # SHA-256 of the canonical request payload, used to detect key reuse.
    payload_hash = Column(String(64), nullable=True)
    # "completed" or "interrupted"
    status = Column(
        String(20),
        nullable=False,
        default="completed",
        server_default="completed"
    )

    transcript = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc)
    )

    patient = relationship("Patient", back_populates="calls")
    symptoms = relationship(
        "Symptom",
        back_populates="call",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        # Same client_call_id cannot be saved twice for one patient.
        # Rows with NULL client_call_id are not restricted.
        Index(
            "uq_calls_patient_client_call_id",
            "patient_id", "client_call_id",
            unique=True
        ),
    )


class Symptom(Base):
    __tablename__ = "symptoms"

    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(
        Integer,
        ForeignKey("calls.id"),
        nullable=False
    )
    name = Column(String(100), nullable=False)
    severity = Column(String(30), nullable=True)
    duration = Column(String(100), nullable=True)
    medication_adherence = Column(String(30), nullable=True)

    # "ongoing" or "resolved"
    status = Column(
        String(20),
        nullable=False,
        default="ongoing",
        server_default="ongoing"
    )
    # new / worsening / stable / improving / resolved; NULL = not recorded
    trend = Column(String(20), nullable=True)

    call = relationship("Call", back_populates="symptoms")