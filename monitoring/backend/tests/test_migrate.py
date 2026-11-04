"""Tests for backend/migrate.py. Uses temporary copies only, never the real DB."""

import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.migrate import migrate, MigrationError
from backend.models import Patient, Call, Symptom


# Exact pre-A2 schema and the real records from health_monitoring.db
LEGACY_SCHEMA = """
CREATE TABLE patients (
    id INTEGER NOT NULL, name VARCHAR(100) NOT NULL, age INTEGER,
    language VARCHAR(50), medical_history TEXT, created_at DATETIME,
    PRIMARY KEY (id)
);
CREATE TABLE calls (
    id INTEGER NOT NULL, patient_id INTEGER NOT NULL, transcript TEXT,
    summary TEXT, created_at DATETIME, PRIMARY KEY (id),
    FOREIGN KEY(patient_id) REFERENCES patients (id)
);
CREATE TABLE symptoms (
    id INTEGER NOT NULL, call_id INTEGER NOT NULL, name VARCHAR(100) NOT NULL,
    severity VARCHAR(30), duration VARCHAR(100), medication_adherence VARCHAR(30),
    PRIMARY KEY (id), FOREIGN KEY(call_id) REFERENCES calls (id)
);
INSERT INTO patients VALUES
 (1, 'Demo Patient', 55, 'English', 'Demo data for testing', '2026-10-24 10:32:12.029287');
INSERT INTO calls VALUES
 (1, 1, 'I have been feeling tired for two days.', 'Patient reports fatigue.', '2026-10-24 10:33:10.959992'),
 (2, 1, 'string', 'Patient reports ongoing fatigue and needs follow-up to assess symptom progression.', '2026-10-26 07:42:37.834228'),
 (3, 1, 'Patient reports a headache for two days and feeling tired.', 'Patient reports a headache and fatigue lasting two days.', '2026-10-26 07:42:59.247792');
INSERT INTO symptoms VALUES
 (1, 1, 'fatigue', 'mild', '2 days', NULL),
 (2, 3, 'headache', 'moderate', '2 days', NULL),
 (3, 3, 'fatigue', 'mild', '2 days', NULL);
"""

ORIGINAL_QUERIES = {
    "patients": "SELECT id, name, age, language, medical_history, created_at FROM patients ORDER BY id",
    "calls": "SELECT id, patient_id, transcript, summary, created_at FROM calls ORDER BY id",
    "symptoms": "SELECT id, call_id, name, severity, duration, medication_adherence FROM symptoms ORDER BY id",
}


@pytest.fixture
def legacy_db(tmp_path) -> Path:
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(LEGACY_SCHEMA)
    return path


def snapshot(path):
    with sqlite3.connect(path) as conn:
        return {t: conn.execute(q).fetchall() for t, q in ORIGINAL_QUERIES.items()}


def columns(path, table):
    with sqlite3.connect(path) as conn:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


# ---------- Dry run and backup ----------

def test_dry_run_makes_no_changes(legacy_db, tmp_path):
    report = migrate(legacy_db, dry_run=True)
    assert len(report["changes"]) == 8          # 6 columns + 2 indexes
    assert report["applied"] is False
    assert "external_id" not in columns(legacy_db, "patients")
    assert list(tmp_path.glob("*.bak-*")) == []


def test_backup_created_before_migration(legacy_db):
    original = snapshot(legacy_db)
    report = migrate(legacy_db)

    backup = Path(report["backup_path"])
    assert backup.exists()
    assert snapshot(backup) == original                      # same data
    assert "external_id" not in columns(backup, "patients")  # pre-migration schema


# ---------- Existing records preserved ----------

def test_existing_records_preserved(legacy_db):
    original = snapshot(legacy_db)
    report = migrate(legacy_db)

    assert report["applied"] is True
    assert report["rows_before"] == report["rows_after"] == {
        "patients": 1, "calls": 3, "symptoms": 3
    }
    assert snapshot(legacy_db) == original


def test_existing_records_get_safe_defaults(legacy_db):
    migrate(legacy_db)
    with sqlite3.connect(legacy_db) as conn:
        assert conn.execute("SELECT external_id FROM patients").fetchall() == [(None,)]
        assert set(conn.execute(
            "SELECT client_call_id, payload_hash, status FROM calls").fetchall()
        ) == {(None, None, "completed")}
        assert set(conn.execute(
            "SELECT status, trend FROM symptoms").fetchall()
        ) == {("ongoing", None)}


def test_rerun_is_noop(legacy_db):
    migrate(legacy_db)
    second = migrate(legacy_db)
    assert second["changes"] == []
    assert second["applied"] is False
    assert second["backup_path"] is None


def test_empty_string_ids_become_null(legacy_db):
    # Simulate a partially migrated DB that stored '' instead of NULL
    with sqlite3.connect(legacy_db) as conn:
        conn.execute("ALTER TABLE patients ADD COLUMN external_id VARCHAR(64)")
        conn.execute("UPDATE patients SET external_id = ''")
        conn.execute("INSERT INTO patients (id, name, external_id) VALUES (2, 'B', '  ')")

    migrate(legacy_db)
    with sqlite3.connect(legacy_db) as conn:
        assert conn.execute(
            "SELECT external_id FROM patients ORDER BY id").fetchall() == [(None,), (None,)]


# ---------- New constraints ----------

def test_multiple_null_external_ids_allowed(legacy_db):
    migrate(legacy_db)
    with sqlite3.connect(legacy_db) as conn:
        conn.execute("INSERT INTO patients (name) VALUES ('A')")
        conn.execute("INSERT INTO patients (name) VALUES ('B')")
        assert conn.execute(
            "SELECT COUNT(*) FROM patients WHERE external_id IS NULL").fetchone()[0] == 3


def test_duplicate_external_id_rejected(legacy_db):
    migrate(legacy_db)
    with sqlite3.connect(legacy_db) as conn:
        conn.execute("INSERT INTO patients (name, external_id) VALUES ('A', 'P001')")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO patients (name, external_id) VALUES ('B', 'P001')")


def test_multiple_null_client_call_ids_allowed(legacy_db):
    migrate(legacy_db)
    with sqlite3.connect(legacy_db) as conn:
        conn.execute("INSERT INTO calls (patient_id) VALUES (1)")
        conn.execute("INSERT INTO calls (patient_id) VALUES (1)")
        assert conn.execute(
            "SELECT COUNT(*) FROM calls WHERE client_call_id IS NULL").fetchone()[0] == 5


def test_duplicate_client_call_id_same_patient_rejected(legacy_db):
    migrate(legacy_db)
    with sqlite3.connect(legacy_db) as conn:
        conn.execute("INSERT INTO calls (patient_id, client_call_id) VALUES (1, 'c-1')")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO calls (patient_id, client_call_id) VALUES (1, 'c-1')")


def test_same_client_call_id_different_patients_allowed(legacy_db):
    migrate(legacy_db)
    with sqlite3.connect(legacy_db) as conn:
        conn.execute("INSERT INTO patients (id, name) VALUES (2, 'Other')")
        conn.execute("INSERT INTO calls (patient_id, client_call_id) VALUES (1, 'c-1')")
        conn.execute("INSERT INTO calls (patient_id, client_call_id) VALUES (2, 'c-1')")


# ---------- Safety ----------

def test_failure_rolls_back_everything(legacy_db):
    # Duplicate real external IDs make the unique index fail mid-migration
    with sqlite3.connect(legacy_db) as conn:
        conn.execute("ALTER TABLE patients ADD COLUMN external_id VARCHAR(64)")
        conn.execute("UPDATE patients SET external_id = 'DUP'")
        conn.execute("INSERT INTO patients (id, name, external_id) VALUES (2, 'B', 'DUP')")

    with pytest.raises(sqlite3.IntegrityError):
        migrate(legacy_db)

    # Columns added in the failed transaction were rolled back
    assert "client_call_id" not in columns(legacy_db, "calls")
    assert "status" not in columns(legacy_db, "symptoms")


def test_missing_database_raises(tmp_path):
    with pytest.raises(MigrationError):
        migrate(tmp_path / "does_not_exist.db")


def test_wrong_database_raises(tmp_path):
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    with pytest.raises(MigrationError):
        migrate(empty)


# ---------- Compatibility with SQLAlchemy models ----------

def test_models_work_on_migrated_db(legacy_db):
    migrate(legacy_db)
    engine = create_engine(f"sqlite:///{legacy_db.as_posix()}")
    try:
        with Session(engine) as session:
            patient = session.get(Patient, 1)
            assert patient.name == "Demo Patient"
            assert patient.external_id is None
            assert [c.status for c in patient.calls] == ["completed"] * 3
            assert session.query(Symptom).filter_by(status="ongoing").count() == 3
            assert session.query(Call).filter(Call.client_call_id.is_(None)).count() == 3
    finally:
        engine.dispose()