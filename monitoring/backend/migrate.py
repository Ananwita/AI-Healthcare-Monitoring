"""
Safe, additive, re-runnable migration for health_monitoring.db (Chunk A2).

- Creates a backup copy BEFORE changing anything.
- Only ADDS columns and indexes. Never deletes or rewrites existing rows
  (except converting '' to NULL in the new ID columns).
- Runs inside one transaction: any error -> full rollback.
- Verifies row counts are unchanged.

Usage (from the monitoring folder):
    python -m backend.migrate --dry-run     # show planned changes only
    python -m backend.migrate               # backup + migrate
"""

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "health_monitoring.db"

TABLES = ("patients", "calls", "symptoms")

# (table, column, column definition)
NEW_COLUMNS = [
    ("patients", "external_id", "VARCHAR(64)"),
    ("calls", "client_call_id", "VARCHAR(64)"),
    ("calls", "payload_hash", "VARCHAR(64)"),
    ("calls", "status", "VARCHAR(20) NOT NULL DEFAULT 'completed'"),
    ("symptoms", "status", "VARCHAR(20) NOT NULL DEFAULT 'ongoing'"),
    ("symptoms", "trend", "VARCHAR(20)"),
]

# Unique indexes allow multiple NULLs in SQLite, so legacy rows are unaffected.
NEW_INDEXES = [
    ("ix_patients_external_id",
     "CREATE UNIQUE INDEX IF NOT EXISTS ix_patients_external_id "
     "ON patients (external_id)"),
    ("uq_calls_patient_client_call_id",
     "CREATE UNIQUE INDEX IF NOT EXISTS uq_calls_patient_client_call_id "
     "ON calls (patient_id, client_call_id)"),
]


class MigrationError(Exception):
    pass


# ---------- Helpers ----------

def _columns(conn, table):
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _indexes(conn):
    return {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
    }


def _row_counts(conn):
    return {
        t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in TABLES
    }


def _check_tables(conn):
    existing = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    missing = [t for t in TABLES if t not in existing]
    if missing:
        raise MigrationError(f"Missing tables: {missing}. Is this the right database?")


def plan(conn):
    """Return the list of changes still needed (empty = up to date)."""
    _check_tables(conn)
    changes = []
    for table, column, ddl in NEW_COLUMNS:
        if column not in _columns(conn, table):
            changes.append(f"ADD COLUMN {table}.{column} {ddl}")
    existing_idx = _indexes(conn)
    for name, _ in NEW_INDEXES:
        if name not in existing_idx:
            changes.append(f"CREATE UNIQUE INDEX {name}")
    return changes


def backup_database(db_path: Path) -> Path:
    """Consistent copy using SQLite's backup API."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup_path = db_path.with_name(f"{db_path.name}.bak-{stamp}")
    src = sqlite3.connect(db_path)
    dst = sqlite3.connect(backup_path)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()

    # Verify backup has the same row counts as the source
    with sqlite3.connect(db_path) as a, sqlite3.connect(backup_path) as b:
        if _row_counts(a) != _row_counts(b):
            raise MigrationError("Backup verification failed: row counts differ.")
    return backup_path


# ---------- Main migration ----------

def migrate(db_path=DEFAULT_DB_PATH, dry_run=False, make_backup=True) -> dict:
    db_path = Path(db_path)
    if not db_path.exists():
        raise MigrationError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path, isolation_level=None)  # manual transactions
    try:
        changes = plan(conn)
        before = _row_counts(conn)
    finally:
        conn.close()

    report = {
        "db_path": str(db_path),
        "changes": changes,
        "rows_before": before,
        "rows_after": before,
        "backup_path": None,
        "applied": False,
    }

    if dry_run or not changes:
        return report

    if make_backup:
        report["backup_path"] = str(backup_database(db_path))

    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        conn.execute("BEGIN")

        for table, column, ddl in NEW_COLUMNS:
            if column not in _columns(conn, table):
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

        # Missing IDs must be NULL (not empty strings) so unique indexes ignore them.
        conn.execute("UPDATE patients SET external_id = NULL WHERE TRIM(external_id) = ''")
        conn.execute("UPDATE calls SET client_call_id = NULL WHERE TRIM(client_call_id) = ''")

        for _, sql in NEW_INDEXES:
            conn.execute(sql)

        after = _row_counts(conn)
        if after != before:
            raise MigrationError(f"Row counts changed: {before} -> {after}")

        conn.execute("COMMIT")
        report["rows_after"] = after
        report["applied"] = True
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()

    return report


def main():
    parser = argparse.ArgumentParser(description="Migrate health_monitoring.db")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to SQLite DB")
    parser.add_argument("--dry-run", action="store_true", help="Show changes only")
    args = parser.parse_args()

    report = migrate(args.db, dry_run=args.dry_run)

    print(f"Database : {report['db_path']}")
    if not report["changes"]:
        print("Already up to date. Nothing to do.")
        return
    print("Changes  :")
    for change in report["changes"]:
        print(f"  - {change}")
    if args.dry_run:
        print("Dry run  : no changes made.")
        return
    print(f"Backup   : {report['backup_path']}")
    print(f"Rows     : {report['rows_before']} -> {report['rows_after']}")
    print("Result   : migration applied successfully.")


if __name__ == "__main__":
    main()