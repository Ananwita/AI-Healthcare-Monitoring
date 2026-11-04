"""
Chunk B7 — Ending a call safely.

Order of operations in finalize_call():
  1. build the payload (deterministic: same session -> identical payload)
  2. write it to data/pending_calls/<client_call_id>.json   (atomic)
  3. delete the draft
  4. send it to the backend
  5. ONLY on 2xx: delete the pending file and report "saved"

Recovery:
  - retry_pending_calls(): resends stored payloads EXACTLY as written.
    Safe even if an earlier attempt succeeded, because the backend
    deduplicates by client_call_id (Chunk A4).
  - recover_interrupted_calls(): leftover drafts (call never finished)
    are saved with status="interrupted".
  - 409 / 422 / 404 are never resent automatically: the file moves to
    pending_calls/failed/ for manual review.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.backend.client import BackendError, BackendUnavailableError
from src.conversation.session import CallSession

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DRAFTS_DIR = DATA_DIR / "drafts"
PENDING_DIR = DATA_DIR / "pending_calls"
FAILED_SUBDIR = "failed"

CALL_STATUSES = ("completed", "interrupted")

SYMPTOM_LABELS = {
    "breathing_difficulty": "Difficulty breathing",
    "sleep_problems": "Sleep trouble",
    "appetite_changes": "Appetite change",
}


# ── Result ───────────────────────────────────────────────────────────────────

@dataclass
class FinalizeResult:
    client_call_id: str
    saved: bool                       # True ONLY when the backend confirmed (2xx)
    call_id: Optional[int] = None
    duplicate: bool = False           # backend already had this exact call
    failed: bool = False              # rejected by backend -> needs manual review
    local_path: Optional[str] = None  # where the call is kept if not saved
    message: str = ""


# ── File helpers ─────────────────────────────────────────────────────────────

def _write_json_atomic(path: Path, data: dict) -> None:
    """Write to a temp file, then rename: never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _move_to_failed(path: Path, record: Optional[dict], error: str) -> Path:
    failed_path = path.parent / FAILED_SUBDIR / path.name
    if record is None:
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, failed_path)
    else:
        record["last_error"] = error
        _write_json_atomic(failed_path, record)
        path.unlink(missing_ok=True)
    return failed_path


# ── Summary and payload ──────────────────────────────────────────────────────

def _label(name: str) -> str:
    return SYMPTOM_LABELS.get(name, name.replace("_", " ").capitalize())


def build_summary(session: CallSession, status: str = "completed") -> str:
    """Deterministic, template-based summary. Only facts the patient stated."""
    updates = {u.name: u for u in session.symptom_updates()}
    sentences = []

    if status == "interrupted":
        sentences.append("Call ended unexpectedly; information is partial.")

    if not session.symptoms:
        sentences.append("No symptoms reported.")

    for name, symptom in session.symptoms.items():
        update = updates[name]
        trend_text = "returned after being resolved" if update.recurred else update.trend
        facts = []
        if symptom.severity:
            facts.append(symptom.severity)
        if symptom.duration:
            facts.append(f"duration: {symptom.duration}")
        if symptom.details:
            facts.append(symptom.details)
        if symptom.medication_adherence:
            facts.append(f"medication: {symptom.medication_adherence.replace('_', ' ')}")
        extra = f" ({'; '.join(facts)})" if facts else ""
        sentences.append(f"{_label(name)}: {trend_text}{extra}.")

    not_discussed = session.unmentioned_previous_symptoms()
    if not_discussed:
        names = ", ".join(_label(n).lower() for n in not_discussed)
        sentences.append(f"Not discussed in this call (status unchanged): {names}.")

    return " ".join(sentences)


def build_call_payload(session: CallSession, status: str = "completed") -> dict:
    """Request body for POST /patients/{id}/calls (matches backend CallCreate)."""
    if status not in CALL_STATUSES:
        raise ValueError(f"status must be one of {CALL_STATUSES}")

    updates = {u.name: u for u in session.symptom_updates()}
    return {
        "client_call_id": session.client_call_id,
        "status": status,
        "transcript": session.transcript(),
        "summary": build_summary(session, status),
        "symptoms": [
            {
                "name": name,
                "severity": s.severity,
                "duration": s.duration,
                "medication_adherence": s.medication_adherence,
                "status": updates[name].status,
                "trend": updates[name].trend,
            }
            for name, s in session.symptoms.items()
        ],
    }


# ── Drafts (for interrupted calls) ───────────────────────────────────────────

def save_draft(session: CallSession, drafts_dir: Path = DRAFTS_DIR) -> Path:
    """Call after every turn. Lets an interrupted call be recovered later."""
    path = Path(drafts_dir) / f"{session.client_call_id}.json"
    _write_json_atomic(path, session.to_dict())
    return path


def delete_draft(client_call_id: str, drafts_dir: Path = DRAFTS_DIR) -> None:
    (Path(drafts_dir) / f"{client_call_id}.json").unlink(missing_ok=True)


# ── Sending ──────────────────────────────────────────────────────────────────

def _send_pending(path: Path, client) -> FinalizeResult:
    """Send one stored pending call exactly as written."""
    client_call_id = path.stem
    try:
        record = _read_json(path)
        patient_id = record["patient_id"]
        payload = record["payload"]
    except Exception as e:
        failed_path = _move_to_failed(path, None, f"Unreadable file: {e}")
        return FinalizeResult(client_call_id, saved=False, failed=True,
                              local_path=str(failed_path),
                              message="Stored call file is unreadable; moved for manual review.")

    try:
        result = client.save_call(patient_id, payload)
    except BackendUnavailableError as e:
        record["attempts"] = record.get("attempts", 0) + 1
        record["last_error"] = str(e)
        _write_json_atomic(path, record)
        return FinalizeResult(client_call_id, saved=False, local_path=str(path),
                              message="Backend unavailable. Call kept on this computer and will be retried.")
    except BackendError as e:          # 409 conflict, 422 invalid, 404 unknown patient
        failed_path = _move_to_failed(path, record, f"{e} | detail={e.detail}")
        return FinalizeResult(client_call_id, saved=False, failed=True,
                              local_path=str(failed_path),
                              message="Backend rejected the call. Kept for manual review.")
    except Exception as e:             # unexpected: keep it, never lose the call
        record["attempts"] = record.get("attempts", 0) + 1
        record["last_error"] = f"{type(e).__name__}: {e}"
        _write_json_atomic(path, record)
        return FinalizeResult(client_call_id, saved=False, local_path=str(path),
                              message="Unexpected error. Call kept on this computer and will be retried.")

    path.unlink(missing_ok=True)       # confirmed by backend -> safe to remove
    return FinalizeResult(
        client_call_id, saved=True, call_id=result.call_id, duplicate=result.duplicate,
        message="Call already saved earlier (duplicate ignored)." if result.duplicate else "Call saved.",
    )


def finalize_call(
    session: CallSession,
    client,
    status: str = "completed",
    pending_dir: Path = PENDING_DIR,
    drafts_dir: Path = DRAFTS_DIR,
) -> FinalizeResult:
    """End the call and save it. Never reports saved unless the backend confirmed."""
    if not session.ended:
        session.end()

    payload = build_call_payload(session, status)
    path = Path(pending_dir) / f"{session.client_call_id}.json"
    _write_json_atomic(path, {
        "patient_id": session.patient_id,
        "payload": payload,
        "attempts": 0,
        "last_error": None,
    })
    delete_draft(session.client_call_id, drafts_dir)
    return _send_pending(path, client)


def retry_pending_calls(client, pending_dir: Path = PENDING_DIR) -> list[FinalizeResult]:
    """Resend every stored pending call (oldest name order). Safe to run any time."""
    pending_dir = Path(pending_dir)
    if not pending_dir.exists():
        return []
    return [_send_pending(path, client) for path in sorted(pending_dir.glob("*.json"))]


def recover_interrupted_calls(
    client,
    drafts_dir: Path = DRAFTS_DIR,
    pending_dir: Path = PENDING_DIR,
) -> list[FinalizeResult]:
    """
    Save leftover drafts as interrupted calls. Run at app start, BEFORE
    retry_pending_calls().
    - Draft whose call already reached pending/failed -> just delete the draft.
    - Draft without any patient message -> nothing to save, delete it.
    """
    drafts_dir, pending_dir = Path(drafts_dir), Path(pending_dir)
    if not drafts_dir.exists():
        return []

    results = []
    for path in sorted(drafts_dir.glob("*.json")):
        try:
            session = CallSession.from_dict(_read_json(path))
        except Exception as e:
            _move_to_failed(path, None, f"Unreadable draft: {e}")
            continue

        cid = session.client_call_id
        already_finalized = (
            (pending_dir / f"{cid}.json").exists()
            or (pending_dir / FAILED_SUBDIR / f"{cid}.json").exists()
        )
        has_patient_turn = any(t.role == "patient" for t in session.turns)

        if already_finalized or not has_patient_turn:
            path.unlink(missing_ok=True)
            continue

        results.append(finalize_call(session, client, "interrupted", pending_dir, drafts_dir))
    return results