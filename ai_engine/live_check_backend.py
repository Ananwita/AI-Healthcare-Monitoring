"""Manual check for Chunk B6 against the RUNNING backend. Not part of pytest."""

import uuid

from src.backend.client import BackendClient, CallConflictError

with BackendClient() as client:
    print("Backend reachable:", client.health())

    first = client.create_or_get_patient("CHECK-B6", "Check Patient", age=40)
    again = client.create_or_get_patient("CHECK-B6", "Check Patient")
    pid = first["patient_id"]
    print("Patient created:", first["created"], "| second time created:", again["created"],
          "| same id:", pid == again["patient_id"])

    payload = {
        "client_call_id": str(uuid.uuid4()),
        "transcript": "Patient: I have a fever.",
        "summary": "Fever reported.",
        "symptoms": [{"name": "fever", "status": "ongoing", "trend": "new"}],
    }
    saved = client.save_call(pid, payload)
    resent = client.save_call(pid, payload)
    print("First save duplicate:", saved.duplicate, "| resend duplicate:", resent.duplicate,
          "| same call_id:", saved.call_id == resent.call_id)

    try:
        client.save_call(pid, {**payload, "transcript": "changed"})
        print("Conflict check: FAILED (no error)")
    except CallConflictError:
        print("Conflict check: 409 raised correctly")

    print("Calls stored:", len(client.get_calls(pid)))
    print("Symptom state:", client.get_symptom_state(pid))