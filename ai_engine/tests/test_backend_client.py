"""Chunk B6: backend client with a fake HTTP server (httpx.MockTransport). No real backend."""

import json

import httpx
import pytest

from src.backend.client import (
    BackendClient,
    BackendError,
    BackendUnavailableError,
    CallConflictError,
    PatientConflictError,
)


class FakeServer:
    """Replies from a queue; records every request."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        status, data = reply
        return httpx.Response(status, json=data)


def make_client(server, max_retries=2):
    sleeps = []
    client = BackendClient(
        base_url="http://test",
        transport=httpx.MockTransport(server),
        max_retries=max_retries,
        sleep=sleeps.append,          # no real waiting in tests
    )
    client.sleeps = sleeps
    return client


SAVED = {"call_id": 7, "patient_id": 1, "client_call_id": "c-1",
         "status": "completed", "symptoms_saved": 2, "duplicate": False}
PAYLOAD = {"client_call_id": "c-1", "transcript": "t", "symptoms": []}


# ── Patients ─────────────────────────────────────────────────────────────────

def test_create_patient_sends_only_given_fields():
    server = FakeServer((200, {"patient_id": 1, "created": True}))
    result = make_client(server).create_or_get_patient("P001", "Asha")
    assert result["created"] is True
    assert server.requests == [("POST", "/patients", {"external_id": "P001", "name": "Asha"})]


def test_create_patient_with_optional_fields():
    server = FakeServer((200, {"patient_id": 1, "created": True}))
    make_client(server).create_or_get_patient("P001", "Asha", age=30, language="Hindi")
    assert server.requests[0][2] == {"external_id": "P001", "name": "Asha", "age": 30, "language": "Hindi"}


def test_existing_patient_returned():
    server = FakeServer((200, {"patient_id": 1, "created": False}))
    assert make_client(server).create_or_get_patient("P001", "Asha")["created"] is False


def test_patient_conflict_raises_and_is_not_retried():
    server = FakeServer((409, {"detail": {"conflicting_fields": ["name"]}}))
    client = make_client(server)
    with pytest.raises(PatientConflictError) as err:
        client.create_or_get_patient("P001", "Other")
    assert err.value.detail == {"conflicting_fields": ["name"]}
    assert len(server.requests) == 1


def test_get_patient_by_external_found_and_missing():
    server = FakeServer((200, {"id": 1, "external_id": "P001"}), (404, {"detail": "Patient not found"}))
    client = make_client(server)
    assert client.get_patient_by_external("P001")["id"] == 1
    assert client.get_patient_by_external("NOPE") is None
    assert server.requests[0][1] == "/patients/by-external/P001"


# ── History reads ────────────────────────────────────────────────────────────

def test_history_endpoints_paths():
    server = FakeServer(*[(200, [])] * 4, (200, {"patient": {}}), (200, {"id": 1}))
    client = make_client(server)
    client.get_symptom_state(1)
    client.get_calls(1)
    client.get_symptoms(1)
    client.get_symptom_state(2)
    client.get_context(1)
    client.get_patient(1)
    assert [p for _, p, _ in server.requests] == [
        "/patients/1/symptom-state", "/patients/1/calls", "/patients/1/symptoms",
        "/patients/2/symptom-state", "/patients/1/context", "/patients/1",
    ]


def test_get_404_raises_backend_error():
    server = FakeServer((404, {"detail": "Patient not found"}))
    with pytest.raises(BackendError) as err:
        make_client(server).get_context(99)
    assert err.value.status_code == 404
    assert not isinstance(err.value, BackendUnavailableError)


def test_get_retries_on_500_then_succeeds():
    server = FakeServer((500, {}), (200, []))
    client = make_client(server)
    assert client.get_symptom_state(1) == []
    assert len(server.requests) == 2
    assert client.sleeps == [0.5]


# ── Save call: success and duplicate ─────────────────────────────────────────

def test_save_call_success():
    server = FakeServer((200, SAVED))
    result = make_client(server).save_call(1, PAYLOAD)
    assert (result.call_id, result.duplicate, result.symptoms_saved) == (7, False, 2)
    assert server.requests == [("POST", "/patients/1/calls", PAYLOAD)]


def test_save_call_duplicate_reported():
    server = FakeServer((200, {**SAVED, "duplicate": True}))
    assert make_client(server).save_call(1, PAYLOAD).duplicate is True


def test_save_call_requires_client_call_id():
    with pytest.raises(ValueError):
        make_client(FakeServer()).save_call(1, {"transcript": "t"})


# ── Save call: retries are safe ──────────────────────────────────────────────

def test_save_retries_network_error_with_same_payload():
    server = FakeServer(httpx.ConnectError("refused"), (200, {**SAVED, "duplicate": True}))
    client = make_client(server)
    result = client.save_call(1, PAYLOAD)
    assert result.call_id == 7
    assert len(server.requests) == 2
    assert server.requests[0][2] == server.requests[1][2]      # identical payload resent


def test_save_retries_timeout():
    server = FakeServer(httpx.ReadTimeout("slow"), (200, SAVED))
    assert make_client(server).save_call(1, PAYLOAD).call_id == 7


def test_save_retries_on_503():
    server = FakeServer((503, {}), (200, SAVED))
    assert make_client(server).save_call(1, PAYLOAD).call_id == 7


def test_save_gives_up_after_all_retries():
    server = FakeServer(httpx.ConnectError("x"), httpx.ConnectError("x"), httpx.ConnectError("x"))
    client = make_client(server)
    with pytest.raises(BackendUnavailableError):
        client.save_call(1, PAYLOAD)
    assert len(server.requests) == 3
    assert client.sleeps == [0.5, 1.0]


def test_save_conflict_not_retried():
    server = FakeServer((409, {"detail": {"call_id": 7}}))
    with pytest.raises(CallConflictError) as err:
        make_client(server).save_call(1, PAYLOAD)
    assert err.value.detail == {"call_id": 7}
    assert len(server.requests) == 1


def test_save_validation_error_not_retried():
    server = FakeServer((422, {"detail": [{"msg": "bad status"}]}))
    with pytest.raises(BackendError) as err:
        make_client(server).save_call(1, PAYLOAD)
    assert err.value.status_code == 422
    assert len(server.requests) == 1


def test_unknown_patient_on_save_not_retried():
    server = FakeServer((404, {"detail": "Patient not found"}))
    with pytest.raises(BackendError):
        make_client(server).save_call(99, PAYLOAD)
    assert len(server.requests) == 1


# ── Health ───────────────────────────────────────────────────────────────────

def test_health_true_and_false():
    assert make_client(FakeServer((200, {"message": "ok"}))).health() is True
    down = FakeServer(httpx.ConnectError("x"), httpx.ConnectError("x"))
    assert make_client(down, max_retries=1).health() is False


def test_context_manager_closes():
    with make_client(FakeServer((200, {"message": "ok"}))) as client:
        assert client.health()