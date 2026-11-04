"""
Chunk B6 — HTTP client for Member 3's FastAPI backend.

Safety rules:
- A call counts as SAVED only when the backend answers 2xx.
- Network errors, timeouts and 5xx are retried. save_call is retried with
  the SAME payload (same client_call_id), so the backend's idempotency
  (Chunk A4) guarantees no duplicate call is created.
- 409 Conflict and other 4xx errors are never retried.
- If every attempt fails, BackendUnavailableError is raised so the caller
  can keep the call locally and retry later (Chunk B7).
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import httpx
from dotenv import load_dotenv

# Always read ai_engine/.env, whichever folder the program is started from
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8001")
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_RETRIES = 2          # retries AFTER the first attempt
DEFAULT_BACKOFF_SECONDS = 0.5    # 0.5s, 1s, ...


# ── Errors ───────────────────────────────────────────────────────────────────

class BackendError(Exception):
    """Backend answered with an error that retrying will not fix (4xx)."""

    def __init__(self, message: str, status_code: Optional[int] = None, detail=None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class BackendUnavailableError(BackendError):
    """Backend could not be reached (network error, timeout, 5xx) after all retries."""


class PatientConflictError(BackendError):
    """Same external_id already exists with different patient data (409)."""


class CallConflictError(BackendError):
    """Same client_call_id already used with different call data (409)."""


# ── Results ──────────────────────────────────────────────────────────────────

@dataclass
class SaveResult:
    call_id: int
    patient_id: int
    client_call_id: Optional[str]
    status: str
    symptoms_saved: int
    duplicate: bool          # True = backend already had this exact call


# ── Client ───────────────────────────────────────────────────────────────────

class BackendClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BACKEND_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        transport: Optional[httpx.BaseTransport] = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ── Core request with retry ──────────────────────────────────────────────

    def _request(self, method: str, path: str, json=None) -> httpx.Response:
        """
        Send a request. Retries only on network errors, timeouts and 5xx.
        Returns the response for any status < 500 (caller decides what it means).
        """
        last_problem = None

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                self._sleep(self.backoff_seconds * attempt)
            try:
                response = self._http.request(method, path, json=json)
            except httpx.TransportError as e:          # includes timeouts
                last_problem = f"{type(e).__name__}: {e}"
                continue

            if response.status_code >= 500:
                last_problem = f"HTTP {response.status_code}"
                continue
            return response

        raise BackendUnavailableError(
            f"Backend unavailable after {self.max_retries + 1} attempt(s) "
            f"({method} {path}): {last_problem}"
        )

    @staticmethod
    def _detail(response: httpx.Response):
        try:
            return response.json().get("detail")
        except Exception:
            return response.text

    def _raise_for_status(self, response: httpx.Response, method: str, path: str) -> None:
        if response.is_success:
            return
        raise BackendError(
            f"{method} {path} failed with HTTP {response.status_code}",
            status_code=response.status_code,
            detail=self._detail(response),
        )

    def _get(self, path: str):
        response = self._request("GET", path)
        self._raise_for_status(response, "GET", path)
        return response.json()

    # ── Health ───────────────────────────────────────────────────────────────

    def health(self) -> bool:
        """True if the backend answers. Never raises."""
        try:
            return self._request("GET", "/").is_success
        except BackendError:
            return False

    # ── Patients ─────────────────────────────────────────────────────────────

    def create_or_get_patient(
        self,
        external_id: str,
        name: str,
        age: Optional[int] = None,
        language: Optional[str] = None,
        medical_history: Optional[str] = None,
    ) -> dict:
        """
        Idempotent (Chunk A3). Returns the backend response, which contains
        patient_id and created (True = new, False = already existed).
        Only fields that are given are sent, so a returning patient can be
        identified with external_id + name alone.
        """
        payload = {"external_id": external_id, "name": name}
        if age is not None:
            payload["age"] = age
        if language is not None:
            payload["language"] = language
        if medical_history is not None:
            payload["medical_history"] = medical_history

        response = self._request("POST", "/patients", json=payload)
        if response.status_code == 409:
            raise PatientConflictError(
                f"Patient '{external_id}' already exists with different data.",
                status_code=409, detail=self._detail(response),
            )
        self._raise_for_status(response, "POST", "/patients")
        return response.json()

    def get_patient(self, patient_id: int) -> dict:
        return self._get(f"/patients/{patient_id}")

    def get_patient_by_external(self, external_id: str) -> Optional[dict]:
        """Patient dict, or None if no patient has this external_id."""
        path = f"/patients/by-external/{external_id}"
        response = self._request("GET", path)
        if response.status_code == 404:
            return None
        self._raise_for_status(response, "GET", path)
        return response.json()

    # ── History ──────────────────────────────────────────────────────────────

    def get_context(self, patient_id: int) -> dict:
        return self._get(f"/patients/{patient_id}/context")

    def get_symptom_state(self, patient_id: int) -> list[dict]:
        return self._get(f"/patients/{patient_id}/symptom-state")

    def get_calls(self, patient_id: int) -> list[dict]:
        return self._get(f"/patients/{patient_id}/calls")

    def get_symptoms(self, patient_id: int) -> list[dict]:
        return self._get(f"/patients/{patient_id}/symptoms")

    # ── Save call ────────────────────────────────────────────────────────────

    def save_call(self, patient_id: int, payload: dict) -> SaveResult:
        """
        Save a call. payload must contain client_call_id so retries are safe.
        Returns SaveResult ONLY when the backend confirms with 2xx.
        """
        if not payload.get("client_call_id"):
            raise ValueError("save_call requires payload['client_call_id'] for safe retries.")

        path = f"/patients/{patient_id}/calls"
        response = self._request("POST", path, json=payload)

        if response.status_code == 409:
            raise CallConflictError(
                f"client_call_id '{payload['client_call_id']}' was already used with different data.",
                status_code=409, detail=self._detail(response),
            )
        self._raise_for_status(response, "POST", path)

        body = response.json()
        return SaveResult(
            call_id=body["call_id"],
            patient_id=body["patient_id"],
            client_call_id=body.get("client_call_id"),
            status=body.get("status", "completed"),
            symptoms_saved=body.get("symptoms_saved", 0),
            duplicate=bool(body.get("duplicate", False)),
        )