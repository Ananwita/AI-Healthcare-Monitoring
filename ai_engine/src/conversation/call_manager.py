"""
Chunk B8 — CallManager: one simple entry point for a whole call.
Used by the CLI (main.py) and by Member 1's voice app.

    manager = CallManager()
    manager.startup()                                   # recover/retry old calls
    session, greeting = manager.start_call("P001", "Asha")
    reply = manager.handle_patient_message(session, "I have a fever.")
    ...                                                 # until reply.done
    result = manager.end_call(session)                  # result.saved only if backend confirmed
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from src.backend.client import BackendClient
from src.conversation.adaptive_questions import (
    closing_message,
    next_question,
    opening_message,
)
from src.conversation.finalize import (
    DRAFTS_DIR,
    PENDING_DIR,
    FinalizeResult,
    delete_draft,
    finalize_call,
    recover_interrupted_calls,
    retry_pending_calls,
    save_draft,
)
from src.conversation.session import CallSession

NOT_HEARD_MESSAGE = "Sorry, I didn't catch that. Could you say it again?"


@dataclass
class AssistantReply:
    text: str                       # what the assistant says next
    done: bool                      # True = closing message; end the call
    extraction_failed: bool = False # AI could not process the last answer


class CallManager:
    def __init__(
        self,
        client=None,
        extractor: Optional[Callable] = None,
        pending_dir: Path = PENDING_DIR,
        drafts_dir: Path = DRAFTS_DIR,
    ):
        self.client = client if client is not None else BackendClient()
        self.extractor = extractor          # None = real LLM extraction
        self.pending_dir = Path(pending_dir)
        self.drafts_dir = Path(drafts_dir)

    # ── App start ────────────────────────────────────────────────────────────

    def startup(self) -> list[FinalizeResult]:
        """Save leftover interrupted calls, then resend pending saves."""
        results = recover_interrupted_calls(self.client, self.drafts_dir, self.pending_dir)
        results += retry_pending_calls(self.client, self.pending_dir)
        return results

    # ── Call lifecycle ───────────────────────────────────────────────────────

    def start_call(
        self,
        external_id: str,
        name: str,
        age: Optional[int] = None,
        language: Optional[str] = None,
        medical_history: Optional[str] = None,
    ) -> tuple[CallSession, str]:
        """
        Identify the patient, load ALL-history symptom state, return the
        session and the greeting. Raises BackendError subclasses if the
        backend is down or the patient data conflicts (no session created).
        """
        patient = self.client.create_or_get_patient(
            external_id, name, age=age, language=language, medical_history=medical_history
        )
        patient_id = patient["patient_id"]
        history = self.client.get_symptom_state(patient_id)

        session = CallSession(
            patient_id=patient_id,
            external_id=external_id,
            previous_symptoms=history,
        )
        opening = opening_message(session)
        session.add_assistant_message(opening.text, opening.symptom, opening.field)
        save_draft(session, self.drafts_dir)
        return session, opening.text

    def handle_patient_message(self, session: CallSession, text: str) -> AssistantReply:
        """Process one patient answer and return what the assistant says next."""
        result = session.process_patient_message(text, self.extractor)

        if result.ignored:
            # Nothing recorded; the previous question stays linked for the next answer
            again = f" {session.last_question}" if session.last_question else ""
            return AssistantReply(NOT_HEARD_MESSAGE + again, done=False)

        planned = next_question(session)
        if planned is not None:
            session.add_assistant_message(planned.text, planned.symptom, planned.field)
            save_draft(session, self.drafts_dir)
            return AssistantReply(planned.text, done=False,
                                  extraction_failed=result.extraction_failed)

        closing = closing_message(session)
        session.add_assistant_message(closing)
        save_draft(session, self.drafts_dir)
        return AssistantReply(closing, done=True, extraction_failed=result.extraction_failed)

    def end_call(self, session: CallSession, interrupted: bool = False) -> FinalizeResult:
        """
        Save the call. interrupted=True for dropped connections.
        A call where the patient never spoke is not saved.
        """
        if not any(t.role == "patient" for t in session.turns):
            session.end()
            delete_draft(session.client_call_id, self.drafts_dir)
            return FinalizeResult(
                client_call_id=session.client_call_id, saved=False,
                message="Nothing to save: the patient did not say anything.",
            )
        status = "interrupted" if interrupted else "completed"
        return finalize_call(session, self.client, status, self.pending_dir, self.drafts_dir)