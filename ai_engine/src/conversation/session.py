"""
Chunk B4 — CallSession: the memory of ONE call.

- Keeps the transcript (assistant + patient turns).
- Remembers the last question (which symptom / which field) for B2 context.
- Merges extracted observations into one record per symptom for this call.
- Applies B3 trend/status rules at the end.
- Can be saved/restored as JSON (used for interrupted-call recovery in B7).
"""

import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from pydantic import BaseModel, Field

from src.analysis.trend import (
    SymptomUpdate,
    determine_update,
    unmentioned_ongoing_symptoms,
)
from src.llm import extraction
from src.llm.extraction import (
    ConversationTurn,
    ExtractionContext,
    ExtractionResult,
    PreviousSymptom,
    SymptomObservation,
)

# Fields merged per symptom. A newer non-empty value replaces an older one;
# an empty (None) value never erases a known value.
REPLACE_FIELDS = ("severity", "duration", "medication_adherence", "reported_change")
ASKABLE_FIELDS = ("severity", "duration", "medication_adherence", "reported_change", "details")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize(name) -> str:
    return "_".join(str(name or "").strip().lower().replace("-", " ").split())


# ── Data models ──────────────────────────────────────────────────────────────

class CallSymptom(BaseModel):
    """Everything learned about one symptom during THIS call."""
    name: str
    severity: Optional[str] = None
    duration: Optional[str] = None
    medication_adherence: Optional[str] = None
    reported_change: Optional[str] = None
    details: Optional[str] = None
    evidence: list[str] = []
    first_turn: int
    last_turn: int


class AskedQuestion(BaseModel):
    symptom: Optional[str] = None
    field: Optional[str] = None
    text: str


class TurnResult(BaseModel):
    """What one patient message produced."""
    observations: list[SymptomObservation] = []
    extraction_failed: bool = False
    ignored: bool = False          # empty message, nothing processed


# ── Session ──────────────────────────────────────────────────────────────────

class CallSession(BaseModel):
    patient_id: int
    external_id: Optional[str] = None
    client_call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = Field(default_factory=_now_iso)

    previous_symptoms: list[PreviousSymptom] = []
    turns: list[ConversationTurn] = []
    symptoms: dict[str, CallSymptom] = {}
    asked: list[AskedQuestion] = []

    last_question: Optional[str] = None
    last_question_symptom: Optional[str] = None
    last_question_field: Optional[str] = None

    extraction_failures: int = 0
    ended: bool = False

    # ── Assistant side ───────────────────────────────────────────────────────

    def add_assistant_message(
        self,
        text: str,
        symptom: Optional[str] = None,
        field: Optional[str] = None,
    ) -> None:
        """Record what the assistant said. symptom/field describe the question, if any."""
        self._ensure_active()
        symptom = _normalize(symptom) or None
        if field is not None and field not in ASKABLE_FIELDS:
            raise ValueError(f"Unknown question field '{field}'. Use one of {ASKABLE_FIELDS}.")

        self.turns.append(ConversationTurn(role="assistant", text=text))
        self.last_question = text
        self.last_question_symptom = symptom
        self.last_question_field = field
        self.asked.append(AskedQuestion(symptom=symptom, field=field, text=text))

    # ── Patient side ─────────────────────────────────────────────────────────

    def build_context(self) -> ExtractionContext:
        """Context for the NEXT patient message (turns so far, last question, history)."""
        return ExtractionContext(
            previous_symptoms=self.previous_symptoms,
            conversation=list(self.turns),
            last_question=self.last_question,
            last_question_symptom=self.last_question_symptom,
            last_question_field=self.last_question_field,
        )

    def process_patient_message(
        self,
        text: str,
        extractor: Optional[Callable[..., Optional[ExtractionResult]]] = None,
    ) -> TurnResult:
        """
        Add the patient's message, extract with context, merge into this call.
        The turn is always kept in the transcript, even if extraction fails.
        """
        self._ensure_active()
        text = (text or "").strip()
        if not text:
            return TurnResult(ignored=True)

        extractor = extractor or extraction.extract_symptoms
        context = self.build_context()          # built BEFORE adding this turn

        self.turns.append(ConversationTurn(role="patient", text=text))
        turn_index = len(self.turns) - 1

        # The question has now been answered (or skipped) -> do not reuse it
        self.last_question = None
        self.last_question_symptom = None
        self.last_question_field = None

        try:
            result = extractor(text, context)
        except Exception as e:  # never crash the call because of the AI layer
            print(f"  [Session] Extraction error: {type(e).__name__}: {e}")
            result = None

        if result is None:
            self.extraction_failures += 1
            return TurnResult(extraction_failed=True)

        for obs in result.symptoms:
            self._merge(obs, turn_index)
        return TurnResult(observations=result.symptoms)

    # ── Merge ────────────────────────────────────────────────────────────────

    def _merge(self, obs: SymptomObservation, turn_index: int) -> None:
        name = _normalize(obs.name)
        current = self.symptoms.get(name)
        if current is None:
            current = CallSymptom(name=name, first_turn=turn_index, last_turn=turn_index)
            self.symptoms[name] = current

        for field in REPLACE_FIELDS:
            value = getattr(obs, field)
            if value is not None:
                setattr(current, field, value)

        if obs.details:
            if not current.details:
                current.details = obs.details
            elif obs.details.lower() not in current.details.lower():
                current.details = f"{current.details}; {obs.details}"

        if obs.evidence and obs.evidence not in current.evidence:
            current.evidence.append(obs.evidence)

        current.last_turn = turn_index

    # ── Queries used by question generation (B5) and saving (B7) ─────────────

    def previous_state(self, symptom: str) -> Optional[PreviousSymptom]:
        key = _normalize(symptom)
        for prev in self.previous_symptoms:
            if _normalize(prev.name) == key:
                return prev
        return None

    def was_asked(self, symptom: Optional[str], field: Optional[str]) -> bool:
        key = _normalize(symptom) or None
        return any(q.symptom == key and q.field == field for q in self.asked)

    def known_value(self, symptom: str, field: str):
        """Value from this call first, otherwise from previous calls (or None)."""
        current = self.symptoms.get(_normalize(symptom))
        if current is not None and getattr(current, field, None) is not None:
            return getattr(current, field)
        prev = self.previous_state(symptom)
        if prev is not None:
            return getattr(prev, f"latest_{field}", None)
        return None

    def mentioned_symptoms(self) -> list[str]:
        return list(self.symptoms.keys())

    def unmentioned_previous_symptoms(self) -> list[str]:
        """Ongoing symptoms from earlier calls not yet mentioned (never auto-resolved)."""
        return unmentioned_ongoing_symptoms(self.previous_symptoms, self.mentioned_symptoms())

    def symptom_updates(self) -> list[SymptomUpdate]:
        """B3 trend/status for every symptom mentioned in this call."""
        return [
            determine_update(self.previous_state(name), symptom)
            for name, symptom in self.symptoms.items()
        ]

    def transcript(self) -> str:
        return "\n".join(
            f"{'Assistant' if t.role == 'assistant' else 'Patient'}: {t.text}"
            for t in self.turns
        )

    # ── Lifecycle and persistence ────────────────────────────────────────────

    def end(self) -> None:
        self.ended = True

    def _ensure_active(self) -> None:
        if self.ended:
            raise RuntimeError("This call has already ended.")

    def to_dict(self) -> dict:
        """JSON-safe snapshot (for drafts / interrupted-call recovery)."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> "CallSession":
        return cls.model_validate(data)