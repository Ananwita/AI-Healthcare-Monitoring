import json
import re
from typing import Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)

from src.config import (
    GROQ_MODEL,
    GROQ_REASONING_EFFORT,
    SUPPORTED_SYMPTOMS,
    SEVERITY_LEVELS,
    MEDICATION_ADHERENCE_LEVELS,
    REPORTED_CHANGES,
    MAX_RETRIES,
    MAX_TOKENS,
)
from src.llm.client import get_client


# ── Helpers ──────────────────────────────────────────────────────────────────

# Values the LLM may use to mean "not stated". Always stored as None.
_EMPTY_VALUES = {"", "null", "none", "unknown", "n/a", "na", "not mentioned", "not stated"}


def _normalize_name(v: str) -> str:
    """'Breathing Difficulty' -> 'breathing_difficulty'"""
    return "_".join(str(v).strip().lower().replace("-", " ").split())


def _empty_to_none(v):
    if v is None:
        return None
    text = str(v).strip()
    return None if text.lower() in _EMPTY_VALUES else text


# ── Pydantic schema ──────────────────────────────────────────────────────────

class SymptomObservation(BaseModel):
    """
    One symptom from the patient's words.
    Every field except `name` is optional: None means the patient did not say it.
    The AI must never fill a value the patient did not provide.
    """
    name: str
    severity: Optional[str] = None             # mild | moderate | severe
    duration: Optional[str] = None             # e.g. "since yesterday", "2 days"
    medication_adherence: Optional[str] = None # taking | missed | not_taking
    reported_change: Optional[str] = None      # new | worse | better | same | gone
    details: Optional[str] = None              # e.g. "temperature 101°F"
    evidence: Optional[str] = None             # patient's own words

    @field_validator("name")
    @classmethod
    def name_must_be_supported(cls, v: str) -> str:
        v = _normalize_name(v)
        if v not in SUPPORTED_SYMPTOMS:
            raise ValueError(
                f"'{v}' is not in the supported symptoms list.\n"
                f"Supported: {SUPPORTED_SYMPTOMS}"
            )
        return v

    @field_validator("severity", mode="before")
    @classmethod
    def severity_must_be_valid(cls, v):
        v = _empty_to_none(v)
        if v is None:
            return None
        v = v.lower()
        if v not in SEVERITY_LEVELS:
            raise ValueError(
                f"'{v}' is not a valid severity.\n"
                f"Valid values: {SEVERITY_LEVELS}"
            )
        return v

    @field_validator("medication_adherence", mode="before")
    @classmethod
    def adherence_must_be_valid(cls, v):
        v = _empty_to_none(v)
        if v is None:
            return None
        v = _normalize_name(v)
        if v not in MEDICATION_ADHERENCE_LEVELS:
            raise ValueError(
                f"'{v}' is not a valid medication adherence.\n"
                f"Valid values: {MEDICATION_ADHERENCE_LEVELS}"
            )
        return v

    @field_validator("reported_change", mode="before")
    @classmethod
    def change_must_be_valid(cls, v):
        v = _empty_to_none(v)
        if v is None:
            return None
        v = v.lower()
        if v not in REPORTED_CHANGES:
            raise ValueError(
                f"'{v}' is not a valid reported change.\n"
                f"Valid values: {REPORTED_CHANGES}"
            )
        return v

    @field_validator("duration", "details", "evidence", mode="before")
    @classmethod
    def blank_text_is_none(cls, v):
        return _empty_to_none(v)


class ExtractionResult(BaseModel):
    symptoms: list[SymptomObservation] = []
    # Symptom names the LLM returned that are not supported (kept for logging)
    dropped_symptoms: list[str] = []
    # Context mode only: symptoms removed because they were not supported by
    # the NEW patient statement (e.g. repeated from earlier turns)
    ungrounded_symptoms: list[str] = []

    @model_validator(mode="before")
    @classmethod
    def drop_unsupported_symptoms(cls, data):
        """
        Remove unsupported symptoms one by one instead of failing the whole
        extraction (e.g. 'fever and stomach ache' still keeps fever).
        Invalid severity/adherence/change values still raise -> retry.
        """
        if not isinstance(data, dict):
            return data

        kept = []
        dropped = list(data.get("dropped_symptoms") or [])

        for item in data.get("symptoms") or []:
            name = item.get("name") if isinstance(item, dict) else None
            if isinstance(name, str) and _normalize_name(name) in SUPPORTED_SYMPTOMS:
                kept.append(item)
            else:
                dropped.append(str(name))

        return {**data, "symptoms": kept, "dropped_symptoms": dropped}


# ── Context (Chunk B2) ───────────────────────────────────────────────────────

MAX_CONTEXT_TURNS = 8


class PreviousSymptom(BaseModel):
    """One entry from the backend's /symptom-state (extra keys ignored)."""
    model_config = ConfigDict(extra="ignore")

    name: str
    status: Optional[str] = None
    latest_severity: Optional[str] = None
    latest_duration: Optional[str] = None


class ConversationTurn(BaseModel):
    role: Literal["assistant", "patient"]
    text: str


class ExtractionContext(BaseModel):
    """
    Information used ONLY to interpret the new statement.
    - previous_symptoms: latest state from earlier calls
    - conversation: turns of the CURRENT call BEFORE the new statement
    - last_question: what the assistant just asked
    - last_question_symptom: the symptom that question was about (or None)
    """
    previous_symptoms: list[PreviousSymptom] = []
    conversation: list[ConversationTurn] = []
    last_question: Optional[str] = None
    last_question_symptom: Optional[str] = None
    # Which detail the last question asked for (set by the question generator)
    last_question_field: Optional[
        Literal["severity", "duration", "medication_adherence", "reported_change", "details"]
    ] = None

    @field_validator("last_question", mode="before")
    @classmethod
    def blank_question_is_none(cls, v):
        return _empty_to_none(v)

    @field_validator("last_question_symptom", mode="before")
    @classmethod
    def question_symptom_supported(cls, v):
        v = _empty_to_none(v)
        if v is None:
            return None
        v = _normalize_name(v)
        return v if v in SUPPORTED_SYMPTOMS else None

    def is_empty(self) -> bool:
        return not (self.previous_symptoms or self.conversation or self.last_question)


def _format_context_block(context: ExtractionContext) -> str:
    lines = ["CONTEXT (use ONLY to understand the new statement; do NOT extract from it):", ""]

    lines.append("Previously recorded symptoms for this patient:")
    if context.previous_symptoms:
        for s in context.previous_symptoms:
            parts = [f"status {s.status or 'unknown'}"]
            if s.latest_severity:
                parts.append(f"last severity {s.latest_severity}")
            if s.latest_duration:
                parts.append(f"last duration {s.latest_duration}")
            lines.append(f"- {_normalize_name(s.name)}: " + ", ".join(parts))
    else:
        lines.append("- none")
    lines.append("")

    lines.append("Earlier in this call:")
    turns = context.conversation[-MAX_CONTEXT_TURNS:]
    if turns:
        for t in turns:
            speaker = "Assistant" if t.role == "assistant" else "Patient"
            lines.append(f"{speaker}: {t.text}")
    else:
        lines.append("(no earlier turns)")
    lines.append("")

    if context.last_question:
        about = context.last_question_symptom or "no specific symptom"
        if context.last_question_symptom and context.last_question_field:
            about += f", asking for: {context.last_question_field}"
        lines.append(f'Last question asked: "{context.last_question}" (about: {about})')
    else:
        lines.append("Last question asked: none")
    lines.append("")

    if context.last_question and context.last_question_symptom:
        symptom = context.last_question_symptom
        if context.last_question_field:
            asked = f"the {context.last_question_field} of {symptom}"
            fill = f"set {context.last_question_field} from the answer"
        else:
            asked = symptom
            fill = "fill the field the answer gives (duration, details, severity, medication_adherence or reported_change)"
        short_answer_rule = (
            f"- The last question asked about {asked}. A direct answer to it, even a single word "
            f"(e.g. \"Yesterday\", \"Yes, 101\", \"A little\"), IS information about {symptom}: "
            f"output {symptom} and {fill}, even if the patient does not repeat the symptom name. "
            "Use the answer as evidence. Apply it ONLY to that symptom. "
            "If the answer is unclear or says they don't know, do not guess.\n"
            "  Example (different patient): question \"When did your cough start?\" (duration of cough), "
            "answer \"Two days ago.\" -> "
            '{"symptoms": [{"name": "cough", "duration": "two days ago", "evidence": "Two days ago"}]}'
        )
    else:
        short_answer_rule = (
            "- The last question was not about a specific symptom. Do not attach a short answer "
            "(e.g. \"Yesterday\", \"Two days\") to any symptom; do not guess."
        )

    lines += [
        "Context rules:",
        "- Extract ONLY information stated in the NEW patient statement below. "
        "Never repeat symptoms or details from the context or earlier turns.",
        short_answer_rule,
        "- Words like \"it\", \"that\" or \"still\" may refer to a symptom from the context only "
        "when the reference is clear. If unclear, ignore it.",
        "- reported_change \"gone\" ONLY when the new statement clearly says the symptom is gone, "
        "resolved or no longer present. A symptom that is not mentioned is NOT gone.",
        "- Never copy severity, duration or details from the context into the output.",
        "- evidence must be words from the NEW statement.",
        "",
        "",
    ]

    return "\n".join(lines)


# ── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a clinical information extractor for a health monitoring system. "
    "You only record what the patient explicitly says. You never guess, "
    "diagnose, or add information. Return ONLY valid JSON. "
    "No explanation, no markdown, no extra text."
)

# {context_block} is "" without context, so the prompt is identical to Chunk B1.
USER_PROMPT_TEMPLATE = """Extract health symptoms from the patient statement below.

Supported symptoms (use ONLY these exact names):
{symptom_list}
Common wording: tired/exhausted -> fatigue; weak/no strength -> weakness;
temperature/feverish -> fever; short of breath -> breathing_difficulty;
can't sleep -> sleep_problems; not hungry/eating less -> appetite_changes.

Rules:
- Include a symptom only if the patient says they have it, or says it changed or went away.
- If the patient says a symptom is gone or no longer present, include it with reported_change "gone".
- Do NOT include symptoms the patient denies having.
- severity: "mild", "moderate" or "severe" ONLY if the patient describes intensity
  (e.g. slight/a little -> mild, quite/fairly -> moderate, very bad/terrible -> severe). Otherwise null.
- duration: only if the patient states when it started or how long (e.g. "since yesterday"). Otherwise null.
- medication_adherence: "taking", "missed" or "not_taking" ONLY if the patient talks about medicine. Otherwise null.
- reported_change: "worse", "better", "same", "gone" or "new" ONLY if the patient explicitly
  says so (e.g. "worse than yesterday", "a bit better", "still the same", "it's gone",
  "it just started"). Otherwise null. Never infer change.
- details: other facts the patient stated about the symptom (e.g. "temperature 101°F"). Otherwise null.
- evidence: the patient's exact words for this symptom.

Return a JSON object in this exact format:
{{
  "symptoms": [
    {{
      "name": "<symptom_name>",
      "severity": null,
      "duration": null,
      "medication_adherence": null,
      "reported_change": null,
      "details": null,
      "evidence": "<patient's words>"
    }}
  ]
}}

If no supported symptom is present, return: {{"symptoms": []}}

{context_block}Patient statement: "{patient_text}"
"""


def build_user_prompt(patient_text: str, context: Optional[ExtractionContext] = None) -> str:
    use_context = context is not None and not context.is_empty()
    return USER_PROMPT_TEMPLATE.format(
        symptom_list=", ".join(SUPPORTED_SYMPTOMS),
        context_block=_format_context_block(context) if use_context else "",
        patient_text=patient_text,
    )


# ── Context guards (deterministic safety net, context mode only) ─────────────

_GROUNDING_THRESHOLD = 0.6


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _is_grounded(fragment: Optional[str], source: str) -> bool:
    """True if most words of `fragment` appear in `source`."""
    if not fragment:
        return False
    fragment_words = _words(fragment)
    if not fragment_words:
        return False
    source_words = set(_words(source))
    hits = sum(1 for w in fragment_words if w in source_words)
    return hits / len(fragment_words) >= _GROUNDING_THRESHOLD


def apply_context_guards(
    result: ExtractionResult,
    patient_text: str,
    context: ExtractionContext,
) -> ExtractionResult:
    """
    Enforce the context rules in code, not only in the prompt:
    1. evidence must come from the NEW statement, otherwise it is cleared.
    2. A symptom without grounded evidence is kept only if it is the symptom
       the last question was about (short answers like "Yesterday").
       Otherwise it is treated as repeated from context and removed.
    3. "gone" requires grounded evidence (explicit resolution only).
    4. A duration identical to the previous record that is not in the new
       statement is treated as copied from context and cleared.
    """
    previous_duration = {
        _normalize_name(p.name): (p.latest_duration or "").strip().lower()
        for p in context.previous_symptoms
    }

    kept: list[SymptomObservation] = []
    ungrounded = list(result.ungrounded_symptoms)

    for obs in result.symptoms:
        obs = obs.model_copy()
        grounded = _is_grounded(obs.evidence, patient_text)
        if not grounded:
            obs.evidence = None

        if not grounded and obs.name != context.last_question_symptom:
            ungrounded.append(obs.name)
            continue

        if obs.reported_change == "gone" and not grounded:
            obs.reported_change = None

        if (
            obs.duration
            and obs.duration.strip().lower() == previous_duration.get(obs.name)
            and not _is_grounded(obs.duration, patient_text)
        ):
            obs.duration = None

        kept.append(obs)

    return result.model_copy(update={"symptoms": kept, "ungrounded_symptoms": ungrounded})


# ── Response cleaner ─────────────────────────────────────────────────────────

def _clean_response(raw: str) -> str:
    """
    Remove <think>...</think> blocks produced by reasoning models.
    Also strips markdown code fences if present.
    """
    # Remove thinking blocks
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)

    # Remove markdown code fences
    raw = re.sub(r"```(?:json)?", "", raw)
    raw = raw.replace("```", "")

    # If there's a JSON object embedded in extra text, extract it
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        raw = match.group(0)

    return raw.strip()


# ── Request builder ──────────────────────────────────────────────────────────

def _supports_reasoning_effort(model: str) -> bool:
    return model.startswith("openai/gpt-oss")


def build_request_kwargs(user_prompt: str, model: str = GROQ_MODEL) -> dict:
    kwargs = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    if _supports_reasoning_effort(model):
        kwargs["reasoning_effort"] = GROQ_REASONING_EFFORT
    return kwargs


# ── Extractor ─────────────────────────────────────────────────────────────────

def extract_symptoms(
    patient_text: str,
    context: Optional[ExtractionContext] = None,
) -> Optional[ExtractionResult]:
    """
    Send patient text to the LLM and return a validated ExtractionResult.
    `context` is optional; without it, behaviour is identical to Chunk B1.
    Returns ExtractionResult on success, None if all retries exhausted.
    """
    client = get_client()
    use_context = context is not None and not context.is_empty()
    user_prompt = build_user_prompt(patient_text, context)
    request_kwargs = build_request_kwargs(user_prompt)

    for attempt in range(1, MAX_RETRIES + 1):
        raw_text = ""
        try:
            response = client.chat.completions.create(**request_kwargs)

            raw_text = (response.choices[0].message.content or "").strip()
            if not raw_text:
                raise ValueError("Empty response (output may have been cut off).")

            cleaned = _clean_response(raw_text)
            parsed_json = json.loads(cleaned)
            result = ExtractionResult(**parsed_json)

            if use_context:
                result = apply_context_guards(result, patient_text, context)

            if result.dropped_symptoms:
                print(f"  [Info] Unsupported symptoms ignored: {result.dropped_symptoms}")
            if result.ungrounded_symptoms:
                print(f"  [Info] Not in new statement, ignored: {result.ungrounded_symptoms}")
            return result

        except json.JSONDecodeError as e:
            print(f"  [Attempt {attempt}] JSON parse error: {e}")
            print(f"  Raw response was:\n{raw_text}")

        except ValidationError as e:
            print(f"  [Attempt {attempt}] Validation error:\n{e}")

        except Exception as e:
            print(f"  [Attempt {attempt}] Unexpected error: {type(e).__name__}: {e}")

    print("  All retries exhausted. Returning None.")
    return None