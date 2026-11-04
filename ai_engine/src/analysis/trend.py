from typing import Optional
from dataclasses import dataclass

# Severity mapped to numeric values for comparison
SEVERITY_SCALE = {
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}


@dataclass
class TrendResult:
    """
    Output of trend detection for one symptom.

    Fields:
        trend   — one of: new_symptom, stable, worsening, improving
        reason  — human-readable explanation (used in adaptive question generation)
        previous_severity — None if first observation
        current_severity  — current observation
    """
    trend: str
    reason: str
    previous_severity: Optional[str]
    current_severity: str


def detect_trend(
    symptom: str,
    current_severity: str,
    previous_severity: Optional[str],
) -> TrendResult:
    """
    Compare current and previous severity to determine trend.

    Rules:
        - No previous observation        → new_symptom
        - current == previous            → stable
        - current scale > previous scale → worsening
        - current scale < previous scale → improving

    Args:
        symptom           : symptom name (used only in reason string)
        current_severity  : severity from current call
        previous_severity : severity from previous call, or None

    Returns:
        TrendResult with trend label and plain-English reason.
    """
    current_severity = current_severity.lower().strip()

    # Validate current severity
    if current_severity not in SEVERITY_SCALE:
        raise ValueError(
            f"Invalid severity '{current_severity}'. "
            f"Must be one of: {list(SEVERITY_SCALE.keys())}"
        )

    # Case 1 — No previous observation
    if previous_severity is None:
        return TrendResult(
            trend="new_symptom",
            reason=(
                f"{symptom.replace('_', ' ').title()} was reported for the first time "
                f"with {current_severity} severity."
            ),
            previous_severity=None,
            current_severity=current_severity,
        )

    previous_severity = previous_severity.lower().strip()

    # Validate previous severity
    if previous_severity not in SEVERITY_SCALE:
        raise ValueError(
            f"Invalid previous severity '{previous_severity}'. "
            f"Must be one of: {list(SEVERITY_SCALE.keys())}"
        )

    current_score = SEVERITY_SCALE[current_severity]
    previous_score = SEVERITY_SCALE[previous_severity]

    # Case 2 — Stable
    if current_score == previous_score:
        return TrendResult(
            trend="stable",
            reason=(
                f"{symptom.replace('_', ' ').title()} severity remains "
                f"{current_severity}, unchanged from the previous call."
            ),
            previous_severity=previous_severity,
            current_severity=current_severity,
        )

    # Case 3 — Worsening
    if current_score > previous_score:
        return TrendResult(
            trend="worsening",
            reason=(
                f"{symptom.replace('_', ' ').title()} severity increased "
                f"from {previous_severity} to {current_severity}."
            ),
            previous_severity=previous_severity,
            current_severity=current_severity,
        )

    # Case 4 — Improving
    return TrendResult(
        trend="improving",
        reason=(
            f"{symptom.replace('_', ' ').title()} severity decreased "
            f"from {previous_severity} to {current_severity}."
        ),
        previous_severity=previous_severity,
        current_severity=current_severity,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Chunk B3 — Trend and status rules based on what the patient EXPLICITLY said.
# detect_trend() above is kept unchanged for backward compatibility.
# ══════════════════════════════════════════════════════════════════════════════

TREND_VALUES = ("new", "worsening", "stable", "improving", "resolved")
STATUS_VALUES = ("ongoing", "resolved")


@dataclass
class SymptomUpdate:
    """
    Result of applying one new observation to a symptom's previous state.

    trend    — new | worsening | stable | improving | resolved
    status   — ongoing | resolved
    recurred — True if the symptom came back after being resolved
    reason   — plain-English explanation (for logs, summaries, questions)
    """
    name: str
    trend: str
    status: str
    recurred: bool
    reason: str


def _normalize_symptom_name(name) -> str:
    return "_".join(str(name or "").strip().lower().replace("-", " ").split())


def _field(previous, key):
    """Read a field from a dict (backend JSON) or an object (PreviousSymptom)."""
    if previous is None:
        return None
    if isinstance(previous, dict):
        return previous.get(key)
    return getattr(previous, key, None)


def _label(name: str) -> str:
    return name.replace("_", " ").capitalize()


def determine_update(previous, observation) -> SymptomUpdate:
    """
    Decide trend and status for ONE symptom mentioned in the current call.

    Args:
        previous    : latest state of this symptom from earlier calls
                      (backend /symptom-state entry or PreviousSymptom), or None
        observation : SymptomObservation from extraction
                      (uses .name, .reported_change, .severity)

    Rules (explicit patient statements only, never guessed):
        - no previous record                  -> new / ongoing
          (unless the patient says it is gone -> resolved / resolved)
        - patient says gone                   -> resolved / resolved
        - previously resolved, mentioned again -> new / ongoing, recurred=True
        - patient says worse / better / same  -> worsening / improving / stable
        - no change word, but patient states a different severity than the
          previous recorded severity          -> worsening / improving
        - otherwise                           -> stable / ongoing

    Symptoms NOT mentioned in the call must not be passed here: they keep
    their previous state and are never resolved automatically.
    """
    name = _normalize_symptom_name(observation.name)
    label = _label(name)
    change = observation.reported_change
    severity = observation.severity

    # ── No previous record ───────────────────────────────────────────────────
    if previous is None:
        if change == "gone":
            return SymptomUpdate(name, "resolved", "resolved", False,
                                 f"{label} reported as gone (no earlier record).")
        return SymptomUpdate(name, "new", "ongoing", False,
                             f"{label} reported for the first time.")

    previous_status = _field(previous, "status") or "ongoing"
    previous_severity = _field(previous, "latest_severity")

    # ── Explicit resolution ──────────────────────────────────────────────────
    if change == "gone":
        if previous_status == "resolved":
            return SymptomUpdate(name, "resolved", "resolved", False,
                                 f"{label} still reported as gone.")
        return SymptomUpdate(name, "resolved", "resolved", False,
                             f"Patient said {label.lower()} is gone.")

    # ── Recurrence after resolution ──────────────────────────────────────────
    if previous_status == "resolved":
        return SymptomUpdate(name, "new", "ongoing", True,
                             f"{label} reported again after it had resolved.")

    # ── Explicit change words ────────────────────────────────────────────────
    if change == "worse":
        return SymptomUpdate(name, "worsening", "ongoing", False,
                             f"Patient said {label.lower()} is worse.")
    if change == "better":
        return SymptomUpdate(name, "improving", "ongoing", False,
                             f"Patient said {label.lower()} is better.")
    if change == "same":
        return SymptomUpdate(name, "stable", "ongoing", False,
                             f"Patient said {label.lower()} is unchanged.")

    # ── Patient-stated severity compared with previous patient-stated severity
    if (
        severity in SEVERITY_SCALE
        and previous_severity in SEVERITY_SCALE
        and severity != previous_severity
    ):
        if SEVERITY_SCALE[severity] > SEVERITY_SCALE[previous_severity]:
            return SymptomUpdate(name, "worsening", "ongoing", False,
                                 f"{label} described as {severity} (previously {previous_severity}).")
        return SymptomUpdate(name, "improving", "ongoing", False,
                             f"{label} described as {severity} (previously {previous_severity}).")

    # ── Mentioned with no reported change ────────────────────────────────────
    if change == "new":
        return SymptomUpdate(name, "stable", "ongoing", False,
                             f"Patient called {label.lower()} new, but it is already being tracked.")
    return SymptomUpdate(name, "stable", "ongoing", False,
                         f"{label} mentioned again with no reported change.")


def unmentioned_ongoing_symptoms(previous_states, mentioned_names) -> list[str]:
    """
    Ongoing symptoms from earlier calls that were NOT mentioned in this call.
    They keep their previous state (never auto-resolved); the question
    generator can ask about them.
    """
    mentioned = {_normalize_symptom_name(n) for n in mentioned_names}
    result = []
    for prev in previous_states or []:
        name = _normalize_symptom_name(_field(prev, "name"))
        if name and name not in mentioned and (_field(prev, "status") or "ongoing") == "ongoing":
            result.append(name)
    return result