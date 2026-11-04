from dataclasses import dataclass
from typing import Optional


@dataclass
class QuestionResult:
    """
    Output of adaptive question generation.

    Fields:
        question        — the follow-up question to ask the patient
        strategy        — label explaining why this question was chosen
        symptom         — symptom this question is about
        trend           — trend that triggered this question
    """
    question: str
    strategy: str
    symptom: str
    trend: str


# ── Question templates ────────────────────────────────────────────────────────
#
# Structure:
#   TEMPLATES[symptom][trend] = (question, strategy)
#
# If a symptom is not in the table, FALLBACK_TEMPLATES[trend] is used.
# This makes it easy to add more symptoms in Evaluation 2 without
# changing any logic — just add a new key to TEMPLATES.

TEMPLATES: dict[str, dict[str, tuple[str, str]]] = {
    "fatigue": {
        "new_symptom": (
            "How long have you been feeling fatigued?",
            "Establish duration of new fatigue symptom.",
        ),
        "worsening": (
            "Has your increased fatigue started affecting your daily activities?",
            "Assess functional impact of worsening fatigue.",
        ),
        "improving": (
            "That is good to hear. Are you still experiencing any fatigue at all?",
            "Confirm whether fatigue is fully resolving.",
        ),
        "stable": (
            "Your fatigue has persisted. Has it been affecting your sleep or energy levels?",
            "Explore impact of ongoing fatigue on daily function.",
        ),
    },
    "headache": {
        "new_symptom": (
            "Where exactly is the headache located, and how long have you had it?",
            "Characterise new headache symptom.",
        ),
        "worsening": (
            "Has your headache become more frequent, or is it constant now?",
            "Assess whether worsening headache is frequent or continuous.",
        ),
        "improving": (
            "Good to hear your headache is easing. Are you still taking any pain relief?",
            "Confirm medication use as headache improves.",
        ),
        "stable": (
            "Your headache has continued at the same level. Is it affecting your concentration?",
            "Assess cognitive impact of persistent headache.",
        ),
    },
    "fever": {
        "new_symptom": (
            "Have you measured your temperature, and when did the fever start?",
            "Establish fever onset and confirm measurement.",
        ),
        "worsening": (
            "Has your fever increased further? Have you been able to take any medication for it?",
            "Assess fever escalation and medication response.",
        ),
        "improving": (
            "Good. Has your temperature returned to normal, or is it still slightly elevated?",
            "Confirm whether fever has fully resolved.",
        ),
        "stable": (
            "Your fever has persisted. Have you been drinking enough fluids?",
            "Check hydration during ongoing fever.",
        ),
    },
    "cough": {
        "new_symptom": (
            "Is the cough dry or producing any phlegm, and when did it start?",
            "Characterise new cough symptom.",
        ),
        "worsening": (
            "Has the cough been disturbing your sleep or making it hard to breathe?",
            "Assess functional impact of worsening cough.",
        ),
        "improving": (
            "Good to hear the cough is improving. Is it still present at certain times of day?",
            "Check for residual cough pattern.",
        ),
        "stable": (
            "Your cough has continued. Has it been accompanied by any chest discomfort?",
            "Screen for chest symptoms alongside persistent cough.",
        ),
    },
    "pain": {
        "new_symptom": (
            "Can you describe where the pain is and when it started?",
            "Localise and establish onset of new pain.",
        ),
        "worsening": (
            "Has the pain started limiting your movement or daily activities?",
            "Assess functional limitation from worsening pain.",
        ),
        "improving": (
            "Good. Is the pain now manageable without medication?",
            "Confirm pain management as symptoms improve.",
        ),
        "stable": (
            "The pain has continued at the same level. Is it affecting your ability to rest or sleep?",
            "Assess sleep impact of persistent pain.",
        ),
    },
    "dizziness": {
        "new_symptom": (
            "When does the dizziness occur — when standing up, moving, or all the time?",
            "Characterise triggers of new dizziness.",
        ),
        "worsening": (
            "Has the dizziness caused any difficulty walking or a risk of falling?",
            "Assess fall risk from worsening dizziness.",
        ),
        "improving": (
            "Good. Does the dizziness still occur in certain situations?",
            "Confirm residual dizziness triggers.",
        ),
        "stable": (
            "Your dizziness has persisted. Have you noticed if it is worse at any particular time of day?",
            "Identify time-of-day pattern in persistent dizziness.",
        ),
    },
    "breathing_difficulty": {
        "new_symptom": (
            "Is the difficulty breathing constant, or does it come and go?",
            "Establish pattern of new breathing difficulty.",
        ),
        "worsening": (
            "Has your breathing difficulty increased to the point where it affects walking or speaking?",
            "Assess severity of worsening breathing difficulty.",
        ),
        "improving": (
            "Good to hear your breathing is easier. Do you still feel any shortness of breath during activity?",
            "Confirm resolution of breathing difficulty during exertion.",
        ),
        "stable": (
            "Your breathing difficulty has continued. Are you avoiding any activities because of it?",
            "Assess activity avoidance from persistent breathing difficulty.",
        ),
    },
    "sleep_problems": {
        "new_symptom": (
            "Are you having trouble falling asleep, staying asleep, or both?",
            "Characterise type of new sleep problem.",
        ),
        "worsening": (
            "Has the poor sleep started affecting your energy or concentration during the day?",
            "Assess daytime impact of worsening sleep problems.",
        ),
        "improving": (
            "Good to hear your sleep is improving. How many hours are you getting now?",
            "Confirm sleep duration as problems resolve.",
        ),
        "stable": (
            "Your sleep difficulties have continued. Have you tried any changes to your routine?",
            "Explore sleep hygiene alongside persistent sleep problems.",
        ),
    },
    "appetite_changes": {
        "new_symptom": (
            "Have you been eating less than usual, or has your appetite changed in another way?",
            "Characterise nature of new appetite change.",
        ),
        "worsening": (
            "Has the change in appetite led to any noticeable weight loss?",
            "Screen for weight loss alongside worsening appetite change.",
        ),
        "improving": (
            "Good. Are you managing to eat regular meals now?",
            "Confirm return to normal eating pattern.",
        ),
        "stable": (
            "Your appetite has remained changed. Are you managing to get enough nutrition each day?",
            "Assess nutritional adequacy during persistent appetite change.",
        ),
    },
}

# Used when a symptom is not in TEMPLATES
FALLBACK_TEMPLATES: dict[str, tuple[str, str]] = {
    "new_symptom": (
        "How long have you been experiencing this symptom?",
        "Generic duration question for new symptom.",
    ),
    "worsening": (
        "Has this symptom started affecting your daily activities?",
        "Generic functional impact question for worsening symptom.",
    ),
    "improving": (
        "Good to hear. Are you still experiencing this symptom at all?",
        "Generic resolution check for improving symptom.",
    ),
    "stable": (
        "This symptom has continued. Is it affecting your daily routine?",
        "Generic impact question for stable symptom.",
    ),
}


# ── Generator ─────────────────────────────────────────────────────────────────

def generate_question(
    symptom: str,
    trend: str,
    previous_severity: Optional[str] = None,
    current_severity: Optional[str] = None,
) -> QuestionResult:
    """
    Select and return an adaptive follow-up question based on symptom and trend.

    Args:
        symptom           : normalised symptom name (e.g. "fatigue")
        trend             : one of new_symptom, worsening, stable, improving
        previous_severity : optional, used for future personalisation
        current_severity  : optional, used for future personalisation

    Returns:
        QuestionResult with question text and strategy label.

    Design note:
        Currently deterministic template lookup.
        In Evaluation 3, this can be replaced with LLM-generated questions
        using the same interface — nothing outside this function needs to change.
    """
    symptom = symptom.lower().strip()
    trend = trend.lower().strip()

    valid_trends = {"new_symptom", "worsening", "stable", "improving"}
    if trend not in valid_trends:
        raise ValueError(
            f"Invalid trend '{trend}'. Must be one of: {valid_trends}"
        )

    # Look up symptom-specific template, fall back to generic if not found
    symptom_templates = TEMPLATES.get(symptom, FALLBACK_TEMPLATES)

    if isinstance(symptom_templates, dict) and trend in symptom_templates:
        question, strategy = symptom_templates[trend]
    else:
        question, strategy = FALLBACK_TEMPLATES[trend]

    return QuestionResult(
        question=question,
        strategy=strategy,
        symptom=symptom,
        trend=trend,
    )



# ══════════════════════════════════════════════════════════════════════════════
# Chunk B5 — Slot-based adaptive questions for a whole call.
# Uses CallSession (B4). Deterministic templates only: no LLM, no invented
# medical content. generate_question() above is kept unchanged.
# ══════════════════════════════════════════════════════════════════════════════

MAX_QUESTIONS_PER_CALL = 8

# Friendlier wording for symptom names in sentences
SYMPTOM_LABELS = {
    "breathing_difficulty": "difficulty breathing",
    "sleep_problems": "sleep trouble",
    "appetite_changes": "appetite change",
}

# One symptom-specific follow-up (stored in the "details" field)
DETAIL_QUESTIONS = {
    "fever": "Have you measured your temperature? If so, what was it?",
    "pain": "Where exactly is the pain?",
    "headache": "Where is the headache located?",
    "cough": "Is the cough dry, or are you bringing up phlegm?",
}

SAFETY_LINE = "If any symptom gets worse or you feel unwell, please contact your doctor."


@dataclass
class PlannedQuestion:
    """
    A question to say next. Pass symptom/field to
    session.add_assistant_message(text, symptom, field) so the answer is
    linked to it (B2) and it is never asked twice (B4).
    """
    text: str
    symptom: Optional[str]
    field: Optional[str]
    strategy: str


def _label(name: str) -> str:
    return SYMPTOM_LABELS.get(name, name.replace("_", " "))


def _join(names: list[str]) -> str:
    labels = [_label(n) for n in names]
    if len(labels) <= 1:
        return "".join(labels)
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def _questions_asked(session) -> int:
    """Follow-up questions asked so far (greetings without a field do not count)."""
    return sum(1 for q in session.asked if q.field is not None)


def _ask(session, symptom, field, text, strategy) -> Optional[PlannedQuestion]:
    if session.was_asked(symptom, field):
        return None
    return PlannedQuestion(text=text, symptom=symptom, field=field, strategy=strategy)


def opening_message(session) -> PlannedQuestion:
    """First thing the assistant says. General, not tied to one symptom."""
    ongoing = [
        p.name for p in session.previous_symptoms
        if (p.status or "ongoing") == "ongoing"
    ]
    if not ongoing:
        return PlannedQuestion(
            text="Hello, I'm your health monitoring assistant. How are you feeling today?",
            symptom=None, field=None, strategy="Open the first call.",
        )
    how_now = "how is it now" if len(ongoing) == 1 else "how are those symptoms now"
    return PlannedQuestion(
        text=(
            f"Hello again. Last time, you mentioned {_join(ongoing)}. "
            f"How are you feeling today, and {how_now}?"
        ),
        symptom=None, field=None,
        strategy="Follow up on ongoing symptoms from earlier calls.",
    )


def next_question(session) -> Optional[PlannedQuestion]:
    """
    Choose the next question, or None when there is nothing left to ask.

    Never asks for information already known (this call or earlier calls
    where relevant) and never repeats a question already asked in this call.
    """
    if _questions_asked(session) >= MAX_QUESTIONS_PER_CALL:
        return None

    updates = {u.name: u for u in session.symptom_updates()}

    # 1. Symptoms mentioned in this call (in order of first mention)
    for name in session.mentioned_symptoms():
        update = updates[name]
        if update.status == "resolved":
            continue                                   # nothing to ask
        current = session.symptoms[name]
        label = _label(name)

        if update.trend == "new":                      # first report or recurrence
            if current.duration is None:
                verb = "come back" if update.recurred else "start"
                q = _ask(session, name, "duration", f"When did your {label} {verb}?",
                         "Establish onset of new or returning symptom.")
                if q:
                    return q
            if current.severity is None:
                q = _ask(session, name, "severity",
                         f"How bad is your {label}: mild, moderate, or severe?",
                         "Establish severity of new or returning symptom.")
                if q:
                    return q
            if name in DETAIL_QUESTIONS and current.details is None:
                q = _ask(session, name, "details", DETAIL_QUESTIONS[name],
                         "Symptom-specific detail.")
                if q:
                    return q
            continue

        # Symptom already tracked from earlier calls and still ongoing
        if update.trend == "worsening" and current.severity is None:
            q = _ask(session, name, "severity",
                     f"I'm sorry it's worse. How bad is your {label} now: mild, moderate, or severe?",
                     "Assess severity of worsening symptom.")
            if q:
                return q

        if update.trend == "stable" and current.reported_change is None:
            q = _ask(session, name, "reported_change",
                     f"Compared with last time, is your {label} better, worse, or about the same?",
                     "Clarify change of ongoing symptom (never assumed).")
            if q:
                return q

    # 2. Ongoing symptoms from earlier calls not mentioned yet (never auto-resolved)
    for name in session.unmentioned_previous_symptoms():
        q = _ask(session, name, "reported_change",
                 f"Last time, you mentioned {_label(name)}. How is it now: better, worse, the same, or gone?",
                 "Follow up on earlier symptom not mentioned today.")
        if q:
            return q

    # 3. Medication adherence, once per call, for the first ongoing symptom
    ongoing_now = [n for n in session.mentioned_symptoms() if updates[n].status == "ongoing"]
    already_asked_medication = any(q.field == "medication_adherence" for q in session.asked)
    already_known = any(s.medication_adherence for s in session.symptoms.values())
    if ongoing_now and not already_asked_medication and not already_known:
        name = ongoing_now[0]
        return PlannedQuestion(
            text=f"Are you taking any medicine for your {_label(name)} as advised, or have you missed any doses?",
            symptom=name, field="medication_adherence",
            strategy="Check medication adherence once per call.",
        )

    return None


_TREND_PHRASES = {
    "new": "{label} (new)",
    "worsening": "{label} is worse",
    "improving": "{label} is better",
    "stable": "{label} is about the same",
    "resolved": "{label} has gone",
}


def closing_message(session) -> str:
    """Summary of what was noted + a general safety line. No diagnosis."""
    parts = []
    for update in session.symptom_updates():
        label = _label(update.name)
        if update.recurred:
            parts.append(f"{label} has come back")
        else:
            parts.append(_TREND_PHRASES[update.trend].format(label=label))
    if parts:
        noted = "; ".join(parts)
        return f"Thank you for the update. I've noted: {noted}. {SAFETY_LINE}"
    return f"Thank you for talking with me today. {SAFETY_LINE}"