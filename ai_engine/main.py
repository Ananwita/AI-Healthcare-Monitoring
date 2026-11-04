"""
AI Healthcare Monitoring — Member 2 CLI.

    python main.py               Interactive text call (real Groq + real backend)
    python main.py --eval1-demo  Original Evaluation 1 demo (JSON storage only)

The backend must be running for the interactive call:
    cd monitoring
    python -m uvicorn backend.main:app --port 8001
"""

import argparse
import sys

from src.llm.extraction import extract_symptoms
from src.history.storage import (
    save_observation,
    get_previous_observation,
    clear_patient_history,
)
from src.analysis.trend import detect_trend
from src.conversation.adaptive_questions import generate_question


# ══════════════════════════════════════════════════════════════════════════════
# Interactive call (Chunk B8)
# ══════════════════════════════════════════════════════════════════════════════

END_WORDS = {"quit", "exit", "bye"}


def run_interactive() -> int:
    from src.backend.client import BackendError, PatientConflictError
    from src.conversation.call_manager import CallManager

    manager = CallManager()

    if not manager.client.health():
        print(f"Backend is not reachable at {manager.client.base_url}.")
        print("Start it first:  cd monitoring  then  python -m uvicorn backend.main:app --port 8001")
        return 1

    for result in manager.startup():
        print(f"  [Recovery] call {result.client_call_id[:8]}: {result.message}")

    external_id = input("Patient ID (e.g. P001): ").strip()
    name = input("Patient name: ").strip()
    if not external_id or not name:
        print("Patient ID and name are required.")
        return 1

    try:
        session, greeting = manager.start_call(external_id, name)
    except PatientConflictError as e:
        print(f"Patient ID '{external_id}' already belongs to a patient with different details: {e.detail}")
        return 1
    except BackendError as e:
        print(f"Could not start the call: {e}")
        return 1

    print(f"\n(Type your answers. '{'/'.join(sorted(END_WORDS))}' ends the call, Ctrl+C simulates a dropped call.)\n")
    print(f"AI : {greeting}")

    try:
        while True:
            text = input("You: ")
            if text.strip().lower() in END_WORDS:
                break
            reply = manager.handle_patient_message(session, text)
            if reply.extraction_failed:
                print("     (the AI could not process that answer; continuing)")
            print(f"AI : {reply.text}")
            if reply.done:
                break
        result = manager.end_call(session)
    except (KeyboardInterrupt, EOFError):
        print("\n\nCall interrupted.")
        result = manager.end_call(session, interrupted=True)

    print(f"\nSave status: {result.message}")
    if result.saved:
        print(f"Backend call_id: {result.call_id}")
    elif result.local_path:
        print(f"Kept locally at: {result.local_path}")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# Original Evaluation 1 demo (unchanged behaviour, JSON storage)
# ══════════════════════════════════════════════════════════════════════════════

PATIENT_ID = "P001"

# Four calls demonstrating: new_symptom → worsening → stable → improving
CALLS = [
    {"call_number": 1, "text": "I feel slightly tired."},
    {"call_number": 2, "text": "I feel more tired today."},
    {"call_number": 3, "text": "I am still feeling quite tired."},
    {"call_number": 4, "text": "I feel a little better today, less tired."},
]


def process_call(patient_id: str, call_number: int, patient_text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  CALL {call_number}")
    print(f"{'='*60}")
    print(f"  Patient  : {patient_text}")

    # Step 1 — Extract
    result = extract_symptoms(patient_text)
    if result is None or not result.symptoms:
        print("  Output   : No symptoms extracted.")
        return

    # Step 2 — For each symptom: retrieve → trend → question → store
    for symptom_obj in result.symptoms:
        symptom = symptom_obj.name
        severity = symptom_obj.severity
        if severity is None:
            print(f"\n  Symptom  : {symptom}")
            print("  Severity : not stated (skipped; trend rules arrive in Chunk B3)")
            continue

        print(f"\n  Symptom  : {symptom}")
        print(f"  Severity : {severity}")

        # Retrieve previous BEFORE saving
        previous = get_previous_observation(patient_id, symptom)
        previous_severity = previous["severity"] if previous else None

        # Detect trend
        trend_result = detect_trend(
            symptom=symptom,
            current_severity=severity,
            previous_severity=previous_severity,
        )

        # Generate adaptive question
        question_result = generate_question(
            symptom=symptom,
            trend=trend_result.trend,
            previous_severity=previous_severity,
            current_severity=severity,
        )

        # Store current observation
        save_observation(
            patient_id=patient_id,
            symptom=symptom,
            severity=severity,
            call_number=call_number,
        )

        # Print full pipeline output
        if previous_severity:
            print(f"  Previous : {previous_severity}")
        print(f"  Trend    : {trend_result.trend}")
        print(f"  Reason   : {trend_result.reason}")
        print(f"  Strategy : {question_result.strategy}")
        print(f"\n  ➤ AI asks : \"{question_result.question}\"")


def run():
    print(f"Resetting demo history for patient '{PATIENT_ID}'...")
    clear_patient_history(PATIENT_ID)

    for call in CALLS:
        process_call(
            patient_id=PATIENT_ID,
            call_number=call["call_number"],
            patient_text=call["text"],
        )

    print(f"\n{'='*60}")
    print("  Demo complete.")
    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Healthcare Monitoring CLI")
    parser.add_argument("--eval1-demo", action="store_true",
                        help="Run the original Evaluation 1 demo (JSON storage)")
    args = parser.parse_args()

    if args.eval1_demo:
        run()
    else:
        sys.exit(run_interactive())