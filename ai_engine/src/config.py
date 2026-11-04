import os
from pathlib import Path

from dotenv import load_dotenv

# Always read ai_engine/.env, whichever folder the program is started from
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# gpt-oss models "think" before answering. "low" keeps that short and fast.
# Only sent to models that support it (see extraction.py).
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")

if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set.\n"
        "Copy .env.example to .env and add your key from https://console.groq.com"
    )

# Symptoms the system supports (Evaluation 2).
# Extend this list in later evaluations.
SUPPORTED_SYMPTOMS = [
    "fatigue",
    "weakness",
    "headache",
    "pain",
    "cough",
    "fever",
    "dizziness",
    "breathing_difficulty",
    "sleep_problems",
    "appetite_changes",
]

SEVERITY_LEVELS = ["mild", "moderate", "severe"]

# Only filled when the patient talks about medication
MEDICATION_ADHERENCE_LEVELS = ["taking", "missed", "not_taking"]

# What the patient EXPLICITLY said about change. Never inferred.
REPORTED_CHANGES = ["new", "worse", "better", "same", "gone"]

MAX_RETRIES = 2
# Reasoning models use part of this budget for thinking; 512 was too small.
MAX_TOKENS = 2048