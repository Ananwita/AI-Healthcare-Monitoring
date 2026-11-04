# AI-Powered Healthcare Monitoring System

A voice-based health **monitoring** assistant. Patients talk to it; it understands their
symptoms, asks relevant follow-up questions, remembers earlier calls, tracks how each
symptom changes over time, and stores everything in a backend with a dashboard.

> ⚠️ **Not a medical device.** This is an academic monitoring prototype. It does not
> diagnose, does not replace a doctor, and never invents medical information.

**Status:** Evaluation 2 complete — 406 automated tests passing.

---

## How it works

```
 Patient (voice)
      │
      ▼
 voice_assistant  (Member 1)   Streamlit app · Whisper speech-to-text · gTTS speech
      │  imports
      ▼
 ai_engine        (Member 2)   Groq LLM extraction · call memory · trends · questions
      │  HTTP (httpx)
      ▼
 monitoring       (Member 3)   FastAPI backend · SQLite · SQLAlchemy · Streamlit dashboard
```

### Example
- **Day 1** — *"I have a fever and feel weak."* → asks when it started, how bad it is,
  the temperature; a short answer like *"Yesterday."* is linked to the right question.
- **Day 2** — *"Last time you mentioned fever and weakness…"* → *"My fever is gone, but I
  still feel weak."* → fever marked **resolved**, weakness **stable**; history kept.
- **Later** — fever reported again → marked as **returned after being resolved**.

## Features (Evaluation 2)

| Area | What it does |
|---|---|
| Voice | Browser recording → Whisper (local) → AI → spoken reply (gTTS); typed fallback |
| Extraction | Symptoms, severity, duration, medication adherence, details — only what the patient said (no invented values) |
| Call memory | Short answers linked to the last question; nothing asked twice |
| History | Uses **all** previous calls, not just the latest few |
| Trends | new · worsening · stable · improving · resolved — only from explicit patient statements |
| Status | ongoing / resolved; recurrence detected; unmentioned symptoms never auto-resolved |
| Reliability | Idempotent saves (`client_call_id`), 409 on conflicting data, safe retries |
| Interruptions | Dropped calls saved as `interrupted`; unsent calls kept locally and retried once |
| Dashboard | Patient overview, current symptom status, symptom timeline, call history |

Supported symptoms: fatigue, weakness, headache, pain, cough, fever, dizziness,
breathing difficulty, sleep problems, appetite changes.

## Tech stack
Python · Groq API (`openai/gpt-oss-20b`) · Pydantic · OpenAI Whisper · gTTS ·
FastAPI · SQLite · SQLAlchemy · Streamlit · httpx · pytest

---

## Setup (Windows, PowerShell)

Requirements: **Python 3.11+**, **Git**, **ffmpeg** (for Whisper: `winget install Gyan.FFmpeg`),
and a free **Groq API key** from <https://console.groq.com>.

```powershell
git clone <this-repository-url>
cd <repository-folder>

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -r ai_engine\requirements.txt
python -m pip install -r monitoring\requirements.txt
python -m pip install -r voice_assistant\requirements.txt

copy ai_engine\.env.example ai_engine\.env
# then open ai_engine\.env and paste your Groq API key
```

## Run (three terminals, each from the project root with `.venv` active)

| # | Component | Command | Open |
|---|---|---|---|
| 1 | Backend | `cd monitoring` → `python -m uvicorn backend.main:app --port 8001` | http://127.0.0.1:8001/docs |
| 2 | Voice app | `cd voice_assistant` → `python -m streamlit run app.py` | http://localhost:8501 |
| 3 | Dashboard | `cd monitoring` → `python -m streamlit run dashboard\app.py --server.port 8502` | http://localhost:8502 |

The database (`monitoring/health_monitoring.db`) is created automatically on first run.
To upgrade an **older** database safely (backup + additive changes only):
`cd monitoring` → `python -m backend.migrate --dry-run`, then `python -m backend.migrate`.

A text-only version of the call (no microphone) is also available:
`cd ai_engine` → `python main.py`

## Tests

```powershell
cd monitoring;      python -m pytest backend\tests -q;   cd ..   # 82 tests
cd ai_engine;       python -m pytest -q;                 cd ..   # 285 tests
cd voice_assistant; python -m pytest -q;                 cd ..   # 31 tests
python -m pytest tests_integration -q                            # 8 end-to-end tests
```

Tests never call the real Groq API, never download the Whisper model, and never touch the
real database (temporary databases and fakes are used). The end-to-end tests run the real
backend over HTTP together with the real AI engine and voice bridge.

## Project structure

```
ai_engine/          Member 2 — AI processing
  src/llm/            extraction (Groq + Pydantic validation, context-aware)
  src/analysis/       trend and status rules
  src/conversation/   call session, adaptive questions, saving/recovery, call manager
  src/backend/        HTTP client for the backend
  tests/
monitoring/         Member 3 — backend and dashboard
  backend/            FastAPI app, models, safe migration script, tests
  dashboard/          Streamlit dashboard
voice_assistant/    Member 1 — voice interface
  app.py              Streamlit voice app
  conversation.py     bridge to ai_engine
  speech_to_text.py · text_to_speech.py · tests/
tests_integration/  end-to-end tests across all three components
```

## Known limitations
- Whisper `base` can mis-hear words (`set WHISPER_MODEL=small` for better accuracy, slower).
- Groq and gTTS need an internet connection (speech falls back to text if gTTS fails).
- Only the 10 symptoms above are tracked; patients are identified by a typed ID (no login).
- One active call at a time in the voice app.

## Team
| Member | Component |
|---|---|
| Member 1 | Voice interaction and conversation layer |
| Member 2 | AI processing and symptom intelligence |
| Member 3 | Backend, database and dashboard |