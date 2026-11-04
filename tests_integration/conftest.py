"""
Chunk E1 — integration test setup.

REAL backend (Member 3: FastAPI over real HTTP, temporary SQLite database),
REAL AI engine (Member 2) and REAL voice bridge (Member 1).
Only the LLM, Whisper and gTTS are replaced by fakes (offline, no API quota).

Run from the project root:
    python -m pytest tests_integration -q
"""

import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

# ── Temporary database: must be set BEFORE the backend is imported ───────────
_TMP_DIR = tempfile.mkdtemp(prefix="hm_integration_")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP_DIR, "integration.db").replace(os.sep, "/")

# ── Make all three members' code importable ──────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
for folder in ("monitoring", "ai_engine", "voice_assistant"):
    path = str(ROOT / folder)
    if path not in sys.path:
        sys.path.insert(0, path)

import uvicorn  # noqa: E402

from backend.database import Base, engine  # noqa: E402
from backend import models  # noqa: E402,F401
from backend.main import app  # noqa: E402

# Safety guard: never run against the real database
assert "hm_integration_" in str(engine.url), f"Refusing to use real DB: {engine.url}"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _ThreadedServer(uvicorn.Server):
    def install_signal_handlers(self):  # not allowed outside the main thread
        pass


@pytest.fixture(scope="session")
def backend_url():
    """Start the real FastAPI backend on a free port for the whole test run."""
    port = free_port()
    server = _ThreadedServer(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 15
    while not server.started:
        if time.time() > deadline or not thread.is_alive():
            raise RuntimeError("Test backend did not start.")
        time.sleep(0.05)

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(autouse=True)
def fresh_db():
    """Every test starts with an empty database."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def dirs(tmp_path):
    """Separate local draft/pending folders per test (never ai_engine/data)."""
    return {"pending_dir": tmp_path / "pending", "drafts_dir": tmp_path / "drafts"}