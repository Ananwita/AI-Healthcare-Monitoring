"""
Chunk C2 — Bridge between the voice app (Member 1) and the AI engine (Member 2).

    convo = VoiceConversation()
    convo.startup()                                   # recover/retry earlier calls
    greeting, audio = convo.start("P001", "Asha")
    turn = convo.handle_audio(recorded_bytes)         # Whisper -> AI -> reply text + audio
    ...                                               # until turn.done
    result = convo.end()                              # result.saved only if backend confirmed
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# ── Make ai_engine importable (no code is copied) ────────────────────────────
AI_ENGINE_DIR = Path(
    os.getenv("AI_ENGINE_DIR", Path(__file__).resolve().parent.parent / "ai_engine")
).resolve()
if str(AI_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_ENGINE_DIR))

from src.conversation.call_manager import CallManager  # noqa: E402
from src.conversation.finalize import FinalizeResult  # noqa: E402


@dataclass
class VoiceTurn:
    patient_text: str               # what Whisper heard ("" if nothing)
    assistant_text: str             # what the assistant says next
    assistant_audio: Optional[bytes]  # MP3 bytes, or None if speech failed
    done: bool                      # True = closing message; end the call
    heard_nothing: bool = False     # nothing understood from the audio
    extraction_failed: bool = False # AI could not process the answer
    error: Optional[str] = None     # audio processing problem, if any


def _default_transcriber(audio_bytes: bytes, suffix: str) -> str:
    from speech_to_text import transcribe_bytes   # loads Whisper only when needed
    return transcribe_bytes(audio_bytes, suffix=suffix)


def _default_synthesizer(text: str) -> Optional[bytes]:
    from text_to_speech import generate_speech_bytes
    return generate_speech_bytes(text)


class VoiceConversation:
    def __init__(
        self,
        manager: Optional[CallManager] = None,
        transcriber: Optional[Callable[[bytes, str], str]] = None,
        synthesizer: Optional[Callable[[str], Optional[bytes]]] = None,
    ):
        self.manager = manager if manager is not None else CallManager()
        self.transcribe = transcriber or _default_transcriber
        self.speak = synthesizer or _default_synthesizer
        self.session = None

    # ── State ────────────────────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        return self.session is not None

    def _require_session(self):
        if self.session is None:
            raise RuntimeError("No active call. Call start() first.")
        return self.session

    def transcript(self) -> str:
        return self.session.transcript() if self.session else ""

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def startup(self) -> list[FinalizeResult]:
        """Save interrupted calls from last time and resend pending saves."""
        return self.manager.startup()

    def start(self, external_id: str, name: str) -> tuple[str, Optional[bytes]]:
        """Begin a call. Raises backend errors (no call is started if it fails)."""
        if self.session is not None:
            raise RuntimeError("A call is already in progress. End it first.")
        self.session, greeting = self.manager.start_call(external_id, name)
        return greeting, self.speak(greeting)

    def handle_text(self, text: str) -> VoiceTurn:
        session = self._require_session()
        text = (text or "").strip()
        reply = self.manager.handle_patient_message(session, text)
        return VoiceTurn(
            patient_text=text,
            assistant_text=reply.text,
            assistant_audio=self.speak(reply.text),
            done=reply.done,
            heard_nothing=not text,
            extraction_failed=reply.extraction_failed,
        )

    def handle_audio(self, audio_bytes: bytes, suffix: str = ".wav") -> VoiceTurn:
        self._require_session()
        error = None
        try:
            text = self.transcribe(audio_bytes, suffix)
        except Exception as e:                       # e.g. ffmpeg missing, bad audio
            print(f"[Voice] Transcription failed: {type(e).__name__}: {e}")
            text, error = "", f"Could not process the audio ({type(e).__name__})."
        turn = self.handle_text(text)
        turn.error = error
        return turn

    def end(self, interrupted: bool = False) -> FinalizeResult:
        """Save the call. interrupted=True when the connection dropped."""
        session = self._require_session()
        try:
            return self.manager.end_call(session, interrupted=interrupted)
        finally:
            self.session = None