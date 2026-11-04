"""
Speech-to-text with OpenAI Whisper (runs locally).

Chunk C1 changes:
- The model is loaded ONCE and reused (previously reloaded on every call).
- Model size and language come from environment variables.
- Silence / non-speech segments are dropped (Whisper can "hear" words in silence).
- transcribe_bytes() for audio recorded in the browser (Streamlit).
"""

import os
import tempfile
from typing import Optional

import whisper

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
# "en" by default; set WHISPER_LANGUAGE=auto to let Whisper detect the language
_language_setting = os.getenv("WHISPER_LANGUAGE", "en").strip().lower()
WHISPER_LANGUAGE = None if _language_setting == "auto" else _language_setting

# Segments Whisper itself rates as probably "no speech" are ignored
NO_SPEECH_THRESHOLD = 0.6

_model = None
_model_name = None


def load_model(name: Optional[str] = None):
    """
    Load the Whisper speech recognition model once and reuse it.
    """
    global _model, _model_name
    name = name or WHISPER_MODEL
    if _model is None or _model_name != name:
        _model = whisper.load_model(name)
        _model_name = name
    return _model


def _text_from_result(result: dict) -> str:
    segments = result.get("segments")
    if segments:
        spoken = [
            (s.get("text") or "").strip()
            for s in segments
            if s.get("no_speech_prob", 0.0) <= NO_SPEECH_THRESHOLD
        ]
        return " ".join(t for t in spoken if t).strip()
    return (result.get("text") or "").strip()


def transcribe_audio(audio_path, language: Optional[str] = WHISPER_LANGUAGE) -> str:
    """
    Convert an audio file into text using Whisper.
    Returns "" when nothing was said.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = load_model()
    result = model.transcribe(audio_path, language=language, fp16=False)
    return _text_from_result(result)


def transcribe_bytes(
    audio_bytes: bytes,
    suffix: str = ".wav",
    language: Optional[str] = WHISPER_LANGUAGE,
) -> str:
    """
    Transcribe audio held in memory (e.g. from Streamlit's st.audio_input).
    Uses a temporary file that is always deleted afterwards.
    """
    if not audio_bytes:
        return ""

    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
        return transcribe_audio(path, language=language)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass