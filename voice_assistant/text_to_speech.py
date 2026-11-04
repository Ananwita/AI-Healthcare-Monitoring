"""
Text-to-speech with gTTS (needs an internet connection).

Chunk C1 changes:
- generate_speech_bytes(): audio returned in memory, so every reply is
  separate (nothing is overwritten) and nothing is left on disk.
- Failures return None so the app can still show the text.
- generate_speech() keeps working as before.
"""

import io
import os
from typing import Optional

from gtts import gTTS

DEFAULT_LANG = "en"


def generate_speech_bytes(text: str, lang: str = DEFAULT_LANG) -> Optional[bytes]:
    """Return MP3 audio for `text`, or None if it could not be generated."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        buffer = io.BytesIO()
        gTTS(text=text, lang=lang).write_to_fp(buffer)
        return buffer.getvalue()
    except Exception as e:  # no internet, gTTS service error, ...
        print(f"[TTS] Could not generate speech: {type(e).__name__}: {e}")
        return None


def generate_speech(text, output_path="audio/assistant_response.mp3"):
    """Save speech to an MP3 file and return its path (original behaviour)."""
    folder = os.path.dirname(output_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    tts = gTTS(text=text, lang=DEFAULT_LANG)
    tts.save(output_path)
    return output_path