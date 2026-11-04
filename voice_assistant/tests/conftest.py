"""
Install FAKE whisper and gtts modules before the app modules are imported,
so tests run fast, offline, and without downloading the Whisper model.
"""

import sys
import types

# ── Fake whisper ─────────────────────────────────────────────────────────────

fake_whisper = types.ModuleType("whisper")
fake_whisper.load_calls = []


class FakeWhisperModel:
    def __init__(self, name):
        self.name = name
        self.result = {"text": " hello ", "segments": None}
        self.transcribe_calls = []

    def transcribe(self, path, **kwargs):
        with open(path, "rb") as f:
            content = f.read()
        self.transcribe_calls.append({"path": path, "content": content, **kwargs})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def fake_load_model(name):
    fake_whisper.load_calls.append(name)
    return FakeWhisperModel(name)


fake_whisper.load_model = fake_load_model
sys.modules["whisper"] = fake_whisper

# ── Fake gtts ────────────────────────────────────────────────────────────────

fake_gtts = types.ModuleType("gtts")
fake_gtts.fail = False


class FakeGTTS:
    def __init__(self, text, lang="en"):
        if fake_gtts.fail:
            raise ConnectionError("no internet")
        self.text, self.lang = text, lang

    def _audio(self):
        return f"MP3[{self.lang}]:{self.text}".encode()

    def write_to_fp(self, fp):
        fp.write(self._audio())

    def save(self, path):
        with open(path, "wb") as f:
            f.write(self._audio())


fake_gtts.gTTS = FakeGTTS
sys.modules["gtts"] = fake_gtts