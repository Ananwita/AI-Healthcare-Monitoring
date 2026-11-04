"""
Chunk C2: voice app <-> AI engine bridge.
Runs the REAL ai_engine logic with a fake backend, fake LLM, fake Whisper and fake gTTS.
"""

import json

import pytest

import conversation
from conversation import VoiceConversation

from src.backend.client import BackendUnavailableError, SaveResult
from src.conversation.call_manager import CallManager
from src.llm.extraction import ExtractionResult


# ── Fakes ────────────────────────────────────────────────────────────────────

class FakeBackend:
    def __init__(self):
        self.patients, self.calls, self.down = {}, {}, False

    def create_or_get_patient(self, external_id, name, **kwargs):
        if self.down:
            raise BackendUnavailableError("down")
        pid = self.patients.setdefault(external_id, len(self.patients) + 1)
        return {"patient_id": pid, "created": True}

    def get_symptom_state(self, patient_id):
        state = {}
        for (pid, _), payload in self.calls.items():
            if pid == patient_id:
                for s in payload["symptoms"]:
                    state[s["name"]] = {"name": s["name"], "status": s["status"]}
        return list(state.values())

    def save_call(self, patient_id, payload):
        if self.down:
            raise BackendUnavailableError("down")
        key = (patient_id, payload["client_call_id"])
        duplicate = key in self.calls
        self.calls.setdefault(key, json.loads(json.dumps(payload)))
        return SaveResult(len(self.calls), patient_id, key[1], payload["status"],
                          len(payload["symptoms"]), duplicate)


SCRIPT = {
    "I have a fever.": [{"name": "fever"}],
    "Since yesterday.": [{"name": "fever", "duration": "since yesterday"}],
    "My fever is gone.": [{"name": "fever", "reported_change": "gone"}],
}


def fake_llm(text, context):
    return ExtractionResult(symptoms=SCRIPT.get(text, []))


def fake_whisper(audio_bytes, suffix):
    return audio_bytes.decode() if audio_bytes else ""     # "audio" is just the text


def fake_tts(text):
    return b"MP3:" + text.encode()


@pytest.fixture
def setup(tmp_path):
    backend = FakeBackend()

    def make(transcriber=fake_whisper, synthesizer=fake_tts):
        manager = CallManager(client=backend, extractor=fake_llm,
                              pending_dir=tmp_path / "pending", drafts_dir=tmp_path / "drafts")
        return VoiceConversation(manager, transcriber, synthesizer)
    return backend, make


# ── Wiring ───────────────────────────────────────────────────────────────────

def test_ai_engine_path_points_to_ai_engine_folder():
    assert (conversation.AI_ENGINE_DIR / "src" / "conversation" / "call_manager.py").exists()


def test_default_manager_uses_real_call_manager():
    assert isinstance(VoiceConversation(manager=CallManager(client=FakeBackend())).manager, CallManager)


# ── Full voice call ──────────────────────────────────────────────────────────

def test_voice_call_day1_then_day2(setup):
    backend, make = setup

    convo = make()
    greeting, audio = convo.start("P001", "Asha")
    assert "How are you feeling today?" in greeting
    assert audio == b"MP3:" + greeting.encode()

    turn = convo.handle_audio(b"I have a fever.")
    assert turn.patient_text == "I have a fever."
    assert turn.assistant_text == "When did your fever start?"
    assert turn.assistant_audio == b"MP3:When did your fever start?"

    convo.handle_audio(b"Since yesterday.")
    result = convo.end()
    assert result.saved is True
    assert convo.active is False

    convo2 = make()
    greeting2, _ = convo2.start("P001", "Asha")
    assert "Last time, you mentioned fever" in greeting2
    turn = convo2.handle_audio(b"My fever is gone.")
    assert turn.done is True
    assert "fever has gone" in turn.assistant_text
    assert convo2.end().saved is True
    assert len(backend.calls) == 2


def test_transcript_kept(setup):
    _, make = setup
    convo = make()
    convo.start("P001", "Asha")
    convo.handle_audio(b"I have a fever.")
    assert "Patient: I have a fever." in convo.transcript()


# ── Problems during a call ───────────────────────────────────────────────────

def test_silence_asks_again(setup):
    _, make = setup
    convo = make()
    convo.start("P001", "Asha")
    turn = convo.handle_audio(b"")
    assert turn.heard_nothing is True
    assert turn.assistant_text.startswith("Sorry, I didn't catch that.")
    assert turn.done is False


def test_transcription_error_does_not_end_call(setup):
    _, make = setup

    def broken(audio, suffix):
        raise RuntimeError("ffmpeg not found")

    convo = make(transcriber=broken)
    convo.start("P001", "Asha")
    turn = convo.handle_audio(b"x")
    assert turn.heard_nothing is True
    assert "Could not process the audio" in turn.error
    assert convo.active is True


def test_tts_failure_still_returns_text(setup):
    _, make = setup
    convo = make(synthesizer=lambda text: None)
    greeting, audio = convo.start("P001", "Asha")
    assert greeting and audio is None
    turn = convo.handle_text("I have a fever.")
    assert turn.assistant_text == "When did your fever start?"
    assert turn.assistant_audio is None


# ── Lifecycle rules ──────────────────────────────────────────────────────────

def test_cannot_talk_before_start(setup):
    _, make = setup
    with pytest.raises(RuntimeError):
        make().handle_text("hello")
    with pytest.raises(RuntimeError):
        make().end()


def test_cannot_start_twice(setup):
    _, make = setup
    convo = make()
    convo.start("P001", "Asha")
    with pytest.raises(RuntimeError):
        convo.start("P001", "Asha")


def test_backend_down_at_start_leaves_no_call(setup):
    backend, make = setup
    backend.down = True
    convo = make()
    with pytest.raises(BackendUnavailableError):
        convo.start("P001", "Asha")
    assert convo.active is False


def test_interrupted_call_saved_as_interrupted(setup):
    backend, make = setup
    convo = make()
    convo.start("P001", "Asha")
    convo.handle_audio(b"I have a fever.")
    result = convo.end(interrupted=True)
    assert result.saved is True
    assert list(backend.calls.values())[0]["status"] == "interrupted"


def test_backend_down_at_end_reports_not_saved_then_startup_recovers(setup):
    backend, make = setup
    convo = make()
    convo.start("P001", "Asha")
    convo.handle_audio(b"I have a fever.")
    backend.down = True
    result = convo.end()
    assert result.saved is False
    assert convo.active is False

    backend.down = False
    assert [r.saved for r in make().startup()] == [True]
    assert len(backend.calls) == 1