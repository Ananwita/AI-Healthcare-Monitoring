"""Chunk C1: speech-to-text and text-to-speech (fake Whisper and fake gTTS)."""

import os
import sys

import pytest

import speech_to_text as stt
import text_to_speech as tts

fake_whisper = sys.modules["whisper"]
fake_gtts = sys.modules["gtts"]


@pytest.fixture(autouse=True)
def reset_fakes():
    stt._model = None
    stt._model_name = None
    fake_whisper.load_calls.clear()
    fake_gtts.fail = False
    yield


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "clip.wav"
    path.write_bytes(b"RIFF-fake-audio")
    return str(path)


# ── Speech-to-text ───────────────────────────────────────────────────────────

def test_model_loaded_once_and_reused(audio_file):
    stt.transcribe_audio(audio_file)
    stt.transcribe_audio(audio_file)
    stt.transcribe_audio(audio_file)
    assert fake_whisper.load_calls == ["base"]


def test_different_model_name_reloads():
    stt.load_model("base")
    stt.load_model("small")
    assert fake_whisper.load_calls == ["base", "small"]


def test_transcribe_options(audio_file):
    stt.transcribe_audio(audio_file)
    call = stt.load_model().transcribe_calls[0]
    assert call["language"] == "en"
    assert call["fp16"] is False


def test_text_is_stripped(audio_file):
    assert stt.transcribe_audio(audio_file) == "hello"


def test_silence_returns_empty(audio_file):
    stt.load_model().result = {"text": "  ", "segments": []}
    assert stt.transcribe_audio(audio_file) == ""


def test_no_speech_segments_dropped(audio_file):
    stt.load_model().result = {
        "text": "I have a fever. Thank you.",
        "segments": [
            {"text": " I have a fever.", "no_speech_prob": 0.05},
            {"text": " Thank you.", "no_speech_prob": 0.92},     # silence "heard" as words
        ],
    }
    assert stt.transcribe_audio(audio_file) == "I have a fever."


def test_all_segments_silent_returns_empty(audio_file):
    stt.load_model().result = {"text": " Thank you.",
                               "segments": [{"text": " Thank you.", "no_speech_prob": 0.95}]}
    assert stt.transcribe_audio(audio_file) == ""


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        stt.transcribe_audio("does/not/exist.wav")
    assert fake_whisper.load_calls == []


def test_transcribe_bytes_uses_and_deletes_temp_file():
    text = stt.transcribe_bytes(b"recorded-audio", suffix=".wav")
    call = stt.load_model().transcribe_calls[0]
    assert text == "hello"
    assert call["content"] == b"recorded-audio"
    assert call["path"].endswith(".wav")
    assert not os.path.exists(call["path"])


def test_transcribe_bytes_deletes_temp_file_on_error():
    stt.load_model().result = RuntimeError("ffmpeg missing")
    with pytest.raises(RuntimeError):
        stt.transcribe_bytes(b"audio")
    assert not os.path.exists(stt.load_model().transcribe_calls[0]["path"])


@pytest.mark.parametrize("empty", [b"", None])
def test_transcribe_bytes_empty_audio(empty):
    assert stt.transcribe_bytes(empty) == ""
    assert fake_whisper.load_calls == []


# ── Text-to-speech ───────────────────────────────────────────────────────────

def test_speech_bytes_returned():
    audio = tts.generate_speech_bytes("How are you feeling today?")
    assert audio == b"MP3[en]:How are you feeling today?"


def test_each_reply_has_its_own_audio():
    first = tts.generate_speech_bytes("Question one?")
    second = tts.generate_speech_bytes("Question two?")
    assert first != second


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_text_gives_none(text):
    assert tts.generate_speech_bytes(text) is None


def test_tts_failure_returns_none():
    fake_gtts.fail = True
    assert tts.generate_speech_bytes("Hello") is None


def test_generate_speech_file_still_works(tmp_path):
    path = tmp_path / "new_folder" / "reply.mp3"
    result = tts.generate_speech("Hello", str(path))
    assert result == str(path)
    assert path.read_bytes() == b"MP3[en]:Hello"