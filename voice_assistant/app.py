"""
AI Healthcare Voice Assistant — Streamlit app (Member 1, Chunk C3).

Run (backend must be running on port 8001):
    cd voice_assistant
    python -m streamlit run app.py

All conversation logic lives in conversation.py (tested); this file is only the UI.
"""

import streamlit as st

from conversation import VoiceConversation   # also makes ai_engine importable
from src.backend.client import BackendClient, BackendError, PatientConflictError

st.set_page_config(page_title="AI Healthcare Voice Assistant", page_icon="🩺")


# ── Cached resources (once per app process) ──────────────────────────────────

@st.cache_resource(show_spinner="Loading speech recognition model (first time only)...")
def load_whisper() -> bool:
    import speech_to_text
    speech_to_text.load_model()
    return True


@st.cache_resource(show_spinner="Checking for unfinished calls from earlier...")
def recover_earlier_calls() -> list[dict]:
    """Save interrupted drafts and resend pending calls (runs once per app start)."""
    results = VoiceConversation().startup()
    return [{"id": r.client_call_id[:8], "saved": r.saved, "message": r.message} for r in results]


@st.cache_data(ttl=10, show_spinner=False)
def backend_is_up() -> bool:
    return BackendClient(max_retries=0, timeout=3).health()


# ── Session state ────────────────────────────────────────────────────────────

DEFAULTS = {
    "convo": None,          # VoiceConversation during a call
    "messages": [],         # [(role, text)]
    "notice": None,         # (level, text) shown once
    "last_result": None,    # FinalizeResult of the last call
    "audio_key": 0,         # changes after each recording -> recorder resets
    "play_audio": None,     # MP3 bytes to autoplay once
    "patient_label": "",
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
state = st.session_state


# ── Actions ──────────────────────────────────────────────────────────────────

def start_call(external_id: str, name: str) -> None:
    convo = VoiceConversation()
    try:
        greeting, audio = convo.start(external_id, name)
    except PatientConflictError as e:
        state.notice = ("error", f"Patient ID '{external_id}' already belongs to a patient "
                                 f"with different details: {e.detail}")
        return
    except BackendError as e:
        state.notice = ("error", f"Could not start the call. Is the backend running? ({e})")
        return
    state.convo = convo
    state.messages = [("assistant", greeting)]
    state.play_audio = audio
    state.last_result = None
    state.patient_label = f"{name} ({external_id})"


def finish_call(interrupted: bool) -> None:
    state.last_result = state.convo.end(interrupted=interrupted)
    state.convo = None


def handle_turn(turn) -> None:
    if turn.patient_text:
        state.messages.append(("user", turn.patient_text))
    state.messages.append(("assistant", turn.assistant_text))
    state.play_audio = turn.assistant_audio

    if turn.error:
        state.notice = ("warning", turn.error)
    elif turn.heard_nothing:
        state.notice = ("info", "Nothing was heard. Please try recording again.")
    elif turn.extraction_failed:
        state.notice = ("warning", "The AI could not process that answer. The call continues.")

    if turn.done:
        finish_call(interrupted=False)


# ── Startup ──────────────────────────────────────────────────────────────────

load_whisper()
recovered = recover_earlier_calls()


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🩺 Call control")

    if backend_is_up():
        st.success("Backend connected")
    else:
        st.error("Backend not reachable. Start it: `cd monitoring` then "
                 "`python -m uvicorn backend.main:app --port 8001`")

    if state.convo is None:
        external_id = st.text_input("Patient ID", placeholder="e.g. P001").strip()
        name = st.text_input("Patient name").strip()
        if st.button("📞 Start call", type="primary", disabled=not (external_id and name)):
            start_call(external_id, name)
            st.rerun()
    else:
        st.write(f"**In call with:** {state.patient_label}")
        if st.button("✅ End call", type="primary"):
            finish_call(interrupted=False)
            st.rerun()
        if st.button("⚠️ Simulate dropped call"):
            finish_call(interrupted=True)
            st.rerun()

    if recovered:
        with st.expander(f"Earlier calls processed at startup ({len(recovered)})"):
            for r in recovered:
                st.write(("✅ " if r["saved"] else "⏳ ") + f"{r['id']}: {r['message']}")

    st.caption("This assistant records symptoms for monitoring. "
               "It does not diagnose. In an emergency, contact a doctor immediately.")


# ── Main area ────────────────────────────────────────────────────────────────

st.title("AI Healthcare Voice Assistant")

if state.notice:
    level, text = state.notice
    getattr(st, level)(text)
    state.notice = None

result = state.last_result
if result is not None:
    if result.saved:
        extra = " (it was already saved earlier)" if result.duplicate else ""
        st.success(f"Call saved. Call ID: {result.call_id}{extra}")
    elif result.failed:
        st.error(f"{result.message} File: {result.local_path}")
    elif result.local_path:
        st.warning(f"{result.message} File: {result.local_path}")
    else:
        st.info(result.message)

for role, text in state.messages:
    with st.chat_message(role):
        st.write(text)

if state.play_audio:
    st.audio(state.play_audio, format="audio/mp3", autoplay=True)
    state.play_audio = None           # play once; not again on the next rerun

if state.convo is not None:
    recording = st.audio_input("🎤 Record your answer", key=f"audio_{state.audio_key}")
    typed = st.chat_input("…or type your answer")

    if recording is not None:
        with st.spinner("Listening and thinking..."):
            turn = state.convo.handle_audio(recording.getvalue(), suffix=".wav")
        state.audio_key += 1          # fresh recorder: this recording is never reprocessed
        handle_turn(turn)
        st.rerun()
    elif typed:
        with st.spinner("Thinking..."):
            turn = state.convo.handle_text(typed)
        handle_turn(turn)
        st.rerun()
elif result is None:
    st.info("Enter the patient ID and name in the sidebar, then press **Start call**.")