
import os
import re
import html
from datetime import datetime, timezone
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CAREPULSE — PATIENT MONITORING DASHBOARD
# Streamlit: http://localhost:8502
# FastAPI:   http://127.0.0.1:8001
# ============================================================

API_URL = os.getenv(
    "CAREPULSE_API_URL",
    "http://127.0.0.1:8001"
).rstrip("/")

TIMEOUT = 8

st.set_page_config(
    page_title="CarePulse | Patient Monitoring",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --page: #F3F6FA;
        --card: #FFFFFF;
        --navy: #17365D;
        --blue: #2864A5;
        --text: #24364B;
        --muted: #62748A;
        --border: #DCE5EF;
        --soft-blue: #EAF3FF;
    }

    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background: var(--page) !important;
        color: var(--text) !important;
        font-family: "Segoe UI", Arial, sans-serif;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1600px;
        padding: 2rem 2.3rem 3rem;
    }

    h1, h2, h3, h4 {
        color: var(--navy) !important;
        letter-spacing: -0.35px;
    }

    p, label, li,
    [data-testid="stMarkdownContainer"] {
        color: var(--text);
    }

    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        color: var(--muted) !important;
    }

    [data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 1.6rem 1.1rem;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: var(--text) !important;
    }

    .brand {
        font-size: 27px;
        font-weight: 800;
        color: #245B97 !important;
        letter-spacing: -.7px;
    }

    .brand-caption {
        color: #62748A !important;
        font-size: 10px;
        letter-spacing: 1.4px;
        margin: 3px 0 28px;
    }

    .hero {
        background: linear-gradient(120deg, #17365D, #2864A5);
        border-radius: 16px;
        padding: 29px 32px;
        margin-bottom: 25px;
    }

    .hero-kicker {
        color: #DCEAFF !important;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        margin-bottom: 9px;
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-size: 30px;
        margin: 0 0 8px;
    }

    .hero p {
        color: #E9F1FB !important;
        margin: 0;
        font-size: 15px;
    }

    .page-subtitle {
        color: #62748A !important;
        font-size: 14px;
        margin: -5px 0 22px;
    }

    .metric {
        background: #FFFFFF;
        border: 1px solid #DCE5EF;
        border-radius: 12px;
        padding: 18px;
        min-height: 105px;
        box-shadow: 0 3px 12px rgba(23, 54, 93, .04);
    }

    .metric-label {
        color: #62748A !important;
        font-size: 13px;
        margin-bottom: 11px;
    }

    .metric-value {
        color: #17365D !important;
        font-size: 26px;
        font-weight: 750;
        line-height: 1.3;
        overflow-wrap: anywhere;
    }

    .metric-detail {
        color: #62748A !important;
        font-size: 12px;
        margin-top: 5px;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF !important;
        border: 1px solid #DCE5EF !important;
        border-radius: 13px !important;
        box-shadow: 0 4px 16px rgba(23, 54, 93, .035);
    }

    .section-title {
        color: #17365D !important;
        font-size: 22px;
        font-weight: 750;
        margin: 15px 0;
    }

    .card-title {
        color: #244C79 !important;
        font-size: 16px;
        font-weight: 750;
        padding-bottom: 13px;
        border-bottom: 1px solid #E6ECF4;
        margin-bottom: 18px;
    }

    .field-label {
        color: #62748A !important;
        font-size: 11px;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: .6px;
        margin-bottom: 6px;
    }

    .field-value {
        color: #24364B !important;
        font-size: 15px;
        font-weight: 600;
        line-height: 1.5;
        overflow-wrap: anywhere;
        margin-bottom: 20px;
    }

    .history-box {
        background: #F7F9FC;
        border: 1px solid #DCE5EF;
        border-radius: 9px;
        padding: 15px;
        color: #24364B !important;
        line-height: 1.7;
        overflow-wrap: anywhere;
    }

    .call-banner {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 15px;
        background: #F0F6FF;
        border: 1px solid #D5E5FA;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 20px;
    }

    .summary {
        background: #EAF3FF;
        border: 1px solid #D6E6FA;
        border-left: 4px solid #2864A5;
        border-radius: 8px;
        padding: 15px 17px;
        color: #24364B !important;
        line-height: 1.75;
        overflow-wrap: anywhere;
        margin: 8px 0 20px;
    }

    .symptom {
        background: #FFFFFF;
        border: 1px solid #DCE5EF;
        border-radius: 9px;
        padding: 14px 16px;
        margin: 9px 0;
    }

    .symptom-name {
        color: #24364B !important;
        font-size: 15px;
        font-weight: 700;
    }

    .severity {
        color: #62748A !important;
        font-size: 13px;
        margin-top: 6px;
    }

    .timeline-note {
        background: #F0F6FF;
        border: 1px solid #D5E5FA;
        border-radius: 9px;
        padding: 13px 16px;
        color: #344A65 !important;
        font-size: 13px;
        line-height: 1.6;
        margin: 8px 0 18px;
    }

    /* Readable input controls */
    div[data-baseweb="select"] > div {
        background: #FFFFFF !important;
        border-color: #C8D5E5 !important;
    }

    div[data-baseweb="select"] input,
    div[data-baseweb="select"] span {
        color: #24364B !important;
        -webkit-text-fill-color: #24364B !important;
    }

    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input {
        background: #FFFFFF !important;
        color: #24364B !important;
        -webkit-text-fill-color: #24364B !important;
        border: 1px solid #C8D5E5 !important;
        caret-color: #2864A5 !important;
        opacity: 1 !important;
    }

    [data-testid="stTextArea"] textarea:disabled {
        background: #F7F9FC !important;
        color: #24364B !important;
        -webkit-text-fill-color: #24364B !important;
        opacity: 1 !important;
    }

    /* Native tables */
    [data-testid="stTable"],
    [data-testid="stDataFrame"] {
        color: #24364B !important;
    }

    [data-testid="stAlert"] {
        border-radius: 9px;
    }

    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span {
        color: #24364B !important;
    }

    .footer {
        border-top: 1px solid #DCE5EF;
        margin-top: 30px;
        padding-top: 16px;
        color: #62748A !important;
        font-size: 12px;
        line-height: 1.8;
    }

    hr {
        border-color: #DCE5EF !important;
    }

    footer {
        visibility: hidden;
    }

    @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1.1rem 1rem 2rem;
        }

        .hero {
            padding: 23px 20px;
        }

        .hero h1 {
            font-size: 25px;
        }

        .call-banner {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API HELPERS
# ============================================================

@st.cache_data(ttl=20, show_spinner=False)
def get_api_paths(base_url):
    try:
        response = requests.get(
            f"{base_url}/openapi.json",
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json().get("paths", {})
    except (requests.RequestException, ValueError, AttributeError):
        return {}


def get_json(path):
    try:
        response = requests.get(
            f"{API_URL}{path}",
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)
    except ValueError:
        return None, "The endpoint did not return valid JSON."


def get_first(item, *keys, default="Not provided"):
    if not isinstance(item, dict):
        return default

    for key in keys:
        result = item.get(key)
        if result is not None and result != "":
            return result

    return default


def as_list(data, keys=()):
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in keys:
            if isinstance(data.get(key), list):
                return data[key]

    return []


def safe(value):
    return html.escape(str(value))


def format_date(raw):
    if raw in (None, "", "Not provided"):
        return "Not available"

    try:
        parsed = datetime.fromisoformat(
            str(raw).replace("Z", "+00:00")
        )
        # Backend stores UTC without a timezone marker -> show local time
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone().strftime("%d %b %Y, %I:%M %p")
    except (ValueError, TypeError):
        return str(raw)


def get_patient_id(patient):
    return get_first(patient, "id", "patient_id", default=1)


def get_patient_name(patient):
    return get_first(
        patient, "name", "patient_name", "full_name",
        default="Unnamed patient",
    )


def get_symptom_name(item):
    return get_first(
        item, "symptom", "symptom_name", "name", "condition",
        default="Unspecified symptom",
    )


def get_severity(item):
    return get_first(
        item, "severity", "reported_severity", "severity_level",
        default="Not recorded",
    )


def get_call_date(call):
    return get_first(
        call, "created_at", "call_date", "date", "timestamp",
        default=None,
    )


def get_call_summary(call):
    return get_first(
        call, "summary", "health_summary", "conversation_summary",
        default="No conversation summary available.",
    )


def get_transcript(call):
    return get_first(
        call, "transcript", "conversation", "full_transcript",
        default="No transcript available.",
    )

STATUS_LABELS = {
    "ongoing": "🟠 Ongoing",
    "resolved": "🟢 Resolved",
}

TREND_LABELS = {
    "new": "🆕 New",
    "worsening": "🔺 Worsening",
    "stable": "➖ Stable",
    "improving": "🔻 Improving",
    "resolved": "✅ Resolved",
}

CALL_STATUS_LABELS = {
    "completed": "✅ Completed",
    "interrupted": "⚠️ Interrupted",
}


def get_status_label(item):
    raw = str(get_first(item, "status", default="")).lower()
    return STATUS_LABELS.get(raw, "Not recorded")


def get_trend_label(item, *keys):
    raw = str(get_first(item, *(keys or ("trend",)), default="")).lower()
    return TREND_LABELS.get(raw, "Not recorded")


def get_call_status_label(call):
    raw = str(get_first(call, "status", default="completed")).lower()
    return CALL_STATUS_LABELS.get(raw, raw.title())

def show_field(label, content):
    st.markdown(
        f'<div class="field-label">{safe(label)}</div>'
        f'<div class="field-value">{safe(content)}</div>',
        unsafe_allow_html=True,
    )


def show_table(rows, columns):
    if not rows:
        st.info("No records are available.")
        return

    st.dataframe(
        pd.DataFrame(rows, columns=columns),
        use_container_width=True,
        hide_index=True,
    )


def endpoint_candidates(paths, resource):
    """Find likely GET endpoints using the actual FastAPI OpenAPI schema."""
    found = []

    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue

        if "get" not in methods:
            continue

        clean = path.lower()
        if any(x in clean for x in ("openapi", "docs", "redoc")):
            continue

        segments = clean.strip("/").split("/")
        resource_match = any(
            resource in segment
            or (
                resource == "calls"
                and segment in ("call", "call-history", "call_history")
            )
            or (
                resource == "symptoms"
                and segment in ("symptom", "symptom-records", "symptom_records")
            )
            for segment in segments
        )

        if resource_match:
            score = 0

            if "{" in path:
                score += 2
            if "patient" in clean:
                score += 3
            if clean.endswith("/" + resource):
                score += 1

            found.append((score, path))

    found.sort(key=lambda item: (-item[0], item[1]))
    return [path for _, path in found]


def fetch_resource(resource, patient_id, paths):
    """
    Try discovered GET endpoints without assuming a fixed API route.
    Only return data from an endpoint that responds successfully.
    """
    candidates = endpoint_candidates(paths, resource)
    errors = []

    for template in candidates:
        path = template

        for name in ("patient_id", "id"):
            path = re.sub(
                r"\{" + name + r"\}",
                str(patient_id),
                path,
            )

        # Skip routes requiring other unresolved path parameters.
        if "{" in path:
            continue

        data, error = get_json(path)

        if error is None:
            return data, None, path

        errors.append(f"{path}: {error}")

    detail = (
        "; ".join(errors[:2])
        if errors
        else f"No matching GET endpoint found for {resource}."
    )

    return None, detail, None


# ============================================================
# SIDEBAR AND PATIENT SELECTION
# ============================================================

paths = get_api_paths(API_URL)

with st.sidebar:
    st.markdown(
        '<div class="brand">🏥 CarePulse</div>'
        '<div class="brand-caption">PATIENT MONITORING</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Patient workspace")


    patient_data, patient_error = get_json("/patients")
    patient_endpoint = "/patients" if patient_error is None else None


    patients = as_list(
        patient_data,
        ("patients", "items", "results", "data"),
    )

    # Clearly identified fallback preview if the API has no patient route.
    if not patients:
        patients = [{
            "id": 1,
            "name": "Demo Patient",
            "age": 55,
            "language": "English",
            "medical_history": "Demo data for testing",
            "_demo": True,
        }]

    patient_options = {}

    for index, item in enumerate(patients):
        pid = get_first(item, "id", "patient_id", default=index + 1)
        name = get_patient_name(item)
        patient_options[f"{name} · #{pid}"] = item

    selected_label = st.selectbox(
        "Select patient",
        list(patient_options.keys()),
        key="carepulse_patient",
    )

    patient = patient_options[selected_label]
    patient_id = get_patient_id(patient)

    st.divider()
    st.markdown("### Navigation")

    page = st.radio(
        "Dashboard sections",
        [
            "Patient overview",
            "Symptom timeline",
            "Call history",
            "Monitoring status",
        ],
        label_visibility="collapsed",
        key="carepulse_navigation",
    )

    st.divider()
    st.caption("CarePulse · Research prototype")
    st.caption("Informational monitoring, not a diagnostic system.")


# ============================================================
# LOAD SELECTED PATIENT DATA
# ============================================================

is_demo = bool(patient.get("_demo", False))

patient_details, patient_detail_error, patient_detail_endpoint = (
    fetch_resource("patients", patient_id, paths)
)

if isinstance(patient_details, dict):
    patient = {**patient, **patient_details}

calls_data, calls_error, calls_endpoint = fetch_resource(
    "calls", patient_id, paths
)

symptoms_data, symptoms_error, symptoms_endpoint = fetch_resource(
    "symptoms", patient_id, paths
)

calls = as_list(calls_data, ("calls", "items", "results", "data"))
symptoms = as_list(
    symptoms_data, ("symptoms", "items", "results", "data")
)

# Recover embedded symptoms from call responses when necessary.
if not symptoms:
    for call in calls:
        embedded = call.get("symptoms", [])

        if isinstance(embedded, list):
            for item in embedded:
                if isinstance(item, dict):
                    symptoms.append(item)
                else:
                    symptoms.append({
                        "symptom": str(item),
                        "severity": "Not recorded",
                        "created_at": get_call_date(call),
                    })

# Latest state of every symptom across ALL calls (backend Chunk A5).
# Optional: sections using it are hidden if the endpoint is unavailable.
symptom_state = []
if not is_demo:
    state_data, state_error = get_json(f"/patients/{patient_id}/symptom-state")
    if state_error is None:
        symptom_state = as_list(state_data)

patient_name = get_patient_name(patient)
external_id = get_first(patient, "external_id", default="Not provided")
patient_age = get_first(patient, "age", default="Not provided")
patient_language = get_first(
    patient, "language", "preferred_language",
    default="Not provided",
)
medical_history = get_first(
    patient, "medical_history", "history",
    default="No medical history recorded.",
)

# Sort calls chronologically when timestamp data is available.
def sort_key(record):
    raw = get_call_date(record)
    if not raw:
        return ""
    return str(raw)


calls = sorted(calls, key=sort_key)
latest_call = calls[-1] if calls else {}


# ============================================================
# COMMON HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">CARE MANAGEMENT WORKSPACE</div>
        <h1>Patient Monitoring Center</h1>
        <p>Patient history, conversation records and symptom tracking.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f'<h2 style="margin-bottom:5px">{safe(patient_name)}'
    f' <span style="font-size:17px;font-weight:500;color:#62748A">'
    f'Patient ID: {safe(patient_id)}</span></h2>'
    f'<div class="page-subtitle">Preferred language: '
    f'{safe(patient_language)} &nbsp; · &nbsp; '
    f'Age: {safe(patient_age)}</div>',
    unsafe_allow_html=True,
)

if is_demo:
    st.info(
        "Preview patient shown because the patient list could not be loaded. "
        "Live patient data will appear when a matching backend endpoint "
        "is available."
    )


# ============================================================
# PATIENT OVERVIEW
# ============================================================

if page == "Patient overview":

    latest_date = get_call_date(latest_call) if latest_call else None

    unique_symptoms = len({
        str(get_symptom_name(item)).strip().lower()
        for item in symptoms
    })

    metrics = [
        ("Total calls", len(calls)),
        ("Symptom records", len(symptoms)),
        ("Symptoms tracked", unique_symptoms),
        (
            "Latest interaction",
            format_date(latest_date) if latest_date else "No calls yet",
        ),
    ]

    metric_columns = st.columns(4, gap="medium")

    for col, (label, number) in zip(metric_columns, metrics):
        with col:
            font_size = "17px" if label == "Latest interaction" else "27px"

            st.markdown(
                f'<div class="metric">'
                f'<div class="metric-label">{safe(label)}</div>'
                f'<div class="metric-value" style="font-size:{font_size}">'
                f'{safe(number)}</div></div>',
                unsafe_allow_html=True,
            )

    st.write("")

    left, right = st.columns([0.95, 1.05], gap="large")

    # Patient profile
    with left:
        st.markdown(
            '<div class="section-title">👤 Patient profile</div>',
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown(
                '<div class="card-title">Personal information</div>',
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2, gap="medium")

            with c1:
                show_field("Patient name", patient_name)
                show_field(
                    "Age",
                    f"{patient_age} years"
                    if str(patient_age).isdigit()
                    else patient_age,
                )

            with c2:
                show_field("Patient ID", f"#{patient_id}")
                show_field("External ID", external_id)
                show_field("Preferred language", patient_language)

            st.markdown(
                '<div class="card-title">🩺 Medical background</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="history-box">{safe(medical_history)}</div>',
                unsafe_allow_html=True,
            )

    # Latest conversation
    with right:
        st.markdown(
            '<div class="section-title">☎️ Latest conversation</div>',
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown(
                '<div class="card-title">Call information</div>',
                unsafe_allow_html=True,
            )

            if latest_call:
                call_id = get_first(
                    latest_call, "id", "call_id", default="—"
                )
                call_date = get_call_date(latest_call)
                summary = get_call_summary(latest_call)
                transcript = get_transcript(latest_call)

                st.markdown(
                    f'<div class="call-banner">'
                    f'<div><div class="field-label">Call ID</div>'
                    f'<div class="field-value">#{safe(call_id)}</div></div>'
                    f'<div><div class="field-label">Date and time</div>'
                    f'<div class="field-value">'
                    f'{safe(format_date(call_date))}</div></div>'
                    f'<div><div class="field-label">Call status</div>'
                    f'<div class="field-value">'
                    f'{safe(get_call_status_label(latest_call))}</div></div></div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    '<div class="field-label">Conversation summary</div>'
                    f'<div class="summary">{safe(summary)}</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("**🩺 Symptoms reported**")

                latest_symptoms = [
                    item for item in latest_call.get("symptoms", [])
                    if isinstance(item, dict)
                ]

                if latest_symptoms:
                    for item in latest_symptoms:
                        st.markdown(
                            f'<div class="symptom">'
                            f'<div class="symptom-name">'
                            f'{safe(get_symptom_name(item))}</div>'
                            f'<div class="severity">Severity: '
                            f'{safe(get_severity(item))} · '
                            f'{safe(get_status_label(item))} · '
                            f'{safe(get_trend_label(item))}</div></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No symptoms recorded in this call.")

                with st.expander("Read full conversation transcript"):
                    st.text_area(
                        "Conversation transcript",
                        value=str(transcript),
                        height=220,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"overview_transcript_{patient_id}",
                    )
            else:
                st.info("No calls have been recorded for this patient.")

    if symptom_state:
        st.markdown(
            '<div class="section-title">🩺 Current symptom status</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Latest state of each symptom across all calls. A symptom is "
            "only marked resolved when the patient said so."
        )

        state_rows = []
        for item in symptom_state:
            state_rows.append({
                "Symptom": get_symptom_name(item),
                "Status": get_status_label(item),
                "Latest trend": get_trend_label(item, "latest_trend"),
                "Last severity": get_first(
                    item, "latest_severity", default="Not recorded"
                ),
                "Times reported": get_first(
                    item, "times_reported", default="—"
                ),
                "Came back": "Yes" if item.get("recurred") else "No",
                "Last reported": format_date(
                    get_first(item, "last_reported_at", default=None)
                ),
            })

        show_table(
            state_rows,
            ["Symptom", "Status", "Latest trend", "Last severity",
             "Times reported", "Came back", "Last reported"],
        )

    st.markdown(
        '<div class="section-title">Recent activity</div>',
        unsafe_allow_html=True,
    )

    if calls:
        activity = []

        for call in reversed(calls[-10:]):
            activity.append({
                "Call ID": get_first(call, "id", "call_id", default="—"),
                "Date and time": format_date(get_call_date(call)),
                "Status": get_call_status_label(call),
                "Summary": get_call_summary(call),
            })

        show_table(
            activity,
            ["Call ID", "Date and time", "Status", "Summary"],
        )
    else:
        st.info("No previous interactions are available.")


# ============================================================
# SYMPTOM TIMELINE
# ============================================================

elif page == "Symptom timeline":

    st.markdown(
        '<div class="section-title">📈 Symptom timeline</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Track reported symptom severity across recorded interactions. '
        'The timeline displays recorded information only.'
        '</div>',
        unsafe_allow_html=True,
    )

    if symptoms:
        symptom_rows = []

        for index, item in enumerate(symptoms):
            raw_date = get_first(
                item,
                "created_at", "recorded_at", "date", "timestamp",
                default=None,
            )

            symptom_rows.append({
                "Symptom": get_symptom_name(item),
                "Severity": str(get_severity(item)).strip().title(),
                "Status": get_status_label(item),
                "Trend": get_trend_label(item),
                "Date": format_date(raw_date),
                "Raw date": str(raw_date) if raw_date else "",
                "Record": index + 1,
            })

        df = pd.DataFrame(symptom_rows)

        # Severity map for the visualization.
        severity_map = {
            "mild": 1,
            "low": 1,
            "moderate": 2,
            "medium": 2,
            "severe": 3,
            "high": 3,
            "critical": 3,
        }

        st.markdown(
            '<div class="section-title">Recorded symptoms</div>',
            unsafe_allow_html=True,
        )

        show_table(
            df[["Symptom", "Severity", "Status", "Trend", "Date"]].to_dict("records"),
            ["Symptom", "Severity", "Status", "Trend", "Date"],
        )

        symptom_names = sorted(
            df["Symptom"].dropna().astype(str).unique().tolist()
        )

        selected_symptom = st.selectbox(
            "Select a symptom to view its timeline",
            symptom_names,
            key="timeline_symptom",
        )

        chart_df = df[df["Symptom"] == selected_symptom].copy()

        chart_df["Score"] = (
            chart_df["Severity"].str.lower().map(severity_map)
        )

        chart_df = chart_df.dropna(subset=["Score"])

        if not chart_df.empty:

            # Sort by date if the backend supplies parseable timestamps.
            chart_df["Parsed date"] = pd.to_datetime(
                chart_df["Raw date"],
                errors="coerce",
                utc=True,
            )

            if chart_df["Parsed date"].notna().any():
                chart_df = chart_df.sort_values(
                    ["Parsed date", "Record"],
                    na_position="last",
                )

            # A real date is displayed where available; otherwise use
            # a record number rather than inventing a date.
            chart_df["Timeline label"] = chart_df.apply(
                lambda row: (
                    row["Date"]
                    if row["Raw date"]
                    and row["Date"] != "Not available"
                    else f"Record {int(row['Record'])}"
                ),
                axis=1,
            )

            chart_df = chart_df.reset_index(drop=True)

            # Use evenly spaced points to keep labels readable.
            x_values = list(range(1, len(chart_df) + 1))

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=chart_df["Score"],
                    mode="lines+markers",
                    line=dict(color="#2864A5", width=3),
                    marker=dict(
                        color="#2864A5",
                        size=11,
                        line=dict(color="#FFFFFF", width=2),
                    ),
                    customdata=chart_df[
                        ["Timeline label", "Severity"]
                    ].values,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Severity: %{customdata[1]}"
                        "<extra></extra>"
                    ),
                    name=selected_symptom,
                )
            )

            fig.update_layout(
                title=dict(
                    text=f"Severity over time — {selected_symptom}",
                    font=dict(size=19, color="#17365D"),
                    x=0.02,
                ),
                template="plotly_white",
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(
                    family="Segoe UI, Arial, sans-serif",
                    color="#24364B",
                    size=13,
                ),
                height=440,
                margin=dict(l=65, r=30, t=85, b=95),
                xaxis=dict(
                    title="Recorded interaction",
                    tickmode="array",
                    tickvals=x_values,
                    ticktext=chart_df["Timeline label"].tolist(),
                    tickangle=-25 if len(chart_df) > 3 else 0,
                    tickfont=dict(color="#344A65", size=11),
                    title_font=dict(color="#24364B", size=13),
                    showgrid=False,
                    linecolor="#C8D5E5",
                    automargin=True,
                ),
                yaxis=dict(
                    title="Reported severity",
                    tickmode="array",
                    tickvals=[1, 2, 3],
                    ticktext=["Mild", "Moderate", "Severe"],
                    range=[0.65, 3.35],
                    tickfont=dict(color="#344A65", size=12),
                    title_font=dict(color="#24364B", size=13),
                    gridcolor="#E4EAF2",
                    linecolor="#C8D5E5",
                    zeroline=False,
                    automargin=True,
                ),
                hoverlabel=dict(
                    bgcolor="#FFFFFF",
                    bordercolor="#C8D5E5",
                    font=dict(color="#24364B", size=13),
                ),
                showlegend=False,
            )

            st.markdown(
                '<div class="timeline-note">'
                '<strong>How to read this chart:</strong> '
                'Mild = 1, Moderate = 2, Severe = 3. '
                'Points represent recorded symptom reports; a line between '
                'points does not imply a medical trend or diagnosis.'
                '</div>',
                unsafe_allow_html=True,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "responsive": True,
                    "scrollZoom": False,
                },
                theme="streamlit",
            )

        else:
            st.warning(
                "The records do not contain recognized severity values "
                "(mild, moderate, severe, low, medium, high or critical). "
                "The symptom records are shown above, but a severity "
                "timeline cannot be plotted."
            )

    else:
        st.info(
            "No symptom records are available for this patient. "
            "The timeline will appear when symptom records are returned "
            "by the backend."
        )


# ============================================================
# CALL HISTORY
# ============================================================

elif page == "Call history":

    st.markdown(
        '<div class="section-title">☎️ Call history</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Review previous interactions, summaries and full transcripts.'
        '</div>',
        unsafe_allow_html=True,
    )

    if calls:
        rows = []

        for call in reversed(calls):
            rows.append({
                "Call ID": get_first(
                    call, "id", "call_id", default="—"
                ),
                "Date and time": format_date(get_call_date(call)),
                "Status": get_call_status_label(call),
                "Summary": get_call_summary(call),
            })

        show_table(
            rows,
            ["Call ID", "Date and time", "Status", "Summary"],
        )

        st.markdown(
            '<div class="section-title">Conversation details</div>',
            unsafe_allow_html=True,
        )

        indices = list(range(len(calls)))

        chosen_index = st.selectbox(
            "Select an interaction",
            indices,
            format_func=lambda i: (
                f"Call #{get_first(calls[i], 'id', 'call_id', default=i + 1)}"
                f" — {format_date(get_call_date(calls[i]))}"
            ),
            key="history_call",
        )

        selected_call = calls[chosen_index]

        st.markdown(f"**Call status:** {get_call_status_label(selected_call)}")

        st.markdown("**Conversation summary**")

        st.markdown(
            f'<div class="summary">'
            f'{safe(get_call_summary(selected_call))}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("**Full transcript**")

        st.text_area(
            "Full transcript",
            value=str(get_transcript(selected_call)),
            height=300,
            disabled=True,
            label_visibility="collapsed",
            key=f"history_transcript_{patient_id}_{chosen_index}",
        )

    else:
        st.info("No call history is available for this patient.")


# ============================================================
# MONITORING STATUS
# ============================================================

elif page == "Monitoring status":

    st.markdown(
        '<div class="section-title">🔔 Monitoring status</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Review recorded symptom severity and backend connectivity.'
        '</div>',
        unsafe_allow_html=True,
    )

    if symptom_state:
        # Only symptoms that are CURRENTLY ongoing with a severe last report
        high_severity = [
            {**item, "severity": item.get("latest_severity")}
            for item in symptom_state
            if str(item.get("status", "")).lower() == "ongoing"
            and str(item.get("latest_severity") or "").strip().lower()
            in {"severe", "high", "critical"}
        ]
    else:
        high_severity = [
            item for item in symptoms
            if str(get_severity(item)).strip().lower()
            in {"severe", "high", "critical"}
        ]

    if high_severity:
        st.warning(
            f"{len(high_severity)} symptom record(s) are marked high, "
            "severe or critical. Review these records with an appropriate "
            "healthcare professional."
        )

        show_table(
            [
                {
                    "Symptom": get_symptom_name(item),
                    "Reported severity": get_severity(item),
                }
                for item in high_severity
            ],
            ["Symptom", "Reported severity"],
        )
    elif symptoms:
        st.success(
            "No records are explicitly marked high, severe or critical."
        )
    else:
        st.info(
            "No symptom records are available to assess monitoring status."
        )

    st.caption(
        "Prototype rules only. This dashboard does not predict medical risk "
        "or replace professional assessment."
    )

    st.markdown(
        '<div class="section-title">Backend connection</div>',
        unsafe_allow_html=True,
    )

    try:
        response = requests.get(f"{API_URL}/", timeout=TIMEOUT)

        if response.ok:
            st.success(f"API is reachable at {API_URL}.")
        else:
            st.error(f"API returned HTTP {response.status_code}.")

    except requests.RequestException as exc:
        st.error(f"Cannot connect to the API: {exc}")

    st.markdown("**Data endpoint status**")

    for label, endpoint, error in [
        ("Patients", patient_endpoint, patient_error),
        ("Patient details", patient_detail_endpoint, patient_detail_error),
        ("Calls", calls_endpoint, calls_error),
        ("Symptoms", symptoms_endpoint, symptoms_error),
    ]:
        if endpoint and error is None:
            st.success(f"{label}: connected via {endpoint}")
        else:
            st.warning(f"{label}: {error or 'No matching endpoint found.'}")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <strong>CarePulse</strong> · Patient Monitoring Workspace
        · Academic research prototype
        <br>
        Informational monitoring only — not a diagnostic system.
    </div>
    """,
    unsafe_allow_html=True,
)
