import os
import io
import base64
import hashlib
import importlib
from datetime import datetime
from uuid import uuid4

import streamlit as st
from custom_favicon import FAVICON_PATH, apply_custom_favicon
from dotenv import dotenv_values, load_dotenv
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from PIL import Image
from streamlit.errors import StreamlitSecretNotFoundError

load_dotenv("key.env", override=True)

st.set_page_config(
    page_title="Doctor Assistant - AI Medical Assistant",
    page_icon=str(FAVICON_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_favicon()

# Inject Google Fonts and improved CSS
st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    :root {
        --bg-main: #eef4f8;
        --bg-surface: rgba(255, 255, 255, 0.94);
        --bg-surface-strong: #ffffff;
        --bg-sidebar: linear-gradient(180deg, #0d2741 0%, #133b5a 100%);
        --bg-sidebar-card: linear-gradient(180deg, rgba(255, 255, 255, 0.14) 0%, rgba(255, 255, 255, 0.08) 100%);
        --bg-brand: linear-gradient(135deg, #0d5a78 0%, #0f7690 45%, #26a7a2 100%);
        --bg-brand-soft: linear-gradient(135deg, #edf8f7 0%, #f9fcff 100%);
        --text-primary: #0f2234;
        --text-secondary: #4f667b;
        --text-inverse: #f8fbff;
        --border-soft: rgba(15, 34, 52, 0.10);
        --border-accent: rgba(18, 119, 146, 0.20);
        --shadow-soft: 0 20px 55px rgba(15, 39, 66, 0.08);
        --shadow-card: 0 16px 36px rgba(15, 39, 66, 0.10);
        --brand: #137792;
        --brand-strong: #0d5971;
        --brand-soft: #e4f3f7;
        --success-soft: #d9f3ef;
        --warning-soft: #fff6de;
        --warning-border: #f0c36d;
        --danger-soft: #fff0ef;
        --radius-xl: 28px;
        --radius-lg: 20px;
        --radius-md: 14px;
    }
    html, body, .stApp {
        font-family: 'Manrope', 'Segoe UI', sans-serif !important;
        color: var(--text-primary);
        background:
            radial-gradient(circle at top left, rgba(37, 167, 160, 0.10), transparent 26%),
            radial-gradient(circle at top right, rgba(13, 76, 115, 0.10), transparent 22%),
            linear-gradient(180deg, #f8fbfd 0%, var(--bg-main) 100%);
    }
    [data-testid="stAppViewContainer"] {
        background: transparent;
    }
    [data-testid="stHeader"] {
        background: rgba(248, 251, 253, 0.78);
        backdrop-filter: blur(10px);
    }
    [data-testid="stSidebar"] {
        background: var(--bg-sidebar);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    [data-testid="stSidebar"] * {
        color: var(--text-inverse);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: inherit;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.12);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.16);
        box-shadow: none;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 255, 255, 0.18);
        transform: translateY(-1px);
    }
    .main .block-container {
        max-width: 1200px;
        padding-top: 2.2rem;
        padding-bottom: 2.5rem;
    }
    .custom-header {
        display: flex;
        align-items: center;
        background:
            radial-gradient(circle at top right, rgba(255, 255, 255, 0.24), transparent 22%),
            var(--bg-brand);
        padding: 32px;
        border-radius: var(--radius-xl);
        margin-bottom: 22px;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 26px 64px rgba(14, 53, 80, 0.20);
        animation: fade-slide-up 0.55s ease-out;
    }
    .custom-header::before {
        content: "";
        position: absolute;
        inset: 0;
        background:
            linear-gradient(110deg, transparent 0%, rgba(255, 255, 255, 0.10) 48%, transparent 100%);
        transform: translateX(-120%);
        animation: shimmer 7s linear infinite;
    }
    .custom-header-content {
        position: relative;
        z-index: 1;
        flex: 1;
        min-width: 0;
    }
    .custom-header-content h1 {
        color: #ffffff;
        font-size: 2.45rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin: 0 0 6px 0;
    }
    .custom-header-content p {
        color: rgba(245, 250, 255, 0.92);
        font-size: 1.02rem;
        line-height: 1.7;
        max-width: 760px;
        margin: 0;
    }
    .hero-bullets {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 18px;
    }
    .hero-chip,
    .quick-pill,
    .suggestion-chip,
    .attachment-chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border-radius: 999px;
        padding: 9px 14px;
        font-size: 0.86rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }
    .hero-chip {
        color: #ffffff;
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.22);
        backdrop-filter: blur(8px);
    }
    .sidebar-panel,
    .composer-shell,
    [data-testid="stForm"],
    .footer-bar {
        background: var(--bg-surface);
        border: 1px solid var(--border-soft);
        box-shadow: var(--shadow-soft);
        backdrop-filter: blur(12px);
    }
    .sidebar-panel {
        border-radius: 22px;
        padding: 18px 18px 16px 18px;
        margin-bottom: 6px;
        background: var(--bg-sidebar-card);
        border: 1px solid rgba(255, 255, 255, 0.10);
        box-shadow: 0 18px 34px rgba(2, 18, 31, 0.20);
    }
    .sidebar-panel h4 {
        margin: 0 0 8px 0;
        font-size: 1.02rem;
        font-weight: 800;
        color: #ffffff;
    }
    .sidebar-panel p,
    .sidebar-section-caption,
    .sidebar-delete-note {
        color: rgba(240, 247, 252, 0.88);
        font-size: 0.9rem;
        line-height: 1.55;
    }
    .sidebar-note {
        display: inline-flex;
        margin-top: 12px;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.18);
        font-size: 0.82rem;
        font-weight: 700;
        color: #ffffff;
    }
    .sidebar-section-title {
        margin-top: 18px;
        margin-bottom: 6px;
        font-size: 1rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: 0.01em;
    }
    .sidebar-divider {
        height: 1px;
        margin: 18px 0 14px 0;
        background: linear-gradient(90deg, rgba(255,255,255,0.04), rgba(255,255,255,0.28), rgba(255,255,255,0.04));
    }
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stFileUploader > div,
    .stPopover button {
        border-radius: var(--radius-md) !important;
    }
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] input {
        background: rgba(255, 255, 255, 0.14) !important;
        color: #ffffff !important;
        border-color: rgba(255, 255, 255, 0.16) !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] svg {
        fill: #ffffff !important;
    }
    .stTextInput > div > div > input,
    .stSelectbox [data-baseweb="select"] > div,
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.98) !important;
        color: var(--text-primary) !important;
        border: 1px solid rgba(18, 38, 58, 0.12) !important;
        box-shadow: none !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea textarea:focus {
        border-color: rgba(22, 124, 145, 0.42) !important;
        box-shadow: 0 0 0 3px rgba(22, 124, 145, 0.10) !important;
    }
    .stButton > button,
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #176b8c 0%, #1d96a0 100%);
        color: #ffffff;
        border: none;
        border-radius: 14px;
        min-height: 46px;
        font-weight: 800;
        letter-spacing: 0.01em;
        box-shadow: 0 14px 30px rgba(23, 107, 140, 0.18);
        transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
    }
    .stButton > button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-1px);
        filter: brightness(1.02);
        box-shadow: 0 18px 34px rgba(23, 107, 140, 0.24);
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 16px;
        margin: 4px 0 12px 0;
    }
    .stat-card {
        background: var(--bg-surface-strong);
        border: 1px solid var(--border-accent);
        border-radius: 20px;
        padding: 20px 20px 18px 20px;
        box-shadow: var(--shadow-card);
    }
    .stat-card h3 {
        margin: 0;
        color: var(--brand-strong);
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
    }
    .stat-card p {
        margin: 8px 0 0 0;
        color: var(--text-secondary);
        font-size: 0.94rem;
        font-weight: 600;
    }
    .quick-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 18px;
    }
    .quick-pill {
        background: var(--bg-surface);
        border: 1px solid var(--border-soft);
        color: var(--text-primary);
        box-shadow: 0 10px 24px rgba(15, 39, 66, 0.05);
    }
    .stChatMessage.user,
    .stChatMessage.assistant {
        padding: 18px 20px;
        border-radius: 20px;
        margin-bottom: 12px;
        border: 1px solid transparent;
        box-shadow: var(--shadow-card);
        line-height: 1.7;
    }
    .stChatMessage.user {
        background: linear-gradient(135deg, #1b7898 0%, #26a4a1 100%);
        color: #ffffff;
        border-top-right-radius: 8px;
    }
    .stChatMessage.assistant {
        background: rgba(255, 255, 255, 0.92);
        color: var(--text-primary);
        border-color: rgba(18, 38, 58, 0.08);
        border-top-left-radius: 8px;
    }
    .attachment-preview {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 10px 0 2px 0;
    }
    .attachment-chip {
        background: var(--brand-soft);
        color: var(--brand-strong);
        border: 1px solid rgba(22, 124, 145, 0.14);
    }
    .source-doc {
        background: #f8fbff;
        border: 1px solid rgba(18, 38, 58, 0.10);
        border-left: 4px solid #1d96a0;
        border-radius: 14px;
        padding: 14px 16px;
        margin: 8px 0;
        color: var(--text-primary);
        box-shadow: 0 10px 24px rgba(15, 39, 66, 0.05);
    }
    .composer-label {
        margin: 24px 0 4px 2px;
        color: var(--brand-strong);
        font-size: 0.92rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .composer-caption {
        color: var(--text-secondary);
        font-size: 1rem;
        line-height: 1.6;
        margin: 0 0 14px 2px;
        max-width: 760px;
    }
    .composer-shell {
        border-radius: 24px;
        padding: 18px 18px 12px 18px;
        margin-bottom: 12px;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(247, 251, 254, 0.96) 100%);
    }
    .suggestion-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 14px;
    }
    .suggestion-chip {
        background: var(--bg-brand-soft);
        color: var(--brand-strong);
        border: 1px solid rgba(22, 124, 145, 0.14);
    }
    [data-testid="stForm"] {
        border-radius: 22px;
        padding: 8px 10px;
        border: 1px solid rgba(18, 38, 58, 0.12);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7), 0 10px 24px rgba(15, 39, 66, 0.06);
        background: rgba(255, 255, 255, 0.98);
    }
    [data-testid="stForm"]:focus-within {
        border-color: rgba(22, 124, 145, 0.34);
        box-shadow: 0 0 0 4px rgba(22, 124, 145, 0.10), 0 12px 28px rgba(15, 39, 66, 0.08);
    }
    [data-testid="stForm"] .stTextInput > div,
    [data-testid="stForm"] .stTextInput > div > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stForm"] .stTextInput > div > div > input {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding-left: 0.2rem !important;
        padding-right: 0.2rem !important;
    }
    [data-testid="stForm"] .stTextInput > div > div > input:focus {
        box-shadow: none !important;
    }
    [data-testid="stForm"] .stPopover button {
        min-height: 44px;
        border-radius: 16px !important;
        box-shadow: none;
    }
    [data-testid="stForm"] div[data-testid="stFormSubmitButton"] button {
        min-height: 44px;
        border-radius: 16px !important;
        box-shadow: none;
    }
    .stPopover button {
        border: 1px solid rgba(18, 38, 58, 0.10) !important;
        background: #ffffff !important;
        color: var(--brand-strong) !important;
        font-weight: 800 !important;
    }
    .stFileUploader > div {
        background: #f9fcff !important;
        border: 1px dashed rgba(22, 124, 145, 0.26) !important;
    }
    .stAlert {
        border-radius: 16px;
        border: 1px solid rgba(18, 38, 58, 0.08);
    }
    .footer-bar {
        width: 100%;
        border-radius: 26px;
        padding: 22px 24px 18px 24px;
        margin-top: 28px;
        text-align: center;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(245, 249, 252, 0.96) 100%);
    }
    .footer-bar p {
        margin: 0;
        color: var(--text-primary);
    }
    .footer-meta {
        margin-top: 10px;
        color: var(--text-secondary);
        font-size: 0.92rem;
    }
    .footer-links {
        margin-top: 8px;
        font-size: 0.9rem;
    }
    .footer-links a {
        color: var(--brand);
        text-decoration: none;
        font-weight: 700;
    }
    .footer-links a:hover {
        text-decoration: underline;
    }
    .disclaimer-banner {
        margin: 14px auto 0 auto;
        max-width: 760px;
        padding: 14px 16px;
        border-radius: 16px;
        background: var(--warning-soft);
        border: 1px solid var(--warning-border);
        color: #7a5a11;
        text-align: left;
        line-height: 1.6;
        font-size: 0.94rem;
    }
    #MainMenu,
    footer {
        visibility: hidden;
    }
    @media (max-width: 900px) {
        .custom-header {
            flex-direction: column;
            align-items: flex-start;
            padding: 24px;
        }
        .metric-grid {
            grid-template-columns: 1fr;
        }
        .composer-shell {
            padding: 16px 14px 10px 14px;
        }
    }
    @keyframes fade-slide-up {
        from {
            opacity: 0;
            transform: translateY(18px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    @keyframes soft-pulse {
        0%, 100% {
            box-shadow: 0 18px 46px rgba(16, 94, 116, 0.18);
        }
        50% {
            box-shadow: 0 24px 58px rgba(16, 94, 116, 0.24);
        }
    }
    @keyframes shimmer {
        from {
            transform: translateX(-120%);
        }
        to {
            transform: translateX(120%);
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

DB_FAISS_PATH = "vectorstore/db_faiss"


def get_pdf_reader():
    for module_name in ("pypdf", "PyPDF2"):
        try:
            module = importlib.import_module(module_name)
            return getattr(module, "PdfReader", None)
        except ImportError:
            continue
    return None


def create_chat():
    chat_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid4().hex[:6]}"
    return chat_id, {
        "messages": [],
        "title": "New Chat",
        "created": datetime.now(),
    }


def format_chat_label(chat_data):
    title = chat_data["title"] or "New Chat"
    created = chat_data["created"].strftime("%d %b %H:%M")
    return f"{title} [{created}]"


def count_chat_attachments(messages):
    return sum(len(message.get("attachments", [])) for message in messages)


def _fingerprint_text(value):
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()[:12]


def _key_env_mtime():
    try:
        return os.path.getmtime("key.env")
    except OSError:
        return 0


@st.cache_data(show_spinner=False)
def _load_key_env(_mtime):
    return dotenv_values("key.env")


def get_active_api_key():
    try:
        secret_key = (st.secrets.get("GROQ_API_KEY", "") or "").strip()
    except StreamlitSecretNotFoundError:
        secret_key = ""
    file_key = (_load_key_env(_key_env_mtime()).get("GROQ_API_KEY", "") or "").strip()
    env_key = (os.environ.get("GROQ_API_KEY", "") or "").strip()
    session_key = (st.session_state.get("api_key", "") or "").strip()
    active_key = secret_key or file_key or env_key or session_key
    st.session_state.api_key = active_key
    if active_key:
        os.environ["GROQ_API_KEY"] = active_key
    return active_key


def build_attachment_metadata(uploaded_images, uploaded_files, recorded_audio):
    attachments = []
    for image in uploaded_images or []:
        attachments.append(
            {
                "kind": "image",
                "name": image.name,
                "size": image.size,
                "type": image.type or "image",
            }
        )
    for doc in uploaded_files or []:
        attachments.append(
            {
                "kind": "file",
                "name": doc.name,
                "size": doc.size,
                "type": doc.type or "file",
            }
        )
    if recorded_audio is not None:
        attachments.append(
            {
                "kind": "voice",
                "name": getattr(recorded_audio, "name", "voice-message.wav"),
                "size": getattr(recorded_audio, "size", 0),
                "type": getattr(recorded_audio, "type", "audio/wav"),
            }
        )
    return attachments


def _truncate_text(text, limit=4000):
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _uploaded_file_bytes(uploaded_file):
    return uploaded_file.getvalue()


@st.cache_data(show_spinner=False)
def _extract_text_from_document_bytes(file_name, file_type, file_bytes):
    file_type = (file_type or "").lower()
    file_name = (file_name or "").lower()

    if file_type == "application/pdf" or file_name.endswith(".pdf"):
        pdf_reader = get_pdf_reader()
        if pdf_reader is None:
            return (
                "PDF support is unavailable because neither 'pypdf' nor 'PyPDF2' is installed "
                "in the active environment."
            )
        reader = pdf_reader(io.BytesIO(file_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return _truncate_text(text, 5000)

    if file_type.startswith("text/") or file_name.endswith((".txt", ".csv")):
        try:
            return _truncate_text(file_bytes.decode("utf-8"), 5000)
        except UnicodeDecodeError:
            return _truncate_text(file_bytes.decode("latin-1", errors="ignore"), 5000)

    return ""


def extract_text_from_document(uploaded_file):
    return _extract_text_from_document_bytes(
        uploaded_file.name,
        uploaded_file.type,
        _uploaded_file_bytes(uploaded_file),
    )


@st.cache_data(show_spinner=False)
def _describe_image_bytes(image_name, image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return (
            f"Image '{image_name}' with size {image.width}x{image.height} pixels, "
            f"mode {image.mode}, format {image.format or 'unknown'}."
        )
    except Exception:
        return f"Image '{image_name}' uploaded."


def describe_image(uploaded_image):
    return _describe_image_bytes(uploaded_image.name, _uploaded_file_bytes(uploaded_image))


@st.cache_data(show_spinner=False)
def _analyze_image_bytes(image_name, mime_type, image_bytes, api_key_fingerprint):
    if not api_key_fingerprint:
        return _describe_image_bytes(image_name, image_bytes)

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return _describe_image_bytes(image_name, image_bytes)

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.2,
        max_tokens=500,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical assistant analyzing an uploaded image. "
                    "Describe only what is visible. If the image appears non-medical, say that clearly. "
                    "If text is visible, include it. Do not diagnose with certainty."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this uploaded image and summarize medically relevant visible information.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}",
                        },
                    },
                ],
            },
        ],
    )
    content = response.choices[0].message.content if response.choices else ""
    return _truncate_text(content or _describe_image_bytes(image_name, image_bytes), 3000)


def analyze_image_with_groq(uploaded_image, api_key):
    return _analyze_image_bytes(
        uploaded_image.name,
        uploaded_image.type or "image/jpeg",
        _uploaded_file_bytes(uploaded_image),
        _fingerprint_text(api_key),
    )


@st.cache_data(show_spinner=False)
def _transcribe_audio_bytes(audio_name, audio_bytes, api_key_fingerprint):
    if not audio_bytes or not api_key_fingerprint:
        return ""

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return ""

    client = Groq(api_key=api_key)
    transcript = client.audio.transcriptions.create(
        file=(audio_name, audio_bytes),
        model="whisper-large-v3-turbo",
        response_format="verbose_json",
    )
    return _truncate_text(getattr(transcript, "text", "") or "", 5000)


def transcribe_audio(recorded_audio, api_key):
    if recorded_audio is None or not api_key:
        return ""
    return _transcribe_audio_bytes(
        getattr(recorded_audio, "name", "voice-message.wav"),
        _uploaded_file_bytes(recorded_audio),
        _fingerprint_text(api_key),
    )


def build_attachment_context(uploaded_images, uploaded_files, recorded_audio, api_key):
    attachment_blocks = []

    for image in uploaded_images or []:
        image_analysis = analyze_image_with_groq(image, api_key)
        attachment_blocks.append(f"[Image: {image.name}]\n{image_analysis}")

    for doc in uploaded_files or []:
        extracted_text = extract_text_from_document(doc)
        if extracted_text:
            attachment_blocks.append(f"[File: {doc.name}]\n{extracted_text}")
        else:
            attachment_blocks.append(f"[File: {doc.name}]\nUnable to extract readable text from this file.")

    if recorded_audio is not None:
        audio_text = transcribe_audio(recorded_audio, api_key)
        if audio_text:
            attachment_blocks.append(f"[Voice Transcript: {getattr(recorded_audio, 'name', 'voice-message.wav')}]\n{audio_text}")
        else:
            attachment_blocks.append("[Voice Transcript]\nUnable to transcribe the audio.")

    return attachment_blocks


def format_attachment_context(attachments, attachment_blocks):
    if not attachments:
        return ""
    parts = [
        f"{item['kind']}: {item['name']}"
        for item in attachments
    ]
    context = "\n\nAttached items:\n- " + "\n- ".join(parts)
    if attachment_blocks:
        context += "\n\nAttachment contents:\n" + "\n\n".join(attachment_blocks)
    return context


def build_recent_history(messages, limit=6):
    """Keep a compact conversation window for follow-up questions."""
    history_lines = []
    for message in messages[-limit:]:
        role = "User" if message.get("role") == "user" else "Assistant"
        content = _truncate_text(message.get("content", ""), 700)
        history_lines.append(f"{role}: {content}")
    return "\n".join(history_lines) if history_lines else "None"


if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_count" not in st.session_state:
    st.session_state.chat_count = 0

if "vectorstore_loaded" not in st.session_state:
    st.session_state.vectorstore_loaded = False

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

st.session_state.api_key = get_active_api_key()

if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if st.session_state.current_chat_id is None:
    chat_id, chat_data = create_chat()
    st.session_state.conversations[chat_id] = chat_data
    st.session_state.current_chat_id = chat_id


@st.cache_resource
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        return FAISS.load_local(
            DB_FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None


@st.cache_resource(show_spinner=False)
def get_cached_qa_chain(model_name, temperature):
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return None

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return None

    llm = ChatGroq(
        model_name=model_name,
        temperature=temperature,
        groq_api_key=api_key,
    )

    class SimpleQAChain:
        def __init__(self, llm, vectorstore):
            self.llm = llm
            self.vectorstore = vectorstore

        def invoke(self, input_dict):
            query = input_dict.get("query", "")
            attachment_context = input_dict.get("attachment_context", [])
            recent_history = input_dict.get("recent_history", "None")
            try:
                docs = self.vectorstore.max_marginal_relevance_search(query, k=4, fetch_k=10)
            except Exception:
                docs = self.vectorstore.similarity_search(query, k=4)
            context = "\n".join([f"Document: {doc.page_content}" for doc in docs])
            attachment_text = "\n\n".join(attachment_context) if attachment_context else "None"

            custom_prompt_template = """
    You are a careful medical AI assistant. Behave like a doctor doing basic clinical history taking, but do not claim to replace a doctor or provide a final diagnosis.

    Core rules:
    - Use only the provided evidence
    - Answer only what the user actually asked
    - If the evidence is incomplete, say that clearly
    - Use simple, direct language
    - Keep the response concise and clinically organized
    - Do not claim certain diagnosis from image, audio, or upload alone
    - Use uploaded attachment contents first when directly relevant
    - Ignore retrieved context if it does not match the exact user question
    - Use recent conversation history only to understand follow-up questions
    - Do not restart from zero if the user is continuing the same topic
    - Do not pad the answer with generic advice

    Doctor-like information flow:
    1. Chief complaint: identify the main problem and since when it started
    2. Symptom exploration: ask about severity, duration, and associated symptoms if important details are missing
    3. Medical background: ask about chronic illness, medicines, allergies, age, and gender only when relevant
    4. Symptom analysis: explain the likely clinical pattern in simple words
    5. Differential diagnosis: mention possible conditions, not a single certain diagnosis
    6. Risk assessment: clearly say whether the situation sounds low-risk, urgent, or emergency based on the available evidence
    7. Advice: give practical next steps and basic precautions
    8. Disclaimer: remind the user this is not a confirmed diagnosis
    9. Follow-up: ask the next most useful question when more information is needed

    Response style:
    - If the user gives only a short complaint like "I have fever", ask 2 to 4 focused follow-up questions before giving possible causes
    - If enough information is already available, give a short explanation first
    - Then include `Precautions:` with simple, practical safety advice
    - When relevant, include `Possible causes:` with 2 to 4 likely conditions
    - When relevant, include `Risk level:` as one of: Low, Urgent, Emergency
    - End with a brief medical disclaimer and one follow-up question if more history is needed
    - Prefer short bullets over long paragraphs
    - Avoid long lists unless the user asked for a list

    Preferred section headings:
    - Explanation:
    - Possible causes:
    - Precautions:
    - Risk level:
    - When to see a doctor:
    - Follow-up question:
    - Disclaimer:

    Extra rules:
    - If the user asks about a medicine, include dose strength only if clearly supported by the evidence
    - If the user asks about an uploaded image or file, mention the visible or extracted findings first
    - If something cannot be confirmed, explicitly say that it cannot be confirmed from the provided information
    - If the user describes red-flag symptoms like chest pain, breathing trouble, severe dehydration, confusion, stroke signs, or heavy bleeding, treat that as urgent or emergency
    - Never present differential diagnosis as certainty
    - If the user asks for images and no image generation or image source is available, say that clearly in plain language

    User question: {question}

    Recent conversation:
    {recent_history}

    Uploaded attachment contents:
    {attachment_text}

    Retrieved knowledge base context:
    {context}

    Write the response now:
    """
            prompt_text = custom_prompt_template.format(
                context=context,
                question=query,
                recent_history=recent_history,
                attachment_text=attachment_text,
            )
            response = self.llm.invoke(prompt_text)
            content = response.content if hasattr(response, "content") else str(response)
            return {"answer": content, "context": docs, "result": content}

    return SimpleQAChain(llm, vectorstore)


def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"],
    )


def get_qa_chain(api_key, temperature=0.0, model_name="llama-3.3-70b-versatile"):
    try:
        os.environ["GROQ_API_KEY"] = api_key
        return get_cached_qa_chain(model_name, temperature)
    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return None


with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-panel">
            <h4>Workspace</h4>
            <p>Manage ongoing chats, keep the medical context active, and move between threads without losing history.</p>
            <div class="sidebar-note">Active Clinical Workspace</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("+ New Chat", use_container_width=True, key="new_chat_btn"):
        chat_id, chat_data = create_chat()
        st.session_state.conversations[chat_id] = chat_data
        st.session_state.current_chat_id = chat_id
        st.rerun()

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">Chat History</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sidebar-section-caption">{len(st.session_state.conversations)} saved conversation'
        f'{"s" if len(st.session_state.conversations) != 1 else ""} ready to resume.</div>',
        unsafe_allow_html=True,
    )

    sorted_chats = sorted(
        st.session_state.conversations.items(),
        key=lambda x: x[1]["created"],
        reverse=True,
    )

    if sorted_chats:
        chat_labels = [format_chat_label(chat_data) for chat_id, chat_data in sorted_chats]
        chat_ids = [chat_id for chat_id, _chat_data in sorted_chats]

        current_index = (
            chat_ids.index(st.session_state.current_chat_id)
            if st.session_state.current_chat_id in chat_ids
            else 0
        )

        selected_label = st.selectbox(
            "Select a chat:",
            chat_labels,
            index=current_index,
            label_visibility="collapsed",
            key="chat_selector",
        )

        selected_chat_id = chat_ids[chat_labels.index(selected_label)]
        if selected_chat_id != st.session_state.current_chat_id:
            st.session_state.current_chat_id = selected_chat_id
            st.rerun()

        if st.button("Delete Current Chat", key="delete_current_chat", use_container_width=True):
            del st.session_state.conversations[st.session_state.current_chat_id]
            if st.session_state.conversations:
                remaining_chats = sorted(
                    st.session_state.conversations.items(),
                    key=lambda x: x[1]["created"],
                    reverse=True,
                )
                st.session_state.current_chat_id = remaining_chats[0][0]
            else:
                new_chat_id, chat_data = create_chat()
                st.session_state.conversations[new_chat_id] = chat_data
                st.session_state.current_chat_id = new_chat_id
            st.rerun()

        st.markdown(
            '<div class="sidebar-delete-note">Removes only the currently selected conversation.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">Settings</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-section-caption">Choose the response model for this workspace.</div>',
        unsafe_allow_html=True,
    )

    model_options = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]

    selected_model = st.selectbox(
        "Model",
        model_options,
        help="Select the Groq model to use",
    )


st.markdown(
    """
<div class="custom-header">
    <div class="custom-header-content">
        <h1>Doctor Assistant</h1>
        <p>A professional clinical workspace for symptom guidance, document review, retrieval-backed answers, and clearer patient communication.</p>
        <div class="hero-bullets">
            <span class="hero-chip">Evidence-aware responses</span>
            <span class="hero-chip">Image, file, and voice intake</span>
            <span class="hero-chip">Fast triage-friendly workflow</span>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

current_messages = st.session_state.conversations[st.session_state.current_chat_id]["messages"]
message_count = len(current_messages)
attachment_count = count_chat_attachments(current_messages)
assistant_count = sum(1 for message in current_messages if message["role"] == "assistant")

st.markdown(
    f"""
<div class="metric-grid">
    <div class="stat-card"><h3>{len(st.session_state.conversations)}</h3><p>Active case threads</p></div>
    <div class="stat-card"><h3>{message_count}</h3><p>Messages in current workspace</p></div>
    <div class="stat-card"><h3>{attachment_count}</h3><p>Clinical files reviewed</p></div>
</div>
<div class="quick-actions">
    <span class="quick-pill">Use symptom follow-ups for better triage</span>
    <span class="quick-pill">Upload reports for concise interpretation</span>
    <span class="quick-pill">Capture questions hands-free with voice</span>
    <span class="quick-pill">{assistant_count} assistant replies in this case</span>
</div>
""",
    unsafe_allow_html=True,
)


# Professional chat bubbles with role-based styling
for message in current_messages:
    role = message["role"]
    avatar = "You" if role == "user" else "AI"
    bubble_class = f"stChatMessage {role}"
    with st.chat_message(role, avatar=avatar):
        st.markdown(f'<div class="{bubble_class}">{message["content"]}</div>', unsafe_allow_html=True)

        if message.get("attachments"):
            chips = []
            for item in message["attachments"]:
                icon = "IMG" if item["kind"] == "image" else "FILE" if item["kind"] == "file" else "MIC"
                chips.append(f"<span class='attachment-chip'>{icon} {item['name']}</span>")
            st.markdown(
                f"<div class='attachment-preview'>{''.join(chips)}</div>",
                unsafe_allow_html=True,
            )
            if message.get("attachment_context"):
                with st.expander("Attachment details"):
                    for block in message["attachment_context"]:
                        st.markdown(block)

        if role == "assistant" and "sources" in message:
            with st.expander("Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(
                        f"""
                        <div class="source-doc">
                            <strong>Source {i}:</strong><br>
                            {source.page_content[:300]}...
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


st.markdown(
    """
    <div class='composer-label'>Ask Smarter</div>
    <div class='composer-caption'>Ask about symptoms, compare treatments, summarize uploaded reports, or rewrite medical language into something easier to understand.</div>
    <div class='composer-shell'>
        <div class='suggestion-row'>
            <span class='suggestion-chip'>Symptoms and next steps</span>
            <span class='suggestion-chip'>Medication comparison</span>
            <span class='suggestion-chip'>Report summary</span>
            <span class='suggestion-chip'>Plain-language explanation</span>
        </div>
    """,
    unsafe_allow_html=True,
)
with st.form("chat_composer_form", clear_on_submit=False):
    composer_col1, composer_col2, composer_col3, composer_col4 = st.columns(
        [0.8, 5.8, 0.9, 1.0],
        gap="small",
    )
    with composer_col1:
        with st.popover("+", use_container_width=True):
            uploaded_images = st.file_uploader(
                "Photo",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
                key="chat_image_upload",
            )
            uploaded_files = st.file_uploader(
                "File",
                type=["pdf", "txt", "doc", "docx", "csv"],
                accept_multiple_files=True,
                key="chat_file_upload",
            )
    with composer_col2:
        prompt = st.text_input(
            "Ask anything",
            placeholder="Search or ask anything about symptoms, medicines, reports, or care...",
            key="chat_prompt_text",
            label_visibility="collapsed",
        )
    with composer_col3:
        with st.popover("Mic", use_container_width=True):
            recorded_audio = st.audio_input(
                "Voice",
                key="chat_voice_input",
            )
    with composer_col4:
        submitted = st.form_submit_button("Send", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

pending_attachments = build_attachment_metadata(uploaded_images, uploaded_files, recorded_audio)
if pending_attachments:
    preview_chips = []
    for item in pending_attachments:
        icon = "IMG" if item["kind"] == "image" else "FILE" if item["kind"] == "file" else "MIC"
        preview_chips.append(f"<span class='attachment-chip'>{icon} {item['name']}</span>")
    st.markdown(
        f"<div class='attachment-preview'>{''.join(preview_chips)}</div>",
        unsafe_allow_html=True,
    )

prompt = (prompt or "").strip()
if submitted and (prompt or pending_attachments):
    active_api_key = get_active_api_key()
    if not active_api_key:
        st.error("Missing GROQ_API_KEY. Add it to key.env as GROQ_API_KEY=your_key and refresh.")
        st.stop()

    if not prompt and pending_attachments:
        prompt = "Please review the attached items and help me understand them."

    try:
        attachment_blocks = build_attachment_context(
            uploaded_images,
            uploaded_files,
            recorded_audio,
            active_api_key,
        )
    except Exception as e:
        st.error(f"Failed to process an attachment: {str(e)}")
        st.stop()

    user_message = {
        "role": "user",
        "content": prompt,
        "attachments": pending_attachments,
        "attachment_context": attachment_blocks,
        "timestamp": datetime.now().strftime("%H:%M"),
    }
    current_chat_messages = st.session_state.conversations[st.session_state.current_chat_id]["messages"]
    recent_history = build_recent_history(current_chat_messages)
    current_chat_messages.append(user_message)

    if len(st.session_state.conversations[st.session_state.current_chat_id]["messages"]) == 1:
        title = prompt[:40].replace("\n", " ")
        if len(prompt) > 40:
            title += "..."
        st.session_state.conversations[st.session_state.current_chat_id]["title"] = title

    with st.chat_message("user", avatar="You"):
        st.markdown(prompt)
        if pending_attachments:
            preview_chips = []
            for item in pending_attachments:
                icon = "IMG" if item["kind"] == "image" else "FILE" if item["kind"] == "file" else "MIC"
                preview_chips.append(f"<span class='attachment-chip'>{icon} {item['name']}</span>")
            st.markdown(
                f"<div class='attachment-preview'>{''.join(preview_chips)}</div>",
                unsafe_allow_html=True,
            )

    with st.chat_message("assistant", avatar="AI"):
        with st.spinner("Thinking..."):
            try:
                qa_chain = get_qa_chain(
                    active_api_key,
                    temperature=0.0,
                    model_name=selected_model,
                )

                if qa_chain:
                    response = qa_chain.invoke(
                        {
                            "query": prompt,
                            "attachment_context": attachment_blocks,
                            "recent_history": recent_history,
                        }
                    )
                    result = response.get("answer", response.get("result", ""))
                    source_documents = response.get("context", response.get("source_documents", []))

                    st.markdown(result)

                    assistant_message = {
                        "role": "assistant",
                        "content": result,
                        "sources": source_documents,
                        "timestamp": datetime.now().strftime("%H:%M"),
                    }
                    st.session_state.conversations[st.session_state.current_chat_id]["messages"].append(
                        assistant_message
                    )

                    st.session_state.chat_count += 1

                    if source_documents:
                        with st.expander("View Source Documents"):
                            for i, source in enumerate(source_documents, 1):
                                st.markdown(
                                    f"""
                                <div class="source-doc">
                                    <strong>Source {i}:</strong><br>
                                    {source.page_content[:300]}...
                                </div>
                                """,
                                    unsafe_allow_html=True,
                                )
                    st.rerun()
                else:
                    st.error("Failed to initialize the QA system. Please check your API key and try again.")
                    st.rerun()

            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                if "invalid_api_key" in str(e):
                    error_msg = (
                        "Groq rejected the API key. The app is reading key.env now, "
                        "so this means the key itself is invalid, expired, or revoked."
                    )
                st.error(error_msg)
                error_message = {
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now().strftime("%H:%M"),
                }
                st.session_state.conversations[st.session_state.current_chat_id]["messages"].append(
                    error_message
                )
                st.rerun()


st.markdown("---")
st.markdown(
    """
<div class='footer-bar'>
    <p><strong>Doctor Assistant</strong> is built for structured medical Q&amp;A, multimodal review, and faster knowledge retrieval.</p>
    <div class='footer-meta'>Powered by Streamlit, LangChain, and Groq</div>
    <div class='disclaimer-banner'>
        <strong>Medical Disclaimer:</strong> This assistant provides general informational support only and does not replace licensed clinical judgment, diagnosis, or treatment.
    </div>
    <div class='footer-meta'><strong>Developed by Kunal Chaudhari</strong></div>
    <div class='footer-links'>
        <a href='https://www.linkedin.com/in/kunal-chaudhari-8311542a6' target='_blank'>LinkedIn</a>&nbsp;&nbsp;|&nbsp;&nbsp;<a href='https://github.com/Nice-Kunal' target='_blank'>GitHub</a>
    </div>
</div>
""",
    unsafe_allow_html=True,
)
