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

st.markdown(
    """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(16, 163, 127, 0.18), transparent 26%),
            radial-gradient(circle at top right, rgba(59, 130, 246, 0.10), transparent 24%),
            linear-gradient(180deg, #1b1e28 0%, #14161d 100%);
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
            box-shadow: 0 16px 40px rgba(16, 163, 127, 0.18);
        }
        50% {
            box-shadow: 0 20px 52px rgba(16, 163, 127, 0.3);
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

    .main {
        background:
            radial-gradient(circle at top, rgba(16, 163, 127, 0.12), transparent 30%),
            linear-gradient(180deg, #343541 0%, #2f3038 100%);
        animation: fade-slide-up 0.45s ease-out;
    }

    [data-testid="stSidebar"] {
        background-color: #202123;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    [data-testid="stSidebar"] .css-1d391kg {
        color: #FFFFFF;
    }

    .stChatMessage {
        background-color: #444654;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.04);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18);
        animation: fade-slide-up 0.35s ease-out;
        transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
    }

    .stChatMessage:hover {
        transform: translateY(-2px);
        border-color: rgba(16, 163, 127, 0.25);
        box-shadow: 0 18px 34px rgba(0, 0, 0, 0.24);
    }

    [data-testid="stChatMessageContent"] {
        color: #ECECF1;
    }

    .stTextInput > div > div > input {
        background-color: #40414F;
        color: #ECECF1;
        border: 1px solid #565869;
        border-radius: 8px;
    }

    .stButton > button {
        background-color: #10A37F;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
        font-weight: 600;
        transition: transform 0.18s ease, background-color 0.18s ease, box-shadow 0.18s ease;
        box-shadow: 0 10px 24px rgba(16, 163, 127, 0.18);
    }

    .stButton > button:hover {
        background-color: #0D8C6B;
        transform: translateY(-1px);
        box-shadow: 0 14px 28px rgba(16, 163, 127, 0.28);
    }

    h1 {
        color: #ECECF1;
        font-weight: 700;
    }

    h2, h3 {
        color: #ECECF1;
    }

    .stMarkdown {
        color: #ECECF1;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .custom-header {
        background:
            radial-gradient(circle at top right, rgba(255, 255, 255, 0.16), transparent 24%),
            linear-gradient(135deg, rgba(16, 163, 127, 0.95) 0%, rgba(11, 96, 108, 0.94) 100%);
        padding: 28px;
        border-radius: 26px;
        margin-bottom: 20px;
        text-align: left;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 22px 58px rgba(0, 0, 0, 0.22);
        animation: fade-slide-up 0.55s ease-out, soft-pulse 4s ease-in-out infinite;
    }

    .custom-header::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(100deg, transparent 20%, rgba(255, 255, 255, 0.14) 50%, transparent 80%);
        animation: shimmer 6s linear infinite;
    }

    .custom-header h1 {
        color: white;
        margin: 0;
        font-size: 2.65rem;
        position: relative;
        z-index: 1;
    }

    .custom-header p {
        color: #E5E5E5;
        margin: 10px 0 0 0;
        font-size: 1rem;
        position: relative;
        z-index: 1;
    }

    .stat-card {
        background: linear-gradient(180deg, rgba(42, 43, 50, 0.96), rgba(31, 34, 43, 0.94));
        padding: 16px;
        border-radius: 18px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.07);
        box-shadow: 0 16px 30px rgba(0, 0, 0, 0.18);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .stat-card:hover {
        transform: translateY(-3px);
        border-color: rgba(16, 163, 127, 0.26);
    }

    .hero-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.55fr) minmax(260px, 0.85fr);
        gap: 18px;
        position: relative;
        z-index: 1;
    }

    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        width: fit-content;
        padding: 8px 14px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.12);
        color: #ecfdf5;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .hero-bullets {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 16px;
    }

    .hero-chip {
        padding: 10px 14px;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.22);
        border: 1px solid rgba(255, 255, 255, 0.14);
        color: #f8fafc;
        font-size: 0.88rem;
        font-weight: 600;
    }

    .hero-sidecard {
        background: rgba(11, 18, 32, 0.24);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 22px;
        padding: 18px;
        backdrop-filter: blur(10px);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
    }

    .hero-sidecard h3 {
        margin: 0 0 10px 0;
        color: #f8fafc;
    }

    .hero-sidecard p {
        margin: 0;
        color: #d1fae5;
        line-height: 1.6;
        font-size: 0.92rem;
    }

    .stat-card h3 {
        margin: 0;
        color: #10A37F;
        font-size: 1.8rem;
    }

    .stat-card p {
        margin: 5px 0 0 0;
        color: #ECECF1;
        font-size: 0.9rem;
    }

    .source-doc {
        background-color: #2A2B32;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-size: 0.85rem;
        border-left: 3px solid #565869;
        animation: fade-slide-up 0.3s ease-out;
    }

    .disclaimer-banner {
        margin: 18px auto;
        max-width: 900px;
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid #f59e0b;
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.18), rgba(239, 68, 68, 0.12));
        color: #FFF7ED;
        font-size: 0.95rem;
        font-weight: 600;
        line-height: 1.5;
        text-align: left;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.22);
    }

    .disclaimer-banner strong {
        color: #FDE68A;
    }

    .attachment-toolbar {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 10px 0 6px 0;
        padding: 12px 14px;
        border-radius: 18px;
        background: linear-gradient(90deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.02));
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 18px 42px rgba(0, 0, 0, 0.18);
        animation: fade-slide-up 0.3s ease-out;
    }

    .toolbar-title {
        color: #f3f4f6;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .toolbar-note {
        color: #9ca3af;
        font-size: 0.78rem;
    }

    .attachment-chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin: 6px 8px 0 0;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(16, 163, 127, 0.14);
        border: 1px solid rgba(16, 163, 127, 0.28);
        color: #e5f9f3;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .attachment-preview {
        margin: 4px 0 12px 0;
        animation: fade-slide-up 0.3s ease-out;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin: 14px 0 18px 0;
    }

    .quick-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 10px 0 18px 0;
    }

    .quick-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 14px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #d1d5db;
        font-size: 0.84rem;
        font-weight: 600;
        transition: transform 0.18s ease, background-color 0.18s ease, border-color 0.18s ease;
    }

    .quick-pill:hover {
        transform: translateY(-1px);
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 255, 255, 0.14);
    }

    .sidebar-panel {
        position: relative;
        overflow: hidden;
        background:
            radial-gradient(circle at top right, rgba(16, 163, 127, 0.14), transparent 34%),
            linear-gradient(180deg, rgba(44, 46, 53, 0.96), rgba(29, 31, 37, 0.96));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 22px;
        padding: 18px 18px 16px 18px;
        margin: 12px 0 18px 0;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.22);
    }

    .sidebar-panel::before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 4px;
        background: linear-gradient(180deg, #34d399 0%, #10a37f 100%);
        opacity: 0.95;
    }

    .sidebar-panel h4 {
        margin: 0 0 10px 0;
        color: #f9fafb;
        font-size: 1.02rem;
        font-weight: 700;
    }

    .sidebar-panel p {
        margin: 0;
        color: #c6ced8;
        font-size: 0.88rem;
        line-height: 1.72;
    }

    .sidebar-panel .sidebar-note {
        margin-top: 14px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #ecfdf5;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    .sidebar-divider {
        height: 1px;
        margin: 20px 0 18px 0;
        background: linear-gradient(90deg, rgba(255, 255, 255, 0.16), rgba(255, 255, 255, 0.02));
    }

    .sidebar-section-title {
        margin: 0 0 6px 0;
        color: #f8fafc;
        font-size: 1.38rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .sidebar-section-caption {
        margin: 0 0 14px 0;
        color: #8f98a3;
        font-size: 0.8rem;
        line-height: 1.5;
    }

    .sidebar-delete-note {
        margin: 10px 0 0 0;
        color: #8f98a3;
        font-size: 0.77rem;
        line-height: 1.45;
    }

    .composer-shell {
        position: sticky;
        bottom: 16px;
        z-index: 20;
        margin: 14px 0 18px 0;
        padding: 14px;
        border-radius: 28px;
        background:
            radial-gradient(circle at top, rgba(52, 211, 153, 0.08), transparent 46%),
            linear-gradient(180deg, rgba(18, 22, 32, 0.96), rgba(13, 16, 24, 0.98));
        border: 1px solid rgba(148, 163, 184, 0.16);
        box-shadow: 0 28px 60px rgba(0, 0, 0, 0.34);
        backdrop-filter: blur(18px);
        animation: fade-slide-up 0.3s ease-out;
    }

    .composer-label {
        margin: 0 0 8px 6px;
        color: #94a3b8;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }

    .composer-caption {
        margin: 0 0 12px 6px;
        color: #64748b;
        font-size: 0.86rem;
    }

    .command-bar {
        padding: 10px;
        border-radius: 22px;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.02));
        border: 1px solid rgba(148, 163, 184, 0.08);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }

    .suggestion-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 0 0 12px 0;
        padding: 0 2px;
    }

    .suggestion-chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(148, 163, 184, 0.12);
        color: #cbd5e1;
        font-size: 0.8rem;
        font-weight: 600;
        transition: transform 0.18s ease, background-color 0.18s ease, border-color 0.18s ease;
    }

    .suggestion-chip:hover {
        transform: translateY(-1px);
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(52, 211, 153, 0.28);
    }

    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px dashed rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 8px;
    }

    [data-testid="stAudioInput"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px dashed rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 8px;
    }

    div[data-testid="stTextInput"] input {
        min-height: 56px;
        border-radius: 18px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        background: linear-gradient(180deg, rgba(70, 73, 92, 0.96), rgba(60, 63, 80, 0.96));
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
        padding: 0 18px;
        font-size: 1rem;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: rgba(34, 197, 94, 0.55);
        box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.12);
    }

    div[data-testid="stPopover"] button {
        min-height: 56px;
        border-radius: 999px;
        background: linear-gradient(180deg, rgba(28, 33, 45, 0.96), rgba(23, 27, 37, 0.96));
        border: 1px solid rgba(148, 163, 184, 0.16);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
        font-weight: 600;
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease, background 0.18s ease;
    }

    div[data-testid="stPopover"] button:hover {
        background: linear-gradient(180deg, rgba(38, 44, 58, 1), rgba(29, 33, 44, 1));
        transform: translateY(-1px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.18);
    }

    div[data-testid="stFormSubmitButton"] button {
        min-height: 56px;
        border-radius: 16px;
        background: linear-gradient(135deg, #111827 0%, #0f766e 100%);
        border: 1px solid rgba(94, 234, 212, 0.2);
        color: #f8fafc;
        font-weight: 700;
        letter-spacing: 0.02em;
        box-shadow: 0 18px 26px rgba(15, 118, 110, 0.22);
        transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-1px);
        background: linear-gradient(135deg, #0f172a 0%, #115e59 100%);
        box-shadow: 0 22px 32px rgba(15, 118, 110, 0.32);
        filter: brightness(1.04);
    }

    .stChatInputContainer {
        background: rgba(52, 53, 65, 0.82);
        backdrop-filter: blur(10px);
        border-top: 1px solid rgba(255, 255, 255, 0.06);
    }

    ::-webkit-scrollbar {
        width: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #202123;
    }

    ::-webkit-scrollbar-thumb {
        background: #565869;
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #6E6F7F;
    }

    @media (max-width: 900px) {
        .hero-grid,
        .metric-grid {
            grid-template-columns: 1fr;
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
    <div class="hero-kicker">Live Clinical Knowledge Workspace</div>
    <div class="hero-grid"><div><h1>Doctor Assistant</h1><p>Your AI Medical Assistant with animated chat, live retrieval, and multimodal review.</p><div class="hero-bullets"><span class="hero-chip">Grounded vector search</span><span class="hero-chip">Voice, file, and image input</span><span class="hero-chip">Fast model switching</span></div></div><div class="hero-sidecard"><h3>Interactive session</h3><p>Use the composer below to ask follow-up questions, upload reports, or review symptoms with attached context.</p></div></div>
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
    <div class="stat-card"><h3>{len(st.session_state.conversations)}</h3><p>Active chat threads</p></div>
    <div class="stat-card"><h3>{message_count}</h3><p>Messages in this thread</p></div>
    <div class="stat-card"><h3>{attachment_count}</h3><p>Attachments reviewed</p></div>
</div>
<div class="quick-actions">
    <span class="quick-pill">Try symptom check follow-ups</span>
    <span class="quick-pill">Upload reports and ask for a summary</span>
    <span class="quick-pill">Use voice input for hands-free prompts</span>
    <span class="quick-pill">{assistant_count} assistant replies in this session</span>
</div>
""",
    unsafe_allow_html=True,
)

for message in current_messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🩺"):
        st.markdown(message["content"])

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

        if message["role"] == "assistant" and "sources" in message:
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
    <div class='composer-caption'>Search symptoms, compare medications, or upload a report for contextual analysis.</div>
    <div class='composer-shell'>
        <div class='suggestion-row'>
            <span class='suggestion-chip'>Symptoms and next steps</span>
            <span class='suggestion-chip'>Compare two medicines</span>
            <span class='suggestion-chip'>Summarize uploaded report</span>
            <span class='suggestion-chip'>Explain in simple language</span>
        </div>
        <div class='command-bar'>
    """,
    unsafe_allow_html=True,
)
with st.form("chat_composer_form", clear_on_submit=False):
    composer_col1, composer_col2, composer_col3, composer_col4 = st.columns(
        [0.9, 5.4, 0.9, 1.0],
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
st.markdown("</div></div>", unsafe_allow_html=True)

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

    with st.chat_message("user", avatar="👤"):
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

    with st.chat_message("assistant", avatar="🩺"):
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
                        with st.expander("📚 View Source Documents"):
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
<div style='text-align: center; color: #ECECF1; padding: 20px;'>
    <p>Made with ❤️ using Streamlit, LangChain, and Groq | Doctor Assistant v1.0</p>
    <div class='disclaimer-banner'>
        <strong>⚠️ Medical Disclaimer:</strong> This chatbot provides general medical information only. Always consult qualified healthcare professionals for medical advice.
    </div>
    <hr style='margin: 15px 0; border: none; border-top: 1px solid #565869;'>
    <p style='font-size: 0.85rem; color: #ECECF1;'><strong>Developed by Mayur Panchbhai</strong></p>
    <p style='font-size: 0.8rem; color: #10A37F;'>
        <a href='https://www.linkedin.com/in/mayur-panchbhai-bb86723a0' target='_blank' style='color: #10A37F; text-decoration: none;'>🔗 LinkedIn</a>&nbsp;&nbsp;|&nbsp;&nbsp;<a href='https://github.com/mayurpanchbhai03-source' target='_blank' style='color: #10A37F; text-decoration: none;'>🐙 GitHub</a>
    </p>
</div>
""",
    unsafe_allow_html=True,
)
