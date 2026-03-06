import os
import hashlib
import numpy as np
import streamlit as st
import time
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import json

try:
    from groq import Groq
except ImportError:
    st.error("Installing packages... Please refresh in 2 minutes.")
    st.stop()

from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import docx

# Page config
st.set_page_config(
    page_title="Kivi - Intelligent Document Assistant",
    page_icon="🥝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get API key
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("Please configure API key in Secrets")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.1-8b-instant"

# =============================
# SOUTH INDIAN LANGUAGES CONFIG
# =============================
SOUTH_LANGUAGES = {
    "English (Default)": None,
    "తెలుగు (Telugu)": "Telugu",
    "தமிழ் (Tamil)": "Tamil",
    "ಕನ್ನಡ (Kannada)": "Kannada",
    "മലയാളം (Malayalam)": "Malayalam",
}

# =============================
# SESSION STATE
# =============================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "meta" not in st.session_state:
    st.session_state.meta = []
if "files_hash" not in st.session_state:
    st.session_state.files_hash = None
if "current_chat_name" not in st.session_state:
    st.session_state.current_chat_name = "New Chat"
if "saved_chats" not in st.session_state:
    st.session_state.saved_chats = []
if "selected_language" not in st.session_state:
    st.session_state.selected_language = "English (Default)"
if "toast_message" not in st.session_state:
    st.session_state.toast_message = None

# =============================
# UPGRADED PROFESSIONAL CSS
# =============================
st.markdown("""
<style>
/* ── Fonts ───────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Playfair+Display:wght@600;700&family=Noto+Sans+Telugu&family=Noto+Sans+Tamil&family=Noto+Sans+Kannada&family=Noto+Sans+Malayalam&display=swap');

/* ── CSS Variables ───────────────────────────── */
:root {
    --green-dark:    #065F46;
    --green-mid:     #059669;
    --green-light:   #10B981;
    --green-pale:    #D1FAE5;
    --green-xpale:   #ECFDF5;
    --surface:       #FFFFFF;
    --surface-2:     #F8FAFC;
    --border:        #E2E8F0;
    --border-dark:   #CBD5E1;
    --text-primary:  #0F172A;
    --text-secondary:#475569;
    --text-muted:    #94A3B8;
    --shadow-sm:     0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.06);
    --shadow-md:     0 4px 16px rgba(0,0,0,.08), 0 2px 4px rgba(0,0,0,.06);
    --shadow-lg:     0 12px 40px rgba(0,0,0,.12);
    --radius-sm:     8px;
    --radius-md:     12px;
    --radius-lg:     20px;
    --radius-full:   100px;
}

/* ── Base ────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

.main .block-container {
    padding: 2rem 2.5rem 4rem;
    max-width: 1180px;
    margin: 0 auto;
}

#MainMenu, footer, header { display: none; }

/* ── Sidebar ─────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F0FDF4 0%, #FFFFFF 100%) !important;
    border-right: 1px solid var(--border) !important;
    width: 290px !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 1.75rem 1.1rem !important;
    max-width: 100% !important;
}

/* ── Sidebar Brand ───────────────────────────── */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding-bottom: 1.25rem;
    margin-bottom: 1.25rem;
    border-bottom: 1px solid var(--border);
}
.brand-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, var(--green-mid), var(--green-dark));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    color: white; font-weight: 700; font-size: 1rem;
    box-shadow: 0 2px 8px rgba(5,150,105,.3);
}
.brand-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem; font-weight: 700;
    color: var(--green-dark);
    letter-spacing: -0.01em;
}

/* ── Sidebar Section Labels ──────────────────── */
.sidebar-label {
    font-size: 0.68rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--text-muted);
    margin: 1.25rem 0 0.6rem;
}

/* ── Chat List Items ─────────────────────────── */
.stButton > button {
    border-radius: var(--radius-md) !important;
    transition: all 0.18s ease !important;
}

/* ── Language Badge ──────────────────────────── */
.lang-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: var(--green-xpale);
    border: 1px solid var(--green-pale);
    color: var(--green-dark);
    font-size: 0.75rem; font-weight: 600;
    padding: 3px 10px; border-radius: var(--radius-full);
    margin-left: 8px;
}

/* ── Stats Grid ──────────────────────────────── */
.stats-row {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 0.65rem; margin-top: 0.5rem;
}
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 0.75rem 0.5rem;
    text-align: center;
    box-shadow: var(--shadow-sm);
}
.stat-num {
    font-size: 1.6rem; font-weight: 700;
    color: var(--green-mid); line-height: 1;
}
.stat-lbl {
    font-size: 0.68rem; color: var(--text-muted);
    margin-top: 2px;
}

/* ── Main Header ─────────────────────────────── */
.main-header {
    display: flex; align-items: center;
    justify-content: space-between;
    margin-bottom: 1.75rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--border);
}

.logo-wrap {
    display: flex; align-items: center; gap: 12px;
}
.logo-mark {
    width: 44px; height: 44px; position: relative;
}
.logo-outer {
    position: absolute; inset: 0;
    background: linear-gradient(135deg, var(--green-mid), var(--green-dark));
    border-radius: 40% 60% 60% 40% / 50% 50% 50% 50%;
    transform: rotate(-5deg);
    box-shadow: 0 4px 12px rgba(5,150,105,.25);
}
.logo-inner {
    position: absolute;
    width: 30px; height: 30px;
    background: linear-gradient(135deg, #34D399, #6EE7B7);
    border-radius: 45% 55% 55% 45%;
    top: 7px; left: 7px;
    transform: rotate(5deg); opacity: .88;
}
.logo-dot {
    position: absolute;
    width: 5px; height: 5px;
    background: var(--green-dark);
    border-radius: 50%;
    top: 19px; left: 19px;
}
.logo-text {
    font-family: 'Playfair Display', serif;
    font-size: 1.75rem; font-weight: 700;
    color: var(--green-dark);
    letter-spacing: -0.02em;
}
.logo-tagline {
    font-size: 0.78rem; color: var(--text-muted);
    margin-top: -2px; letter-spacing: 0.01em;
}

/* ── Header Action Buttons ───────────────────── */
.header-actions { display: flex; gap: 0.5rem; }

/* ── Upload Area ─────────────────────────────── */
.upload-zone {
    background: linear-gradient(135deg, var(--green-xpale) 0%, #F8FAFC 100%);
    border: 2px dashed #A7F3D0;
    border-radius: var(--radius-lg);
    padding: 2rem 2.5rem;
    text-align: center;
    margin: 1.25rem 0 0.5rem;
    transition: border-color 0.2s, background 0.2s;
    position: relative; overflow: hidden;
}
.upload-zone::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 30% 50%, rgba(16,185,129,.06) 0%, transparent 70%);
    pointer-events: none;
}
.upload-zone:hover {
    border-color: var(--green-mid);
    background: linear-gradient(135deg, #D1FAE5 0%, #F0FDF4 100%);
}
.upload-icon {
    font-size: 2.25rem; margin-bottom: 0.75rem;
    filter: drop-shadow(0 2px 4px rgba(5,150,105,.2));
}
.upload-title {
    font-size: 1rem; font-weight: 600;
    color: var(--green-dark); margin-bottom: 0.35rem;
}
.upload-sub {
    font-size: 0.8rem; color: var(--text-secondary);
    display: flex; justify-content: center; gap: 8px;
}
.file-tag {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1px 8px;
    font-size: 0.72rem; font-weight: 600;
    color: var(--text-secondary);
}

/* ── Chat Messages ───────────────────────────── */
.stChatMessage {
    margin-bottom: 0.85rem !important;
    animation: fadeSlideUp 0.3s ease forwards;
}

@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* User bubble */
[data-testid="stChatMessage"][data-testid*="user"] > div,
.stChatMessage.user > div {
    background: linear-gradient(135deg, var(--green-mid), var(--green-dark)) !important;
    color: white !important;
    border-radius: 20px 20px 4px 20px !important;
    padding: 0.9rem 1.2rem !important;
    max-width: 78%;
    box-shadow: 0 4px 16px rgba(5,150,105,.25) !important;
}

/* Assistant bubble */
[data-testid="stChatMessage"][data-testid*="assistant"] > div,
.stChatMessage.assistant > div {
    background: var(--surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px 20px 20px 4px !important;
    padding: 0.9rem 1.2rem !important;
    max-width: 78%;
    box-shadow: var(--shadow-sm) !important;
}

/* ── Confidence Badge ────────────────────────── */
.conf-badge {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 0.73rem; font-weight: 600;
    padding: 3px 10px; border-radius: var(--radius-full);
    margin-top: 8px;
}
.conf-high   { background: #D1FAE5; color: #065F46; }
.conf-medium { background: #FEF3C7; color: #92400E; }
.conf-low    { background: #FEE2E2; color: #991B1B; }

/* ── Translation Badge ───────────────────────── */
.trans-badge {
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 0.7rem; font-weight: 600;
    padding: 3px 10px; border-radius: var(--radius-full);
    background: #EEF2FF; color: #3730A3;
    border: 1px solid #C7D2FE;
    margin-top: 4px; margin-left: 6px;
}

/* ── Source Cards ────────────────────────────── */
.source-card {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--green-light);
    border-radius: var(--radius-md);
    padding: 0.85rem 1rem;
    margin-bottom: 0.65rem;
    transition: box-shadow 0.15s;
}
.source-card:hover { box-shadow: var(--shadow-sm); }
.source-file {
    font-size: 0.8rem; font-weight: 600;
    color: var(--green-dark); margin-bottom: 0.4rem;
    display: flex; align-items: center; gap: 5px;
}
.source-score {
    margin-left: auto;
    font-size: 0.7rem; font-weight: 500;
    background: var(--green-pale); color: var(--green-dark);
    padding: 1px 7px; border-radius: 99px;
}
.source-text {
    font-size: 0.8rem; color: var(--text-secondary);
    line-height: 1.55;
}

/* ── Chat Input ──────────────────────────────── */
.stChatInputContainer {
    border: 2px solid var(--border) !important;
    border-radius: var(--radius-full) !important;
    padding: 0.3rem 0.3rem 0.3rem 1.4rem !important;
    background: var(--surface) !important;
    box-shadow: var(--shadow-md) !important;
    transition: border-color 0.2s !important;
}
.stChatInputContainer:focus-within {
    border-color: var(--green-mid) !important;
    box-shadow: 0 0 0 3px rgba(5,150,105,.1), var(--shadow-md) !important;
}
.stChatInputContainer button {
    background: linear-gradient(135deg, var(--green-light), var(--green-mid)) !important;
    border-radius: var(--radius-full) !important;
    padding: 0.5rem 1.5rem !important;
    box-shadow: 0 2px 8px rgba(5,150,105,.3) !important;
}

/* ── Expander ────────────────────────────────── */
.streamlit-expanderHeader {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* ── Success / Info ──────────────────────────── */
.stSuccess {
    background: var(--green-xpale) !important;
    color: var(--green-dark) !important;
    border: 1px solid var(--green-pale) !important;
    border-radius: var(--radius-md) !important;
}

/* ── Selectbox ───────────────────────────────── */
div[data-testid="stSelectbox"] > div {
    border-radius: var(--radius-md) !important;
    border-color: var(--border) !important;
    font-size: 0.88rem !important;
}

/* ── Slider ──────────────────────────────────── */
.stSlider > div > div > div { background: var(--green-light) !important; }

/* ── Divider ─────────────────────────────────── */
hr { border-color: var(--border) !important; }

/* ── Native language text ────────────────────── */
.native-answer {
    font-family: 'Noto Sans Telugu', 'Noto Sans Tamil', 'Noto Sans Kannada', 'Noto Sans Malayalam', 'DM Sans', sans-serif;
    line-height: 1.75;
}
</style>
""", unsafe_allow_html=True)

# =============================
# EMBEDDER
# =============================
@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = get_embedder()

# =============================
# HELPER FUNCTIONS
# =============================
def extract_text(file):
    try:
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            return "\n".join([p.extract_text() or "" for p in reader.pages])
        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            return "\n".join([p.text for p in doc.paragraphs if p.text])
        else:
            return file.read().decode("utf-8", errors="ignore")
    except:
        return ""

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks

def process_files(files):
    all_chunks, all_meta = [], []
    for f in files:
        text = extract_text(f)
        if text.strip():
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            all_meta.extend([{"file": f.name}] * len(chunks))
    if not all_chunks:
        return None, [], []
    embeddings = embedder.encode(all_chunks, convert_to_numpy=True)
    return embeddings, all_chunks, all_meta

def find_similar(query_emb, embeddings, chunks, meta, k=5):
    if embeddings is None or len(embeddings) == 0:
        return []
    sims = cosine_similarity([query_emb], embeddings)[0]
    top_idx = np.argsort(sims)[-k:][::-1]
    return [(chunks[i], meta[i]["file"], sims[i]) for i in top_idx]

def get_answer(context, question, target_language=None):
    try:
        lang_instruction = ""
        if target_language:
            lang_instruction = f"\n\nIMPORTANT: Provide your answer in {target_language} script. The user wants the response in {target_language}."

        system_prompt = f"""You are Kivi, an intelligent document assistant. Answer based ONLY on the provided context.
Be concise, accurate, and helpful. If information is not in the context, say so clearly.{lang_instruction}"""

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def translate_text(text, target_language):
    """Translate existing English text to target language."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": f"You are a professional translator. Translate the following text to {target_language}. Output ONLY the translated text, nothing else."},
                {"role": "user", "content": text},
            ],
            temperature=0.2,
            max_tokens=1200,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return text  # Fallback to original

def stream_response(text, placeholder):
    words = text.split()
    displayed = ""
    for i, word in enumerate(words):
        displayed += word + " "
        placeholder.markdown(f'<div class="native-answer">{displayed}{"▌" if i < len(words)-1 else ""}</div>', unsafe_allow_html=True)
        time.sleep(0.02)

def save_current_chat():
    if st.session_state.messages:
        for chat in st.session_state.saved_chats:
            if chat['messages'] == st.session_state.messages:
                return False, "Chat already saved"
        st.session_state.saved_chats.append({
            "name": st.session_state.current_chat_name,
            "messages": st.session_state.messages.copy(),
            "date": datetime.now().strftime("%b %d, %H:%M")
        })
        return True, "Chat saved!"
    return False, "Nothing to save"

# =============================
# SIDEBAR
# =============================
def show_sidebar():
    with st.sidebar:

        # Brand
        st.markdown("""
        <div class="sidebar-brand">
            <div class="brand-icon">K</div>
            <div class="brand-name">Kivi</div>
        </div>
        """, unsafe_allow_html=True)

        # New Chat
        if st.button("✦ New Chat", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.embeddings = None
            st.session_state.chunks = []
            st.session_state.meta = []
            st.session_state.current_chat_name = f"Chat {len(st.session_state.saved_chats) + 1}"
            st.rerun()

        # ── Language Selector ──────────────────────
        st.markdown('<div class="sidebar-label">🌐 Response Language</div>', unsafe_allow_html=True)
        selected_lang = st.selectbox(
            "Language",
            options=list(SOUTH_LANGUAGES.keys()),
            index=list(SOUTH_LANGUAGES.keys()).index(st.session_state.selected_language),
            label_visibility="collapsed",
            key="lang_selector"
        )
        st.session_state.selected_language = selected_lang

        lang_value = SOUTH_LANGUAGES[selected_lang]
        if lang_value:
            st.markdown(f"""
            <div style="font-size:0.75rem; color:#065F46; background:#D1FAE5; 
                        border-radius:8px; padding:6px 10px; margin-top:4px;">
                🗣️ Answers will be in <strong>{lang_value}</strong>
            </div>
            """, unsafe_allow_html=True)

        # ── Saved Chats ────────────────────────────
        st.markdown('<div class="sidebar-label">💬 Saved Chats</div>', unsafe_allow_html=True)

        if st.session_state.saved_chats:
            chats_to_delete = []
            for i, chat in enumerate(st.session_state.saved_chats):
                cols = st.columns([0.82, 0.18])
                with cols[0]:
                    if st.button(f"📄 {chat['name']}", key=f"chat_load_{i}", use_container_width=True):
                        st.session_state.messages = chat['messages']
                        st.session_state.current_chat_name = chat['name']
                        st.rerun()
                with cols[1]:
                    if st.button("🗑", key=f"chat_del_{i}", use_container_width=True):
                        chats_to_delete.append(i)
            if chats_to_delete:
                for i in sorted(chats_to_delete, reverse=True):
                    st.session_state.saved_chats.pop(i)
                st.rerun()
        else:
            st.caption("No saved chats yet")

        st.divider()

        # ── Settings ──────────────────────────────
        with st.expander("⚙️ Settings", expanded=False):
            TOP_K = st.slider("Chunks to retrieve", 3, 10, 5)
            show_sources = st.checkbox("Show sources", value=True)

        # ── Stats ─────────────────────────────────
        st.markdown('<div class="sidebar-label">📊 Stats</div>', unsafe_allow_html=True)
        num_docs = len(set([m['file'] for m in st.session_state.meta])) if st.session_state.meta else 0
        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-num">{num_docs}</div>
                <div class="stat-lbl">Documents</div>
            </div>
            <div class="stat-card">
                <div class="stat-num">{len(st.session_state.chunks)}</div>
                <div class="stat-lbl">Chunks</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        return TOP_K, show_sources

# =============================
# MAIN APP
# =============================
TOP_K, show_sources = show_sidebar()

# ── Header ────────────────────────────────────
col1, col2 = st.columns([0.58, 0.42])
with col1:
    lang_value = SOUTH_LANGUAGES[st.session_state.selected_language]
    lang_tag = f'<span class="lang-badge">🌐 {lang_value}</span>' if lang_value else ""
    st.markdown(f"""
    <div class="logo-wrap">
        <div class="logo-mark">
            <div class="logo-outer"></div>
            <div class="logo-inner"></div>
            <div class="logo-dot"></div>
        </div>
        <div>
            <div class="logo-text">Kivi{lang_tag}</div>
            <div class="logo-tagline">Intelligent Document Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("💾 Save", use_container_width=True):
            ok, msg = save_current_chat()
            st.toast(msg, icon="✅" if ok else "ℹ️")
            if ok:
                st.rerun()
    with b2:
        if st.button("🧹 Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with b3:
        if st.button("📁 New", use_container_width=True):
            st.session_state.messages = []
            st.session_state.embeddings = None
            st.session_state.chunks = []
            st.session_state.meta = []
            st.session_state.current_chat_name = f"Chat {len(st.session_state.saved_chats) + 1}"
            st.rerun()

st.markdown("<div style='height:1px; background:var(--border); margin: 1rem 0 1.5rem;'></div>", unsafe_allow_html=True)

# ── Upload Zone ────────────────────────────────
st.markdown("""
<div class="upload-zone">
    <div class="upload-icon">📄</div>
    <div class="upload-title">Drop your documents here</div>
    <div class="upload-sub">
        <span class="file-tag">PDF</span>
        <span class="file-tag">DOCX</span>
        <span class="file-tag">TXT</span>
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

if uploaded_files:
    new_hash = hashlib.sha256(str([f.name for f in uploaded_files]).encode()).hexdigest()
    if st.session_state.files_hash != new_hash:
        with st.spinner("🔮 Processing documents..."):
            embeddings, chunks, meta = process_files(uploaded_files)
            st.session_state.embeddings = embeddings
            st.session_state.chunks = chunks
            st.session_state.meta = meta
            st.session_state.files_hash = new_hash
        if embeddings is not None:
            st.success(f"✅ Ready! {len(chunks)} chunks indexed from {len(uploaded_files)} file(s)")
        else:
            st.warning("⚠️ No readable text found in the uploaded files.")

# ── Chat History ───────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(f'<div class="native-answer">{msg["content"]}</div>', unsafe_allow_html=True)

# ── Chat Input ─────────────────────────────────
lang_value = SOUTH_LANGUAGES[st.session_state.selected_language]
placeholder_text = f"Ask about your documents... (reply in {lang_value})" if lang_value else "Ask about your documents..."
question = st.chat_input(placeholder_text)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if not uploaded_files or st.session_state.embeddings is None:
        st.warning("⚠️ Please upload documents first.")
        st.stop()

    # Search
    with st.spinner("🔍 Searching documents..."):
        q_emb = embedder.encode([question])[0]
        retrieved = find_similar(q_emb, st.session_state.embeddings,
                                 st.session_state.chunks, st.session_state.meta, k=TOP_K)

    # Confidence
    if retrieved:
        avg_conf = sum([s for _, _, s in retrieved]) / len(retrieved)
        if avg_conf > 0.5:
            conf_label, conf_class = "High", "conf-high"
        elif avg_conf > 0.3:
            conf_label, conf_class = "Medium", "conf-medium"
        else:
            conf_label, conf_class = "Low", "conf-low"
    else:
        avg_conf, conf_label, conf_class = 0, "None", "conf-low"

    context = "\n\n".join([f"[From {fn}]\n{chunk}" for chunk, fn, _ in retrieved]) if retrieved else "No relevant information found."

    # Answer
    with st.chat_message("assistant"):
        lang_value = SOUTH_LANGUAGES[st.session_state.selected_language]

        with st.spinner("💭 Thinking..."):
            answer = get_answer(context, question, target_language=lang_value)

        placeholder = st.empty()
        stream_response(answer, placeholder)

        # Meta row
        lang_badge = f'<span class="trans-badge">🌐 {lang_value}</span>' if lang_value else ""
        st.markdown(f"""
        <div style="margin-top:8px; display:flex; align-items:center; flex-wrap:wrap; gap:4px;">
            <span class="conf-badge {conf_class}">🎯 {conf_label} confidence</span>
            {lang_badge}
        </div>
        """, unsafe_allow_html=True)

        # Source cards
        if show_sources and retrieved:
            with st.expander(f"📄 {len(retrieved)} source{'s' if len(retrieved) > 1 else ''} used"):
                for i, (chunk, fname, score) in enumerate(retrieved, 1):
                    preview = chunk[:280] + "..." if len(chunk) > 280 else chunk
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-file">
                            📁 {fname}
                            <span class="source-score">{score:.0%} match</span>
                        </div>
                        <div class="source-text">{preview}</div>
                    </div>
                    """, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    if len(st.session_state.messages) == 2:
        save_current_chat()

    st.rerun()

# ── Welcome Screen ─────────────────────────────
if len(st.session_state.messages) == 0:
    if st.session_state.embeddings is not None:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; color:var(--text-secondary);">
            <div style="font-size:3rem; margin-bottom:1rem; filter:drop-shadow(0 4px 8px rgba(5,150,105,.2));">🥝</div>
            <h3 style="color:var(--green-dark); font-family:'Playfair Display',serif; margin-bottom:0.5rem;">Ready to help!</h3>
            <p style="font-size:0.9rem;">Documents indexed — ask me anything.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; color:var(--text-secondary);">
            <div style="font-size:3.5rem; margin-bottom:1rem; filter:drop-shadow(0 4px 8px rgba(5,150,105,.2));">🥝</div>
            <h3 style="color:var(--green-dark); font-family:'Playfair Display',serif; margin-bottom:0.75rem;">Welcome to Kivi</h3>
            <p style="font-size:0.92rem; max-width:420px; margin:0 auto; line-height:1.65;">
                Upload your documents above, choose a language, and start asking questions — in English or your preferred South Indian language.
            </p>
            <div style="margin-top:1.5rem; display:flex; justify-content:center; gap:1rem; flex-wrap:wrap;">
                <div style="background:var(--green-xpale); border:1px solid var(--green-pale); border-radius:12px; padding:0.6rem 1.2rem; font-size:0.82rem; color:var(--green-dark); font-weight:500;">📄 PDF support</div>
                <div style="background:var(--green-xpale); border:1px solid var(--green-pale); border-radius:12px; padding:0.6rem 1.2rem; font-size:0.82rem; color:var(--green-dark); font-weight:500;">🌐 4 South Indian languages</div>
                <div style="background:var(--green-xpale); border:1px solid var(--green-pale); border-radius:12px; padding:0.6rem 1.2rem; font-size:0.82rem; color:var(--green-dark); font-weight:500;">🔍 Semantic search</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
