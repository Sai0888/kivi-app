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
# LANGUAGES - CONSISTENT NAMES
# =============================
LANGUAGES = {
    "English": None,
    "తెలుగు (Telugu)": "Telugu",
    "தமிழ் (Tamil)": "Tamil",
    "ಕನ್ನಡ (Kannada)": "Kannada",
    "മലയാളം (Malayalam)": "Malayalam",
}

# =============================
# SESSION STATE - FIXED
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
    st.session_state.selected_language = "English"
else:
    # FIX: Validate existing language against current LANGUAGES
    if st.session_state.selected_language not in LANGUAGES:
        st.session_state.selected_language = "English"

# =============================
# PROFESSIONAL CSS
# =============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

/* ===== RESET & BASE ===== */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.main .block-container {
    padding: 2rem 2.5rem;
    max-width: 1200px;
    margin: 0 auto;
}

#MainMenu, footer, header {
    display: none;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: #F8FAFC !important;
    border-right: 1px solid #E2E8F0 !important;
}

section[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.25rem !important;
}

/* Sidebar header */
.sidebar-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #E2E8F0;
}

.sidebar-icon {
    width: 32px;
    height: 32px;
    background: #0F172A;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 600;
    font-size: 16px;
}

.sidebar-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #0F172A;
}

/* Section labels */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748B;
    margin: 1.5rem 0 0.75rem 0;
}

/* Language badge */
.lang-badge {
    background: #F1F5F9;
    color: #0F172A;
    font-size: 0.7rem;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 100px;
    margin-left: 6px;
}

/* Stats cards */
.stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin: 0.5rem 0;
}

.stat-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0.75rem;
    text-align: center;
}

.stat-number {
    font-size: 1.5rem;
    font-weight: 600;
    color: #0F172A;
    line-height: 1.2;
}

.stat-label {
    font-size: 0.65rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.02em;
}

/* ===== MAIN HEADER ===== */
.main-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #E2E8F0;
}

.logo-area {
    display: flex;
    align-items: center;
    gap: 12px;
}

.logo-mark {
    width: 40px;
    height: 40px;
    background: #0F172A;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 600;
    font-size: 18px;
}

.logo-text {
    font-size: 1.5rem;
    font-weight: 600;
    color: #0F172A;
    letter-spacing: -0.02em;
}

.logo-tagline {
    font-size: 0.75rem;
    color: #64748B;
    margin-top: 2px;
}

/* Header buttons */
.header-buttons {
    display: flex;
    gap: 0.5rem;
}

/* ===== UPLOAD AREA ===== */
.upload-area {
    background: #F8FAFC;
    border: 2px dashed #CBD5E1;
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    margin: 1.5rem 0;
    transition: all 0.2s;
}

.upload-area:hover {
    border-color: #0F172A;
    background: #F1F5F9;
}

.upload-icon {
    font-size: 2rem;
    margin-bottom: 0.75rem;
    opacity: 0.7;
}

.upload-title {
    font-size: 1rem;
    font-weight: 500;
    color: #0F172A;
    margin-bottom: 0.25rem;
}

.upload-sub {
    font-size: 0.8rem;
    color: #64748B;
}

/* ===== CHAT MESSAGES ===== */
.stChatMessage {
    margin-bottom: 0.75rem !important;
}

.stChatMessage > div {
    border-radius: 14px !important;
    padding: 0.9rem 1.2rem !important;
    max-width: 80%;
}

.stChatMessage.user > div {
    background: #0F172A !important;
    color: white !important;
    margin-left: auto !important;
}

.stChatMessage.assistant > div {
    background: #F8FAFC !important;
    color: #0F172A !important;
    border: 1px solid #E2E8F0 !important;
}

/* Confidence badge */
.conf-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 500;
    padding: 2px 10px;
    border-radius: 100px;
    margin-right: 6px;
}

.conf-high {
    background: #E6F7E6;
    color: #0F4D2E;
}

.conf-medium {
    background: #FEF9C3;
    color: #854D0E;
}

.conf-low {
    background: #FEE2E2;
    color: #991B1B;
}

/* Translation badge */
.trans-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 500;
    padding: 2px 10px;
    border-radius: 100px;
    background: #E0F2FE;
    color: #0369A1;
}

/* Source cards */
.source-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    border-left: 3px solid #0F172A;
}

.source-file {
    font-size: 0.75rem;
    font-weight: 600;
    color: #0F172A;
    margin-bottom: 0.25rem;
    display: flex;
    align-items: center;
    gap: 4px;
}

.source-score {
    font-size: 0.65rem;
    background: #E2E8F0;
    padding: 1px 6px;
    border-radius: 100px;
    margin-left: auto;
}

.source-text {
    font-size: 0.75rem;
    color: #475569;
    line-height: 1.5;
}

/* Chat input */
.stChatInputContainer {
    border: 1px solid #E2E8F0 !important;
    border-radius: 100px !important;
    padding: 0.2rem 0.2rem 0.2rem 1.2rem !important;
    background: white !important;
}

.stChatInputContainer button {
    background: #0F172A !important;
    border-radius: 100px !important;
    padding: 0.4rem 1.2rem !important;
}

/* Buttons */
.stButton > button {
    border-radius: 8px !important;
    border: 1px solid #E2E8F0 !important;
    background: white !important;
    color: #0F172A !important;
    font-size: 0.85rem !important;
    padding: 0.3rem 0 !important;
}

.stButton > button:hover {
    border-color: #0F172A !important;
    background: #F8FAFC !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
}

/* Success message */
.stSuccess {
    background: #F0FDF4 !important;
    color: #166534 !important;
    border: 1px solid #BBF7D0 !important;
}

/* Selectbox */
div[data-testid="stSelectbox"] > div {
    border-radius: 8px !important;
    border-color: #E2E8F0 !important;
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
        if target_language and target_language != "English":
            lang_instruction = f"\n\nIMPORTANT: Provide your answer in {target_language} language."

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

def stream_response(text, placeholder):
    words = text.split()
    displayed = ""
    for i, word in enumerate(words):
        displayed += word + " "
        placeholder.markdown(displayed + ("▌" if i < len(words)-1 else ""))
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
# SIDEBAR - FIXED
# =============================
with st.sidebar:
    # Header
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-icon">K</div>
        <div class="sidebar-title">Kivi</div>
    </div>
    """, unsafe_allow_html=True)
    
    # New Chat
    if st.button("+ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.embeddings = None
        st.session_state.chunks = []
        st.session_state.meta = []
        st.session_state.current_chat_name = f"Chat {len(st.session_state.saved_chats) + 1}"
        st.rerun()
    
    # Language Selection - FIXED with validation
    st.markdown('<div class="section-label">🌐 Language</div>', unsafe_allow_html=True)
    
    # Ensure selected language is valid
    if st.session_state.selected_language not in LANGUAGES:
        st.session_state.selected_language = "English"
    
    selected_lang = st.selectbox(
        "Language",
        options=list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(st.session_state.selected_language),
        label_visibility="collapsed",
        key="lang_selector_main"
    )
    st.session_state.selected_language = selected_lang
    
    # Saved Chats
    st.markdown('<div class="section-label">💬 Saved Chats</div>', unsafe_allow_html=True)
    
    if st.session_state.saved_chats:
        chats_to_delete = []
        for i, chat in enumerate(st.session_state.saved_chats):
            cols = st.columns([0.8, 0.2])
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
        st.caption("No saved chats")
    
    st.divider()
    
    # Settings
    with st.expander("⚙️ Settings"):
        TOP_K = st.slider("Chunks to retrieve", 3, 10, 5)
        show_sources = st.checkbox("Show sources", value=True)
    
    # Stats
    st.markdown('<div class="section-label">📊 Stats</div>', unsafe_allow_html=True)
    num_docs = len(set([m['file'] for m in st.session_state.meta])) if st.session_state.meta else 0
    st.markdown(f"""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-number">{num_docs}</div>
            <div class="stat-label">Documents</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{len(st.session_state.chunks)}</div>
            <div class="stat-label">Chunks</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================
# MAIN HEADER
# =============================
col1, col2 = st.columns([0.6, 0.4])
with col1:
    lang = LANGUAGES[st.session_state.selected_language]
    lang_display = f'<span class="lang-badge">{lang}</span>' if lang and lang != "English" else ""
    st.markdown(f"""
    <div class="logo-area">
        <div class="logo-mark">K</div>
        <div>
            <div class="logo-text">Kivi{lang_display}</div>
            <div class="logo-tagline">Document Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("💾 Save", use_container_width=True):
            ok, msg = save_current_chat()
            st.toast(msg)
            if ok:
                time.sleep(0.5)
                st.rerun()
    with b2:
        if st.button("🗑 Clear", use_container_width=True):
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

st.divider()

# =============================
# UPLOAD AREA
# =============================
st.markdown("""
<div class="upload-area">
    <div class="upload-icon">📄</div>
    <div class="upload-title">Drop your documents here</div>
    <div class="upload-sub">PDF • DOCX • TXT</div>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

# Process files
if uploaded_files:
    new_hash = hashlib.sha256(str([f.name for f in uploaded_files]).encode()).hexdigest()
    if st.session_state.files_hash != new_hash:
        with st.spinner("Processing documents..."):
            embeddings, chunks, meta = process_files(uploaded_files)
            if embeddings is not None:
                st.session_state.embeddings = embeddings
                st.session_state.chunks = chunks
                st.session_state.meta = meta
                st.session_state.files_hash = new_hash
                st.success(f"✅ Ready! {len(chunks)} chunks indexed")
            else:
                st.warning("No readable text found")

# =============================
# CHAT HISTORY
# =============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =============================
# CHAT INPUT
# =============================
lang = LANGUAGES[st.session_state.selected_language]
placeholder = f"Ask in {lang}..." if lang and lang != "English" else "Ask about your documents..."
question = st.chat_input(placeholder)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    
    if st.session_state.embeddings is None:
        st.warning("Please upload documents first")
        st.stop()
    
    # Search
    with st.spinner("Searching..."):
        q_emb = embedder.encode([question])[0]
        retrieved = find_similar(q_emb, st.session_state.embeddings,
                                 st.session_state.chunks, st.session_state.meta, k=TOP_K)
    
    # Confidence
    if retrieved:
        avg_conf = sum([s for _, _, s in retrieved]) / len(retrieved)
        if avg_conf > 0.5:
            conf_class, conf_text = "conf-high", "High"
        elif avg_conf > 0.3:
            conf_class, conf_text = "conf-medium", "Medium"
        else:
            conf_class, conf_text = "conf-low", "Low"
    else:
        conf_class, conf_text = "conf-low", "None"
    
    # Context
    context = "\n\n".join([f"[{fn}]\n{chunk}" for chunk, fn, _ in retrieved]) if retrieved else ""
    
    # Answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = get_answer(context, question, LANGUAGES[st.session_state.selected_language])
        
        placeholder = st.empty()
        stream_response(answer, placeholder)
        
        # Badges
        badges = f'<span class="conf-badge {conf_class}">🎯 {conf_text}</span>'
        if lang and lang != "English":
            badges += f' <span class="trans-badge">🌐 {lang}</span>'
        st.markdown(f'<div style="margin-top: 8px;">{badges}</div>', unsafe_allow_html=True)
        
        # Sources
        if show_sources and retrieved:
            with st.expander(f"📄 {len(retrieved)} sources"):
                for chunk, fn, score in retrieved:
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-file">
                            📁 {fn} <span class="source-score">{score:.0%}</span>
                        </div>
                        <div class="source-text">{chunk[:200]}...</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    if len(st.session_state.messages) == 2:
        save_current_chat()
    
    st.rerun()

# Welcome
if len(st.session_state.messages) == 0 and st.session_state.embeddings is None:
    st.markdown("""
    <div style="text-align: center; padding: 3rem; color: #64748B;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🥝</div>
        <h3 style="color: #0F172A; margin-bottom: 0.5rem;">Welcome to Kivi</h3>
        <p style="font-size: 0.9rem;">Upload documents and start asking questions</p>
    </div>
    """, unsafe_allow_html=True)
