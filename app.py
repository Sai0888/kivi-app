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
    page_title="Kivi - GenAI Document Assistant",
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
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []
if "genai_mode" not in st.session_state:
    st.session_state.genai_mode = "Q&A"  # Default mode

# =============================
# CLEAN PROFESSIONAL CSS
# =============================
st.markdown("""
<style>
/* Import fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Reset */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main container */
.main .block-container {
    padding: 2rem 2.5rem;
    max-width: 1400px;
    margin: 0 auto;
}

/* Hide Streamlit branding */
#MainMenu, footer, header {
    display: none;
}

/* ===================================== */
/* KIWI LOGO */
/* ===================================== */
.kivi-logo-container {
    display: flex;
    align-items: center;
    gap: 12px;
}

.kivi-logo-mark {
    width: 40px;
    height: 40px;
    position: relative;
}

.kivi-shape-outer {
    position: absolute;
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #059669, #10B981);
    border-radius: 40% 60% 60% 40% / 50% 50% 50% 50%;
    transform: rotate(-5deg);
}

.kivi-shape-inner {
    position: absolute;
    width: 28px;
    height: 28px;
    background: linear-gradient(135deg, #34D399, #6EE7B7);
    border-radius: 45% 55% 55% 45% / 45% 50% 50% 55%;
    top: 6px;
    left: 6px;
    transform: rotate(5deg);
    opacity: 0.9;
}

.kivi-dot {
    position: absolute;
    width: 5px;
    height: 5px;
    background: #064E3B;
    border-radius: 50%;
    top: 17px;
    left: 17px;
}

.kivi-logo-text {
    font-size: 28px;
    font-weight: 600;
    color: #111827;
    letter-spacing: -0.02em;
}

/* ===================================== */
/* GENAI MODE SELECTOR */
/* ===================================== */
.genai-selector {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 100px;
    padding: 0.25rem;
    display: flex;
    margin: 1rem 0 2rem 0;
}

.genai-option {
    flex: 1;
    text-align: center;
    padding: 0.5rem;
    border-radius: 100px;
    font-size: 0.9rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
}

.genai-option.active {
    background: #059669;
    color: white;
}

.genai-option:not(.active):hover {
    background: #F3F4F6;
}

/* ===================================== */
/* DOCUMENT PREVIEW */
/* ===================================== */
.doc-preview {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1rem;
    margin: 1rem 0;
    max-height: 200px;
    overflow-y: auto;
}

.doc-preview-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0.5rem;
    border-bottom: 1px solid #E5E7EB;
}

.doc-preview-item:last-child {
    border-bottom: none;
}

.doc-preview-name {
    flex: 1;
    font-size: 0.9rem;
    color: #111827;
}

.doc-preview-size {
    font-size: 0.8rem;
    color: #6B7280;
}

.doc-preview-status {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #059669;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #F9FAFB !important;
    border-right: 1px solid #E5E7EB !important;
    width: 300px !important;
}

section[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem !important;
}

.sidebar-header {
    padding: 0 0 1.5rem 0;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid #E5E7EB;
}

.sidebar-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.02em;
    color: #9CA3AF;
    margin-bottom: 1rem;
}

/* Chat list */
.chat-list {
    max-height: 300px;
    overflow-y: auto;
    margin-bottom: 1rem;
}

.chat-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem;
    border-radius: 10px;
    margin-bottom: 0.25rem;
    cursor: pointer;
    transition: all 0.2s;
    border: 1px solid transparent;
}

.chat-item:hover {
    background: #F3F4F6;
    border-color: #E5E7EB;
}

.chat-item.active {
    background: #E6F7E6;
    border-color: #059669;
}

.chat-item-content {
    flex: 1;
    overflow: hidden;
}

.chat-item-title {
    font-size: 0.9rem;
    font-weight: 500;
    color: #111827;
    margin-bottom: 0.25rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.chat-item-meta {
    font-size: 0.7rem;
    color: #9CA3AF;
}

.chat-delete-btn {
    opacity: 0;
    background: none;
    border: none;
    color: #9CA3AF;
    font-size: 1.1rem;
    cursor: pointer;
    padding: 0.25rem 0.5rem;
    border-radius: 6px;
}

.chat-item:hover .chat-delete-btn {
    opacity: 1;
}

.chat-delete-btn:hover {
    color: #DC2626;
    background: #FEE2E2;
}

/* Stats grid */
.stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin: 1rem 0;
}

.stat-card {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 0.75rem;
    text-align: center;
}

.stat-number {
    font-size: 1.5rem;
    font-weight: 600;
    color: #059669;
}

.stat-label {
    font-size: 0.7rem;
    color: #6B7280;
}

/* Main header */
.main-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
}

.header-buttons {
    display: flex;
    gap: 0.5rem;
}

.header-btn {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    font-size: 0.9rem;
    color: #374151;
    cursor: pointer;
}

.header-btn:hover {
    background: #F3F4F6;
    border-color: #059669;
}

/* Upload area */
.upload-area {
    background: #F9FAFB;
    border: 2px dashed #E5E7EB;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}

.upload-area:hover {
    border-color: #059669;
    background: #F3F4F6;
}

/* Chat messages */
.stChatMessage {
    margin-bottom: 1rem !important;
}

.stChatMessage > div {
    border-radius: 16px !important;
    padding: 1rem 1.25rem !important;
    max-width: 80%;
}

.stChatMessage.user > div {
    background: #059669 !important;
    color: white !important;
    margin-left: auto !important;
}

.stChatMessage.assistant > div {
    background: #F3F4F6 !important;
    color: #111827 !important;
    border: 1px solid #E5E7EB !important;
}

/* Chat input */
.stChatInputContainer {
    border: 2px solid #E5E7EB !important;
    border-radius: 100px !important;
    padding: 0.25rem 0.25rem 0.25rem 1.25rem !important;
}

.stChatInputContainer button {
    background: #059669 !important;
    border-radius: 100px !important;
    padding: 0.5rem 1.5rem !important;
}
</style>
""", unsafe_allow_html=True)

# =============================
# SIDEBAR
# =============================
def show_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 32px; height: 32px; background: #059669; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 600;">K</div>
                <span style="font-size: 1.2rem; font-weight: 600; color: #111827;">Kivi</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.embeddings = None
            st.session_state.chunks = []
            st.session_state.meta = []
            st.session_state.current_chat_name = f"Chat {len(st.session_state.saved_chats) + 1}"
            st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">💬 SAVED CHATS</div>', unsafe_allow_html=True)
        
        if st.session_state.saved_chats:
            chats_to_delete = []
            for i, chat in enumerate(st.session_state.saved_chats):
                cols = st.columns([0.85, 0.15])
                with cols[0]:
                    if st.button(f"📄 {chat['name']}", key=f"chat_load_{i}", use_container_width=True):
                        st.session_state.messages = chat['messages']
                        st.session_state.current_chat_name = chat['name']
                        st.rerun()
                with cols[1]:
                    if st.button("🗑️", key=f"chat_delete_{i}", use_container_width=True):
                        chats_to_delete.append(i)
            
            if chats_to_delete:
                for i in sorted(chats_to_delete, reverse=True):
                    st.session_state.saved_chats.pop(i)
                st.rerun()
        else:
            st.caption("No saved chats yet")
        
        st.divider()
        
        with st.expander("⚙️ Settings", expanded=False):
            TOP_K = st.slider("Chunks to retrieve", 3, 10, 5)
            show_sources = st.checkbox("Show sources", value=True)
        
        st.markdown('<div class="sidebar-title">📊 STATS</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{len(st.session_state.uploaded_file_names)}</div>
                <div class="stat-label">Files</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(st.session_state.chunks)}</div>
                <div class="stat-label">Chunks</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        return TOP_K, show_sources

# =============================
# KIWI LOGO
# =============================
def show_logo():
    st.markdown("""
    <div class="kivi-logo-container">
        <div class="kivi-logo-mark">
            <div class="kivi-shape-outer"></div>
            <div class="kivi-shape-inner"></div>
            <div class="kivi-dot"></div>
        </div>
        <div class="kivi-logo-text">Kivi</div>
    </div>
    """, unsafe_allow_html=True)

# =============================
# GENAI MODE SELECTOR
# =============================
def show_genai_selector():
    modes = ["Q&A", "Summarize", "Translate", "Analyze", "Extract"]
    
    html = '<div class="genai-selector">'
    for mode in modes:
        active_class = "active" if st.session_state.genai_mode == mode else ""
        html += f'<div class="genai-option {active_class}" onclick="alert(\'{mode}\')">{mode}</div>'
    html += '</div>'
    
    st.markdown(html, unsafe_allow_html=True)
    
    # Radio buttons for actual functionality (hidden but functional)
    selected = st.radio(
        "Mode",
        modes,
        index=modes.index(st.session_state.genai_mode),
        label_visibility="collapsed",
        key="mode_selector"
    )
    if selected != st.session_state.genai_mode:
        st.session_state.genai_mode = selected
        st.rerun()

# =============================
# DOCUMENT PREVIEW
# =============================
def show_document_preview():
    if st.session_state.uploaded_file_names:
        st.markdown('<div class="doc-preview">', unsafe_allow_html=True)
        for fname in st.session_state.uploaded_file_names:
            st.markdown(f"""
            <div class="doc-preview-item">
                <span>📄</span>
                <span class="doc-preview-name">{fname}</span>
                <span class="doc-preview-status"></span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# =============================
# GENAI FUNCTIONS
# =============================
def summarize_document(context):
    prompt = f"""Please provide a comprehensive summary of the following document. 
    Include main points, key findings, and important details:
    
    {context[:3000]}"""
    
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,
        )
        return resp.choices[0].message.content
    except:
        return "Error generating summary"

def translate_text(text, target_language="Spanish"):
    prompt = f"Translate the following text to {target_language}:\n\n{text[:2000]}"
    
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,
        )
        return resp.choices[0].message.content
    except:
        return "Error translating text"

def analyze_document(context):
    prompt = f"""Analyze this document and provide:
    1. Main topic
    2. Key themes
    3. Sentiment
    4. Complexity level
    5. Target audience
    
    Document: {context[:2000]}"""
    
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
        )
        return resp.choices[0].message.content
    except:
        return "Error analyzing document"

def extract_key_info(context, info_type="key points"):
    prompt = f"Extract and list the most important {info_type} from this document:\n\n{context[:2000]}"
    
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
        )
        return resp.choices[0].message.content
    except:
        return f"Error extracting {info_type}"

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
    file_names = []
    
    for f in files:
        file_names.append(f.name)
        text = extract_text(f)
        if text.strip():
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            all_meta.extend([{"file": f.name}] * len(chunks))
    
    st.session_state.uploaded_file_names = file_names
    
    if not all_chunks:
        return None, [], []
    
    embeddings = embedder.encode(all_chunks, convert_to_numpy=True)
    return embeddings, all_chunks, all_meta

def find_similar(query_emb, embeddings, chunks, meta, k=5):
    if embeddings is None:
        return []
    similarities = cosine_similarity([query_emb], embeddings)[0]
    top_indices = np.argsort(similarities)[-k:][::-1]
    return [(chunks[i], meta[i]["file"], similarities[i]) for i in top_indices]

def get_answer(context, question, mode="Q&A"):
    if mode == "Q&A":
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer based only on the context:"
    elif mode == "Summarize":
        return summarize_document(context)
    elif mode == "Translate":
        return translate_text(context)
    elif mode == "Analyze":
        return analyze_document(context)
    elif mode == "Extract":
        return extract_key_info(context, question)
    
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
        )
        return resp.choices[0].message.content
    except:
        return "Error generating response"

def save_current_chat():
    if st.session_state.messages:
        for chat in st.session_state.saved_chats:
            if chat['messages'] == st.session_state.messages:
                return False, "Already saved"
        st.session_state.saved_chats.append({
            "name": st.session_state.current_chat_name,
            "messages": st.session_state.messages.copy(),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        return True, "Chat saved!"

# =============================
# MAIN APP
# =============================

TOP_K, show_sources = show_sidebar()

# Header with only 2 buttons (removed duplicate)
col1, col2 = st.columns([0.7, 0.3])
with col1:
    show_logo()
with col2:
    cols = st.columns(2)
    with cols[0]:
        if st.button("💾 Save", use_container_width=True):
            success, msg = save_current_chat()
            st.success(msg) if success else st.info(msg)
            time.sleep(1)
            st.rerun()
    with cols[1]:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# GenAI Mode Selector
show_genai_selector()

# Upload area
st.markdown("""
<div class="upload-area">
    <div style="font-size: 2rem;">📄</div>
    <div style="font-size: 1.1rem; font-weight: 500;">Drop your documents here</div>
    <div style="color: #6B7280;">PDF • DOCX • TXT</div>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

# Show document preview
show_document_preview()

# Process files
if uploaded_files:
    new_hash = hashlib.sha256(str([f.name for f in uploaded_files]).encode()).hexdigest()
    if st.session_state.files_hash != new_hash:
        with st.spinner("Processing..."):
            emb, chunks, meta = process_files(uploaded_files)
            st.session_state.embeddings = emb
            st.session_state.chunks = chunks
            st.session_state.meta = meta
            st.session_state.files_hash = new_hash
        if emb is not None:
            st.success(f"✅ {len(chunks)} chunks ready")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
question = st.chat_input(f"Ask in {st.session_state.genai_mode} mode...")

if question and st.session_state.embeddings is not None:
    st.session_state.messages.append({"role": "user", "content": f"[{st.session_state.genai_mode}] {question}"})
    with st.chat_message("user"):
        st.markdown(f"**{st.session_state.genai_mode}:** {question}")
    
    # Get context for Q&A mode only
    if st.session_state.genai_mode == "Q&A":
        q_emb = embedder.encode([question])[0]
        retrieved = find_similar(q_emb, st.session_state.embeddings, 
                                st.session_state.chunks, st.session_state.meta, k=TOP_K)
        context = "\n\n".join([f"[{fname}]\n{chunk}" for chunk, fname, _ in retrieved]) if retrieved else ""
    else:
        # For other modes, use full document text
        context = " ".join(st.session_state.chunks) if st.session_state.chunks else ""
        retrieved = []
    
    with st.chat_message("assistant"):
        with st.spinner(f"Generating {st.session_state.genai_mode}..."):
            answer = get_answer(context, question, st.session_state.genai_mode)
        
        st.markdown(answer)
        
        if show_sources and retrieved and st.session_state.genai_mode == "Q&A":
            with st.expander("📄 Sources"):
                for chunk, fname, score in retrieved:
                    st.info(f"**{fname}** (score: {score:.2f})\n{chunk[:200]}...")
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    save_current_chat()
    st.rerun()
