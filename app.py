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
    st.session_state.saved_chats = []  # Store chats in session

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
    max-width: 1200px;
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
/* SIDEBAR - PERMANENT */
/* ===================================== */
section[data-testid="stSidebar"] {
    background: #F9FAFB !important;
    border-right: 1px solid #E5E7EB !important;
    width: 280px !important;
}

section[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem !important;
    max-width: 100% !important;
}

/* Sidebar header */
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

/* New Chat button */
.new-chat-btn {
    width: 100%;
    padding: 0.75rem;
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    font-size: 0.9rem;
    font-weight: 500;
    color: #111827;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 1.5rem;
}

.new-chat-btn:hover {
    background: #F3F4F6;
    border-color: #059669;
}

/* Chat list */
.chat-list {
    max-height: 300px;
    overflow-y: auto;
    margin-bottom: 1rem;
}

.chat-item {
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

.chat-item-title {
    font-size: 0.9rem;
    font-weight: 500;
    color: #111827;
    margin-bottom: 0.25rem;
}

.chat-item-meta {
    font-size: 0.7rem;
    color: #9CA3AF;
    display: flex;
    gap: 8px;
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
    line-height: 1;
}

.stat-label {
    font-size: 0.7rem;
    color: #6B7280;
}

/* ===================================== */
/* MAIN CONTENT */
/* ===================================== */
.main-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 2rem;
}

.clear-btn {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    font-size: 0.9rem;
    color: #374151;
    cursor: pointer;
    transition: all 0.2s;
}

.clear-btn:hover {
    background: #F3F4F6;
    border-color: #DC2626;
    color: #DC2626;
}

/* Upload area */
.upload-area {
    background: #F9FAFB;
    border: 2px dashed #E5E7EB;
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    margin: 1.5rem 0;
    transition: all 0.2s;
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

/* Expander */
.streamlit-expanderHeader {
    background: #F9FAFB !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# =============================
# SIDEBAR - PERMANENT
# =============================
def show_sidebar():
    with st.sidebar:
        # Simple header with Kivi
        st.markdown("""
        <div class="sidebar-header">
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 32px; height: 32px; background: #059669; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 600;">K</div>
                <span style="font-size: 1.2rem; font-weight: 600; color: #111827;">Kivi</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # New Chat Button
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.embeddings = None
            st.session_state.chunks = []
            st.session_state.meta = []
            st.session_state.current_chat_name = f"Chat {len(st.session_state.saved_chats) + 1}"
            st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Saved Chats Section
        st.markdown('<div class="sidebar-title">💬 RECENT CHATS</div>', unsafe_allow_html=True)
        
        if st.session_state.saved_chats:
            st.markdown('<div class="chat-list">', unsafe_allow_html=True)
            for i, chat in enumerate(st.session_state.saved_chats[-10:]):
                # Check if this chat is active
                is_active = chat['messages'] == st.session_state.messages
                
                # Create a button for each chat
                if st.button(f"📄 {chat['name']}", key=f"chat_{i}", use_container_width=True):
                    st.session_state.messages = chat['messages']
                    st.session_state.current_chat_name = chat['name']
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.caption("No saved chats yet")
        
        st.divider()
        
        # Settings
        with st.expander("⚙️ Settings", expanded=False):
            TOP_K = st.slider("Chunks to retrieve", 3, 10, 5)
            show_sources = st.checkbox("Show sources", value=True)
        
        # Stats
        st.markdown('<div class="sidebar-title">📊 STATS</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{len(set([m['file'] for m in st.session_state.meta])) if st.session_state.meta else 0}</div>
                <div class="stat-label">Documents</div>
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
    except Exception as e:
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
    
    similarities = cosine_similarity([query_emb], embeddings)[0]
    top_indices = np.argsort(similarities)[-k:][::-1]
    
    return [(chunks[i], meta[i]["file"], similarities[i]) for i in top_indices]

def get_answer(context, question):
    try:
        system_prompt = """You are Kivi, an intelligent document assistant. Answer based ONLY on the provided context. 
        Be concise, accurate, and helpful. If information is not in the context, say so clearly."""
        
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.3,
            max_tokens=800,
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

# =============================
# SAVE CHAT FUNCTION
# =============================
def save_current_chat():
    if st.session_state.messages:
        chat_name = st.session_state.current_chat_name
        # Check if chat already exists
        for chat in st.session_state.saved_chats:
            if chat['messages'] == st.session_state.messages:
                return
        # Save new chat
        st.session_state.saved_chats.append({
            "name": chat_name,
            "messages": st.session_state.messages.copy(),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M")
        })

# =============================
# MAIN APP - DIRECT, NO LOGIN
# =============================

# Show sidebar
TOP_K, show_sources = show_sidebar()

# Main header
col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
with col1:
    show_logo()
with col2:
    if st.button("💾 Save Chat", use_container_width=True):
        save_current_chat()
        st.success("Chat saved!")
        time.sleep(1)
        st.rerun()
with col3:
    if st.button("🗑 Clear", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Upload area
st.markdown("""
<div class="upload-area">
    <div style="font-size: 2rem; margin-bottom: 1rem;">📄</div>
    <div style="font-size: 1.1rem; font-weight: 500; color: #111827; margin-bottom: 0.5rem;">Drop your documents here</div>
    <div style="color: #6B7280; font-size: 0.9rem;">PDF • DOCX • TXT</div>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

# Process uploaded files
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
            st.success(f"✅ Ready! {len(chunks)} chunks from {len(uploaded_files)} files")
        else:
            st.warning("No readable text found")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
question = st.chat_input("Ask about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    
    if not uploaded_files or st.session_state.embeddings is None:
        st.warning("Please upload documents first")
        st.stop()
    
    # Search documents
    with st.spinner("🔍 Searching..."):
        q_emb = embedder.encode([question])[0]
        retrieved = find_similar(q_emb, st.session_state.embeddings, 
                                st.session_state.chunks, st.session_state.meta, k=TOP_K)
    
    # Calculate confidence
    if retrieved:
        avg_conf = sum([s for _,_,s in retrieved]) / len(retrieved)
        confidence = "High" if avg_conf > 0.5 else "Medium" if avg_conf > 0.3 else "Low"
    else:
        avg_conf = 0
        confidence = "None"
    
    # Prepare context
    if not retrieved:
        context = "No relevant information found in the documents."
    else:
        context = "\n\n".join([f"[From {fname}]\n{chunk}" for chunk, fname, _ in retrieved])
    
    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("💭 Thinking..."):
            answer = get_answer(context, question)
        
        placeholder = st.empty()
        stream_response(answer, placeholder)
        
        # Show metadata
        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            st.caption(f"🎯 Confidence: {confidence}")
        
        # Show sources
        if show_sources and retrieved:
            with st.expander(f"📄 View {len(retrieved)} sources"):
                for i, (chunk, fname, score) in enumerate(retrieved, 1):
                    st.markdown(f"**{i}. {fname}** (relevance: {score:.2f})")
                    st.info(chunk[:300] + "..." if len(chunk) > 300 else chunk)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Auto-save chat
    if len(st.session_state.messages) == 2:  # First Q&A
        save_current_chat()
    
    st.rerun()

# Welcome message
if len(st.session_state.messages) == 0 and st.session_state.embeddings is not None:
    st.markdown("""
    <div style="text-align: center; padding: 3rem; color: #6B7280;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🥝</div>
        <h3 style="color: #111827; margin-bottom: 0.5rem;">Ready to help!</h3>
        <p>Ask me anything about your documents.</p>
    </div>
    """, unsafe_allow_html=True)
