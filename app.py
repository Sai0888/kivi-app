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
    st.session_state.saved_chats = []
if "translate_mode" not in st.session_state:
    st.session_state.translate_mode = False
if "target_language" not in st.session_state:
    st.session_state.target_language = "Telugu"

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
/* TRANSLATE TOGGLE - SIMPLE */
/* ===================================== */
.translate-toggle {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 1rem 0;
    padding: 0.75rem;
    background: #F9FAFB;
    border-radius: 12px;
    border: 1px solid #E5E7EB;
}

.language-select {
    flex: 1;
}

/* ===================================== */
/* SIDEBAR - PERMANENT */
/* ===================================== */
section[data-testid="stSidebar"] {
    background: #F9FAFB !important;
    border-right: 1px solid #E5E7EB !important;
    width: 300px !important;
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
    max-height: 400px;
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
    display: flex;
    gap: 8px;
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
    transition: all 0.2s;
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
    transition: all 0.2s;
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

/* Success message */
.stSuccess {
    background: #E6F7E6 !important;
    color: #059669 !important;
    border: 1px solid #059669 !important;
}
</style>
""", unsafe_allow_html=True)

# =============================
# SIDEBAR - WITH DELETE FUNCTIONALITY
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
        st.markdown('<div class="sidebar-title">💬 SAVED CHATS</div>', unsafe_allow_html=True)
        
        if st.session_state.saved_chats:
            st.markdown('<div class="chat-list">', unsafe_allow_html=True)
            
            # Create a list to track which chats to delete
            chats_to_delete = []
            
            for i, chat in enumerate(st.session_state.saved_chats):
                # Create columns for chat item and delete button
                cols = st.columns([0.85, 0.15])
                
                with cols[0]:
                    # Chat button
                    if st.button(f"📄 {chat['name']}", key=f"chat_load_{i}", use_container_width=True):
                        st.session_state.messages = chat['messages']
                        st.session_state.current_chat_name = chat['name']
                        st.rerun()
                
                with cols[1]:
                    # Delete button
                    if st.button("🗑️", key=f"chat_delete_{i}", use_container_width=True):
                        chats_to_delete.append(i)
            
            # Delete chats after the loop
            if chats_to_delete:
                for i in sorted(chats_to_delete, reverse=True):
                    st.session_state.saved_chats.pop(i)
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
# TRANSLATE TOGGLE - SIMPLE & WORKING
# =============================
def show_translate_option():
    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        translate_on = st.toggle("🌎 Translate Mode", value=st.session_state.translate_mode)
        if translate_on != st.session_state.translate_mode:
            st.session_state.translate_mode = translate_on
            st.rerun()
    
    with col2:
        if st.session_state.translate_mode:
            st.session_state.target_language = st.selectbox(
                "Language",
                ["Telugu", "Hindi", "Tamil", "Malayalam", "Kannada"],
                index=0,
                label_visibility="collapsed"
            )

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

# =============================
# FIXED TRANSLATE FUNCTION - WORKS 100%
# =============================
def translate_text(text, target_language):
    """Simple, working translation function"""
    try:
        # Clean and prepare text
        text_preview = text[:3000] if len(text) > 3000 else text
        
        # Simple, clear prompt
        prompt = f"""Translate this document to {target_language}.

Document:
{text_preview}

Translation:"""
        
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temperature for accurate translation
            max_tokens=2000,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Translation error: {str(e)}"

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
                return False, "Chat already saved"
        # Save new chat
        st.session_state.saved_chats.append({
            "name": chat_name,
            "messages": st.session_state.messages.copy(),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        return True, "Chat saved successfully!"

# =============================
# MAIN APP
# =============================

# Show sidebar
TOP_K, show_sources = show_sidebar()

# Main header with logo and buttons
col1, col2 = st.columns([0.6, 0.4])
with col1:
    show_logo()
with col2:
    st.markdown('<div class="header-buttons">', unsafe_allow_html=True)
    button_col1, button_col2 = st.columns(2)
    
    with button_col1:
        if st.button("💾 Save", use_container_width=True):
            success, message = save_current_chat()
            if success:
                st.success(message)
                time.sleep(1)
                st.rerun()
            else:
                st.info(message)
    
    with button_col2:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Simple translate toggle (appears below header)
show_translate_option()

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
prompt = st.chat_input("Ask about your documents..." if not st.session_state.translate_mode else "Type anything to translate...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    if not uploaded_files or st.session_state.embeddings is None:
        st.warning("Please upload documents first")
        st.stop()
    
    # Get full document text
    full_text = " ".join(st.session_state.chunks) if st.session_state.chunks else ""
    
    with st.chat_message("assistant"):
        with st.spinner("💭 Working..."):
            
            if st.session_state.translate_mode:
                # TRANSLATE MODE - SIMPLE & WORKING
                answer = translate_text(full_text, st.session_state.target_language)
                st.caption(f"🌎 Translated to {st.session_state.target_language}")
                
            else:
                # NORMAL Q&A MODE
                q_emb = embedder.encode([prompt])[0]
                retrieved = find_similar(q_emb, st.session_state.embeddings, 
                                        st.session_state.chunks, st.session_state.meta, k=TOP_K)
                
                if retrieved:
                    avg_conf = sum([s for _,_,s in retrieved]) / len(retrieved)
                    confidence = "High" if avg_conf > 0.5 else "Medium" if avg_conf > 0.3 else "Low"
                    context = "\n\n".join([f"[From {fname}]\n{chunk}" for chunk, fname, _ in retrieved])
                    answer = get_answer(context, prompt)
                    
                    # Show confidence
                    col1, col2 = st.columns([0.3, 0.7])
                    with col1:
                        st.caption(f"🎯 Confidence: {confidence}")
                    
                    # Show sources
                    if show_sources:
                        with st.expander(f"📄 View {len(retrieved)} sources"):
                            for i, (chunk, fname, score) in enumerate(retrieved, 1):
                                st.markdown(f"**{i}. {fname}** (relevance: {score:.2f})")
                                st.info(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                else:
                    answer = "No relevant information found in the documents."
        
        # Display answer with streaming
        placeholder = st.empty()
        stream_response(answer, placeholder)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Auto-save chat
    if len(st.session_state.messages) == 2:
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
