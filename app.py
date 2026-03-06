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
    page_title="Kivi - Document Assistant",
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
if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = "❓ Ask Questions"
if "target_language" not in st.session_state:
    st.session_state.target_language = "Telugu"

# =============================
# CLEAN CSS
# =============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    padding: 2rem;
    max-width: 1000px;
    margin: 0 auto;
}

#MainMenu, footer, header {
    display: none;
}

/* Logo */
.kivi-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1rem;
}

.kivi-icon {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #059669, #10B981);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 600;
    font-size: 20px;
}

.kivi-text {
    font-size: 24px;
    font-weight: 600;
    color: #111827;
}

/* Mode selector */
.mode-selector {
    display: flex;
    gap: 0.5rem;
    margin: 1.5rem 0;
    flex-wrap: wrap;
}

.mode-btn {
    padding: 0.5rem 1rem;
    border-radius: 100px;
    border: 1px solid #E5E7EB;
    background: white;
    font-size: 0.9rem;
    cursor: pointer;
}

.mode-btn.active {
    background: #059669;
    color: white;
    border-color: #059669;
}

/* Upload area */
.upload-area {
    background: #F9FAFB;
    border: 2px dashed #E5E7EB;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}

/* Chat messages */
.stChatMessage > div {
    border-radius: 12px !important;
    padding: 1rem !important;
}

.stChatMessage.user > div {
    background: #059669 !important;
    color: white !important;
}

.stChatMessage.assistant > div {
    background: #F3F4F6 !important;
    color: #111827 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #F9FAFB !important;
    border-right: 1px solid #E5E7EB !important;
}

.sidebar-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #6B7280;
    margin: 1rem 0 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 2rem;">
        <div style="width: 32px; height: 32px; background: #059669; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 600;">K</div>
        <span style="font-size: 1.2rem; font-weight: 600;">Kivi</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.embeddings = None
        st.session_state.chunks = []
        st.session_state.meta = []
        st.rerun()
    
    st.markdown('<div class="sidebar-title">SAVED CHATS</div>', unsafe_allow_html=True)
    
    if st.session_state.saved_chats:
        for i, chat in enumerate(st.session_state.saved_chats[-5:]):
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                if st.button(f"📄 {chat['name']}", key=f"chat_{i}", use_container_width=True):
                    st.session_state.messages = chat['messages']
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{i}", use_container_width=True):
                    st.session_state.saved_chats.pop(i)
                    st.rerun()
    else:
        st.caption("No saved chats")
    
    st.divider()
    
    st.markdown('<div class="sidebar-title">SETTINGS</div>', unsafe_allow_html=True)
    TOP_K = st.slider("Chunks to retrieve", 3, 10, 5)
    show_sources = st.checkbox("Show sources", True)

# =============================
# HEADER
# =============================
col1, col2 = st.columns([0.7, 0.3])
with col1:
    st.markdown("""
    <div class="kivi-logo">
        <div class="kivi-icon">🥝</div>
        <div class="kivi-text">Kivi</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# =============================
# MODE SELECTOR - SIMPLE RADIO
# =============================
mode = st.radio(
    "Select Mode",
    ["❓ Ask Questions", "📋 Summarize Document", "🌎 Translate Document"],
    horizontal=True,
    label_visibility="collapsed"
)
st.session_state.selected_mode = mode

# If Translate mode is selected, show language options
if mode == "🌎 Translate Document":
    st.session_state.target_language = st.selectbox(
        "Translate to:",
        ["Telugu", "Hindi", "Malayalam", "Tamil"],
        index=0
    )

# =============================
# UPLOAD AREA
# =============================
st.markdown("""
<div class="upload-area">
    <div style="font-size: 2rem;">📄</div>
    <div style="font-weight: 500;">Drop your documents here</div>
    <div style="color: #6B7280; font-size: 0.9rem;">PDF • DOCX • TXT</div>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
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

def get_groq_response(prompt):
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def save_chat():
    if st.session_state.messages:
        name = f"Chat {len(st.session_state.saved_chats) + 1}"
        st.session_state.saved_chats.append({
            "name": name,
            "messages": st.session_state.messages.copy(),
            "date": datetime.now().strftime("%Y-%m-%d")
        })
        return True
    return False

# =============================
# PROCESS FILES
# =============================
if uploaded_files:
    new_hash = hashlib.sha256(str([f.name for f in uploaded_files]).encode()).hexdigest()
    if st.session_state.files_hash != new_hash:
        with st.spinner("Processing documents..."):
            emb, chunks, meta = process_files(uploaded_files)
            st.session_state.embeddings = emb
            st.session_state.chunks = chunks
            st.session_state.meta = meta
            st.session_state.files_hash = new_hash
        if emb is not None:
            st.success(f"✅ Loaded {len(chunks)} chunks from {len(uploaded_files)} files")

# =============================
# CHAT HISTORY
# =============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =============================
# CHAT INPUT & RESPONSES
# =============================
prompt = st.chat_input("Type your message...")

if prompt:
    # Check if documents are uploaded
    if st.session_state.embeddings is None:
        st.warning("Please upload documents first")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f"**{st.session_state.selected_mode}:** {prompt}")
    
    # Get full document text
    full_text = " ".join(st.session_state.chunks) if st.session_state.chunks else ""
    
    # Generate response based on mode
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            if st.session_state.selected_mode == "❓ Ask Questions":
                # Search for relevant chunks
                q_emb = embedder.encode([prompt])[0]
                retrieved = find_similar(q_emb, st.session_state.embeddings, 
                                        st.session_state.chunks, st.session_state.meta, k=TOP_K)
                
                if retrieved:
                    context = "\n\n".join([f"[From {fname}]\n{chunk}" for chunk, fname, _ in retrieved])
                    qa_prompt = f"""You are a helpful document assistant. Answer the question based ONLY on the context below.
If the answer is not in the context, say "I cannot find this information in the documents."

Context:
{context}

Question: {prompt}

Answer:"""
                    response = get_groq_response(qa_prompt)
                    
                    # Show sources
                    if show_sources:
                        with st.expander(f"📚 Sources ({len(retrieved)} chunks)"):
                            for i, (chunk, fname, score) in enumerate(retrieved, 1):
                                st.markdown(f"**{i}. {fname}** (Relevance: {score:.2f})")
                                st.info(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                else:
                    response = "No relevant information found in the documents."
            
            elif st.session_state.selected_mode == "📋 Summarize Document":
                summary_prompt = f"""Please provide a clear, comprehensive summary of the following document.
Include the main points, key findings, and important details.

Document:
{full_text[:4000]}

Summary:"""
                response = get_groq_response(summary_prompt)
            
            elif st.session_state.selected_mode == "🌎 Translate Document":
                target = st.session_state.target_language
                translate_prompt = f"""Translate the following document to {target}.
Keep the original meaning, tone, and formatting as much as possible.

Document:
{full_text[:3000]}

Translation to {target}:"""
                response = get_groq_response(translate_prompt)
            
            else:
                response = "Please select a valid mode."
        
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Auto-save after first exchange
    if len(st.session_state.messages) == 2:
        save_chat()
    
    st.rerun()
