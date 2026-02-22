import os
import hashlib
import numpy as np
import streamlit as st
import time
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

try:
    from groq import Groq
except ImportError:
    st.error("Installing packages... Please refresh in 2 minutes.")
    st.stop()

from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import docx

# Page config
st.set_page_config(page_title="Kivi - Document Assistant", page_icon="📄", layout="wide")

# Get API key
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("Please add GROQ_API_KEY to Secrets")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.1-8b-instant"

# Session state
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

# =============================
# CLEAN CSS - LIGHT THEME ONLY
# =============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Logo styling */
.kivi-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 20px 0;
}

.kivi-icon {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 700;
    font-size: 24px;
    box-shadow: 0 4px 12px rgba(99,102,241,0.2);
}

.kivi-text {
    font-size: 36px;
    font-weight: 600;
    color: #1F2937;
    letter-spacing: -0.5px;
}

/* Upload area */
.upload-area {
    background: #F9FAFB;
    border: 2px dashed #E5E7EB;
    border-radius: 16px;
    padding: 40px;
    text-align: center;
    margin: 20px 0;
}

.upload-area:hover {
    border-color: #6366F1;
    background: #F3F4F6;
}

/* Chat bubbles */
.stChatMessage.user > div {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    color: white !important;
}

.stChatMessage.assistant > div {
    background: #F3F4F6 !important;
    color: #1F2937 !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: #F9FAFB !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
}

/* Buttons */
.stButton > button {
    border-radius: 8px !important;
    background: #F3F4F6 !important;
    border: 1px solid #E5E7EB !important;
    color: #1F2937 !important;
}

.stButton > button:hover {
    background: #E5E7EB !important;
    border-color: #6366F1 !important;
}

/* Success/Warning messages */
.stAlert {
    background: #F9FAFB !important;
    border-left-color: #6366F1 !important;
    color: #1F2937 !important;
}
</style>
""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.markdown("""
    <div class="kivi-logo">
        <div class="kivi-icon">K</div>
        <div class="kivi-text">Kivi</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown("<p style='color: #6B7280; margin-top: -10px;'>Document Assistant • Ask questions about your files</p>", unsafe_allow_html=True)
st.divider()

# =============================
# UPLOAD AREA
# =============================
st.markdown("""
<div class="upload-area">
    <div style="font-size: 40px; margin-bottom: 10px;">📄</div>
    <div style="font-size: 18px; font-weight: 500; color: #1F2937; margin-bottom: 5px;">Drop your documents here</div>
    <div style="color: #6B7280; font-size: 14px;">PDF • DOCX • TXT</div>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    TOP_K = st.slider("Chunks to retrieve", 3, 10, 5)
    
    st.divider()
    st.markdown("### 📊 Stats")
    st.write(f"📄 Chunks: {len(st.session_state.chunks)}")
    st.write(f"💬 Messages: {len(st.session_state.messages)}")

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
    
    similarities = cosine_similarity([query_emb], embeddings)[0]
    top_indices = np.argsort(similarities)[-k:][::-1]
    
    return [(chunks[i], meta[i]["file"], similarities[i]) for i in top_indices]

def get_answer(system_prompt, user_prompt, question):
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are Kivi, a document assistant. Answer based ONLY on the provided context. If information is not found, say so."},
                {"role": "user", "content": f"Context:\n{user_prompt}\n\nQuestion: {question}"},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except:
        return "Error generating answer"

def stream_response(text, placeholder):
    words = text.split()
    displayed = ""
    for i, word in enumerate(words):
        displayed += word + " "
        placeholder.markdown(displayed + ("▌" if i < len(words)-1 else ""))
        time.sleep(0.02)

# =============================
# PROCESS UPLOADED FILES
# =============================
if uploaded_files:
    new_hash = hashlib.sha256(str([f.name for f in uploaded_files]).encode()).hexdigest()
    
    if st.session_state.files_hash != new_hash:
        with st.spinner("Processing documents..."):
            embeddings, chunks, meta = process_files(uploaded_files)
            st.session_state.embeddings = embeddings
            st.session_state.chunks = chunks
            st.session_state.meta = meta
            st.session_state.files_hash = new_hash
        
        if embeddings is not None:
            st.success(f"✅ Ready! {len(chunks)} chunks from {len(uploaded_files)} files")
        else:
            st.warning("No readable text found")

# =============================
# CHAT INTERFACE
# =============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    
    if not uploaded_files or st.session_state.embeddings is None:
        st.warning("Please upload documents first")
        st.stop()
    
    # Search documents
    q_emb = embedder.encode([question])[0]
    retrieved = find_similar(q_emb, st.session_state.embeddings, 
                            st.session_state.chunks, st.session_state.meta, k=TOP_K)
    
    # Show confidence
    if retrieved:
        avg_conf = sum([s for _,_,s in retrieved]) / len(retrieved)
        if avg_conf > 0.5:
            st.caption(f"📊 Confidence: High ({avg_conf:.2f})")
        elif avg_conf > 0.3:
            st.caption(f"📊 Confidence: Medium ({avg_conf:.2f})")
        else:
            st.caption(f"📊 Confidence: Low ({avg_conf:.2f})")
    
    # Prepare context
    if not retrieved:
        context = "No relevant information found."
    else:
        context = "\n\n".join([f"[{fname}]\n{chunk}" for chunk, fname, _ in retrieved])
    
    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = get_answer("", context, question)
        
        placeholder = st.empty()
        stream_response(answer, placeholder)
        
        # Show sources
        with st.expander(f"📚 Sources ({len(retrieved)} chunks)"):
            for i, (chunk, fname, score) in enumerate(retrieved, 1):
                st.markdown(f"**{i}. {fname}** (relevance: {score:.2f})")
                st.info(chunk[:200] + "..." if len(chunk) > 200 else chunk)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
