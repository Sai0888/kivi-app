import os
import hashlib
import numpy as np
import faiss
import streamlit as st
import time
import pandas as pd

from groq import Groq
from sentence_transformers import SentenceTransformer

from pypdf import PdfReader
import docx


# =============================
# PAGE CONFIG (ONLY ONCE)
# =============================
st.set_page_config(page_title="Kivi", page_icon="📄", layout="wide")


# =============================
# GET API KEY (Streamlit Secrets ONLY - NO dotenv)
# =============================
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception as e:
    st.error("""
    ⚠️ **GROQ_API_KEY not found in Streamlit Secrets!** 
    
    Please add your API key:
    
    1. Go to your app dashboard on [share.streamlit.io](https://share.streamlit.io)
    2. Click on your app → **Manage app** → **Settings** → **Secrets**
    3. Add this EXACT format:
    
    ```
    GROQ_API_KEY = "gsk_your_actual_api_key_here"
    ```
    
    4. Make sure to include the quotes!
    5. Click Save and restart the app
    """)
    st.stop()

if not GROQ_API_KEY or not GROQ_API_KEY.startswith("gsk_"):
    st.error("Invalid GROQ_API_KEY format. It should start with 'gsk_'")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.1-8b-instant"


# =============================
# SESSION STATE (init early)
# =============================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "meta" not in st.session_state:
    st.session_state.meta = []  # list of {"file":...}

if "files_hash" not in st.session_state:
    st.session_state.files_hash = None

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "saved_conversations" not in st.session_state:
    st.session_state.saved_conversations = []

if "suggested_question" not in st.session_state:
    st.session_state.suggested_question = None


# =============================
# DARK MODE TOGGLE & CSS
# =============================
# Define CSS based on dark mode
def get_css(dark_mode):
    if dark_mode:
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0A0A0A;
            color: #FFFFFF;
        }

        .stApp {
            background: #0A0A0A;
        }

        section[data-testid="stSidebar"] > div {
            background: #141414;
            border-right: 1px solid #242424;
        }

        hr{
            border: none;
            height: 1px;
            background: #242424;
        }

        /* Premium card */
        .kivi-card{
            background: #141414;
            border: 1px solid #242424;
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 14px;
        }
        .kivi-card-title{
            font-weight:600;
            font-size:14px;
            color: #FFFFFF;
            margin-bottom:6px;
        }
        .kivi-card-sub{
            font-size:13px;
            color: #888888;
            margin-bottom:10px;
        }

        /* Upload area */
        .upload-area {
            background: #141414;
            border: 1px dashed #333333;
            border-radius: 16px;
            padding: 32px;
            text-align: center;
            margin-bottom: 20px;
        }
        .upload-icon {
            font-size: 32px;
            margin-bottom: 12px;
            opacity: 0.7;
        }
        .upload-title {
            font-size: 15px;
            font-weight: 500;
            color: #FFFFFF;
            margin-bottom: 4px;
        }
        .upload-subtitle {
            color: #888888;
            font-size: 13px;
        }

        /* Buttons */
        div.stButton > button{
            border-radius: 10px !important;
            padding: 6px 12px !important;
            font-weight: 500 !important;
            font-size: 13px !important;
            background: #242424 !important;
            color: #FFFFFF !important;
            border: none !important;
        }
        div.stButton > button:hover{
            background: #333333 !important;
        }

        /* Chat bubbles */
        .stChatMessage>div{
            border-radius: 12px !important;
            padding: 12px 16px;
            margin-bottom: 8px;
            font-size: 14px;
            line-height: 1.5;
        }
        .stChatMessage [data-testid="chatMessageContent"] {
            color: #FFFFFF !important;
        }
        .stChatMessage.user>div{ 
            background: linear-gradient(135deg, #6366F1, #8B5CF6); 
        }
        .stChatMessage.assistant>div{ 
            background: #141414; 
            border: 1px solid #242424;
        }

        /* Expander */
        [data-testid="stExpander"]{
            border-radius: 12px !important;
            border: 1px solid #242424 !important;
            background: #141414 !important;
        }

        /* Text colors */
        .stMarkdown, p, li, h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF !important;
        }

        /* Input field */
        .stChatInputContainer {
            background: #141414 !important;
            border: 1px solid #242424 !important;
            border-radius: 12px !important;
        }
        
        /* Success/Warning messages */
        .stSuccess, .stWarning {
            background: #141414 !important;
            border: 1px solid #242424 !important;
            color: #FFFFFF !important;
        }
        </style>
        """
    else:
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #FFFFFF;
            color: #111111;
        }

        section[data-testid="stSidebar"] > div {
            background: #F8F8F8;
            border-right: 1px solid #EEEEEE;
        }

        hr{
            border: none;
            height: 1px;
            background: #EEEEEE;
        }

        /* Premium card */
        .kivi-card{
            background: #F8F8F8;
            border: 1px solid #EEEEEE;
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 14px;
        }
        .kivi-card-title{
            font-weight:600;
            font-size:14px;
            color: #111111;
            margin-bottom:6px;
        }
        .kivi-card-sub{
            font-size:13px;
            color: #666666;
            margin-bottom:10px;
        }

        /* Upload area */
        .upload-area {
            background: #F8F8F8;
            border: 1px dashed #DDDDDD;
            border-radius: 16px;
            padding: 32px;
            text-align: center;
            margin-bottom: 20px;
        }
        .upload-icon {
            font-size: 32px;
            margin-bottom: 12px;
            opacity: 0.5;
        }
        .upload-title {
            font-size: 15px;
            font-weight: 500;
            color: #111111;
            margin-bottom: 4px;
        }
        .upload-subtitle {
            color: #666666;
            font-size: 13px;
        }

        /* Buttons */
        div.stButton > button{
            border-radius: 10px !important;
            padding: 6px 12px !important;
            font-weight: 500 !important;
            font-size: 13px !important;
            background: #F8F8F8 !important;
            color: #111111 !important;
            border: 1px solid #EEEEEE !important;
        }
        div.stButton > button:hover{
            background: #FFFFFF !important;
        }

        /* Chat bubbles */
        .stChatMessage>div{
            border-radius: 12px !important;
            padding: 12px 16px;
            margin-bottom: 8px;
            font-size: 14px;
            line-height: 1.5;
        }
        .stChatMessage.user>div{ 
            background: linear-gradient(135deg, #6366F1, #8B5CF6); 
            color: #FFFFFF !important;
        }
        .stChatMessage.assistant>div{ 
            background: #F8F8F8; 
            border: 1px solid #EEEEEE;
        }

        /* Expander */
        [data-testid="stExpander"]{
            border-radius: 12px !important;
            border: 1px solid #EEEEEE !important;
        }

        /* Input field */
        .stChatInputContainer {
            background: #F8F8F8 !important;
            border: 1px solid #EEEEEE !important;
            border-radius: 12px !important;
        }
        </style>
        """

# Apply CSS
st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)


# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:20px;">
        <span style="font-size:20px;font-weight:600;letter-spacing:-0.3px;">Kivi</span>
    </div>
    """, unsafe_allow_html=True)

    # Dark mode toggle
    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)

    TOP_K = st.slider("Top chunks to retrieve", 2, 8, 4)
    st.caption("More chunks = better recall but can add noise.")

    st.divider()
    
    # Analytics
    st.markdown("### 📊 Analytics")
    st.write("Chunks:", len(st.session_state.chunks))
    st.write("Messages:", len(st.session_state.messages))

    st.divider()
    
    # Saved Conversations
    st.markdown("### 💾 Saved Chats")
    
    if st.button("📌 Save Current Chat", use_container_width=True):
        if st.session_state.messages:
            st.session_state.saved_conversations.append({
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                "messages": st.session_state.messages.copy()
            })
            st.success("Chat saved!")
    
    # Display saved conversations
    for i, conv in enumerate(st.session_state.saved_conversations[-5:]):
        msg_count = len(conv["messages"])
        if st.button(f"📄 {conv['timestamp']} - {msg_count} msgs", key=f"saved_{i}", use_container_width=True):
            st.session_state.messages = conv["messages"]
            st.rerun()


# =============================
# COOL UNIQUE LOGO (No status chip)
# =============================
st.markdown("""
<style>
@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-5px); }
    100% { transform: translateY(0px); }
}

@keyframes glow {
    0% { filter: drop-shadow(0 0 5px rgba(99,102,241,0.3)); }
    50% { filter: drop-shadow(0 0 15px rgba(139,92,246,0.5)); }
    100% { filter: drop-shadow(0 0 5px rgba(99,102,241,0.3)); }
}

.kivi-logo-container {
    display: flex;
    align-items: center;
    gap: 15px;
    margin: 30px 0 20px 0;
    animation: float 4s ease-in-out infinite;
}

.kivi-logo-icon {
    width: 50px;
    height: 50px;
    position: relative;
    animation: glow 3s ease-in-out infinite;
}

.kivi-logo-shape1 {
    position: absolute;
    width: 50px;
    height: 50px;
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    border-radius: 18px;
    transform: rotate(10deg);
    opacity: 0.8;
}

.kivi-logo-shape2 {
    position: absolute;
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #8B5CF6, #EC4899);
    border-radius: 14px;
    top: 5px;
    left: 5px;
    transform: rotate(-5deg);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 700;
    font-size: 20px;
    backdrop-filter: blur(2px);
}

.kivi-logo-text {
    font-size: 40px;
    font-weight: 700;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #111111, #333333);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    position: relative;
}

.kivi-logo-dot {
    width: 6px;
    height: 6px;
    background: linear-gradient(135deg, #6366F1, #EC4899);
    border-radius: 50%;
    display: inline-block;
    margin-left: 2px;
    animation: glow 2s ease-in-out infinite;
}

/* Dark mode adjustments */
[data-theme="dark"] .kivi-logo-text {
    background: linear-gradient(135deg, #FFFFFF, #AAAAAA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>

<div class="kivi-logo-container">
    <div class="kivi-logo-icon">
        <div class="kivi-logo-shape1"></div>
        <div class="kivi-logo-shape2">K</div>
    </div>
    <div class="kivi-logo-text">
        Kivi<span class="kivi-logo-dot"></span>
    </div>
</div>
""", unsafe_allow_html=True)

# Clear chat button only (no status chip)
col1, col2 = st.columns([0.9, 0.1])
with col2:
    if st.button("🗑 Clear", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.divider()


# =============================
# PREMIUM UPLOAD AREA
# =============================
st.markdown("""
<div class="upload-area">
    <div class="upload-icon">📄</div>
    <div class="upload-title">Drop your documents here</div>
    <div class="upload-subtitle">PDF • DOCX • TXT</div>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)


# =============================
# EMBEDDER (LOCAL = FREE)
# =============================
@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = get_embedder()


# =============================
# HELPERS
# =============================
def extract_text(file) -> str:
    name = file.name.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(file)
        parts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t)
        return "\n".join(parts)

    if name.endswith(".docx"):
        d = docx.Document(file)
        return "\n".join([p.text for p in d.paragraphs if p.text.strip()])

    if name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

    return ""

def chunk_text(text: str, chunk_size=900, overlap=180):
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(text), step):
        c = text[i:i+chunk_size].strip()
        if c:
            chunks.append(c)
    return chunks

def compute_files_hash(files) -> str:
    h = hashlib.sha256()
    for f in files:
        data = f.getvalue()
        h.update(f.name.encode("utf-8"))
        h.update(data)
    return h.hexdigest()

def build_index_from_files(files):
    all_chunks = []
    all_meta = []

    for f in files:
        text = extract_text(f)
        if not text.strip():
            continue

        chunks = chunk_text(text)
        for c in chunks:
            all_chunks.append(c)
            all_meta.append({"file": f.name})

    if not all_chunks:
        return None, [], []

    emb = embedder.encode(all_chunks, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb)
    return index, all_chunks, all_meta

def groq_answer(system_prompt: str, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content

def stream_response(text, placeholder):
    """Simulate streaming for better UX"""
    displayed = ""
    words = text.split()
    
    for i, word in enumerate(words):
        displayed += word + " "
        if i < len(words) - 1:
            placeholder.markdown(displayed + "▌")
        else:
            placeholder.markdown(displayed)
        time.sleep(0.03)


# =============================
# PROCESS FILES
# =============================
if uploaded_files:
    new_hash = compute_files_hash(uploaded_files)
    if st.session_state.files_hash != new_hash:
        with st.spinner("Building search index..."):
            idx, chunks, meta = build_index_from_files(uploaded_files)
            st.session_state.index = idx
            st.session_state.chunks = chunks
            st.session_state.meta = meta
            st.session_state.files_hash = new_hash

        if idx is None:
            st.warning("No readable text found in uploaded files.")
        else:
            st.success(f"Loaded {len(uploaded_files)} file(s) • {len(chunks)} chunks")


# =============================
# CHAT HISTORY
# =============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =============================
# CHAT INPUT
# =============================
question = st.chat_input("Ask something about your documents...")

# Handle suggested question if exists
if st.session_state.suggested_question and not question:
    question = st.session_state.suggested_question
    st.session_state.suggested_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if not uploaded_files or st.session_state.index is None:
        st.warning("Upload documents first.")
        st.stop()

    q_emb = embedder.encode([question], convert_to_numpy=True).astype("float32")
    D, I = st.session_state.index.search(q_emb, k=TOP_K)

    retrieved = []
    for idx in I[0]:
        if 0 <= idx < len(st.session_state.chunks):
            retrieved.append((st.session_state.chunks[idx], st.session_state.meta[idx]["file"]))

    context = "\n\n".join([f"[{fname}]\n{chunk}" for chunk, fname in retrieved])

    system_prompt = (
        "You are a helpful assistant. Answer ONLY using the provided context. "
        "If the answer is not in the context, say exactly: Not found in the document."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = groq_answer(system_prompt, user_prompt)
            except Exception as e:
                st.error(f"Groq call failed: {e}")
                st.stop()

        # Stream the response
        response_placeholder = st.empty()
        stream_response(answer, response_placeholder)

        with st.expander("🔎 Retrieved Context Used"):
            for i, (chunk, fname) in enumerate(retrieved, start=1):
                st.markdown(f"**Chunk {i} • {fname}**")
                st.info(chunk)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Force rerun to show the complete message
    st.rerun()


# =============================
# SUGGESTED QUESTIONS
# =============================
if uploaded_files and st.session_state.index is not None and len(st.session_state.messages) == 0:
    st.markdown("### 💡 Suggested Questions")
    suggestion_cols = st.columns(3)
    suggestions = [
        "Summarize the main points",
        "What are the key findings?",
        "What methodology was used?",
        "Who are the authors?",
        "What is the conclusion?",
        "List important dates mentioned"
    ]
    
    for i, suggestion in enumerate(suggestions[:3]):
        with suggestion_cols[i]:
            if st.button(suggestion, use_container_width=True):
                st.session_state.suggested_question = suggestion
                st.rerun()
