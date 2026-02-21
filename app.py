import os
import hashlib
import numpy as np
import streamlit as st
import time
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Import with error handling
try:
    from groq import Groq
except ImportError:
    st.error("""
    ⚠️ **Groq package is still installing...** 
    
    Please wait 2-3 minutes and refresh the page.
    """)
    st.stop()

from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import docx


# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Kivi - AI Document Assistant", page_icon="📄", layout="wide")


# =============================
# GET API KEY (Streamlit Secrets ONLY)
# =============================
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception as e:
    st.error("""
    ⚠️ **GROQ_API_KEY not found in Streamlit Secrets!** 
    
    Please add your API key in the Secrets section.
    """)
    st.stop()

if not GROQ_API_KEY or not GROQ_API_KEY.startswith("gsk_"):
    st.error("Invalid GROQ_API_KEY format. It should start with 'gsk_'")
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

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "saved_conversations" not in st.session_state:
    st.session_state.saved_conversations = []

if "suggested_question" not in st.session_state:
    st.session_state.suggested_question = None


# =============================
# DARK MODE TOGGLE & CSS (FULLY FIXED)
# =============================
def get_css(dark_mode):
    if dark_mode:
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Base styles */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0A0A0A !important;
            color: #FFFFFF !important;
        }

        .stApp {
            background: #0A0A0A !important;
        }

        /* Fix for all containers */
        .main > div {
            background-color: #0A0A0A !important;
        }

        .block-container {
            background-color: #0A0A0A !important;
            padding-top: 1rem !important;
        }

        .stApp > header {
            background-color: #141414 !important;
            border-bottom: 1px solid #242424 !important;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] > div {
            background: #141414 !important;
            border-right: 1px solid #242424 !important;
        }

        section[data-testid="stSidebar"] * {
            color: #FFFFFF !important;
        }

        /* Dividers */
        hr {
            border: none !important;
            height: 1px !important;
            background: #242424 !important;
            margin: 1rem 0 !important;
        }

        /* Expanders */
        .streamlit-expanderHeader {
            background-color: #141414 !important;
            color: #FFFFFF !important;
            border: 1px solid #242424 !important;
            border-radius: 8px !important;
        }

        .streamlit-expanderContent {
            background-color: #141414 !important;
            border: 1px solid #242424 !important;
            border-top: none !important;
            border-radius: 0 0 8px 8px !important;
        }

        [data-testid="stExpander"] {
            background: transparent !important;
        }

        /* Input fields */
        .stTextInput > div > div > input {
            background-color: #141414 !important;
            color: #FFFFFF !important;
            border-color: #242424 !important;
        }

        .stTextArea > div > div > textarea {
            background-color: #141414 !important;
            color: #FFFFFF !important;
            border-color: #242424 !important;
        }

        .stSelectbox > div > div {
            background-color: #141414 !important;
            color: #FFFFFF !important;
            border-color: #242424 !important;
        }

        .stMultiSelect > div > div {
            background-color: #141414 !important;
            color: #FFFFFF !important;
            border-color: #242424 !important;
        }

        /* Sliders */
        .stSlider > div {
            color: #FFFFFF !important;
        }

        .stSlider [data-baseweb="slider"] {
            background-color: #242424 !important;
        }

        /* Metrics */
        div[data-testid="stMetricValue"] {
            color: #FFFFFF !important;
            background: transparent !important;
        }

        div[data-testid="stMetricLabel"] {
            color: #888888 !important;
            background: transparent !important;
        }

        div[data-testid="stMetricDelta"] {
            color: #888888 !important;
        }

        /* Buttons */
        div.stButton > button {
            border-radius: 10px !important;
            padding: 6px 12px !important;
            font-weight: 500 !important;
            font-size: 13px !important;
            background: #242424 !important;
            color: #FFFFFF !important;
            border: none !important;
        }
        div.stButton > button:hover {
            background: #333333 !important;
            border: none !important;
        }

        /* Chat messages */
        .stChatMessage > div {
            border-radius: 12px !important;
            padding: 12px 16px !important;
            margin-bottom: 8px !important;
            font-size: 14px !important;
            line-height: 1.5 !important;
        }

        .stChatMessage [data-testid="chatMessageContent"] {
            color: #FFFFFF !important;
        }

        .stChatMessage.user > div { 
            background: linear-gradient(135deg, #6366F1, #8B5CF6) !important; 
        }

        .stChatMessage.assistant > div { 
            background: #141414 !important; 
            border: 1px solid #242424 !important;
        }

        /* Chat input */
        .stChatInputContainer {
            background: #141414 !important;
            border: 1px solid #242424 !important;
            border-radius: 12px !important;
        }

        .stChatInputContainer input {
            color: #FFFFFF !important;
        }

        /* Info/Warning/Success boxes */
        .stAlert {
            background: #141414 !important;
            border: 1px solid #242424 !important;
            color: #FFFFFF !important;
        }

        .stInfo {
            background: #141414 !important;
            border-left-color: #6366F1 !important;
        }

        .stWarning {
            background: #141414 !important;
            border-left-color: #F59E0B !important;
        }

        .stError {
            background: #141414 !important;
            border-left-color: #EF4444 !important;
        }

        .stSuccess {
            background: #141414 !important;
            border-left-color: #10B981 !important;
        }

        /* Data frames */
        .dataframe {
            background: #141414 !important;
            color: #FFFFFF !important;
        }

        .dataframe th {
            background: #242424 !important;
            color: #FFFFFF !important;
        }

        .dataframe td {
            background: #141414 !important;
            color: #FFFFFF !important;
            border-color: #242424 !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background: #141414 !important;
            border-color: #242424 !important;
        }

        .stTabs [data-baseweb="tab"] {
            color: #888888 !important;
        }

        .stTabs [aria-selected="true"] {
            color: #FFFFFF !important;
            border-bottom-color: #6366F1 !important;
        }

        /* Radio buttons and checkboxes */
        .stRadio > div {
            color: #FFFFFF !important;
        }

        .stCheckbox > div {
            color: #FFFFFF !important;
        }

        /* Code blocks */
        .stCodeBlock {
            background: #141414 !important;
            border: 1px solid #242424 !important;
        }

        /* File uploader */
        .uploadedFile {
            background: #141414 !important;
            border-color: #242424 !important;
            color: #FFFFFF !important;
        }

        /* Custom classes */
        .kivi-card {
            background: #141414 !important;
            border: 1px solid #242424 !important;
            border-radius: 16px !important;
            padding: 16px !important;
            margin-bottom: 14px !important;
        }

        .kivi-card-title {
            font-weight: 600 !important;
            font-size: 14px !important;
            color: #FFFFFF !important;
            margin-bottom: 6px !important;
        }

        .kivi-card-sub {
            font-size: 13px !important;
            color: #888888 !important;
            margin-bottom: 10px !important;
        }

        .upload-area {
            background: #141414 !important;
            border: 2px dashed #333333 !important;
            border-radius: 16px !important;
            padding: 32px !important;
            text-align: center !important;
            margin-bottom: 20px !important;
        }

        .upload-icon {
            font-size: 32px !important;
            margin-bottom: 12px !important;
            opacity: 0.7 !important;
        }

        .upload-title {
            font-size: 15px !important;
            font-weight: 500 !important;
            color: #FFFFFF !important;
            margin-bottom: 4px !important;
        }

        .upload-subtitle {
            color: #888888 !important;
            font-size: 13px !important;
        }

        /* Fix for any white backgrounds */
        [data-testid="stDecoration"] {
            background-image: none !important;
        }

        div[role="radiogroup"] {
            background: transparent !important;
        }

        /* Make sure all text is visible */
        p, li, h1, h2, h3, h4, h5, h6, span, div {
            color: inherit !important;
        }

        /* Logo animations */
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
            margin: 20px 0 10px 0;
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
            background: linear-gradient(135deg, #FFFFFF, #AAAAAA);
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

        .upload-area {
            background: #F8F8F8;
            border: 2px dashed #DDDDDD;
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

        [data-testid="stExpander"]{
            border-radius: 12px !important;
            border: 1px solid #EEEEEE !important;
        }

        .stChatInputContainer {
            background: #F8F8F8 !important;
            border: 1px solid #EEEEEE !important;
            border-radius: 12px !important;
        }

        /* Logo animations - Light mode */
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-5px); }
            100% { transform: translateY(0px); }
        }

        @keyframes glow {
            0% { filter: drop-shadow(0 0 5px rgba(99,102,241,0.2)); }
            50% { filter: drop-shadow(0 0 15px rgba(139,92,246,0.3)); }
            100% { filter: drop-shadow(0 0 5px rgba(99,102,241,0.2)); }
        }

        .kivi-logo-container {
            display: flex;
            align-items: center;
            gap: 15px;
            margin: 20px 0 10px 0;
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

    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    TOP_K = st.slider("Number of chunks to retrieve", 3, 10, 5)
    st.caption("More chunks = better context")
    st.divider()
    
    st.markdown("### 📊 Analytics")
    st.write("📄 Chunks:", len(st.session_state.chunks))
    st.write("💬 Messages:", len(st.session_state.messages))
    st.divider()
    
    st.markdown("### 💾 Saved Chats")
    if st.button("📌 Save Current Chat", use_container_width=True):
        if st.session_state.messages:
            st.session_state.saved_conversations.append({
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                "messages": st.session_state.messages.copy()
            })
            st.success("Chat saved!")
    
    for i, conv in enumerate(st.session_state.saved_conversations[-5:]):
        msg_count = len(conv["messages"])
        if st.button(f"📄 {conv['timestamp']} - {msg_count} msgs", key=f"saved_{i}", use_container_width=True):
            st.session_state.messages = conv["messages"]
            st.rerun()


# =============================
# LOGO
# =============================
st.markdown("""
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

# Clear chat button
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
# EMBEDDER
# =============================
@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = get_embedder()


# =============================
# HELPER FUNCTIONS
# =============================
def extract_text(file) -> str:
    name = file.name.lower()
    text = ""
    
    try:
        if name.endswith(".pdf"):
            reader = PdfReader(file)
            for page in reader.pages:
                t = page.extract_text() or ""
                if t.strip():
                    text += t + "\n"
                    
        elif name.endswith(".docx"):
            d = docx.Document(file)
            for p in d.paragraphs:
                if p.text.strip():
                    text += p.text + "\n"
                    
        elif name.endswith(".txt"):
            text = file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.warning(f"Error reading {file.name}: {str(e)}")
        return ""
    
    return text

def chunk_text(text: str, chunk_size=500, overlap=100):
    """Split text into smaller chunks for better retrieval"""
    if not text:
        return []
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        if chunk_words:
            chunk = " ".join(chunk_words)
            if len(chunk) > 50:
                chunks.append(chunk)
    
    return chunks

def compute_files_hash(files) -> str:
    h = hashlib.sha256()
    for f in files:
        data = f.getvalue()
        h.update(f.name.encode("utf-8"))
        h.update(data)
    return h.hexdigest()

def build_embeddings_from_files(files):
    """Build embeddings without faiss"""
    all_chunks = []
    all_meta = []
    
    for f in files:
        text = extract_text(f)
        if not text.strip():
            st.warning(f"No readable text found in {f.name}")
            continue
            
        chunks = chunk_text(text)
        for c in chunks:
            all_chunks.append(c)
            all_meta.append({"file": f.name})
        
        st.info(f"📄 {f.name}: {len(chunks)} chunks")
    
    if not all_chunks:
        return None, [], []
    
    # Create embeddings
    with st.spinner("🔮 Creating embeddings..."):
        embeddings = embedder.encode(all_chunks, convert_to_numpy=True, show_progress_bar=False)
    
    return embeddings, all_chunks, all_meta

def find_similar_chunks(query_embedding, embeddings, chunks, meta, k=5):
    """Find similar chunks using cosine similarity"""
    if embeddings is None or len(embeddings) == 0:
        return []
    
    # Calculate cosine similarity
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    
    # Get top k indices
    top_indices = np.argsort(similarities)[-k:][::-1]
    
    # Return all top results
    results = []
    for idx in top_indices:
        results.append((chunks[idx], meta[idx]["file"], similarities[idx]))
    
    return results

def groq_answer(system_prompt: str, user_prompt: str, question: str) -> str:
    """Enhanced answer generation with better prompting"""
    try:
        # Enhanced system prompt for better answers
        enhanced_system = (
            "You are Kivi, an expert document analysis assistant. "
            "Your role is to provide accurate, helpful answers based ONLY on the given context.\n\n"
            "GUIDELINES:\n"
            "1. Be thorough and detailed when information is available\n"
            "2. If information is partially available, explain what you found and what's missing\n"
            "3. If nothing relevant is found, say: 'I cannot find this information in the provided documents'\n"
            "4. Do not make up or assume information\n"
            "5. Quote relevant parts when helpful\n"
            "6. If the context contains lists or tables, present them clearly\n"
            "7. Be concise but comprehensive\n"
            "8. Use a professional, helpful tone\n\n"
            "Remember: Quality over quantity. Be helpful but accurate."
        )
        
        # Enhanced user prompt
        enhanced_user = f"""CONTEXT:
{user_prompt}

QUESTION: {question}

Please provide a comprehensive answer based ONLY on the context above.
If the context lacks information, clearly state what's missing.
If you find relevant information, explain it in detail."""

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": enhanced_system},
                {"role": "user", "content": enhanced_user},
            ],
            temperature=0.4,
            max_tokens=800,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def stream_response(text, placeholder):
    displayed = ""
    words = text.split()
    for i, word in enumerate(words):
        displayed += word + " "
        if i < len(words) - 1:
            placeholder.markdown(displayed + "▌")
        else:
            placeholder.markdown(displayed)
        time.sleep(0.02)


# =============================
# PROCESS FILES
# =============================
if uploaded_files:
    new_hash = compute_files_hash(uploaded_files)
    if st.session_state.files_hash != new_hash:
        with st.spinner("🔄 Processing documents..."):
            embeddings, chunks, meta = build_embeddings_from_files(uploaded_files)
            st.session_state.embeddings = embeddings
            st.session_state.chunks = chunks
            st.session_state.meta = meta
            st.session_state.files_hash = new_hash
        
        if embeddings is None:
            st.error("❌ No readable text found. Please check your documents.")
        else:
            st.success(f"✅ Loaded {len(uploaded_files)} file(s) • {len(chunks)} chunks ready!")


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

if st.session_state.suggested_question and not question:
    question = st.session_state.suggested_question
    st.session_state.suggested_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if not uploaded_files or st.session_state.embeddings is None:
        st.warning("📤 Please upload documents first.")
        st.stop()

    # Get query embedding
    with st.spinner("🔍 Searching documents..."):
        q_emb = embedder.encode([question], convert_to_numpy=True)[0]
        
        # Find similar chunks
        retrieved = find_similar_chunks(q_emb, st.session_state.embeddings, 
                                        st.session_state.chunks, st.session_state.meta, k=TOP_K)

    # Calculate confidence
    if retrieved:
        avg_confidence = sum([score for _, _, score in retrieved]) / len(retrieved)
        if avg_confidence > 0.5:
            confidence_emoji = "🟢 High"
        elif avg_confidence > 0.3:
            confidence_emoji = "🟡 Medium"
        else:
            confidence_emoji = "🟠 Low"
    else:
        avg_confidence = 0
        confidence_emoji = "⚫ None"

    if not retrieved:
        context = "No relevant information found in the documents."
    else:
        context_parts = []
        for chunk, fname, score in retrieved:
            context_parts.append(f"[From {fname}]\n{chunk}")
        context = "\n\n".join(context_parts)

    system_prompt = "You are a helpful document assistant. Answer based only on the context."
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    with st.chat_message("assistant"):
        with st.spinner("💭 Thinking..."):
            answer = groq_answer(system_prompt, user_prompt, question)

        response_placeholder = st.empty()
        stream_response(answer, response_placeholder)

        # Show confidence and retrieved context
        col1, col2 = st.columns([0.2, 0.8])
        with col1:
            st.caption(f"Confidence: {confidence_emoji}")
        
        with st.expander(f"🔍 View {len(retrieved)} retrieved chunks"):
            for i, (chunk, fname, score) in enumerate(retrieved, start=1):
                st.markdown(f"**Chunk {i} from {fname}** (relevance: {score:.2f})")
                st.info(chunk[:300] + "..." if len(chunk) > 300 else chunk)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()


# =============================
# SUGGESTED QUESTIONS
# =============================
if uploaded_files and st.session_state.embeddings is not None and len(st.session_state.messages) == 0:
    st.markdown("### 💡 Try asking:")
    suggestion_cols = st.columns(3)
    suggestions = [
        "What is the main topic?",
        "Summarize the key points",
        "What are the conclusions?",
        "List important findings",
        "Who is the target audience?",
        "What methodology is used?"
    ]
    
    for i, suggestion in enumerate(suggestions[:3]):
        with suggestion_cols[i]:
            if st.button(suggestion, use_container_width=True):
                st.session_state.suggested_question = suggestion
                st.rerun()
