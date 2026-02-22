import os
import hashlib
import numpy as np
import streamlit as st
import time
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import json
import hashlib

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
    st.error("Please configure API key in Settings")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.1-8b-instant"

# =============================
# USER AUTHENTICATION SYSTEM
# =============================
class UserAuth:
    def __init__(self):
        if 'users' not in st.session_state:
            # Simple in-memory user storage (for demo)
            # In production, use a real database
            st.session_state.users = {
                "demo@kivi.ai": {
                    "password": hashlib.sha256("demo123".encode()).hexdigest(),
                    "name": "Demo User",
                    "created_at": datetime.now().isoformat()
                }
            }
        if 'current_user' not in st.session_state:
            st.session_state.current_user = None
        if 'auth_mode' not in st.session_state:
            st.session_state.auth_mode = "login"
    
    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register(self, email, password, name):
        if email in st.session_state.users:
            return False, "Email already exists"
        st.session_state.users[email] = {
            "password": self.hash_password(password),
            "name": name,
            "created_at": datetime.now().isoformat(),
            "saved_chats": [],
            "documents": []
        }
        return True, "Registration successful"
    
    def login(self, email, password):
        if email not in st.session_state.users:
            return False, "User not found"
        if st.session_state.users[email]["password"] != self.hash_password(password):
            return False, "Incorrect password"
        st.session_state.current_user = {
            "email": email,
            "name": st.session_state.users[email]["name"]
        }
        return True, "Login successful"
    
    def logout(self):
        st.session_state.current_user = None
        st.session_state.messages = []
        st.session_state.embeddings = None
        st.session_state.chunks = []
    
    def get_user_data(self):
        if st.session_state.current_user:
            email = st.session_state.current_user["email"]
            return st.session_state.users.get(email, {})
        return {}

auth = UserAuth()

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
    st.session_state.current_chat_name = "Untitled Chat"
if "document_collections" not in st.session_state:
    st.session_state.document_collections = ["All Documents"]

# =============================
# WORLD-CLASS CSS - KIWI INSPIRED
# =============================
st.markdown("""
<style>
/* Import fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Reset and base */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* Main container - clean and spacious */
.main .block-container {
    padding: 2rem 2.5rem;
    max-width: 1400px;
    margin: 0 auto;
}

/* Hide Streamlit branding */
#MainMenu, footer, header {
    visibility: hidden;
    display: none;
}

/* ===================================== */
/* KIWI-INSPIRED LOGO - WORLD CLASS */
/* ===================================== */
.kivi-logo-container {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 2rem;
}

.kivi-logo-mark {
    width: 48px;
    height: 48px;
    position: relative;
    filter: drop-shadow(0 4px 8px rgba(34, 197, 94, 0.15));
}

/* Abstract kiwi shape - minimalist geometric interpretation */
.kivi-shape-outer {
    position: absolute;
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%);
    border-radius: 40% 60% 60% 40% / 50% 50% 50% 50%;
    transform: rotate(-5deg);
    animation: subtleFloat 6s ease-in-out infinite;
}

.kivi-shape-inner {
    position: absolute;
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #86EFAC 0%, #4ADE80 100%);
    border-radius: 45% 55% 55% 45% / 45% 50% 50% 55%;
    top: 8px;
    left: 8px;
    transform: rotate(5deg);
    opacity: 0.9;
}

.kivi-dot {
    position: absolute;
    width: 6px;
    height: 6px;
    background: #14532D;
    border-radius: 50%;
    top: 20px;
    left: 20px;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes subtleFloat {
    0%, 100% { transform: rotate(-5deg) translateY(0); }
    50% { transform: rotate(-5deg) translateY(-3px); }
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.6; }
    50% { transform: scale(1.2); opacity: 1; }
}

/* Text beside logo */
.kivi-logo-text {
    display: flex;
    flex-direction: column;
}

.kivi-logo-main {
    font-size: 32px;
    font-weight: 600;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #111827, #374151);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}

.kivi-logo-tagline {
    font-size: 12px;
    font-weight: 400;
    color: #6B7280;
    letter-spacing: 0.02em;
    margin-top: 2px;
}

/* ===================================== */
/* AUTHENTICATION MODAL */
/* ===================================== */
.auth-container {
    max-width: 400px;
    margin: 4rem auto;
    padding: 2rem;
    background: white;
    border-radius: 24px;
    box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.1);
    border: 1px solid #E5E7EB;
}

.auth-header {
    text-align: center;
    margin-bottom: 2rem;
}

.auth-header h2 {
    font-size: 1.8rem;
    font-weight: 600;
    color: #111827;
    margin-bottom: 0.5rem;
}

.auth-header p {
    color: #6B7280;
    font-size: 0.95rem;
}

.auth-input {
    width: 100%;
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    font-size: 0.95rem;
    transition: all 0.2s;
}

.auth-input:focus {
    outline: none;
    border-color: #22C55E;
    box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.1);
}

.auth-button {
    width: 100%;
    padding: 0.75rem;
    background: linear-gradient(135deg, #22C55E, #16A34A);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 500;
    font-size: 0.95rem;
    cursor: pointer;
    transition: all 0.2s;
    margin-bottom: 1rem;
}

.auth-button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(34, 197, 94, 0.2);
}

.auth-switch {
    text-align: center;
    color: #6B7280;
    font-size: 0.9rem;
}

.auth-switch span {
    color: #22C55E;
    font-weight: 500;
    cursor: pointer;
}

/* ===================================== */
/* SIDEBAR STYLING - PROFESSIONAL */
/* ===================================== */
section[data-testid="stSidebar"] {
    background: #F9FAFB;
    border-right: 1px solid #E5E7EB;
    box-shadow: none;
}

section[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.25rem;
}

/* User profile in sidebar */
.user-profile {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 1rem 0;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid #E5E7EB;
}

.user-avatar {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #22C55E, #16A34A);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 600;
    font-size: 1.1rem;
}

.user-info {
    flex: 1;
}

.user-name {
    font-weight: 600;
    color: #111827;
    font-size: 0.95rem;
}

.user-email {
    font-size: 0.8rem;
    color: #6B7280;
}

/* Sidebar sections */
.sidebar-section {
    margin-bottom: 2rem;
}

.sidebar-section-title {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.02em;
    color: #9CA3AF;
    margin-bottom: 1rem;
}

/* Document collections */
.collection-item {
    padding: 0.5rem 0.75rem;
    border-radius: 8px;
    font-size: 0.9rem;
    color: #374151;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 8px;
}

.collection-item:hover {
    background: #F3F4F6;
}

.collection-item.active {
    background: #E6F7E6;
    color: #16A34A;
    font-weight: 500;
}

/* Chat list */
.chat-item {
    padding: 0.75rem;
    border-radius: 10px;
    margin-bottom: 0.5rem;
    cursor: pointer;
    transition: all 0.2s;
    border: 1px solid transparent;
}

.chat-item:hover {
    background: #F3F4F6;
    border-color: #E5E7EB;
}

.chat-item-title {
    font-weight: 500;
    color: #111827;
    font-size: 0.9rem;
    margin-bottom: 0.25rem;
}

.chat-item-meta {
    font-size: 0.7rem;
    color: #9CA3AF;
    display: flex;
    gap: 8px;
}

/* ===================================== */
/* MAIN CONTENT AREA */
/* ===================================== */
.main-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #E5E7EB;
}

/* Upload area - clean and minimal */
.upload-area {
    background: #F9FAFB;
    border: 2px dashed #E5E7EB;
    border-radius: 20px;
    padding: 3rem;
    text-align: center;
    margin-bottom: 2rem;
    transition: all 0.2s;
}

.upload-area:hover {
    border-color: #22C55E;
    background: #F3F4F6;
}

.upload-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
    opacity: 0.6;
}

.upload-title {
    font-size: 1.2rem;
    font-weight: 500;
    color: #111827;
    margin-bottom: 0.5rem;
}

.upload-subtitle {
    font-size: 0.9rem;
    color: #6B7280;
}

/* Chat messages */
.stChatMessage {
    margin-bottom: 1rem !important;
}

.stChatMessage > div {
    border-radius: 18px !important;
    padding: 1rem 1.25rem !important;
    max-width: 80%;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
}

.stChatMessage.user > div {
    background: #22C55E !important;
    color: white !important;
    margin-left: auto !important;
    border-bottom-right-radius: 4px !important;
}

.stChatMessage.assistant > div {
    background: #F3F4F6 !important;
    color: #111827 !important;
    border: 1px solid #E5E7EB !important;
    border-bottom-left-radius: 4px !important;
}

/* Chat input */
.stChatInputContainer {
    border: 1px solid #E5E7EB !important;
    border-radius: 100px !important;
    padding: 0.25rem 0.25rem 0.25rem 1.25rem !important;
    background: white !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
}

.stChatInputContainer input {
    font-size: 0.95rem !important;
}

.stChatInputContainer button {
    background: #22C55E !important;
    color: white !important;
    border-radius: 100px !important;
    padding: 0.5rem 1.5rem !important;
}

/* Buttons */
.stButton > button {
    border-radius: 10px !important;
    background: white !important;
    border: 1px solid #E5E7EB !important;
    color: #374151 !important;
    font-size: 0.9rem !important;
    padding: 0.4rem 1rem !important;
    transition: all 0.2s;
}

.stButton > button:hover {
    border-color: #22C55E !important;
    background: #F9FAFB !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: #F9FAFB !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 10px !important;
    font-size: 0.9rem !important;
    color: #374151 !important;
}

/* Metrics */
div[data-testid="stMetricValue"] {
    font-size: 1.3rem !important;
    font-weight: 600 !important;
    color: #111827 !important;
}

div[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    color: #6B7280 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.02em !important;
}

/* Loading states */
.stSpinner {
    border-color: #22C55E !important;
}

/* Success messages */
.stAlert {
    background: #F0FDF4 !important;
    border-left: 4px solid #22C55E !important;
    color: #166534 !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
</style>
""", unsafe_allow_html=True)

# =============================
# AUTHENTICATION UI
# =============================
def show_auth_ui():
    if st.session_state.current_user:
        return
    
    st.markdown("""
    <div class="auth-container">
        <div class="auth-header">
            <div style="display: flex; justify-content: center; margin-bottom: 1rem;">
                <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #22C55E, #16A34A); border-radius: 20px; display: flex; align-items: center; justify-content: center;">
                    <span style="color: white; font-size: 30px;">🥝</span>
                </div>
            </div>
            <h2>Welcome to Kivi</h2>
            <p>Your intelligent document assistant</p>
        </div>
    """, unsafe_allow_html=True)
    
    mode = st.session_state.get('auth_mode', 'login')
    
    if mode == "login":
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="hello@example.com")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Sign In", use_container_width=True)
            
            if submitted:
                success, message = auth.login(email, password)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        
        st.markdown(f"""
        <div class="auth-switch">
            Don't have an account? <span onclick="alert('Switching to signup')">Create account</span>
        </div>
        <script>
        document.querySelector('.auth-switch span').onclick = function() {{
            window.parent.postMessage({{type: 'streamlit:setAuthMode', mode: 'signup'}}, '*');
        }}
        </script>
        """, unsafe_allow_html=True)
        
        # Demo credentials
        st.markdown("""
        <div style="margin-top: 2rem; padding: 1rem; background: #F9FAFB; border-radius: 12px;">
            <p style="color: #6B7280; font-size: 0.8rem; margin-bottom: 0.5rem;">Demo credentials:</p>
            <p style="color: #111827; font-size: 0.9rem;">Email: demo@kivi.ai</p>
            <p style="color: #111827; font-size: 0.9rem;">Password: demo123</p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        with st.form("signup_form"):
            name = st.text_input("Full Name", placeholder="John Doe")
            email = st.text_input("Email", placeholder="hello@example.com")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            confirm = st.text_input("Confirm Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Create Account", use_container_width=True)
            
            if submitted:
                if password != confirm:
                    st.error("Passwords don't match")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    success, message = auth.register(email, password, name)
                    if success:
                        st.success(message)
                        st.session_state.auth_mode = "login"
                        st.rerun()
                    else:
                        st.error(message)
        
        st.markdown(f"""
        <div class="auth-switch">
            Already have an account? <span onclick="alert('Switching to login')">Sign in</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# SIDEBAR - Professional with features
# =============================
def show_sidebar():
    with st.sidebar:
        # User profile
        if st.session_state.current_user:
            user = st.session_state.current_user
            initials = user['name'][0].upper() if user['name'] else 'U'
            st.markdown(f"""
            <div class="user-profile">
                <div class="user-avatar">{initials}</div>
                <div class="user-info">
                    <div class="user-name">{user['name']}</div>
                    <div class="user-email">{user['email']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Document Collections
        st.markdown('<div class="sidebar-section-title">📚 COLLECTIONS</div>', unsafe_allow_html=True)
        
        for collection in ["All Documents", "Work", "Personal", "Shared"]:
            active = "active" if collection == "All Documents" else ""
            st.markdown(f"""
            <div class="collection-item {active}">
                <span>📄</span>
                <span>{collection}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<br>', unsafe_allow_html=True)
        
        # Chat Management
        st.markdown('<div class="sidebar-section-title">💬 CHATS</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ New Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.embeddings = None
                st.session_state.chunks = []
                st.rerun()
        
        with col2:
            if st.button("💾 Save", use_container_width=True):
                if st.session_state.messages:
                    # Save to user's data
                    user_data = auth.get_user_data()
                    if user_data:
                        if 'saved_chats' not in user_data:
                            user_data['saved_chats'] = []
                        user_data['saved_chats'].append({
                            "name": st.session_state.current_chat_name,
                            "messages": st.session_state.messages.copy(),
                            "date": datetime.now().isoformat()
                        })
                    st.success("Chat saved!")
        
        # Recent chats
        user_data = auth.get_user_data()
        if user_data and user_data.get('saved_chats'):
            for chat in user_data['saved_chats'][-3:]:
                st.markdown(f"""
                <div class="chat-item">
                    <div class="chat-item-title">{chat['name']}</div>
                    <div class="chat-item-meta">
                        <span>{len(chat['messages'])} messages</span>
                        <span>•</span>
                        <span>{datetime.fromisoformat(chat['date']).strftime('%b %d')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # Settings
        with st.expander("⚙️ Settings", expanded=False):
            TOP_K = st.slider("Chunks to retrieve", 3, 10, 5)
            show_sources = st.checkbox("Show sources", value=True)
            theme = st.selectbox("Theme", ["Light", "Dark"], index=0)
        
        # Stats
        st.markdown("### 📊 Usage")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", len(set([m['file'] for m in st.session_state.meta])) if st.session_state.meta else 0)
        with col2:
            st.metric("Chunks", len(st.session_state.chunks))
        
        # Sign out
        if st.session_state.current_user:
            if st.button("🚪 Sign Out", use_container_width=True):
                auth.logout()
                st.rerun()

# =============================
# KIWI LOGO - WORLD CLASS
# =============================
def show_logo():
    st.markdown("""
    <div class="kivi-logo-container">
        <div class="kivi-logo-mark">
            <div class="kivi-shape-outer"></div>
            <div class="kivi-shape-inner"></div>
            <div class="kivi-dot"></div>
        </div>
        <div class="kivi-logo-text">
            <span class="kivi-logo-main">Kivi</span>
            <span class="kivi-logo-tagline">intelligent document assistant</span>
        </div>
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
            all_meta.extend([{"file": f.name, "date": datetime.now().isoformat()}] * len(chunks))
    
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
# MAIN APP LOGIC
# =============================

# Check authentication
if not st.session_state.current_user:
    show_auth_ui()
    st.stop()

# Show sidebar for authenticated users
show_sidebar()

# =============================
# MAIN CONTENT
# =============================

# Header with logo and actions
col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
with col1:
    show_logo()
with col2:
    if st.button("📤 Export", use_container_width=True):
        if st.session_state.messages:
            # Create export
            export_data = {
                "chat": st.session_state.messages,
                "date": datetime.now().isoformat(),
                "documents": list(set([m['file'] for m in st.session_state.meta])) if st.session_state.meta else []
            }
            st.download_button(
                "Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"kivi_export_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
with col3:
    if st.button("🗑 Clear", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Upload area
st.markdown("""
<div class="upload-area">
    <div class="upload-icon">📄</div>
    <div class="upload-title">Drop your documents here</div>
    <div class="upload-subtitle">PDF • DOCX • TXT (Max 10MB per file)</div>
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
            
            # Save to user's documents
            user_data = auth.get_user_data()
            if user_data:
                if 'documents' not in user_data:
                    user_data['documents'] = []
                for f in uploaded_files:
                    user_data['documents'].append({
                        "name": f.name,
                        "date": datetime.now().isoformat(),
                        "chunks": len([m for m in meta if m['file'] == f.name])
                    })
        else:
            st.warning("No readable text found in documents")

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
                                st.session_state.chunks, st.session_state.meta, k=5)
    
    # Calculate confidence
    if retrieved:
        avg_conf = sum([s for _,_,s in retrieved]) / len(retrieved)
        confidence_level = "High" if avg_conf > 0.5 else "Medium" if avg_conf > 0.3 else "Low"
    else:
        avg_conf = 0
        confidence_level = "None"
    
    # Prepare context
    if not retrieved:
        context = "No relevant information found in the documents."
        sources_display = []
    else:
        context = "\n\n".join([f"[From {fname}]\n{chunk}" for chunk, fname, _ in retrieved])
        sources_display = retrieved
    
    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("💭 Thinking..."):
            answer = get_answer(context, question)
        
        placeholder = st.empty()
        stream_response(answer, placeholder)
        
        # Show metadata
        col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
        with col1:
            st.caption(f"🎯 Confidence: {confidence_level}")
        with col2:
            st.caption(f"📚 Sources: {len(retrieved)} chunks")
        with col3:
            if retrieved and avg_conf < 0.3:
                st.caption("⚠️ Low confidence - verify information")
        
        # Show sources
        if sources_display:
            with st.expander(f"📄 View {len(sources_display)} sources"):
                for i, (chunk, fname, score) in enumerate(sources_display, 1):
                    st.markdown(f"**{i}. {fname}** (relevance: {score:.2f})")
                    st.info(chunk[:300] + "..." if len(chunk) > 300 else chunk)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

# Welcome message if no messages
if len(st.session_state.messages) == 0 and st.session_state.embeddings is not None:
    st.markdown("""
    <div style="text-align: center; padding: 3rem; color: #6B7280;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🥝</div>
        <h3 style="color: #111827; margin-bottom: 0.5rem;">Ready to help!</h3>
        <p>Ask me anything about your documents.</p>
    </div>
    """, unsafe_allow_html=True)
