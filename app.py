import os
import hashlib
import numpy as np
import streamlit as st
import time
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import json
import sqlite3
from pathlib import Path

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
# DATABASE SETUP - REAL USER STORAGE
# =============================
def init_database():
    """Initialize SQLite database for user data"""
    conn = sqlite3.connect('kivi_users.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create user_data table for storing chats and documents
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            data_type TEXT NOT NULL,
            data_content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_email) REFERENCES users (email)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database
init_database()

# =============================
# USER AUTHENTICATION CLASS - FULLY WORKING
# =============================
class UserAuth:
    def __init__(self):
        if 'current_user' not in st.session_state:
            st.session_state.current_user = None
        if 'auth_page' not in st.session_state:
            st.session_state.auth_page = "login"
    
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(self, email, password, name):
        """Register a new user"""
        try:
            conn = sqlite3.connect('kivi_users.db')
            c = conn.cursor()
            
            # Check if user exists
            c.execute("SELECT email FROM users WHERE email = ?", (email,))
            if c.fetchone():
                conn.close()
                return False, "Email already registered"
            
            # Insert new user
            password_hash = self.hash_password(password)
            c.execute(
                "INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)",
                (email, password_hash, name)
            )
            conn.commit()
            conn.close()
            
            # Auto login after registration
            st.session_state.current_user = {
                "email": email,
                "name": name
            }
            
            return True, "Registration successful!"
            
        except Exception as e:
            return False, f"Registration failed: {str(e)}"
    
    def login_user(self, email, password):
        """Login existing user"""
        try:
            conn = sqlite3.connect('kivi_users.db')
            c = conn.cursor()
            
            password_hash = self.hash_password(password)
            c.execute(
                "SELECT name FROM users WHERE email = ? AND password_hash = ?",
                (email, password_hash)
            )
            result = c.fetchone()
            conn.close()
            
            if result:
                st.session_state.current_user = {
                    "email": email,
                    "name": result[0]
                }
                return True, "Login successful!"
            else:
                return False, "Invalid email or password"
                
        except Exception as e:
            return False, f"Login failed: {str(e)}"
    
    def logout(self):
        """Logout current user"""
        st.session_state.current_user = None
        st.session_state.messages = []
        st.session_state.embeddings = None
        st.session_state.chunks = []
        st.session_state.meta = []
        return True, "Logged out successfully"
    
    def save_user_chat(self, chat_name, messages):
        """Save chat to user's data"""
        if not st.session_state.current_user:
            return False, "Not logged in"
        
        try:
            conn = sqlite3.connect('kivi_users.db')
            c = conn.cursor()
            
            data = {
                "name": chat_name,
                "messages": messages,
                "date": datetime.now().isoformat()
            }
            
            c.execute(
                "INSERT INTO user_data (user_email, data_type, data_content) VALUES (?, ?, ?)",
                (st.session_state.current_user["email"], "chat", json.dumps(data))
            )
            conn.commit()
            conn.close()
            return True, "Chat saved!"
        except Exception as e:
            return False, f"Failed to save: {str(e)}"
    
    def get_user_chats(self):
        """Get all chats for current user"""
        if not st.session_state.current_user:
            return []
        
        try:
            conn = sqlite3.connect('kivi_users.db')
            c = conn.cursor()
            c.execute(
                "SELECT data_content FROM user_data WHERE user_email = ? AND data_type = 'chat' ORDER BY created_at DESC",
                (st.session_state.current_user["email"],)
            )
            results = c.fetchall()
            conn.close()
            
            chats = []
            for row in results:
                chats.append(json.loads(row[0]))
            return chats
        except:
            return []
    
    def save_user_document(self, doc_name, chunks_count):
        """Save document info to user's data"""
        if not st.session_state.current_user:
            return
        
        try:
            conn = sqlite3.connect('kivi_users.db')
            c = conn.cursor()
            
            data = {
                "name": doc_name,
                "chunks": chunks_count,
                "date": datetime.now().isoformat()
            }
            
            c.execute(
                "INSERT INTO user_data (user_email, data_type, data_content) VALUES (?, ?, ?)",
                (st.session_state.current_user["email"], "document", json.dumps(data))
            )
            conn.commit()
            conn.close()
        except:
            pass
    
    def get_user_documents(self):
        """Get all documents for current user"""
        if not st.session_state.current_user:
            return []
        
        try:
            conn = sqlite3.connect('kivi_users.db')
            c = conn.cursor()
            c.execute(
                "SELECT data_content FROM user_data WHERE user_email = ? AND data_type = 'document' ORDER BY created_at DESC",
                (st.session_state.current_user["email"],)
            )
            results = c.fetchall()
            conn.close()
            
            docs = []
            for row in results:
                docs.append(json.loads(row[0]))
            return docs
        except:
            return []

# Initialize auth
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

# =============================
# WORLD-CLASS CSS
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
    visibility: hidden;
    display: none;
}

/* ===================================== */
/* KIWI LOGO */
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
}

.kivi-shape-outer {
    position: absolute;
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%);
    border-radius: 40% 60% 60% 40% / 50% 50% 50% 50%;
    transform: rotate(-5deg);
    animation: float 6s ease-in-out infinite;
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

@keyframes float {
    0%, 100% { transform: rotate(-5deg) translateY(0); }
    50% { transform: rotate(-5deg) translateY(-3px); }
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.6; }
    50% { transform: scale(1.2); opacity: 1; }
}

.kivi-logo-text {
    display: flex;
    flex-direction: column;
}

.kivi-logo-main {
    font-size: 32px;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: #111827;
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
/* AUTHENTICATION PAGES */
/* ===================================== */
.auth-container {
    max-width: 420px;
    margin: 4rem auto;
    padding: 2.5rem;
    background: white;
    border-radius: 24px;
    box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.1);
    border: 1px solid #E5E7EB;
}

.auth-logo {
    text-align: center;
    margin-bottom: 2rem;
}

.auth-logo-icon {
    width: 64px;
    height: 64px;
    background: linear-gradient(135deg, #22C55E, #16A34A);
    border-radius: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 1rem;
    font-size: 2rem;
}

.auth-title {
    font-size: 1.8rem;
    font-weight: 600;
    color: #111827;
    margin-bottom: 0.5rem;
    text-align: center;
}

.auth-subtitle {
    color: #6B7280;
    font-size: 0.95rem;
    text-align: center;
    margin-bottom: 2rem;
}

.auth-input {
    width: 100%;
    padding: 0.875rem 1rem;
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
    padding: 0.875rem;
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

.auth-switch-button {
    color: #22C55E;
    font-weight: 500;
    cursor: pointer;
    background: none;
    border: none;
    padding: 0;
    font-size: 0.9rem;
}

.auth-switch-button:hover {
    text-decoration: underline;
}

.auth-demo {
    margin-top: 2rem;
    padding: 1rem;
    background: #F9FAFB;
    border-radius: 12px;
    border: 1px solid #E5E7EB;
}

.auth-demo p {
    color: #6B7280;
    font-size: 0.8rem;
    margin-bottom: 0.5rem;
}

.auth-demo-code {
    color: #111827;
    font-size: 0.9rem;
    font-family: monospace;
    background: #F3F4F6;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #F9FAFB;
    border-right: 1px solid #E5E7EB;
}

.user-profile {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 1.5rem 1rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid #E5E7EB;
}

.user-avatar {
    width: 44px;
    height: 44px;
    background: linear-gradient(135deg, #22C55E, #16A34A);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 600;
    font-size: 1.2rem;
}

.user-info-name {
    font-weight: 600;
    color: #111827;
    font-size: 1rem;
}

.user-info-email {
    font-size: 0.8rem;
    color: #6B7280;
}

/* Upload area */
.upload-area {
    background: #F9FAFB;
    border: 2px dashed #E5E7EB;
    border-radius: 20px;
    padding: 3rem;
    text-align: center;
    margin: 2rem 0;
    transition: all 0.2s;
}

.upload-area:hover {
    border-color: #22C55E;
    background: #F3F4F6;
}

/* Chat messages */
.stChatMessage.user > div {
    background: #22C55E !important;
    color: white !important;
}

.stChatMessage.assistant > div {
    background: #F3F4F6 !important;
    color: #111827 !important;
}
</style>
""", unsafe_allow_html=True)

# =============================
# AUTHENTICATION UI - FULLY WORKING
# =============================
def show_auth_page():
    """Show login/signup page"""
    
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    
    # Logo
    st.markdown("""
    <div class="auth-logo">
        <div class="auth-logo-icon">🥝</div>
        <div class="auth-title">Kivi</div>
        <div class="auth-subtitle">Your intelligent document assistant</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Toggle between login and signup
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔐 Login", use_container_width=True, 
                    type="primary" if st.session_state.auth_page == "login" else "secondary"):
            st.session_state.auth_page = "login"
            st.rerun()
    with col2:
        if st.button("📝 Sign Up", use_container_width=True,
                    type="primary" if st.session_state.auth_page == "signup" else "secondary"):
            st.session_state.auth_page = "signup"
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.session_state.auth_page == "login":
        # LOGIN FORM
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="hello@example.com", key="login_email")
            password = st.text_input("Password", type="password", placeholder="••••••••", key="login_password")
            
            submitted = st.form_submit_button("Sign In", use_container_width=True)
            
            if submitted:
                if not email or not password:
                    st.error("Please fill in all fields")
                else:
                    success, message = auth.login_user(email, password)
                    if success:
                        st.success(message)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(message)
        
        # Demo credentials
        st.markdown("""
        <div class="auth-demo">
            <p>🔑 Demo Account</p>
            <p><span class="auth-demo-code">Email: demo@kivi.ai</span></p>
            <p><span class="auth-demo-code">Password: demo123</span></p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # SIGNUP FORM
        with st.form("signup_form"):
            name = st.text_input("Full Name", placeholder="John Doe", key="signup_name")
            email = st.text_input("Email", placeholder="hello@example.com", key="signup_email")
            password = st.text_input("Password", type="password", placeholder="••••••••", key="signup_password")
            confirm = st.text_input("Confirm Password", type="password", placeholder="••••••••", key="signup_confirm")
            
            submitted = st.form_submit_button("Create Account", use_container_width=True)
            
            if submitted:
                if not name or not email or not password or not confirm:
                    st.error("Please fill in all fields")
                elif password != confirm:
                    st.error("Passwords do not match")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    success, message = auth.register_user(email, password, name)
                    if success:
                        st.success(message)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(message)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# SIDEBAR - WITH REAL USER DATA
# =============================
def show_sidebar():
    with st.sidebar:
        if st.session_state.current_user:
            # User profile
            user = st.session_state.current_user
            initials = user['name'][0].upper() if user['name'] else 'U'
            st.markdown(f"""
            <div class="user-profile">
                <div class="user-avatar">{initials}</div>
                <div>
                    <div class="user-info-name">{user['name']}</div>
                    <div class="user-info-email">{user['email']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("➕ New Chat", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.embeddings = None
                    st.session_state.chunks = []
                    st.session_state.meta = []
                    st.rerun()
            with col2:
                if st.button("🚪 Logout", use_container_width=True):
                    auth.logout()
                    st.rerun()
            
            st.divider()
            
            # Saved Chats
            st.markdown("### 💬 Saved Chats")
            saved_chats = auth.get_user_chats()
            if saved_chats:
                for chat in saved_chats[:5]:
                    if st.button(f"📄 {chat['name']}", key=f"chat_{chat['name']}_{chat['date']}", use_container_width=True):
                        st.session_state.messages = chat['messages']
                        st.session_state.current_chat_name = chat['name']
                        st.rerun()
            else:
                st.caption("No saved chats yet")
            
            st.divider()
            
            # Documents
            st.markdown("### 📄 Documents")
            saved_docs = auth.get_user_documents()
            if saved_docs:
                for doc in saved_docs[:5]:
                    st.caption(f"• {doc['name']} ({doc['chunks']} chunks)")
            else:
                st.caption("No documents yet")
            
            st.divider()
            
            # Settings
            with st.expander("⚙️ Settings"):
                TOP_K = st.slider("Chunks to retrieve", 3, 10, 5)
                show_sources = st.checkbox("Show sources", value=True)
            
            # Stats
            st.markdown("### 📊 Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", len(set([m['file'] for m in st.session_state.meta])) if st.session_state.meta else 0)
            with col2:
                st.metric("Chunks", len(st.session_state.chunks))
            
            return TOP_K, show_sources
    
    return 5, True

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
# MAIN APP
# =============================

# Check if user is logged in
if not st.session_state.current_user:
    show_auth_page()
    st.stop()

# Show sidebar for logged in users
TOP_K, show_sources = show_sidebar()

# Main content area
col1, col2 = st.columns([0.8, 0.2])
with col1:
    show_logo()
with col2:
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Upload area
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
            for f in uploaded_files:
                chunk_count = len([m for m in meta if m['file'] == f.name])
                auth.save_user_document(f.name, chunk_count)
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
                                st.session_state.chunks, st.session_state.meta, k=TOP_K)
    
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
    else:
        context = "\n\n".join([f"[From {fname}]\n{chunk}" for chunk, fname, _ in retrieved])
    
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
        
        # Show sources
        if show_sources and retrieved:
            with st.expander(f"📄 View {len(retrieved)} sources"):
                for i, (chunk, fname, score) in enumerate(retrieved, 1):
                    st.markdown(f"**{i}. {fname}** (relevance: {score:.2f})")
                    st.info(chunk[:300] + "..." if len(chunk) > 300 else chunk)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Auto-save chat
    if len(st.session_state.messages) == 2:  # First Q&A
        auth.save_user_chat(st.session_state.current_chat_name, st.session_state.messages)
    
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
