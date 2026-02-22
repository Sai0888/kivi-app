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
# DATABASE SETUP
# =============================
def init_database():
    conn = sqlite3.connect('kivi_users.db')
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
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

init_database()

# =============================
# USER AUTHENTICATION CLASS
# =============================
class UserAuth:
    def __init__(self):
        if 'current_user' not in st.session_state:
            st.session_state.current_user = None
        if 'auth_page' not in st.session_state:
            st.session_state.auth_page = "login"
        if 'sidebar_collapsed' not in st.session_state:
            st.session_state.sidebar_collapsed = False
    
    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(self, email, password, name):
        try:
            conn = sqlite3.connect('kivi_users.db')
            c = conn.cursor()
            
            c.execute("SELECT email FROM users WHERE email = ?", (email,))
            if c.fetchone():
                conn.close()
                return False, "Email already registered"
            
            password_hash = self.hash_password(password)
            c.execute(
                "INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)",
                (email, password_hash, name)
            )
            conn.commit()
            conn.close()
            
            st.session_state.current_user = {
                "email": email,
                "name": name
            }
            
            return True, "Welcome to Kivi!"
            
        except Exception as e:
            return False, f"Registration failed: {str(e)}"
    
    def login_user(self, email, password):
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
                return True, "Welcome back!"
            else:
                return False, "Invalid email or password"
                
        except Exception as e:
            return False, f"Login failed: {str(e)}"
    
    def logout(self):
        st.session_state.current_user = None
        st.session_state.messages = []
        st.session_state.embeddings = None
        st.session_state.chunks = []
        st.session_state.meta = []
        return True, "Logged out successfully"
    
    def save_user_chat(self, chat_name, messages):
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
    st.session_state.current_chat_name = "New Chat"

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
/* KIWI LOGO - CLEAN */
/* ===================================== */
.kivi-logo-container {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 0 0 2rem 0;
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
/* AUTHENTICATION PAGE - CLEAN */
/* ===================================== */
.auth-wrapper {
    display: flex;
    min-height: 100vh;
    background: linear-gradient(135deg, #F9FAFB 0%, #F3F4F6 100%);
}

.auth-left {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem;
}

.auth-card {
    width: 100%;
    max-width: 440px;
    padding: 2.5rem;
}

.auth-header {
    margin-bottom: 2rem;
}

.auth-header h1 {
    font-size: 2.5rem;
    font-weight: 600;
    color: #111827;
    margin-bottom: 0.5rem;
}

.auth-header p {
    color: #6B7280;
    font-size: 1rem;
}

.auth-tabs {
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
    border-bottom: 2px solid #E5E7EB;
    padding-bottom: 0.5rem;
}

.auth-tab {
    background: none;
    border: none;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    font-weight: 500;
    color: #6B7280;
    cursor: pointer;
    transition: all 0.2s;
    border-radius: 8px;
}

.auth-tab:hover {
    color: #059669;
    background: #F3F4F6;
}

.auth-tab.active {
    color: #059669;
    border-bottom: 2px solid #059669;
}

.auth-input-group {
    margin-bottom: 1.5rem;
}

.auth-input-group label {
    display: block;
    font-size: 0.9rem;
    font-weight: 500;
    color: #374151;
    margin-bottom: 0.5rem;
}

.auth-input {
    width: 100%;
    padding: 0.875rem 1rem;
    border: 2px solid #E5E7EB;
    border-radius: 12px;
    font-size: 0.95rem;
    transition: all 0.2s;
    background: white;
}

.auth-input:focus {
    outline: none;
    border-color: #059669;
    box-shadow: 0 0 0 3px rgba(5, 150, 105, 0.1);
}

.auth-button {
    width: 100%;
    padding: 0.875rem;
    background: #059669;
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.2s;
    margin: 1rem 0;
}

.auth-button:hover {
    background: #047857;
    transform: translateY(-1px);
}

.auth-divider {
    text-align: center;
    margin: 1.5rem 0;
    color: #9CA3AF;
    font-size: 0.9rem;
}

.auth-demo-card {
    background: #F9FAFB;
    border: 2px solid #E5E7EB;
    border-radius: 12px;
    padding: 1rem;
    margin-top: 1rem;
}

.auth-demo-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #374151;
    margin-bottom: 0.5rem;
}

.auth-demo-item {
    background: white;
    padding: 0.5rem 0.75rem;
    border-radius: 8px;
    font-size: 0.9rem;
    color: #059669;
    font-family: monospace;
    border: 1px solid #E5E7EB;
    margin: 0.25rem 0;
}

.auth-right {
    flex: 1;
    background: linear-gradient(135deg, #059669, #10B981);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem;
    color: white;
}

.auth-quote {
    max-width: 400px;
}

.auth-quote h2 {
    font-size: 2rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

.auth-quote p {
    font-size: 1.1rem;
    opacity: 0.9;
    line-height: 1.6;
}

/* ===================================== */
/* SIDEBAR - PERMANENT LIKE CHATGPT */
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

/* User profile - Fixed at top */
.user-profile {
    padding: 0 0 1.5rem 0;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid #E5E7EB;
    display: flex;
    align-items: center;
    gap: 12px;
}

.user-avatar {
    width: 44px;
    height: 44px;
    background: #059669;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 600;
    font-size: 1.2rem;
}

.user-info {
    flex: 1;
}

.user-name {
    font-weight: 600;
    color: #111827;
    font-size: 0.95rem;
    margin-bottom: 0.25rem;
}

.user-email {
    font-size: 0.8rem;
    color: #6B7280;
}

/* Sidebar sections */
.sidebar-section {
    margin-bottom: 2rem;
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
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 0.75rem;
    width: 100%;
    font-size: 0.9rem;
    font-weight: 500;
    color: #111827;
    cursor: pointer;
    transition: all 0.2s;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.new-chat-btn:hover {
    background: #F3F4F6;
    border-color: #059669;
}

/* Chat list - Scrollable */
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

/* Document list */
.doc-list {
    max-height: 200px;
    overflow-y: auto;
}

.doc-item {
    padding: 0.5rem;
    font-size: 0.85rem;
    color: #374151;
    border-bottom: 1px solid #E5E7EB;
}

/* Logout button */
.logout-btn {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 0.75rem;
    width: 100%;
    font-size: 0.9rem;
    color: #DC2626;
    cursor: pointer;
    transition: all 0.2s;
    margin-top: 1rem;
}

.logout-btn:hover {
    background: #FEF2F2;
    border-color: #DC2626;
}

/* Stats */
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
/* MAIN CONTENT AREA */
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
</style>
""", unsafe_allow_html=True)

# =============================
# AUTHENTICATION PAGE - FIXED
# =============================
def show_auth_page():
    # Remove any existing white boxes by using columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="auth-card">
            <div class="auth-header">
                <h1>Welcome to Kivi</h1>
                <p>Your intelligent document assistant</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Tabs
        tab1, tab2 = st.columns(2)
        with tab1:
            if st.button("🔐 Sign In", use_container_width=True, 
                        type="primary" if st.session_state.auth_page == "login" else "secondary"):
                st.session_state.auth_page = "login"
                st.rerun()
        with tab2:
            if st.button("📝 Create Account", use_container_width=True,
                        type="primary" if st.session_state.auth_page == "signup" else "secondary"):
                st.session_state.auth_page = "signup"
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.session_state.auth_page == "login":
            with st.form("login_form"):
                st.markdown('<div class="auth-input-group"><label>Email</label>', unsafe_allow_html=True)
                email = st.text_input("", placeholder="hello@example.com", key="login_email", label_visibility="collapsed")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="auth-input-group"><label>Password</label>', unsafe_allow_html=True)
                password = st.text_input("", type="password", placeholder="••••••••", key="login_password", label_visibility="collapsed")
                st.markdown('</div>', unsafe_allow_html=True)
                
                submitted = st.form_submit_button("Sign In", use_container_width=True)
                
                if submitted:
                    if email and password:
                        success, message = auth.login_user(email, password)
                        if success:
                            st.success(message)
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(message)
            
            st.markdown('<div class="auth-divider">or</div>', unsafe_allow_html=True)
            
            # Demo account
            st.markdown("""
            <div class="auth-demo-card">
                <div class="auth-demo-title">🔑 Demo Account</div>
                <div class="auth-demo-item">demo@kivi.ai</div>
                <div class="auth-demo-item">demo123</div>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            with st.form("signup_form"):
                st.markdown('<div class="auth-input-group"><label>Full Name</label>', unsafe_allow_html=True)
                name = st.text_input("", placeholder="John Doe", key="signup_name", label_visibility="collapsed")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="auth-input-group"><label>Email</label>', unsafe_allow_html=True)
                email = st.text_input("", placeholder="hello@example.com", key="signup_email", label_visibility="collapsed")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="auth-input-group"><label>Password</label>', unsafe_allow_html=True)
                password = st.text_input("", type="password", placeholder="••••••••", key="signup_password", label_visibility="collapsed")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="auth-input-group"><label>Confirm Password</label>', unsafe_allow_html=True)
                confirm = st.text_input("", type="password", placeholder="••••••••", key="signup_confirm", label_visibility="collapsed")
                st.markdown('</div>', unsafe_allow_html=True)
                
                submitted = st.form_submit_button("Create Account", use_container_width=True)
                
                if submitted:
                    if name and email and password and confirm:
                        if password != confirm:
                            st.error("Passwords don't match")
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
    
    with col2:
        st.markdown("""
        <div class="auth-quote">
            <h2>Chat with your documents</h2>
            <p>Upload PDFs, DOCX, or TXT files and ask questions. Kivi finds answers instantly.</p>
            <div style="margin-top: 2rem; opacity: 0.8;">🥝</div>
        </div>
        """, unsafe_allow_html=True)

# =============================
# SIDEBAR - PERMANENT LIKE CHATGPT
# =============================
def show_sidebar():
    with st.sidebar:
        # User Profile - Always visible
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
        
        # New Chat Button
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.embeddings = None
            st.session_state.chunks = []
            st.session_state.meta = []
            st.session_state.current_chat_name = "New Chat"
            st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Saved Chats Section - Always visible
        st.markdown('<div class="sidebar-title">💬 RECENT CHATS</div>', unsafe_allow_html=True)
        
        saved_chats = auth.get_user_chats()
        if saved_chats:
            st.markdown('<div class="chat-list">', unsafe_allow_html=True)
            for chat in saved_chats[:10]:
                # Check if this chat is active
                is_active = False
                if st.session_state.messages and chat['messages'] == st.session_state.messages:
                    is_active = True
                
                chat_date = datetime.fromisoformat(chat['date']).strftime('%b %d')
                msg_count = len(chat['messages'])
                
                # Create a button for each chat
                if st.button(f"📄 {chat['name']}", key=f"chat_{chat['name']}_{chat['date']}", use_container_width=True):
                    st.session_state.messages = chat['messages']
                    st.session_state.current_chat_name = chat['name']
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.caption("No saved chats yet")
        
        st.divider()
        
        # Documents Section
        st.markdown('<div class="sidebar-title">📄 DOCUMENTS</div>', unsafe_allow_html=True)
        saved_docs = auth.get_user_documents()
        if saved_docs:
            st.markdown('<div class="doc-list">', unsafe_allow_html=True)
            for doc in saved_docs[:5]:
                st.markdown(f"""
                <div class="doc-item">
                    📄 {doc['name']} <span style="color: #9CA3AF; font-size: 0.7rem;">({doc['chunks']} chunks)</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.caption("No documents yet")
        
        st.divider()
        
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
        
        # Settings
        with st.expander("⚙️ Settings"):
            TOP_K = st.slider("Chunks to retrieve", 3, 10, 5)
            show_sources = st.checkbox("Show sources", value=True)
        
        # Logout button
        if st.button("🚪 Sign Out", use_container_width=True):
            auth.logout()
            st.rerun()
        
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
# MAIN APP
# =============================

# Show auth page if not logged in
if not st.session_state.current_user:
    show_auth_page()
    st.stop()

# Show sidebar and main content for logged in users
TOP_K, show_sources = show_sidebar()

# Main header
col1, col2 = st.columns([0.8, 0.2])
with col1:
    show_logo()
with col2:
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
            
            for f in uploaded_files:
                chunk_count = len([m for m in meta if m['file'] == f.name])
                auth.save_user_document(f.name, chunk_count)
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
        context = "No relevant information found."
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
            st.caption(f"🎯 {confidence}")
        with col2:
            st.caption(f"📚 {len(retrieved)} sources")
        
        # Show sources
        if show_sources and retrieved:
            with st.expander(f"📄 View sources"):
                for i, (chunk, fname, score) in enumerate(retrieved, 1):
                    st.markdown(f"**{i}. {fname}** (score: {score:.2f})")
                    st.info(chunk[:300] + "..." if len(chunk) > 300 else chunk)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Auto-save chat
    if len(st.session_state.messages) == 2:
        auth.save_user_chat(st.session_state.current_chat_name, st.session_state.messages)
    
    st.rerun()
