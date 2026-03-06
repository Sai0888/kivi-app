import os
import hashlib
import numpy as np
import streamlit as st
import time
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

try:
    from groq import Groq
except ImportError:
    st.error("Installing packages… Please refresh in 2 minutes.")
    st.stop()

from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import docx

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Kivi — Intelligent Document Assistant",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='8' fill='%23059669'/><text y='22' x='7' font-size='18' font-family='serif' fill='white' font-weight='700'>K</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("Please configure GROQ_API_KEY in Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.1-8b-instant"

# ─────────────────────────────────────────────
# SOUTH INDIAN LANGUAGES
# ─────────────────────────────────────────────
SOUTH_LANGUAGES = {
    "English": None,
    "తెలుగు (Telugu)": "Telugu",
    "தமிழ் (Tamil)": "Tamil",
    "ಕನ್ನಡ (Kannada)": "Kannada",
    "മലയാളം (Malayalam)": "Malayalam",
}

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    "messages": [], "embeddings": None, "chunks": [],
    "meta": [], "files_hash": None,
    "current_chat_name": "New Chat",
    "saved_chats": [], "selected_language": "English",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# SVG ICON LIBRARY  (inline, no external deps)
# ─────────────────────────────────────────────
def svg(path_d, size=16, color="currentColor", vb="0 0 24 24", stroke_w=1.75):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="{vb}" fill="none" stroke="{color}" '
        f'stroke-width="{stroke_w}" stroke-linecap="round" stroke-linejoin="round">'
        f'{path_d}</svg>'
    )

# Lucide-style paths
ICON = {
    # sidebar / nav
    "plus":      svg('<line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>'),
    "chat":      svg('<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>'),
    "trash":     svg('<polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4h6v2"/>'),
    "globe":     svg('<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>'),
    "settings":  svg('<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>'),
    "bar-chart": svg('<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>'),
    # header
    "save":      svg('<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/>'),
    "clear":     svg('<polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/>'),
    "new-file":  svg('<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="12" y1="18" x2="12" y2="12"/><line x1="9" y1="15" x2="15" y2="15"/>'),
    # upload
    "upload":    svg('<polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>'),
    # sources
    "file-text": svg('<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/>'),
    # confidence
    "target":    svg('<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>'),
    # language
    "translate": svg('<path d="M5 8l6 6"/><path d="M4 14l6-6 2-3"/><path d="M2 5h12"/><path d="M7 2h1"/><path d="M22 22l-5-10-5 10"/><path d="M14 18h6"/>'),
    # welcome chips
    "search":    svg('<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>'),
    "zap":       svg('<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>'),
    # check
    "check":     svg('<polyline points="20 6 9 17 4 12"/>'),
}

# ─────────────────────────────────────────────
# KIVI WORDMARK LOGO (SVG, no CSS blobs)
# ─────────────────────────────────────────────
KIVI_LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="120" height="36" viewBox="0 0 120 36" fill="none">
  <!-- Icon mark: rounded square + stylised K -->
  <rect width="36" height="36" rx="10" fill="#059669"/>
  <!-- K stem -->
  <line x1="11" y1="9" x2="11" y2="27" stroke="white" stroke-width="2.4" stroke-linecap="round"/>
  <!-- K upper arm -->
  <line x1="11" y1="18" x2="22" y2="9" stroke="white" stroke-width="2.4" stroke-linecap="round"/>
  <!-- K lower arm -->
  <line x1="11" y1="18" x2="22" y2="27" stroke="white" stroke-width="2.4" stroke-linecap="round"/>
  <!-- Wordmark -->
  <text x="44" y="25" font-family="Georgia, 'Times New Roman', serif"
        font-size="22" font-weight="700" letter-spacing="-0.5" fill="#065F46">Kivi</text>
</svg>
"""

KIVI_LOGO_SMALL_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="90" height="28" viewBox="0 0 90 28" fill="none">
  <rect width="28" height="28" rx="8" fill="#059669"/>
  <line x1="8" y1="7" x2="8" y2="21" stroke="white" stroke-width="2" stroke-linecap="round"/>
  <line x1="8" y1="14" x2="17" y2="7" stroke="white" stroke-width="2" stroke-linecap="round"/>
  <line x1="8" y1="14" x2="17" y2="21" stroke="white" stroke-width="2" stroke-linecap="round"/>
  <text x="34" y="20" font-family="Georgia, 'Times New Roman', serif"
        font-size="17" font-weight="700" letter-spacing="-0.3" fill="#065F46">Kivi</text>
</svg>
"""

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Noto+Sans+Telugu&family=Noto+Sans+Tamil&family=Noto+Sans+Kannada&family=Noto+Sans+Malayalam&display=swap');

:root {
    --g-900: #064E3B;
    --g-700: #065F46;
    --g-500: #059669;
    --g-400: #10B981;
    --g-100: #D1FAE5;
    --g-50:  #ECFDF5;
    --surface:  #FFFFFF;
    --surface2: #F8FAFC;
    --border:   #E2E8F0;
    --text-1:   #0F172A;
    --text-2:   #475569;
    --text-3:   #94A3B8;
    --shadow-sm: 0 1px 3px rgba(0,0,0,.07),0 1px 2px rgba(0,0,0,.05);
    --shadow-md: 0 4px 16px rgba(0,0,0,.07),0 2px 4px rgba(0,0,0,.05);
    --r-sm: 8px; --r-md: 12px; --r-lg: 18px; --r-full: 100px;
}

*, *::before, *::after { box-sizing: border-box; margin:0; padding:0; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--text-1); }
.main .block-container { padding: 2rem 2.5rem 5rem; max-width: 1180px; margin: 0 auto; }
#MainMenu, footer, header { display: none; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F0FDF4 0%, #FAFAFA 100%) !important;
    border-right: 1px solid var(--border) !important;
    width: 280px !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem !important; max-width: 100% !important;
}

.sb-brand {
    display: flex; align-items: center;
    padding-bottom: 1.25rem; margin-bottom: 1.25rem;
    border-bottom: 1px solid var(--border);
}

.sb-label {
    display: flex; align-items: center; gap: 6px;
    font-size: 0.68rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--text-3); margin: 1.2rem 0 0.55rem;
}
.sb-label svg { opacity: .7; }

.lang-pill {
    font-size: 0.73rem; font-weight: 500;
    color: var(--g-700); background: var(--g-50);
    border: 1px solid var(--g-100);
    border-radius: var(--r-sm);
    padding: 5px 10px; margin-top: 5px;
    display: flex; align-items: center; gap: 6px;
}

.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .6rem; margin-top:.4rem; }
.stat-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--r-md); padding: .75rem .5rem;
    text-align: center; box-shadow: var(--shadow-sm);
}
.stat-n { font-size: 1.5rem; font-weight: 700; color: var(--g-500); line-height:1; }
.stat-l { font-size: 0.68rem; color: var(--text-3); margin-top:2px; }

/* ── HEADER ── */
.page-header {
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 1.25rem; margin-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}
.logo-sub { font-size: .75rem; color: var(--text-3); margin-top: 2px; letter-spacing:.02em; }

.lang-tag {
    display: inline-flex; align-items: center; gap: 5px;
    background: var(--g-50); border: 1px solid var(--g-100);
    color: var(--g-700); font-size: .72rem; font-weight: 600;
    padding: 2px 9px; border-radius: var(--r-full);
    margin-left: 10px; vertical-align: middle;
}

/* ── UPLOAD ZONE ── */
.upload-zone {
    background: linear-gradient(135deg, var(--g-50) 0%, #F8FAFC 100%);
    border: 2px dashed #A7F3D0;
    border-radius: var(--r-lg);
    padding: 2rem 2.5rem; text-align: center;
    margin: 1.25rem 0 .5rem;
    transition: border-color .2s, background .2s;
}
.upload-zone:hover { border-color: var(--g-500); }
.upload-icon-wrap {
    width: 52px; height: 52px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; box-shadow: var(--shadow-sm);
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 1rem;
}
.upload-title { font-size: .98rem; font-weight: 600; color: var(--text-1); margin-bottom:.4rem; }
.upload-sub { font-size: .78rem; color: var(--text-2); display: flex; justify-content: center; gap:8px; }
.ftag {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 5px; padding: 1px 8px;
    font-size: .7rem; font-weight: 600; color: var(--text-2);
}

/* ── CHAT ── */
.stChatMessage { margin-bottom: .85rem !important; animation: fadeUp .28s ease forwards; }

@keyframes fadeUp { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }

[data-testid="stChatMessage"] > div {
    border-radius: 18px !important;
    padding: .9rem 1.15rem !important;
    max-width: 78%;
}

/* ── ANSWER META ── */
.meta-row { display:flex; align-items:center; flex-wrap:wrap; gap:5px; margin-top:10px; }

.badge {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: .72rem; font-weight: 600;
    padding: 3px 10px; border-radius: var(--r-full);
}
.badge-high     { background: #D1FAE5; color: #065F46; }
.badge-medium   { background: #FEF3C7; color: #92400E; }
.badge-low      { background: #FEE2E2; color: #991B1B; }
.badge-lang     { background: #EEF2FF; color: #3730A3; border: 1px solid #C7D2FE; }

/* ── SOURCE CARDS ── */
.src-card {
    background: var(--surface2); border: 1px solid var(--border);
    border-left: 3px solid var(--g-400);
    border-radius: var(--r-md);
    padding: .8rem 1rem; margin-bottom: .6rem;
}
.src-head {
    display: flex; align-items: center; gap: 6px;
    font-size: .8rem; font-weight: 600; color: var(--g-700);
    margin-bottom: .35rem;
}
.src-score {
    margin-left: auto; font-size: .68rem; font-weight: 500;
    background: var(--g-100); color: var(--g-700);
    padding: 1px 7px; border-radius: 99px;
}
.src-text { font-size: .79rem; color: var(--text-2); line-height: 1.55; }

/* ── CHAT INPUT ── */
.stChatInputContainer {
    border: 2px solid var(--border) !important;
    border-radius: var(--r-full) !important;
    padding: .3rem .3rem .3rem 1.4rem !important;
    box-shadow: var(--shadow-md) !important;
    transition: border-color .18s !important;
}
.stChatInputContainer:focus-within {
    border-color: var(--g-500) !important;
    box-shadow: 0 0 0 3px rgba(5,150,105,.1), var(--shadow-md) !important;
}
.stChatInputContainer button {
    background: linear-gradient(135deg, var(--g-400), var(--g-500)) !important;
    border-radius: var(--r-full) !important;
}

/* ── MISC ── */
.stSuccess { background: var(--g-50) !important; color: var(--g-700) !important; border: 1px solid var(--g-100) !important; border-radius: var(--r-md) !important; }
.streamlit-expanderHeader { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: var(--r-md) !important; font-size: .84rem !important; }
div[data-testid="stSelectbox"] > div { border-radius: var(--r-md) !important; font-size: .87rem !important; }
.stSlider > div > div > div { background: var(--g-400) !important; }
hr { border-color: var(--border) !important; }

.native-answer {
    font-family: 'Noto Sans Telugu','Noto Sans Tamil','Noto Sans Kannada','Noto Sans Malayalam','DM Sans',sans-serif;
    line-height: 1.75;
}

/* Welcome chips */
.chip {
    display: inline-flex; align-items: center; gap: 7px;
    background: var(--g-50); border: 1px solid var(--g-100);
    border-radius: var(--r-md); padding: .55rem 1.1rem;
    font-size: .81rem; color: var(--g-700); font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# EMBEDDER
# ─────────────────────────────────────────────
@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = get_embedder()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def extract_text(file):
    try:
        if file.name.endswith(".pdf"):
            return "\n".join([p.extract_text() or "" for p in PdfReader(file).pages])
        elif file.name.endswith(".docx"):
            return "\n".join([p.text for p in docx.Document(file).paragraphs if p.text])
        return file.read().decode("utf-8", errors="ignore")
    except:
        return ""

def chunk_text(text, size=500, overlap=100):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size-overlap)
            if len(" ".join(words[i:i+size])) > 50]

def process_files(files):
    chunks, meta = [], []
    for f in files:
        t = extract_text(f)
        if t.strip():
            c = chunk_text(t)
            chunks.extend(c)
            meta.extend([{"file": f.name}] * len(c))
    if not chunks:
        return None, [], []
    return embedder.encode(chunks, convert_to_numpy=True), chunks, meta

def find_similar(q_emb, embeddings, chunks, meta, k=5):
    if embeddings is None:
        return []
    sims = cosine_similarity([q_emb], embeddings)[0]
    return [(chunks[i], meta[i]["file"], sims[i]) for i in np.argsort(sims)[-k:][::-1]]

def get_answer(context, question, target_language=None):
    lang = f"\n\nIMPORTANT: Respond entirely in {target_language}." if target_language else ""
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": f"You are Kivi, an intelligent document assistant. Answer based ONLY on the provided context. Be concise and accurate. If the answer isn't in the context, say so.{lang}"},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.3, max_tokens=1000,
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def stream_response(text, placeholder):
    words = text.split()
    out = ""
    for i, w in enumerate(words):
        out += w + " "
        placeholder.markdown(f'<div class="native-answer">{out}{"▌" if i < len(words)-1 else ""}</div>', unsafe_allow_html=True)
        time.sleep(0.02)

def save_current_chat():
    if not st.session_state.messages:
        return False, "Nothing to save"
    for c in st.session_state.saved_chats:
        if c["messages"] == st.session_state.messages:
            return False, "Already saved"
    st.session_state.saved_chats.append({
        "name": st.session_state.current_chat_name,
        "messages": st.session_state.messages.copy(),
        "date": datetime.now().strftime("%b %d, %H:%M"),
    })
    return True, "Chat saved"


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def show_sidebar():
    with st.sidebar:

        # Brand wordmark
        st.markdown(f'<div class="sb-brand">{KIVI_LOGO_SMALL_SVG}</div>', unsafe_allow_html=True)

        # New Chat button
        if st.button("New conversation", use_container_width=True, type="primary"):
            for k in ["messages", "embeddings", "chunks", "meta"]:
                st.session_state[k] = [] if k != "embeddings" else None
            st.session_state.current_chat_name = f"Chat {len(st.session_state.saved_chats)+1}"
            st.rerun()

        # Language selector
        st.markdown(f'<div class="sb-label">{ICON["globe"]} Response language</div>', unsafe_allow_html=True)
        sel = st.selectbox("lang", list(SOUTH_LANGUAGES.keys()),
                           index=list(SOUTH_LANGUAGES.keys()).index(st.session_state.selected_language),
                           label_visibility="collapsed", key="lang_sel")
        st.session_state.selected_language = sel
        lv = SOUTH_LANGUAGES[sel]
        if lv:
            st.markdown(f'<div class="lang-pill">{ICON["translate"]} Replies in <strong>{lv}</strong></div>',
                        unsafe_allow_html=True)

        # Saved chats
        st.markdown(f'<div class="sb-label">{ICON["chat"]} Saved chats</div>', unsafe_allow_html=True)
        if st.session_state.saved_chats:
            to_del = []
            for i, chat in enumerate(st.session_state.saved_chats):
                c1, c2 = st.columns([0.83, 0.17])
                with c1:
                    if st.button(chat["name"], key=f"load_{i}", use_container_width=True):
                        st.session_state.messages = chat["messages"]
                        st.session_state.current_chat_name = chat["name"]
                        st.rerun()
                with c2:
                    if st.button("×", key=f"del_{i}", use_container_width=True):
                        to_del.append(i)
            for i in sorted(to_del, reverse=True):
                st.session_state.saved_chats.pop(i)
            if to_del:
                st.rerun()
        else:
            st.caption("No saved chats yet")

        st.divider()

        # Settings
        with st.expander("Settings", expanded=False):
            TOP_K = st.slider("Chunks to retrieve", 3, 10, 5)
            show_sources = st.checkbox("Show sources", value=True)

        # Stats
        st.markdown(f'<div class="sb-label">{ICON["bar-chart"]} Stats</div>', unsafe_allow_html=True)
        ndocs = len(set(m["file"] for m in st.session_state.meta)) if st.session_state.meta else 0
        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card"><div class="stat-n">{ndocs}</div><div class="stat-l">Documents</div></div>
            <div class="stat-card"><div class="stat-n">{len(st.session_state.chunks)}</div><div class="stat-l">Chunks</div></div>
        </div>""", unsafe_allow_html=True)

        return TOP_K, show_sources


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
TOP_K, show_sources = show_sidebar()

# ── Page header ──────────────────────────────
c1, c2 = st.columns([0.6, 0.4])
with c1:
    lv = SOUTH_LANGUAGES[st.session_state.selected_language]
    lang_tag = f'<span class="lang-tag">{ICON["globe"]} {lv}</span>' if lv else ""
    st.markdown(f"""
    <div class="page-header" style="border:none; padding:0; margin-bottom:.25rem;">
      {KIVI_LOGO_SVG}
    </div>
    <div class="logo-sub">Intelligent Document Assistant{lang_tag}</div>
    """, unsafe_allow_html=True)

with c2:
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Save", use_container_width=True):
            ok, msg = save_current_chat()
            st.toast(msg, icon="✓" if ok else "·")
            if ok: st.rerun()
    with b2:
        if st.button("Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with b3:
        if st.button("New", use_container_width=True):
            for k in ["messages", "embeddings", "chunks", "meta"]:
                st.session_state[k] = [] if k != "embeddings" else None
            st.session_state.current_chat_name = f"Chat {len(st.session_state.saved_chats)+1}"
            st.rerun()

st.markdown("<div style='height:1px;background:var(--border);margin:1rem 0 1.5rem;'></div>", unsafe_allow_html=True)

# ── Upload zone ───────────────────────────────
st.markdown(f"""
<div class="upload-zone">
  <div class="upload-icon-wrap">
    {svg('<polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>', size=24, color="#059669", stroke_w=1.6)}
  </div>
  <div class="upload-title">Drop your documents here</div>
  <div class="upload-sub">
    <span class="ftag">PDF</span>
    <span class="ftag">DOCX</span>
    <span class="ftag">TXT</span>
  </div>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload", type=["pdf", "docx", "txt"],
    accept_multiple_files=True, label_visibility="collapsed"
)

if uploaded_files:
    new_hash = hashlib.sha256(str([f.name for f in uploaded_files]).encode()).hexdigest()
    if st.session_state.files_hash != new_hash:
        with st.spinner("Processing documents…"):
            emb, chunks, meta = process_files(uploaded_files)
            st.session_state.embeddings = emb
            st.session_state.chunks = chunks
            st.session_state.meta = meta
            st.session_state.files_hash = new_hash
        if emb is not None:
            st.success(f"Ready — {len(chunks)} chunks indexed from {len(uploaded_files)} file(s)")
        else:
            st.warning("No readable text found in the uploaded files.")

# ── Chat history ──────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(f'<div class="native-answer">{msg["content"]}</div>', unsafe_allow_html=True)

# ── Chat input ────────────────────────────────
lv = SOUTH_LANGUAGES[st.session_state.selected_language]
hint = f"Ask about your documents… (reply in {lv})" if lv else "Ask about your documents…"
question = st.chat_input(hint)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if not uploaded_files or st.session_state.embeddings is None:
        st.warning("Please upload documents first.")
        st.stop()

    with st.spinner("Searching…"):
        q_emb = embedder.encode([question])[0]
        retrieved = find_similar(q_emb, st.session_state.embeddings,
                                 st.session_state.chunks, st.session_state.meta, k=TOP_K)

    if retrieved:
        avg = sum(s for _, _, s in retrieved) / len(retrieved)
        conf_label = "High" if avg > .5 else "Medium" if avg > .3 else "Low"
        conf_cls   = "badge-high" if avg > .5 else "badge-medium" if avg > .3 else "badge-low"
    else:
        conf_label, conf_cls = "None", "badge-low"

    context = "\n\n".join(f"[From {fn}]\n{ch}" for ch, fn, _ in retrieved) if retrieved else "No relevant information found."

    with st.chat_message("assistant"):
        lv = SOUTH_LANGUAGES[st.session_state.selected_language]
        with st.spinner("Thinking…"):
            answer = get_answer(context, question, target_language=lv)

        ph = st.empty()
        stream_response(answer, ph)

        lang_badge = f'<span class="badge badge-lang">{ICON["translate"]} {lv}</span>' if lv else ""
        st.markdown(f"""
        <div class="meta-row">
          <span class="badge {conf_cls}">{ICON["target"]} {conf_label} confidence</span>
          {lang_badge}
        </div>""", unsafe_allow_html=True)

        if show_sources and retrieved:
            with st.expander(f"{len(retrieved)} source{'s' if len(retrieved)>1 else ''} referenced"):
                for ch, fn, sc in retrieved:
                    preview = ch[:280] + "…" if len(ch) > 280 else ch
                    st.markdown(f"""
                    <div class="src-card">
                      <div class="src-head">
                        {ICON["file-text"]} {fn}
                        <span class="src-score">{sc:.0%} match</span>
                      </div>
                      <div class="src-text">{preview}</div>
                    </div>""", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    if len(st.session_state.messages) == 2:
        save_current_chat()
    st.rerun()

# ── Welcome ───────────────────────────────────
if not st.session_state.messages:
    if st.session_state.embeddings is not None:
        st.markdown(f"""
        <div style="text-align:center;padding:3rem 1rem;color:var(--text-2);">
          <div style="width:56px;height:56px;background:var(--g-50);border:1px solid var(--g-100);
               border-radius:16px;display:flex;align-items:center;justify-content:center;margin:0 auto 1.25rem;">
            {svg('<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>', size=26, color="#059669", stroke_w=1.5)}
          </div>
          <h3 style="color:var(--g-700);font-size:1.1rem;font-weight:600;margin-bottom:.4rem;">Documents indexed — ready to answer</h3>
          <p style="font-size:.88rem;">Ask anything about your uploaded files.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align:center;padding:3.5rem 1rem 2rem;color:var(--text-2);">
          <div style="margin:0 auto 1.5rem;">{KIVI_LOGO_SVG}</div>
          <p style="font-size:.92rem;max-width:400px;margin:0 auto 1.75rem;line-height:1.65;color:var(--text-2);">
            Upload documents above, choose a language, then ask questions in English or any South Indian language.
          </p>
          <div style="display:flex;justify-content:center;gap:.75rem;flex-wrap:wrap;">
            <span class="chip">{ICON["file-text"]} PDF · DOCX · TXT</span>
            <span class="chip">{ICON["globe"]} 4 South Indian languages</span>
            <span class="chip">{ICON["search"]} Semantic search</span>
          </div>
        </div>""", unsafe_allow_html=True)
