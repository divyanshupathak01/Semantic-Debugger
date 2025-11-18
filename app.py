import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
from bs4 import BeautifulSoup
import nltk
import re
import base64
import requests
import time
from streamlit_lottie import st_lottie

# --- NLTK Data Download ---
nltk.download('stopwords')
from nltk.corpus import stopwords

# --- Page Configuration ---
st.set_page_config(
    page_title="Python Debugging Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE SETUP ---
if 'history' not in st.session_state:
    st.session_state.history = []

if 'input_code_val' not in st.session_state:
    st.session_state.input_code_val = ""
if 'input_error_val' not in st.session_state:
    st.session_state.input_error_val = ""

# --- Animation Loader ---
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# --- Load Animations ---
anim_header = load_lottieurl("https://lottie.host/5a630462-535b-4564-a930-5f455114795a/D0Q1t6E5xO.json")
anim_success = load_lottieurl("https://lottie.host/f166f891-691c-4440-b910-b4018d96858d/X7zQ3jF9aW.json")
anim_loading = load_lottieurl("https://lottie.host/c9d9f362-e280-4d94-9f1e-995116893104/z1v7X9qW0a.json")
anim_sidebar = load_lottieurl("https://lottie.host/0d7d7f3a-86d4-463a-847e-973644554882/L4L6XqJ3C2.json")
anim_error = load_lottieurl("https://lottie.host/9e637e83-d52b-4c9a-b2e5-878b64d09139/v7K1vj1tF0.json")

# --- Custom CSS ---
def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as image:
            encoded_string = base64.b64encode(image.read()).decode()
        st.markdown(
            f"""
            <style>
            /* Animation Keyframes */
            @keyframes fadeInUp {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            
            /* Background Image */
            .stApp {{
                background-image: url(data:image/{"png" if image_file.endswith(".png") else "jpeg"};base64,{encoded_string});
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}

            /* Typography Shadows */
            h1, h2, h3, h4, h5, h6 {{
                color: #ffffff !important;
                text-shadow: 0px 3px 6px rgba(0,0,0, 0.9);
            }}
            .stMarkdown p {{
                color: #e0e0e0 !important;
                text-shadow: 0px 2px 4px rgba(0,0,0, 0.8);
                font-size: 1.1rem;
            }}

            /* Dark Glass Input Boxes */
            .stTextArea textarea {{
                background-color: rgba(0, 0, 0, 0.6) !important;
                color: #ffffff !important;
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                backdrop-filter: blur(10px);
                font-family: 'Consolas', 'Courier New', monospace;
            }}
            .stTextArea textarea:focus {{
                border: 1px solid #00f2fe;
                box-shadow: 0 0 20px rgba(0, 242, 254, 0.3);
            }}

            /* MAIN ACTION BUTTON (Neon Style) */
            button[kind="secondaryFormSubmit"] {{
                background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
                color: white;
                border: none;
                padding: 14px 30px;
                border-radius: 30px;
                font-weight: bold;
                font-size: 20px;
                box-shadow: 0 5px 15px rgba(0, 114, 255, 0.4);
                transition: all 0.3s ease-in-out;
                width: 100%;
            }}
            button[kind="secondaryFormSubmit"]:hover {{
                transform: translateY(-3px);
                box-shadow: 0 0 30px rgba(0, 198, 255, 0.8);
                border: none;
                color: white;
            }}
            
            /* SIDEBAR LINK STYLING */
            section[data-testid="stSidebar"] div.stButton button {{
                background-color: transparent !important;
                border: none !important;
                color: #b0b0b0 !important;
                text-align: left !important;
                padding: 0px !important;
                margin-top: 5px !important;
                font-size: 14px !important;
                box-shadow: none !important;
                transition: all 0.2s ease;
                text-decoration: none;
            }}
            
            section[data-testid="stSidebar"] div.stButton button:hover {{
                color: #00f2fe !important;
                text-decoration: underline;
                background-color: transparent !important;
                transform: translateX(5px);
            }}
            
            /* Result Card */
            .result-card {{
                background-color: rgba(15, 15, 25, 0.9);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                padding: 25px;
                backdrop-filter: blur(15px);
                box-shadow: 0 10px 30px rgba(0,0,0,0.6);
                margin-top: 20px;
                border-left: 5px solid #00f2fe;
                animation: fadeInUp 0.5s ease-out;
            }}

            /* Sidebar Styling */
            [data-testid="stSidebar"] {{
                background-color: rgba(10, 10, 10, 0.85);
                backdrop-filter: blur(10px);
                border-right: 1px solid rgba(255, 255, 255, 0.1);
            }}
            
            /* Remove Form Border */
            [data-testid="stForm"] {{
                border: none !important;
                padding: 0 !important;
                margin: 0 !important;
                background-color: transparent !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        pass

add_bg_from_local('background.jpg')

# --- Preprocessing Function ---
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str): return ""
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return ' '.join(tokens)

# --- Model and Data Loading ---
@st.cache_resource
def load_assets():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = np.load('question_embeddings.npy')
    questions_df = pd.read_csv('processed_questions.csv')
    answers_df = pd.read_csv('processed_answers.csv')
    return model, embeddings, questions_df, answers_df

model, question_embeddings, df_questions, df_answers = load_assets()

# --- Core Functions ---
def find_similar_questions(query, top_k=10):
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, question_embeddings, top_k=top_k)
    hit_indices = [hit['corpus_id'] for hit in hits[0]]
    results = df_questions.iloc[hit_indices].copy()
    results['SimilarityScore'] = [hit['score'] for hit in hits[0]]
    return results

def get_suggested_solution(question_id):
    try:
        accepted_answer_id = df_questions.loc[df_questions['QuestionId'] == question_id, 'AcceptedAnswerId'].iloc[0]
        answer_body = df_answers.loc[df_answers['AnswerId'] == accepted_answer_id, 'AnswerBody'].iloc[0]
        soup = BeautifulSoup(answer_body, 'html.parser')
        code_blocks = [code.get_text() for code in soup.find_all('code')]
        return code_blocks if code_blocks else ["No code snippets available in the accepted answer."]
    except (IndexError, KeyError):
        return ["Could not retrieve the corresponding solution."]

# ==========================
#       UI LAYOUT
# ==========================

# --- Sidebar ---
with st.sidebar:
    if anim_sidebar:
        st_lottie(anim_sidebar, height=180, key="sidebar")
    
    st.markdown("### 🛡️ Debug Buddy V2.0")
    
    # --- HISTORY SECTION ---
    st.markdown("#### 🕒 Recent Searches")
    st.caption("Select to load context:")
    
    if len(st.session_state.history) > 0:
        for i, item in enumerate(st.session_state.history[:5]):
            if st.button(f"↳ {item['label']}", key=f"hist_btn_{i}"):
                st.session_state.input_code_val = item['code']
                st.session_state.input_error_val = item['error']
                st.rerun()
    else:
        st.caption("No history yet.")
    
    st.markdown("---")
    
    # --- STATS SECTION ---
    st.markdown("#### 📊 Stats")
    st.info(f"📚 Database: {len(df_questions):,} Issues") 
    st.warning("⚡ Model: Semantic-BERT")
    
    st.markdown("---")

    # --- HOW TO USE SECTION (UPDATED) ---
    st.markdown("### ⚡ How to Use")
    st.markdown("1. **Paste Code:** Insert the buggy snippet.")
    st.markdown("2. **Paste Error:** Add the error message.")
    st.markdown("3. **Scan:** Click 'Analyze and Find Solution'.")

# --- Main Header ---
col_logo, col_title = st.columns([1, 4])
with col_logo:
    if anim_header:
        st_lottie(anim_header, height=140, key="header")
with col_title:
    st.title("Debug Buddy AI")
    st.markdown("#### *Every bug is a lesson. Let's solve this one together.*")

# --- FORM START ---
with st.form(key='analysis_form', clear_on_submit=False):
    
    # --- Input Section ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💻 Input Code Snippet")
        user_code = st.text_area("code", height=220, label_visibility="collapsed", 
                                 placeholder="import pandas as pd...", 
                                 value=st.session_state.input_code_val)

    with col2:
        st.markdown("### ⚠️ Trackback/Error Message")
        user_error = st.text_area("error", height=220, label_visibility="collapsed", 
                                  placeholder="KeyError: 'column_name'...", 
                                  value=st.session_state.input_error_val)

    st.write("") 

    # --- Action Button ---
    col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
    with col_b2:
        analyze_btn = st.form_submit_button("🔍 Analyze and Find Solution", use_container_width=True)

# --- Analysis Logic ---
if analyze_btn:
    if user_code and user_error:
        
        # --- SAVE TO HISTORY ---
        history_label = user_error.split('\n')[0][:40] + "..." if len(user_error) > 40 else user_error.split('\n')[0]
        history_item = {'label': history_label, 'code': user_code, 'error': user_error}
        
        if not st.session_state.history or st.session_state.history[0]['label'] != history_label:
            st.session_state.history.insert(0, history_item)
            st.session_state.input_code_val = user_code
            st.session_state.input_error_val = user_error
            st.rerun()

        # Loading Animation
        loading_col1, loading_col2, loading_col3 = st.columns([1, 1, 1])
        with loading_col2:
             if anim_loading:
                st_lottie(anim_loading, height=150, key="loading")
        
        time.sleep(0.1) 
        
        with st.spinner("Decoding Error Logic..."):
            user_query = user_code + "\n" + user_error
            top_matches = find_similar_questions(user_query)
            
            st.empty() 
            
            # --- Result Card ---
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            
            top_question = top_matches.iloc[0]
            score = top_question['SimilarityScore']
            
            r_c1, r_c2 = st.columns([1, 5])
            with r_c1:
                if anim_success:
                    st_lottie(anim_success, height=80, key="success")
            with r_c2:
                st.markdown(f"## Solution Identified ({score*100:.0f}% Match)")
                st.markdown(f"**Context:** [{top_question['Title']}](https://stackoverflow.com/q/{top_question['QuestionId']})")
            
            st.markdown("---")
            st.markdown("#### 🛠️ Suggested Fix")
            
            solution_code = get_suggested_solution(top_question['QuestionId'])
            if solution_code and "No code snippets available" not in solution_code[0]:
                for snippet in solution_code:
                    st.code(snippet, language='python')
            else:
                st.info("The solution is text-based. Click the link above to view.")
            
            st.markdown('</div>', unsafe_allow_html=True)

            # Related Threads
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📚 View Related Documentation & Threads"):
                for i, row in top_matches.iloc[1:].iterrows():
                    st.markdown(f"- [{row['Title']}](https://stackoverflow.com/q/{row['QuestionId']})")

    else:
        err_c1, err_c2, err_c3 = st.columns([1, 1, 1])
        with err_c2:
            if anim_error:
                st_lottie(anim_error, height=200, key="error")
            st.error("⚠ MISSING DATA: Please provide both code and error message.")