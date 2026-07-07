# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import json
import datetime
from pathlib import Path

# Set Streamlit Page Configuration as first command
st.set_page_config(
    page_title="AI Emotion Detection & Learning Support Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

import config
from src.database.db_manager import DatabaseManager
from src.auth.session_manager import (
    initialize_session, login_user, logout_user, 
    register_user, authenticate_user
)
from src.services.emotion_classifier import EmotionClassifierPipeline
from src.services.gemini_service import GeminiSupportService
from src.analytics.dashboard import (
    generate_emotion_distribution_chart,
    generate_field_analysis_chart,
    generate_confidence_trend_chart,
    generate_model_comparison_chart
)

# --- Resource Caching ---
@st.cache_resource
def get_db():
    return DatabaseManager(config.DB_PATH)

@st.cache_resource
def get_classifier():
    return EmotionClassifierPipeline()

@st.cache_resource
def get_gemini():
    return GeminiSupportService(config.GEMINI_API_KEY)

# Initialize resources
db = get_db()
classifier = get_classifier()
gemini = get_gemini()

# Initialize session state
initialize_session()

# --- Custom Premium Dark CSS ---
st.markdown("""
<style>
    /* Theme color variables */
    :root {
        --bg-color: #0E1117;
        --card-bg: #1A1F2C;
        --border-color: #2D3748;
        --accent-color: #6366F1;
        --text-primary: #F7FAFC;
        --text-secondary: #A0AEC0;
    }
    
    /* Main body adjustments */
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-primary);
    }
    
    /* Premium Title gradients */
    .title-gradient {
        background: linear-gradient(90deg, #6366F1 0%, #A855F7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: var(--text-secondary);
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphic cards */
    .glass-card {
        background: rgba(26, 31, 44, 0.7);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1A1F2C 0%, #11141D 100%);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar styling override */
    section[data-testid="stSidebar"] {
        background-color: #11141D;
        border-right: 1px solid var(--border-color);
    }
    
    /* Styled buttons */
    .stButton>button {
        background: linear-gradient(90deg, #6366F1 0%, #4F46E5 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    
    /* Custom alerts styling */
    .custom-error {
        background-color: rgba(244, 67, 54, 0.1);
        border-left: 5px solid #F44336;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .custom-success {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper logic for logging back up to local CSV ---
def log_records_to_csv(db_records):
    """Saves records to local backup CSV file."""
    csv_path = config.DATA_DIR / "emotion_records_backup.csv"
    
    # Map row dicts to flat structure
    flat_records = []
    ids_to_mark = []
    
    for r in db_records:
        flat_records.append({
            "record_id": r["record_id"],
            "email": r["email"],
            "field": r["field"],
            "input_text": r["input_text"],
            "predicted_emotion": r["predicted_emotion"],
            "secondary_emotion": r["secondary_emotion"],
            "confidence_score": r["confidence_score"],
            "model_used": r["model_used"],
            "response_type": r["response_type"],
            "timestamp": r["timestamp"]
        })
        if not r.get("csv_logged", 0):
            ids_to_mark.append(r["record_id"])
            
    df_new = pd.DataFrame(flat_records)
    
    if csv_path.exists():
        try:
            df_old = pd.read_csv(csv_path)
            # Filter out overlapping records
            df_combined = pd.concat([df_old, df_new]).drop_duplicates(subset=["record_id"], keep="last")
            df_combined.to_csv(csv_path, index=False)
        except Exception:
            df_new.to_csv(csv_path, index=False)
    else:
        df_new.to_csv(csv_path, index=False)
        
    # Mark in SQLite db
    if ids_to_mark:
        db.mark_records_as_csv_logged(ids_to_mark)


# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("<h2 style='text-align: center;'>🎓 EduEmotion AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='margin-top: 0; border-color: var(--border-color);'>", unsafe_allow_html=True)

if st.session_state.logged_in:
    st.sidebar.markdown(f"""
    <div style='text-align: center; margin-bottom: 1.5rem;'>
        <p style='margin: 0; font-size: 0.85rem; color: var(--text-secondary);'>LOGGED IN AS</p>
        <h4 style='margin: 0; color: #A855F7;'>{st.session_state.user_name}</h4>
        <span style='background-color: #2D3748; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem;'>{st.session_state.user_role}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation list for logged-in user
    nav_options = ["Workspace", "History Logs", "Analytics Dashboard"]
    if st.session_state.user_role == "Admin":
        nav_options.append("User Accounts Manager")
        
    page = st.sidebar.radio("Go to:", nav_options, index=0)
    st.session_state.current_page = page
    
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    if st.sidebar.button("Logout 🚪", use_container_width=True):
        logout_user()
        st.rerun()
else:
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem; border: 1px dashed var(--border-color); border-radius: 8px; margin-bottom: 1.5rem;'>
        <p style='margin: 0; font-size: 0.9rem; color: var(--text-secondary);'>Please sign in or register to analyze text queries and generate AI guidance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio("Go to:", ["Home", "Authenticate"])
    st.session_state.current_page = page


# --- PAGE ROUTING ---

# 1. LANDING PAGE
if st.session_state.current_page == "Home":
    st.markdown("<div class='title-gradient'>AI-Driven Emotion Detection & Support</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Supporting students with sentiment-aware generative guidance and deep learning comparisons.</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🌟 Empowering Emotion-Aware Education
        Our platform bridges the gap between study frustration and learning clarity.
        By pasting natural language study problems, we instantly analyze the student's affective state and deliver custom learning strategies.
        
        #### Core Capabilities:
        - **BiLSTM vs BERT side-by-side**: Evaluate neural predictions in real time.
        - **Mixed Emotion Analysis**: Detect compound emotional states (>= 15\\% probability threshold).
        - **Generative Guidance**: Custom empathetic tips, motivational boosters, and next steps powered by Google Gemini AI.
        - **SQLite Database with CSV Backups**: Maintain persistent study records.
        - **Interactive Plotly Visualizations**: Track emotional trends and average confidence curves.
        """)
        if st.button("Get Started →"):
            st.session_state.current_page = "Authenticate"
            st.rerun()
            
    with col2:
        st.markdown("""
        <div class='glass-card' style='text-align: center;'>
            <h3>How it Works</h3>
            <p>1. <b>Register / Log In</b>: Create a student or educator account.</p>
            <p>2. <b>Input Problem</b>: Describe your academic challenge in natural language.</p>
            <p>3. <b>Dual Inference</b>: View neural classifications and confidence probabilities.</p>
            <p>4. <b>Receive Advice</b>: Access immediate generative learning steps.</p>
        </div>
        """, unsafe_allow_html=True)


# 2. AUTHENTICATION PAGE
elif st.session_state.current_page == "Authenticate":
    st.markdown("<div class='title-gradient'>Account Center</div>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔑 Sign In", "📝 Sign Up"])
    
    with tab1:
        st.subheader("Login to your Account")
        with st.form("login_form"):
            email = st.text_input("Email Address")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign In")
            
            if submit:
                success, result = authenticate_user(db, email, password)
                if success:
                    login_user(result)
                    st.toast(f"Welcome back, {result['name']}!", icon="👋")
                    st.rerun()
                else:
                    st.error(result)
                    
    with tab2:
        st.subheader("Register a New Account")
        with st.form("signup_form"):
            new_name = st.text_input("Full Name")
            new_email = st.text_input("Email Address")
            new_password = st.text_input("Password (min 6 characters)", type="password")
            new_role = st.selectbox("Role", ["Student", "Educator", "Admin"])
            submit_signup = st.form_submit_button("Register Account")
            
            if submit_signup:
                success, msg = register_user(db, new_email, new_name, new_password, new_role)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)


# 3. WORKSPACE (EMOTION DETECTOR & GENERATOR)
elif st.session_state.current_page == "Workspace":
    st.markdown("<div class='title-gradient'>Learning Desk</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Submit your academic roadblocks, analyze emotional responses, and retrieve personalized generative guidance.</div>", unsafe_allow_html=True)
    
    # Layout splits into input/options and model outputs
    col_input, col_settings = st.columns([3, 1])
    
    with col_settings:
        st.markdown("### ⚙️ Inference Settings")
        selected_model = st.radio("Primary Classifier", ["BERT", "BiLSTM"], index=0)
        use_gemini_option = st.checkbox("Use Gemini AI API", value=True)
        persist_records = st.checkbox("Save interaction to database", value=True)
        
    with col_input:
        academic_field = st.selectbox(
            "What academic field are you working on?",
            ["Computer Science", "Mathematics", "Physics", "Chemistry", "Biology", 
             "Engineering", "Business", "Literature", "History", "Psychology", "Other"]
        )
        
        student_query = st.text_area(
            "Describe your learning challenge or problem in detail:",
            placeholder="e.g., I've been trying to solve this recursion stack overflow bug for 3 hours and nothing works! The code keeps crashing.",
            height=150
        )
        
        # Analyze Button
        analyze_btn = st.button("Get AI Learning Help 🚀", use_container_width=True)
        
    if analyze_btn:
        if not student_query.strip():
            st.warning("Please enter a valid study challenge query.")
        else:
            with st.spinner("Analyzing sentiments and fetching guidance..."):
                # Run dual model inference
                bilstm_res = classifier.predict_bilstm(student_query)
                bert_res   = classifier.predict_bert(student_query)

                # Determine active model result based on user selection
                if selected_model == "BERT":
                    active_res = bert_res
                else:
                    active_res = bilstm_res

                # Graceful warnings for unavailable models
                if bert_res is None and selected_model == "BERT":
                    st.error("BERT model weights not loaded! Ensure BERT training has run (training/train_bert.py).")
                elif bilstm_res is None and selected_model == "BiLSTM":
                    st.error("BiLSTM model weights not loaded! Ensure BiLSTM training has run (training/train_bilstm.py).")
                elif active_res is None:
                    st.error("Selected model is unavailable. Please train the model first or switch to a different classifier.")
                else:
                    primary_emo   = active_res["primary_emotion"]
                    secondary_emo = active_res["secondary_emotion"]
                    conf          = active_res["confidence_score"]
                    scores        = active_res["scores"]

                    # Store in session state for display and regeneration
                    st.session_state.active_field        = academic_field
                    st.session_state.active_query        = student_query
                    st.session_state.active_res          = active_res
                    st.session_state.use_gemini_option   = use_gemini_option
                    # Cache both model results for dual comparison panel
                    st.session_state.cached_bilstm_res   = bilstm_res
                    st.session_state.cached_bert_res     = bert_res

                    # Generate Gemini / fallback guidance
                    ai_text, response_type = gemini.generate_guidance(
                        academic_field, student_query, primary_emo, secondary_emo, conf
                    )
                    st.session_state.ai_text       = ai_text
                    st.session_state.response_type = response_type

                    # Save to database
                    if persist_records:
                        record_id = db.save_emotion_record(
                            st.session_state.user_email,
                            academic_field,
                            student_query,
                            primary_emo,
                            secondary_emo,
                            conf,
                            selected_model,
                            ai_text,
                            response_type,
                            scores
                        )
                        st.session_state.active_record_id = record_id

                        # Trigger local CSV backup
                        all_recs = db.get_user_records(st.session_state.user_email)
                        log_records_to_csv(all_recs)

            # Render completion banner
            if "active_res" in st.session_state:
                st.success("Analysis Complete!")
            
    # Display results if present in session state
    if "active_res" in st.session_state:
        res = st.session_state.active_res
        
        # Dual Model Side-by-Side Comparison Columns
        st.markdown("---")
        st.markdown("### 📊 Dual Model Comparison")

        col_m1, col_m2 = st.columns(2)

        # Use cached results from the analysis run (avoid redundant inference)
        bilstm_data = st.session_state.get("cached_bilstm_res", None)
        bert_data   = st.session_state.get("cached_bert_res",   None)

        with col_m1:
            if bilstm_data is not None:
                st.markdown(f"""
                <div class='glass-card' style='border-top: 4px solid #6366F1;'>
                    <h4 style='color: #6366F1;'>BiLSTM Student-Adaptive Model</h4>
                    <p style='margin:0;'>Primary Prediction: <b>{bilstm_data['primary_emotion']}</b> ({bilstm_data['confidence_score']:.1%})</p>
                    <p style='margin:0; font-size:0.9rem;'>Secondary: <i>{bilstm_data['secondary_emotion']}</i></p>
                </div>
                """, unsafe_allow_html=True)
                for emo, val in bilstm_data["scores"].items():
                    st.write(f"**{emo}** ({val:.1%})")
                    st.progress(float(val))
            else:
                st.markdown("""
                <div class='glass-card' style='border-top: 4px solid #6366F1; opacity: 0.5;'>
                    <h4 style='color: #6366F1;'>BiLSTM Student-Adaptive Model</h4>
                    <p style='margin:0; color: #A0AEC0;'>Model not available — run training/train_bilstm.py to enable.</p>
                </div>
                """, unsafe_allow_html=True)

        with col_m2:
            if bert_data is not None:
                st.markdown(f"""
                <div class='glass-card' style='border-top: 4px solid #A855F7;'>
                    <h4 style='color: #A855F7;'>BERT Fine-Tuned Transformer</h4>
                    <p style='margin:0;'>Primary Prediction: <b>{bert_data['primary_emotion']}</b> ({bert_data['confidence_score']:.1%})</p>
                    <p style='margin:0; font-size:0.9rem;'>Secondary: <i>{bert_data['secondary_emotion']}</i></p>
                </div>
                """, unsafe_allow_html=True)
                for emo, val in bert_data["scores"].items():
                    st.write(f"**{emo}** ({val:.1%})")
                    st.progress(float(val))
            else:
                st.markdown("""
                <div class='glass-card' style='border-top: 4px solid #A855F7; opacity: 0.5;'>
                    <h4 style='color: #A855F7;'>BERT Fine-Tuned Transformer</h4>
                    <p style='margin:0; color: #A0AEC0;'>Model not available — run training/train_bert.py to enable.</p>
                </div>
                """, unsafe_allow_html=True)
                
        # Mixed Emotion Status Panel
        st.markdown("### 🏷️ Emotion Status Summary")
        emo_str = res["primary_emotion"]
        if res["secondary_emotion"] != "None":
            emo_str += f" + {res['secondary_emotion']} (Mixed State Detected)"
            st.info(f"Detected Emotional State: **{emo_str}**")
        else:
            st.info(f"Detected Emotional State: **{emo_str}** (Single Primary)")
            
        # AI Guidance Text
        st.markdown("---")
        st.markdown("### 💡 AI Tutor Learning Guidance")
        
        st.markdown(f"""
        <div style='background-color: var(--card-bg); padding: 1.5rem; border-radius: 10px; border: 1px solid var(--border-color); margin-bottom:1.5rem;'>
            <p style='font-size:0.8rem; color:var(--text-secondary); margin-bottom:1rem;'>RESPONSE GENERATION TYPE: <b>{st.session_state.response_type}</b></p>
            {st.session_state.ai_text}
        </div>
        """, unsafe_allow_html=True)
        
        # Regeneration Button
        if st.button("Regenerate Learning Guidance 🔄"):
            with st.spinner("Regenerating advice response..."):
                primary_emo = res["primary_emotion"]
                secondary_emo = res["secondary_emotion"]
                conf = res["confidence_score"]
                
                # Fetch new response from Gemini
                new_text, resp_type = gemini.generate_guidance(
                    st.session_state.active_field,
                    st.session_state.active_query,
                    primary_emo,
                    secondary_emo,
                    conf
                )
                
                # Update session state and DB
                st.session_state.ai_text = new_text
                st.session_state.response_type = resp_type
                
                if "active_record_id" in st.session_state:
                    db.update_ai_response(st.session_state.active_record_id, new_text, resp_type)
                    
                st.toast("Guidance response regenerated and saved!", icon="🔄")
                st.rerun()


# 4. HISTORY LOGS PAGE
elif st.session_state.current_page == "History Logs":
    st.markdown("<div class='title-gradient'>Study History Logs</div>", unsafe_allow_html=True)
    
    # Load user's records
    records = db.get_user_records(st.session_state.user_email)
    
    if not records:
        st.info("No study query records saved yet. Navigate to the Workspace to get AI guidance.")
    else:
        df_history = pd.DataFrame(records)
        
        # Search and filter header row
        st.markdown("#### Search and Filter Filters")
        col_f1, col_f2, col_f3, col_search = st.columns([1, 1, 1, 2])
        
        with col_f1:
            field_filter = st.selectbox("Academic Field", ["All"] + list(df_history["field"].unique()))
        with col_f2:
            emotion_filter = st.selectbox("Predicted Emotion", ["All"] + list(df_history["predicted_emotion"].unique()))
        with col_f3:
            model_filter = st.selectbox("Model Used", ["All"] + list(df_history["model_used"].unique()))
        with col_search:
            search_query = st.text_input("Search query text keywords")
            
        # Apply filters
        df_filtered = df_history.copy()
        if field_filter != "All":
            df_filtered = df_filtered[df_filtered["field"] == field_filter]
        if emotion_filter != "All":
            df_filtered = df_filtered[df_filtered["predicted_emotion"] == emotion_filter]
        if model_filter != "All":
            df_filtered = df_filtered[df_filtered["model_used"] == model_filter]
        if search_query.strip():
            df_filtered = df_filtered[df_filtered["input_text"].str.contains(search_query, case=False, na=False)]
            
        # Display history records
        st.markdown(f"Displaying **{len(df_filtered)}** matching interactions of **{len(df_history)}** total records.")
        
        # Format table for cleaner rendering
        render_df = df_filtered[[
            "record_id", "timestamp", "field", "input_text", 
            "predicted_emotion", "secondary_emotion", "confidence_score", "model_used", "response_type"
        ]].copy()
        render_df["confidence_score"] = render_df["confidence_score"].map(lambda x: f"{x:.1%}")
        
        st.dataframe(render_df, use_container_width=True, hide_index=True)
        
        # CSV Export Options
        csv_data = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered History (CSV) 📥",
            data=csv_data,
            file_name="student_emotion_history_filtered.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Accordion preview for detailed responses
        st.markdown("#### Detailed Guidance Inspector")
        for idx, row in df_filtered.head(10).iterrows():
            conf_display = row['confidence_score']
            conf_str = f"{conf_display:.1%}" if isinstance(conf_display, float) else str(conf_display)
            with st.expander(f"Record #{row['record_id']} | {row['timestamp'][:10]} | {row['field']} - Predicted: {row['predicted_emotion']} ({conf_str})"):
                st.write(f"**Student Query:** {row['input_text']}")
                st.markdown(f"**AI Guidance:**\n{row['ai_response']}")
                
                # Delete record action
                if st.button(f"Delete Record #{row['record_id']}", key=f"del_{row['record_id']}"):
                    db.delete_record(row["record_id"])
                    st.toast("Record deleted successfully.")
                    st.rerun()


# 5. ANALYTICS DASHBOARD
elif st.session_state.current_page == "Analytics Dashboard":
    st.markdown("<div class='title-gradient'>Academic Analytics Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Visualizing student emotion states, confidence metrics, and study behaviors.</div>", unsafe_allow_html=True)
    
    # Load data
    if st.session_state.user_role in ["Educator", "Admin"]:
        # Educators and Admins can see aggregated data for all student records
        records = db.get_all_records()
        st.info("Educator/Admin View: Displaying global database analytics.")
    else:
        records = db.get_user_records(st.session_state.user_email)
        st.info("Student View: Displaying your personal learning analytics.")
        
    if not records:
        st.warning("No data records available to generate charts.")
    else:
        df_anal = pd.DataFrame(records)
        
        # KPI Row
        st.markdown("#### Key Performance Indicators (KPI)")
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            st.markdown(f"""
            <div class='metric-card'>
                <p style='margin:0; font-size:0.9rem; color:var(--text-secondary);'>Total Interactions</p>
                <h2 style='margin:0; color:#6366F1;'>{len(df_anal)}</h2>
            </div>
            """, unsafe_allow_html=True)
        with kpi_col2:
            avg_conf = df_anal["confidence_score"].mean()
            st.markdown(f"""
            <div class='metric-card'>
                <p style='margin:0; font-size:0.9rem; color:var(--text-secondary);'>Average Confidence</p>
                <h2 style='margin:0; color:#4CAF50;'>{avg_conf:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        with kpi_col3:
            top_emotion = df_anal["predicted_emotion"].mode()[0]
            st.markdown(f"""
            <div class='metric-card'>
                <p style='margin:0; font-size:0.9rem; color:var(--text-secondary);'>Dominant Emotion</p>
                <h2 style='margin:0; color:#FF9800;'>{top_emotion}</h2>
            </div>
            """, unsafe_allow_html=True)
        with kpi_col4:
            active_fields = df_anal["field"].nunique()
            st.markdown(f"""
            <div class='metric-card'>
                <p style='margin:0; font-size:0.9rem; color:var(--text-secondary);'>Fields Engaged</p>
                <h2 style='margin:0; color:#A855F7;'>{active_fields}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        # Chart Rows
        st.markdown("---")
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            pie_fig = generate_emotion_distribution_chart(df_anal)
            if pie_fig:
                st.plotly_chart(pie_fig, use_container_width=True)
                
        with chart_col2:
            bar_fig = generate_field_analysis_chart(df_anal)
            if bar_fig:
                st.plotly_chart(bar_fig, use_container_width=True)
                
        chart_col3, chart_col4 = st.columns(2)
        
        with chart_col3:
            line_fig = generate_confidence_trend_chart(df_anal)
            if line_fig:
                st.plotly_chart(line_fig, use_container_width=True)
                
        with chart_col4:
            comp_fig = generate_model_comparison_chart(df_anal)
            if comp_fig:
                st.plotly_chart(comp_fig, use_container_width=True)


# 6. ADMIN USER MANAGER PAGE
elif st.session_state.current_page == "User Accounts Manager":
    st.markdown("<div class='title-gradient'>System User Manager</div>", unsafe_allow_html=True)
    st.warning("Admin Access Only")
    
    with st.spinner("Fetching accounts list..."):
        users = db.get_all_users()
            
        if users:
            df_users = pd.DataFrame(users)
            st.dataframe(df_users, use_container_width=True, hide_index=True)
        else:
            st.info("No system users registered.")
