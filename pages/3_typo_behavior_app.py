# =====================================================================
# STREAMLIT APP: 3_typo_behavior_app.py (The View)
# Strict Decoupling: Only UI and Plotly visualizations live here.
# =====================================================================
import streamlit as st
import plotly.express as px
import sys
import os

# --- PATH CONFIGURATION ---
# This tells the script to look one folder up to find the 'loaders' directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders.typo_behavior_loader import load_all_datasets, load_ml_pipeline

from career_hub_sidebar import apply_global_settings, render_sidebar

########################################
#        APPLY GLOBAL SETTINGS         #
########################################
apply_global_settings("Neuro-Edu | What affects gas prices?")

########################################
# RENDER THE SIDEBAR FOR DATA-PROJECTS #
########################################
render_sidebar()

# --- INITIALIZE DATA & MODELS ---
df_cmu, df_keyrecs, df_aalto, df_clarkson = load_all_datasets()
ml_model = load_ml_pipeline()

st.title("⌨️ Keystroke Dynamics: Real-Time Typo Prediction")

# --- THE 4-TAB ANALYTICAL PROGRESSION ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📂 1. Dataset Overview", 
    "📈 2. Macro Benchmarks", 
    "🧠 3. Taxonomy & Features", 
    "⚡ 4. Real-Time Prediction"
])

# ---------------------------------------------------------------------
# TAB 1: PHASE 1 (Dataset Overview)
# ---------------------------------------------------------------------
with tab1:
    st.header("Phase 1: Cloud Pre-Processing & Ingestion")
    st.write("Overview of the massive datasets downsampled via Colab and ingested as Parquet files.")
    # TODO: Display head() of dataframes to show schema unification

# ---------------------------------------------------------------------
# TAB 2: PHASE 2 (Macro-Level Benchmarks)
# ---------------------------------------------------------------------
with tab2:
    st.header("Phase 2: Muscle Memory vs. Cognitive Load")
    st.write("Establishing the statistical boundaries of 'normal' typing.")
    # TODO: Plotly graphs of CMU decay curve and Aalto t-tests

# ---------------------------------------------------------------------
# TAB 3: PHASES 3 & 4 (Taxonomy & Feature Engineering)
# ---------------------------------------------------------------------
with tab3:
    st.header("Phases 3 & 4: Error Categorization & Behavioral Features")
    st.write("Categorizing errors (Spatial vs. Cognitive) and mapping Fatigue U-Curves.")
    # TODO: Plotly correlation matrices and Rolling Variance (Burstiness) plots

# ---------------------------------------------------------------------
# TAB 4: PHASE 5 (Predictive ML Modeling - Latency Mitigated)
# ---------------------------------------------------------------------
with tab4:
    st.header("Phase 5: Real-Time Machine Learning Inference")
    st.write("Type in the box below. The Random Forest model will analyze your rolling WPM and flight times to predict your probability of making a typo on the next keystroke.")
    
    # ⚡ STREAMLIT FRAGMENT (Crucial for Lag-Free UI)
    # MATLAB Analogy: Using a custom callback function tied *only* to a UI element's 
    # 'ValueChangedFcn' so the rest of the App Designer figure doesn't redraw.
    @st.fragment
    def real_time_inference_ui():
        user_input = st.text_input("Type a complex sentence here:", key="ml_input")
        
        # Debounce/Mock Logic (To be replaced with actual ML inference)
        if user_input:
            # TODO: Extract N-grams and latency from user_input here
            # TODO: prob = ml_model.predict_proba(engineered_features)
            
            # Placeholder UI
            st.metric(label="Probability of Impending Typo", value="14%", delta="2% due to burstiness")
            st.progress(0.14)
            
    # Execute the isolated fragment
    real_time_inference_ui()
