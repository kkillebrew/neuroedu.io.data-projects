# =====================================================================
# STREAMLIT APP: 3_typo_behavior_app.py (The View)
# Strict Decoupling: Only UI and Plotly visualizations live here.
# =====================================================================
import streamlit as st
import plotly.express as px
import sys
import os
import pandas as pd

# --- PATH CONFIGURATION ---
# This tells the script to look one folder up to find the 'loaders' directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders.typo_behavior_loader import load_all_datasets, load_ml_pipeline, calculate_muscle_memory_decay

from data_projects_sidebar import apply_global_settings, render_sidebar

########################################
#        APPLY GLOBAL SETTINGS         #
########################################
apply_global_settings("Neuro-Edu | Why do we make typos?")

########################################
# RENDER THE SIDEBAR FOR DATA-PROJECTS #
########################################
render_sidebar()

# --- INITIALIZE DATA & MODELS ---
df_cmu, df_keyrecs, df_aalto, df_clarkson = load_all_datasets()
ml_model = load_ml_pipeline()

st.title("Typo Behavior & Cognitive Load Dashboard")
st.markdown("Analyze microscopic keystroke events, backspace footprints, and cognitive misfires.")

# Local Dataset Controller
dataset_choice = st.selectbox(
    "Select Dataset to Analyze:",
    ("KeyRecs (Micro-Typos)", "Clarkson (Cognitive)", "Aalto (Macro-Baseline)", "CMU (Muscle Memory)")
)

# Route the selected dataframe to our active variable
if dataset_choice == "KeyRecs (Micro-Typos)":
    active_df = df_keyrecs
elif dataset_choice == "Clarkson (Cognitive)":
    active_df = df_clarkson
elif dataset_choice == "Aalto (Macro-Baseline)":
    active_df = df_aalto
else:
    active_df = df_cmu


if active_df is not None and not active_df.empty:
    
    # Calculate high-level metrics safely
    total_events = len(active_df)
    
    if 'Is_Typo' in active_df.columns:
        total_typos = active_df['Is_Typo'].sum()
        typo_rate = (total_typos / total_events) * 100 if total_events > 0 else 0
    else:
        total_typos = 0
        typo_rate = 0.0

    avg_flight = active_df['Flight_Time'].mean()

    # Display Metrics in a 3-column layout
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Keystroke Events", f"{total_events:,}")
    col2.metric("Detected Typos", f"{total_typos:,} ({typo_rate:.2f}%)")
    col3.metric("Avg Flight Time", f"{avg_flight:.1f} ms")

    # Show the Microscopic Event Log
    st.divider()
    
    # Header and toggle switch aligned horizontally
    head_col, toggle_col = st.columns([3, 1])
    with head_col:
        st.subheader(f"Microscopic Event Log: {dataset_choice.split(' ')[0]}")
    with toggle_col:
        show_only_typos = st.checkbox("Show only flagged typos")
    
    if show_only_typos and 'Is_Typo' in active_df.columns:
        display_df = active_df[active_df['Is_Typo'] == True]
    else:
        display_df = active_df
        
    # Render the interactive dataframe
    st.dataframe(display_df, use_container_width=True, height=500)

else:
    st.warning("Data not loaded. Please verify the Parquet files are in the 'documents/' folder.")

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
    
    # --- SCHEMA UNIFICATION PROOF (Check off the TODO!) ---
    with st.expander("🔍 Verify Schema Unification"):
        st.markdown("Proof that disparate datasets were successfully normalized into the Master Data Schema:")
        
        schema_col1, schema_col2 = st.columns(2)
        with schema_col1:
            st.caption("Clarkson Dataset (Raw Keystrokes)")
            if not df_clarkson.empty:
                st.dataframe(df_clarkson.head(3), use_container_width=True)
        with schema_col2:
            st.caption("Aalto Dataset (Macro Baseline)")
            if not df_aalto.empty:
                st.dataframe(df_aalto.head(3), use_container_width=True)

    # --- TAB 1: DATASET OVERVIEW & PHASE 1 ANALYTICS ---
    with tab1:
        st.subheader(f"Microscopic Event Log: {dataset_choice.split(' ')[0]}")
        st.markdown("Analyze raw chronologies, backspace footprints, and behavioral error classifications.")
        
        # --- SMALL MULTIPLES (TRELLIS) CORRELATION CHART ---
        st.markdown("### Cross-Dataset Biometric Correlations")
        st.markdown("This Small Multiples chart plots **Dwell Time vs. Flight Time**. By standardizing the axes across different datasets, we can visually compare the fundamental shape of human typing behavior regardless of the data's origin.")
        
        # Gather a random sample from each dataset to keep the browser lightning fast
        multiples_data = []
        
        # We loop through the 3 datasets that utilize the Dwell/Flight master schema
        for df, name in [(df_clarkson, "Clarkson (Cognitive)"), (df_aalto, "Aalto (Macro)"), (df_keyrecs, "KeyRecs (Digraph)")]:
            if df is not None and not df.empty and 'Flight_Time' in df.columns and 'Dwell_Time' in df.columns:
                
                # Filter out extreme outliers (e.g., getting up for a coffee) to see the true cluster
                clean_df = df[(df['Flight_Time'] > 0) & (df['Flight_Time'] < 800) & 
                              (df['Dwell_Time'] > 0) & (df['Dwell_Time'] < 300)].copy()
                
                if not clean_df.empty:
                    # Sample exactly 2,000 points per dataset so the plot is perfectly balanced
                    sample_size = min(len(clean_df), 2000)
                    sampled = clean_df.sample(n=sample_size, random_state=42)
                    sampled['Source_Dataset'] = name
                    multiples_data.append(sampled)
        
        # Render the Plotly Facet Grid
        if multiples_data:
            combined_multiples = pd.concat(multiples_data, ignore_index=True)
            
            # facet_col is the magic Plotly parameter that creates Small Multiples
            fig_trellis = px.scatter(
                combined_multiples, 
                x="Dwell_Time", 
                y="Flight_Time", 
                color="Source_Dataset",
                facet_col="Source_Dataset",  # <-- CREATES THE GRID
                opacity=0.4,                 # Transparency shows density
                title="Universal Keystroke Signatures (Dwell vs. Flight)",
                labels={"Dwell_Time": "Dwell Time (ms)", "Flight_Time": "Flight Time (ms)"},
                trendline="ols"              # Draws the linear correlation line
            )
            
            # Hide the legend since the column titles already explain which dataset is which
            fig_trellis.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
            fig_trellis.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
            fig_trellis.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
            
            st.plotly_chart(fig_trellis, use_container_width=True)
            st.divider()

        # Phase 1 Visual Analytics
        if 'Is_Typo' in active_df.columns and active_df['Is_Typo'].any():
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Chart 1: Typo Taxonomy Pie Chart
                if 'Typo_Category' in active_df.columns:
                    category_df = active_df[active_df['Typo_Category'] != 'None']
                    if not category_df.empty:
                        cat_counts = category_df['Typo_Category'].value_counts().reset_index()
                        cat_counts.columns = ['Category', 'Count']
                        fig_cat = px.pie(
                            cat_counts, names='Category', values='Count', 
                            title="Typo Behavioral Taxonomy Distribution", 
                            hole=0.4, color_discrete_sequence=px.colors.sequential.Teal
                        )
                        st.plotly_chart(fig_cat, use_container_width=True)
                    else:
                        st.info("No spatial/cognitive categorizations found in this dataset.")
                        
            with col_chart2:
                # Chart 2: Latency Cost of Typos (Box Plot)
                # We cap at 1500ms to filter out pauses/getting up from the desk
                flight_df = active_df[active_df['Flight_Time'] < 1500].copy()
                # Map booleans to readable strings for the chart legend
                flight_df['Event_Type'] = flight_df['Is_Typo'].map({True: 'Typo / Backspace Trigger', False: 'Valid Keystroke'})
                
                fig_flight = px.box(
                    flight_df, x='Event_Type', y='Flight_Time', 
                    title="Flight Time Latency: Valid vs. Typos", 
                    color='Event_Type', color_discrete_sequence=['#00e676', '#ff5252']
                )
                st.plotly_chart(fig_flight, use_container_width=True)
                
        st.divider()
        
        # 2. Interactive Raw Dataframe
        head_col, toggle_col = st.columns([3, 1])
        with head_col:
            st.markdown("### Raw Event Log")
        with toggle_col:
            show_only_typos = st.checkbox("Show only flagged typos", value=True)
        
        if show_only_typos and 'Is_Typo' in active_df.columns:
            display_df = active_df[active_df['Is_Typo'] == True]
        else:
            display_df = active_df
            
        st.dataframe(display_df, use_container_width=True, height=400)

# ---------------------------------------------------------------------
# TAB 2: PHASE 2 (Macro-Level Benchmarks)
# ---------------------------------------------------------------------
with tab2:
    # 1. Run the backend calculation (Now safely cached!)
    decay_df = calculate_muscle_memory_decay(df_cmu)

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
