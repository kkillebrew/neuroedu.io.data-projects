# =====================================================================
# STREAMLIT APP: 3_typo_behavior_app.py (The View)
# Strict Decoupling: Only UI and Plotly visualizations live here.
# =====================================================================
import streamlit as st
import plotly.express as px
import scipy.stats as stats
import sys
import os
import pandas as pd

# --- PATH CONFIGURATION ---
# This tells the script to look one folder up to find the 'loaders' directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders.typo_behavior_loader import calculate_muscle_memory_decay

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
st.title("Typo Behavior & Cognitive Load Dashboard")
st.markdown("Analyze microscopic keystroke events, backspace footprints, and cognitive misfires.")

# --- UNIFIED BIG DATA LOADER ---
base_dir = os.path.join(os.path.dirname(__file__), '..', 'documents')
master_path = os.path.join(base_dir, 'master_dataset.parquet')

@st.cache_data(show_spinner=False)
def load_master_matrix(filepath):
    if os.path.exists(filepath):
        return pd.read_parquet(filepath)
    return pd.DataFrame()

with st.spinner("Initializing Cloud Master Matrix..."):
    active_df = load_master_matrix(master_path)
    
    if not active_df.empty:
        # Isolate subsets using the new 'Source_Dataset' column schema
        df_cmu = active_df[active_df['Source_Dataset'] == 'CMU']
        df_clarkson = active_df[active_df['Source_Dataset'].isin(['Clarkson_I', 'Clarkson_II'])]
        df_aalto = active_df[active_df['Source_Dataset'] == 'Aalto']
        df_keyrecs = active_df[active_df['Source_Dataset'] == 'KeyRecs']
    else:
        df_cmu = df_clarkson = df_aalto = df_keyrecs = pd.DataFrame()
        st.error("Master Dataset not found. Waiting for GitHub ETL pipeline to finish...")

if active_df is not None and not active_df.empty:
    
    # Calculate high-level metrics safely
    total_events = len(active_df)
    
    if 'Is_Typo' in active_df.columns:
        total_typos = active_df['Is_Typo'].sum()
        typo_rate = (total_typos / total_events) * 100 if total_events > 0 else 0
    else:
        total_typos = 0
        typo_rate = 0.0

    # Use the standardized 'Flight_DD_ms'
    avg_flight = active_df['Flight_DD_ms'].mean() if 'Flight_DD_ms' in active_df.columns else 0.0

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
        st.subheader("Microscopic Event Log: Master Matrix")
    with toggle_col:
        show_only_typos = st.checkbox("Show only flagged typos")
    
    # Safely render only the first 1000 rows to prevent Arrow serialization OOM crashes
    if show_only_typos and 'Is_Typo' in active_df.columns:
        display_df = active_df[active_df['Is_Typo'] == True].head(1000)
    else:
        display_df = active_df.head(1000)
        
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
        st.subheader("Microscopic Event Log: Master Matrix")
        st.markdown("Analyze raw chronologies, backspace footprints, and behavioral error classifications.")
        
        # --- SMALL MULTIPLES (TRELLIS) CORRELATION CHART ---
        st.markdown("### Cross-Dataset Biometric Correlations")
        st.markdown("This Small Multiples chart plots **Dwell Time vs. Flight Time**. By standardizing the axes across different datasets, we can visually compare the fundamental shape of human typing behavior regardless of the data's origin.")
        
        # Gather a random sample from each dataset to keep the browser lightning fast
        multiples_data = []
        
        # We loop through the 3 datasets that utilize the Dwell/Flight master schema
        for df, name in [(df_clarkson, "Clarkson (Cognitive)"), (df_aalto, "Aalto (Macro)"), (df_keyrecs, "KeyRecs (Digraph)")]:
            if df is not None and not df.empty and 'Flight_DD_ms' in df.columns and 'Hold_Time_ms' in df.columns:
                
                # Filter out extreme outliers (e.g., getting up for a coffee) to see the true cluster
                clean_df = df[(df['Flight_DD_ms'] > 0) & (df['Flight_DD_ms'] < 800) & 
                              (df['Hold_Time_ms'] > 0) & (df['Hold_Time_ms'] < 300)].copy()
                
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
                x="Hold_Time_ms", 
                y="Flight_DD_ms", 
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
                flight_df = active_df[active_df['Flight_DD_ms'] < 1500].copy()
                # Map booleans to readable strings for the chart legend
                flight_df['Event_Type'] = flight_df['Is_Typo'].map({True: 'Typo / Backspace Trigger', False: 'Valid Keystroke'})
                
                fig_flight = px.box(
                    flight_df, x='Event_Type', y='Flight_DD_ms', 
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
            # Added a unique key to prevent DuplicateWidgetID crash
            show_only_typos_tab1 = st.checkbox("Show only flagged typos", value=True, key="tab1_typo_toggle")
            
        # Safely render only the first 1000 rows to prevent serialization crashes
        if show_only_typos_tab1 and 'Is_Typo' in active_df.columns:
            display_df = active_df[active_df['Is_Typo'] == True].head(1000)
        else:
            display_df = active_df.head(1000)
            
        st.dataframe(display_df, use_container_width=True, height=400)

# ---------------------------------------------------------------------
# TAB 2: PHASE 2 (Macro-Level Benchmarks)
# ---------------------------------------------------------------------
with tab2:
    st.header("Phase 2: Muscle Memory vs. Cognitive Load")
    st.write("Establishing the statistical boundaries of 'normal' typing.")
    
    # --- SECTION A: MUSCLE MEMORY DECAY ---
    st.subheader("1. Muscle Memory Decay (CMU Dataset)")
    st.markdown("This curve tracks 51 subjects typing the same complex password 400 times. Notice the exponential drop in cognitive load before hitting a physical motor-control asymptote.")
    
    # 1. Create the filtered timing dataframe using the standardized columns from Step 4
    # Filter for rows that have valid timing data (Standardized in Step 4)
    df_timing = active_df.dropna(subset=['Flight_DD_ms'])
    df_timing = df_timing[df_timing['Flight_DD_ms'] < 3000]

    st.subheader("Inter-Key Interval (IKI) by Dataset")
    
    # Use the standardized 'Source_Dataset' and 'Flight_DD_ms' columns
    fig_iki = px.box(
        df_timing, 
        x="Source_Dataset", 
        y="Flight_DD_ms", 
        color="Source_Dataset",
        points=False, 
        labels={'Flight_DD_ms': 'Flight Time (ms)', 'Source_Dataset': 'Study Source'},
        title="Macro Latency Benchmarks"
    )
    st.plotly_chart(fig_iki, use_container_width=True)


    # 1. Isolate CMU data from the Master Matrix to calculate the curve
    df_cmu_only = active_df[active_df['Source_Dataset'] == 'CMU']
        
    if not df_cmu_only.empty:
        # Calculate the decay using our standardized subset
        decay_df = calculate_muscle_memory_decay(df_cmu_only)
        
        # 2. Render the interactive Plotly line chart
        fig_decay = px.line(
            decay_df, 
            x='Attempt_Number', 
            y='Avg_Flight_Time',
            title="The '.tie5Roanl' Muscle Memory Curve",
            labels={'Attempt_Number': 'Password Attempt #', 'Avg_Flight_Time': 'Avg Flight Time (ms)'},
            color_discrete_sequence=['#00e676'] 
        )
        fig_decay.update_traces(line=dict(width=3))
        st.plotly_chart(fig_decay, use_container_width=True)
    else:
        st.warning("CMU dataset is required for the Muscle Memory baseline.")

    st.divider()

    # --- SECTION B: INTERFACE & BASELINE VARIANCE ---
    st.subheader("2. Universal Baseline Variance")
    st.markdown("Comparing average typing speeds across different datasets to establish macro-baselines. Notice how free-text typing (Aalto) differs from rigid password entry (CMU).")
    
   # 1. Calculate macro averages by slicing the Unified Master 'df'
    macro_stats = []
    comparison_targets = [
        ('CMU', "CMU (Passwords)"), 
        ('Aalto', "Aalto (Free Text)"), 
        ('KeyRecs', "KeyRecs (Mixed)")
    ]

    for source_key, display_name in comparison_targets:
        # Extract source-specific rows from our master 'active_df'
        subset = active_df[active_df['Source_Dataset'] == source_key]
        
        if not subset.empty and 'Flight_DD_ms' in subset.columns:
            # Use standardized columns: Flight_DD_ms and Hold_Time_ms
            clean_subset = subset[(subset['Flight_DD_ms'] > 0) & (subset['Flight_DD_ms'] < 1500)]
            
            avg_flight = clean_subset['Flight_DD_ms'].mean()
            avg_hold = clean_subset['Hold_Time_ms'].mean() if 'Hold_Time_ms' in clean_subset.columns else 0
            
            macro_stats.append({"Dataset": display_name, "Metric": "Avg Flight Time", "Value (ms)": avg_flight})
            macro_stats.append({"Dataset": display_name, "Metric": "Avg Hold Time", "Value (ms)": avg_hold})
            
    # 2. Render the Grouped Bar Chart
    if macro_stats:
        df_macro = pd.DataFrame(macro_stats)
        fig_bar = px.bar(
            df_macro,
            x="Dataset",
            y="Value (ms)",
            color="Metric",
            barmode="group",
            title="Average Keystroke Latencies by Context",
            color_discrete_sequence=['#29b6f6', '#ab47bc'] # Light Blue and Purple
        )
        fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- SECTION C: STATISTICAL BOUNDARIES (T-TEST) ---
    st.divider()
    st.subheader("3. Statistical Boundaries (Welch's T-Test)")
    st.markdown("Mathematically proving the boundary between Muscle Memory (CMU) and Cognitive Load (Aalto).")
    
    if not df_cmu.empty and not active_df.empty and 'Flight_DD_ms' in active_df.columns:
        # 1. Isolate clean flight time array for CMU
        cmu_array = df_cmu[(df_cmu['Flight_DD_ms'] > 0) & (df_cmu['Flight_DD_ms'] < 1000)]['Flight_DD_ms'].dropna()
        
        # 2. Extract ONLY Aalto from the unified Master Dataset
        aalto_subset = active_df[active_df['Source_Dataset'] == 'Aalto']
        
        if not aalto_subset.empty:
            aalto_array = aalto_subset[(aalto_subset['Flight_DD_ms'] > 0) & (aalto_subset['Flight_DD_ms'] < 1000)]['Flight_DD_ms'].dropna()
            # 2. Run Welch's T-Test (equal_var=False handles the massive sample size difference)
            t_stat, p_val = stats.ttest_ind(aalto_array, cmu_array, equal_var=False)
            
            # 3. Render the statistical readout
            t_col1, t_col2, t_col3, t_col4 = st.columns(4)
            t_col1.metric("CMU Mean (Muscle)", f"{cmu_array.mean():.1f} ms")
            t_col2.metric("Aalto Mean (Cognitive)", f"{aalto_array.mean():.1f} ms")
            t_col3.metric("T-Statistic", f"{t_stat:.2f}")
            
            # Format p-value output
            if p_val < 0.0001:
                p_display = "< 0.0001"
                significance = "Significant Boundary"
            else:
                p_display = f"{p_val:.4f}"
                significance = "Not Significant" if p_val >= 0.05 else "Significant"
                
            t_col4.metric("P-Value", p_display, delta=significance, delta_color="normal" if p_val < 0.05 else "inverse")
            
            # 4. State the findings
            if p_val < 0.05:
                penalty = aalto_array.mean() - cmu_array.mean()
                st.success(f"**Conclusion:** The data proves a statistically significant boundary between motor execution and cognitive generation. Free-text typing requires significantly more mental overhead, resulting in an average latency penalty of **{penalty:.1f} ms** per keystroke.")
            else:
                st.warning("**Conclusion:** No statistically significant difference found. Ensure dataset ingestion is fully complete.")

# ---------------------------------------------------------------------
# TAB 3: PHASES 3 & 4 (Taxonomy & Feature Engineering)
# ---------------------------------------------------------------------
with tab3:
    st.header("Phases 3 & 4: Error Categorization & Behavioral Features")
    st.write("Categorizing errors (Spatial vs. Cognitive) and mapping Fatigue U-Curves.")
    # TODO: Plotly correlation matrices and Rolling Variance (Burstiness) plots

    st.header("Taxonomy Exemplars: Spatial vs. Cognitive Errors")
    st.markdown("Visualizing the behavioral signatures of motor-execution slips versus top-down processing misfires.")
    
    # Ensure the taxonomy columns actually exist in the currently selected dataset
    if 'Is_Typo' in active_df.columns and 'Typo_Category' in active_df.columns:
        
        # 1. Isolate the two error streams using boolean masking
        df_spatial = active_df[active_df['Typo_Category'] == 'Spatial']
        df_cognitive = active_df[active_df['Typo_Category'] == 'Cognitive']
        
        # 2. Build the side-by-side metric comparison
        col_spat, col_cog = st.columns(2)
        
        with col_spat:
            st.success("### Category A: Spatial Error")
            st.markdown("**The Motor Slip:** The brain knew the correct sequence, but the motor execution missed the target by millimeters. Characterized by extremely fast realization and immediate correction.")
            if not df_spatial.empty:
                st.metric("Avg Reaction Time (Flight)", f"{df_spatial['Flight_DD_ms'].mean():.1f} ms")
                st.metric("Avg Hesitation (Hold)", f"{df_spatial['Hold_Time_ms'].mean():.1f} ms")
        
        with col_cog:
            st.error("### Category B: Cognitive Error")
            st.markdown("**The Mental Misfire:** The brain temporarily lost the syntactic or spelling thread. Characterized by a massive latency spike as the brain recalculates, often resulting in multiple backspaces.")
            if not df_cognitive.empty:
                st.metric("Avg Reaction Time (Flight)", f"{df_cognitive['Flight_DD_ms'].mean():.1f} ms")
                st.metric("Avg Hesitation (Hold)", f"{df_cognitive['Hold_Time_ms'].mean():.1f} ms")
        
        st.divider()
        
        # 3. Render the Latency Distribution Comparison
        st.subheader("Latency Signatures: Reaction Time Variance")
        st.markdown("Notice the tighter, faster clustering of Spatial errors compared to the wide, delayed spread of Cognitive errors.")
        
        valid_typos = active_df[active_df['Typo_Category'].isin(['Spatial', 'Cognitive'])]
        
        if not valid_typos.empty:
            # Filter massive outliers (pauses > 2 seconds) for a clean visual distribution
            plot_df = valid_typos[valid_typos['Flight_DD_ms'] < 2000]
            
            fig_box = px.box(
                plot_df, 
                x="Typo_Category", 
                y="Flight_DD_ms",
                color="Typo_Category",
                title="Reaction Time Disparity Between Error Streams",
                labels={"Typo_Category": "Error Origin", "Flight_DD_ms": "Flight Time (ms)"},
                color_discrete_map={"Spatial": "#00e676", "Cognitive": "#ff5252"}
            )
            
            fig_box.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_showgrid=False)
            st.plotly_chart(fig_box, use_container_width=True)
            
    else:
        st.info("Run the Phase 1 Taxonomy pipeline to categorize errors before viewing exemplars.")

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
