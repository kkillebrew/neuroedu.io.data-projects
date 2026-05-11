# =====================================================================
# STREAMLIT APP: pages/4_tech_in_edu_app.py (The View)
# PURPOSE: Interactive visualization of the EdTech "Knowledge Gap".
# STRICT DECOUPLING: Only UI components and Plotly visualizations live here.
# =====================================================================
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# --- PATH CONFIGURATION ---
# MATLAB Bridge: This is the equivalent of addpath('../'). 
# It allows this script (inside /pages) to see the root folders (/loaders).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders.tech_in_edu_loader import load_edtech_master, calculate_knowledge_gap, calculate_correlations
from data_projects_sidebar import apply_global_settings, render_sidebar

# ---------------------------------------------------------------------
# GLOBAL SETTINGS & SIDEBAR ROUTING
# ---------------------------------------------------------------------
# Applying the standardized CSS and Hub configuration
apply_global_settings("Neuro-Edu | The EdTech Paradox")
render_sidebar()

# ---------------------------------------------------------------------
# DATA INGESTION (Via Loader)
# ---------------------------------------------------------------------
base_dir = os.path.join(os.path.dirname(__file__), '..', 'documents')

try:
    # We strictly load the cached memory block. No raw processing occurs here.
    df_raw = load_edtech_master(base_dir)
    df = calculate_knowledge_gap(df_raw)
except Exception as e:
    st.error(f"Failed to load datasets. Please ensure 'tech_in_ed_etl.py' has run. Error: {e}")
    st.stop()

# ---------------------------------------------------------------------
# UI LAYOUT & HEADER
# ---------------------------------------------------------------------
st.title("🎓 The EdTech Paradox: Mastery vs. Overload")
st.markdown("""
Has technology made it easier to learn, or has it simply increased the sheer volume of information we are expected to retain? 
By merging World Bank macroeconomic data with global student assessments, we can visualize the expanding **Knowledge Gap**.
""")

# MATLAB Bridge: st.tabs() replaces uitabgroup. It is an incredibly clean way 
# to organize the DOM without forcing the user to navigate to new pages.
tab1, tab2, tab3, tab4 = st.tabs([
    "🗃️ 1. Master Schema", 
    "📈 2. Macro Divergence", 
    "🔬 3. Acceleration Mechanics", 
    "🕹️ 4. The Sandbox"
])

# ---------------------------------------------------------------------
# TAB 1: DATASET OVERVIEW
# ---------------------------------------------------------------------
with tab1:
    st.header("Phase 1: Ingested Master Schema")
    st.write("This unified `.parquet` artifact is the result of a massive chunked aggregation pipeline, merging World Bank API data with 16GB+ of Kaggle educational proxies.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Macro Rows", len(df))
    with col2:
        st.metric("Tracked Nations", df['Country'].nunique())
        
    st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 2: EXPLORATORY ANALYSIS (The Divergence Curve)
# ---------------------------------------------------------------------
with tab2:
    st.header("Phase 2: The Expanding Knowledge Gap")
    st.write("Comparing the growth of curriculum requirements against biological learning efficiency.")
    
    selected_country = st.selectbox("Select Benchmark Region:", df['Country'].unique(), index=0, key="tab2_country")
    df_filtered = df[df['Country'] == selected_country]
    
    # VISUALIZATION DESIGN MENTOR: 
    # We use contrasting hues (Red for required load, Green for human capacity). 
    # This guides the user's eye to the 'delta' (the gap) between the two lines.
    fig_gap = px.line(
        df_filtered, 
        x='Year', 
        y=['Curriculum_Complexity_Index', 'Learning_Efficiency_Score'],
        title=f"Curriculum Complexity vs. Efficiency ({selected_country})",
        labels={'value': 'Index Score', 'variable': 'Metric'},
        color_discrete_map={
            'Curriculum_Complexity_Index': '#EF553B', 
            'Learning_Efficiency_Score': '#00CC96'
        }
    )
    
    # Adding attention cuing: The iPhone launch marks a massive shift in cognitive offloading.
    fig_gap.add_vline(x=2007, line_width=2, line_dash="dash", line_color="gray", annotation_text="Smartphone Era")
    fig_gap.update_layout(hovermode="x unified") # Shows all values on a single hover line
    
    st.plotly_chart(fig_gap, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 3: TAXONOMY & ENGINEERING (Velocity & Correlation)
# ---------------------------------------------------------------------
with tab3:
    st.header("Phase 3: Technology Acceleration & Gap Velocity")
    st.write("Does the rate of technology adoption directly correlate with the widening of the Knowledge Gap?")
    
    target_country = st.selectbox("Select Region for Correlation:", df['Country'].unique(), index=0, key="tab3_country")
    
    col_chart, col_matrix = st.columns([2, 1])
    
    with col_chart:
        df_target = df[df['Country'] == target_country]
        
        # VISUALIZATION DESIGN MENTOR: 
        # A dual-axis chart is crucial here because Penetration is a percentage (0-100) 
        # while Gap Velocity is a differential index. Plotly's graph_objects (go) handles this beautifully.
        fig_vel = go.Figure()
        
        fig_vel.add_trace(go.Bar(
            x=df_target['Year'], 
            y=df_target['Tech_Acceleration'],
            name="Tech Adoption Acceleration (%)",
            marker_color='rgba(52, 152, 219, 0.6)'
        ))
        
        fig_vel.add_trace(go.Scatter(
            x=df_target['Year'], 
            y=df_target['Gap_Velocity'],
            name="Knowledge Gap Velocity",
            mode='lines+markers',
            line=dict(color='#E67E22', width=3),
            yaxis='y2' # Assigns this trace to the secondary Y-axis
        ))
        
        fig_vel.update_layout(
            title=f"Acceleration Mechanics ({target_country})",
            yaxis=dict(title="Tech Acceleration (%)"),
            yaxis2=dict(title="Gap Velocity", overlaying='y', side='right'),
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_vel, use_container_width=True)
        
    with col_matrix:
        st.subheader("Pearson Correlation")
        st.write("MATLAB equivalent of `corrcoef()` mapping the latent variables.")
        corr_matrix = calculate_correlations(df, target_country)
        
        # Displaying the matrix with a heatmap gradient for scannability
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}"))

# ---------------------------------------------------------------------
# TAB 4: THE SANDBOX (Streamlit Fragment Integration)
# ---------------------------------------------------------------------
with tab4:
    st.header("Phase 4: Theoretical Sandbox")
    st.write("Adjust the theoretical limits of human working memory. If we introduce neural-link technology or perfect AI-assisted recall, how does the gap close?")
    
    # ⚡ STREAMLIT FRAGMENT (Crucial for Lag-Free UI)
    # MATLAB Bridge: This is a localized callback. When the slider moves, ONLY this 
    # specific block of code re-runs. It prevents the massive Plotly charts in Tabs 1-3 
    # from unnecessarily re-rendering and consuming CPU cycles.
    @st.fragment
    def cognitive_sandbox_ui():
        # Baseline data for the sandbox
        df_sandbox = df[df['Country'] == 'USA'].copy()
        
        ai_assist_factor = st.slider("AI Cognitive Offload Factor", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
        
        # Apply theoretical math to the baseline efficiency
        df_sandbox['Theoretical_Efficiency'] = df_sandbox['Learning_Efficiency_Score'] * ai_assist_factor
        
        fig_sandbox = px.line(
            df_sandbox, 
            x='Year', 
            y=['Curriculum_Complexity_Index', 'Theoretical_Efficiency'],
            labels={'value': 'Index Score', 'variable': 'Metric'},
            color_discrete_map={
                'Curriculum_Complexity_Index': '#EF553B', 
                'Theoretical_Efficiency': '#9B59B6' # Purple for theoretical
            }
        )
        fig_sandbox.update_layout(title="Simulated Future: AI-Augmented Learning")
        st.plotly_chart(fig_sandbox, use_container_width=True)
        
    cognitive_sandbox_ui()