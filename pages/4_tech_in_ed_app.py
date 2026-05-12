# =====================================================================
# STREAMLIT APP: pages/4_tech_in_edu_app.py (The View)
# PURPOSE: Interactive visualization of the 22-Year EdTech "Knowledge Gap".
# STRICT DECOUPLING: Only UI components and Plotly visualizations live here.
# =====================================================================
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# --- PATH CONFIGURATION ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders.tech_in_ed_loader import load_edtech_master, calculate_knowledge_gap, calculate_correlations
from data_projects_sidebar import apply_global_settings, render_sidebar

# ---------------------------------------------------------------------
# GLOBAL SETTINGS & SIDEBAR ROUTING
# ---------------------------------------------------------------------
apply_global_settings("Neuro-Edu | The EdTech Paradox")
render_sidebar()

# ---------------------------------------------------------------------
# DATA INGESTION
# ---------------------------------------------------------------------
base_dir = os.path.join(os.path.dirname(__file__), '..', 'documents')

try:
    df_raw = load_edtech_master(base_dir)
    df = calculate_knowledge_gap(df_raw)
except Exception as e:
    st.error(f"Failed to load datasets. Ensure 'tech_in_ed_etl.py' has run. Error: {e}")
    st.stop()

# ---------------------------------------------------------------------
# UI LAYOUT & HEADER
# ---------------------------------------------------------------------
st.title("🎓 The EdTech Paradox: Mastery vs. Overload")
st.markdown("""
By merging 22 years of PISA assessments (2000-2022) with macroeconomic indicators, we examine 
if technology has closed the gap between curriculum complexity and biological learning efficiency.
""")

tab1, tab2, tab3, tab4 = st.tabs([
    "🗃️ 1. Master Timeline", 
    "📈 2. Subject Divergence", 
    "🔬 3. Multi-Domain Velocity", 
    "🕹️ 4. The Sandbox"
])

# ---------------------------------------------------------------------
# TAB 1: DATASET OVERVIEW (22-Year Inventory)
# ---------------------------------------------------------------------
with tab1:
    st.header("Phase 1: The Unified PISA Timeline")
    st.write("Current dataset spans from the first PISA cycle in 2000 to the latest 2022 dataset.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sample Size", f"~{len(df) * 5} Students") # Estimated based on 20% sample
    with col2:
        st.metric("Timeline Span", f"{df['Year'].min()} - {df['Year'].max()}")
    with col3:
        st.metric("Macro Metrics", len(df.columns))
        
    st.dataframe(df.sort_values(['Country', 'Year']), use_container_width=True)

# ---------------------------------------------------------------------
# TAB 2: SUBJECT DIVERGENCE (Math, Science, Reading)
# ---------------------------------------------------------------------
with tab2:
    st.header("Phase 2: Subject-Specific Divergence")
    
    selected_country = st.selectbox("Select Benchmark Region:", sorted(df['Country'].unique()), index=0, key="tab2_country")
    df_filtered = df[df['Country'] == selected_country].sort_values('Year')
    
    # Selection for which score to compare against complexity
    score_to_view = st.radio("Select Cognitive Domain:", 
                             ["Learning_Efficiency_Score", "Math_Score", "Reading_Score", "Science_Score"],
                             horizontal=True)

    fig_gap = px.line(
        df_filtered, 
        x='Year', 
        y=['Curriculum_Complexity_Index', score_to_view],
        title=f"Complexity vs. {score_to_view.replace('_', ' ')} ({selected_country})",
        labels={'value': 'Index Score', 'variable': 'Metric'},
        color_discrete_map={
            'Curriculum_Complexity_Index': '#EF553B', 
            score_to_view: '#00CC96'
        },
        markers=True
    )
    
    # Timeline Milestone markers
    fig_gap.add_vline(x=2007, line_width=2, line_dash="dash", line_color="gray", annotation_text="Mobile Revolution")
    fig_gap.add_vline(x=2020, line_width=2, line_dash="dot", line_color="red", annotation_text="Remote Shift")
    
    st.plotly_chart(fig_gap, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 3: ACCELERATION & ERA MODULES (The High-Tech View)
# ---------------------------------------------------------------------
with tab3:
    st.header("Phase 3: Digital Literacy & Correlation")
    
    target_country = st.selectbox("Select Region for Analysis:", sorted(df['Country'].unique()), index=0, key="tab3_country")
    df_target = df[df['Country'] == target_country].sort_values('Year')
    
    col_chart, col_matrix = st.columns([2, 1])
    
    with col_chart:
        # Dual-axis chart comparing Internet Penetration to the Knowledge Gap
        fig_vel = go.Figure()
        
        fig_vel.add_trace(go.Bar(
            x=df_target['Year'], 
            y=df_target['Internet_Penetration'] if 'Internet_Penetration' in df_target.columns else [0],
            name="Internet Penetration (%)",
            marker_color='rgba(52, 152, 219, 0.3)'
        ))
        
        fig_vel.add_trace(go.Scatter(
            x=df_target['Year'], 
            y=df_target['Knowledge_Gap'],
            name="The Knowledge Gap",
            mode='lines+markers',
            line=dict(color='#E67E22', width=4),
            yaxis='y2'
        ))
        
        fig_vel.update_layout(
            title=f"Tech Saturation vs. Knowledge Gap ({target_country})",
            yaxis=dict(title="Internet Penetration (%)", range=[0, 100]),
            yaxis2=dict(title="Gap Index", overlaying='y', side='right'),
            legend=dict(x=0.01, y=0.99),
            hovermode="x unified"
        )
        st.plotly_chart(fig_vel, use_container_width=True)
        
    with col_matrix:
        st.subheader("Correlation Matrix")
        # Surfacing our new 2003/2009 variables
        corr_matrix = calculate_correlations(df, target_country)
        st.dataframe(corr_matrix.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"))
        st.caption("Note: ERA Digital Reading and Problem Solving scores appear in specific cycle years.")

# ---------------------------------------------------------------------
# TAB 4: THE SANDBOX (Theoretical AI Augmentation)
# ---------------------------------------------------------------------
with tab4:
    st.header("Phase 4: Theoretical Sandbox")
    st.write("Simulate the impact of AI-assisted recall on the 2022-2025 Knowledge Gap.")
    
    @st.fragment
    def cognitive_sandbox_ui():
        # Baseline for USA
        df_sandbox = df[df['Country'] == 'USA'].copy().sort_values('Year')
        
        ai_factor = st.slider("AI Augmentation Factor (Cognitive Offload)", 0.5, 3.0, 1.0, 0.1)
        
        # Calculate theoretical future
        df_sandbox['Theoretical_Efficiency'] = df_sandbox['Learning_Efficiency_Score'] * ai_factor
        
        fig_sandbox = px.line(
            df_sandbox, 
            x='Year', 
            y=['Curriculum_Complexity_Index', 'Theoretical_Efficiency'],
            labels={'value': 'Index Score', 'variable': 'Metric'},
            color_discrete_map={
                'Curriculum_Complexity_Index': '#EF553B', 
                'Theoretical_Efficiency': '#9B59B6'
            },
            title="Future Projection: Neural Augmentation vs. Complexity"
        )
        st.plotly_chart(fig_sandbox, use_container_width=True)
        
        if ai_factor > 1.5:
            st.success("Theoretical Neural Augmentation has closed the Knowledge Gap.")
        elif ai_factor < 1.0:
            st.warning("Cognitive decline (digital distraction) is widening the Gap.")
            
    cognitive_sandbox_ui()