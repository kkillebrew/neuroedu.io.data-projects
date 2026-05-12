# =====================================================================
# STREAMLIT APP: pages/4_tech_in_edu_app.py (The View)
# =====================================================================
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# --- PATH CONFIGURATION ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders.tech_in_ed_loader import (
    load_edtech_master, 
    calculate_knowledge_gap, 
    calculate_correlations, 
    get_pisa_grid_samples,          # FIXED: Was get_pisa_snapshots
    get_benchmark_comparison_data   # ADDED: Required for Phase 2 and 3
)
from data_projects_sidebar import apply_global_settings, render_sidebar

# --- SETTINGS & SIDEBAR ---
st.set_page_config(layout="wide") # Required for a clean 4-column grid
apply_global_settings("Neuro-Edu | The EdTech Paradox")
render_sidebar()

# --- DATA INGESTION ---
base_dir = os.path.join(os.path.dirname(__file__), '..', 'documents')

try:
    df_raw = load_edtech_master(base_dir)
    df = calculate_knowledge_gap(df_raw)
except Exception as e:
    st.error(f"Failed to load datasets. Error: {e}")
    st.stop()

# --- HEADER & BLURB ---
st.title("🎓 The EdTech Paradox: Mastery vs. Overload")
st.markdown("""
### Analyzing 22 Years of Global Educational Evolution
Does the rapid adoption of classroom technology correlate with higher student proficiency, 
or has the digital age simply increased the complexity of what students are expected to manage? 
By fusing **OECD PISA** datasets with **World Bank** economic indicators, we visualize the 
shifting boundary between human cognitive capacity and academic demand.
""")

tab1, tab2, tab3, tab4 = st.tabs([
    "🗃️ 1. Data Display & Confirmation", 
    "📈 2. Subject Divergence", 
    "🔬 3. Multi-Domain Velocity", 
    "🕹️ 4. The Sandbox"
])

# ---------------------------------------------------------------------
# TAB 1: DATA DISPLAY & CONFIRMATION
# ---------------------------------------------------------------------
with tab1:
    st.header("Pipeline Integrity Verification")
    st.write("""
    **Verification Strategy:** Randomly sampling 1 student record from 10 benchmark nations 
    across 4 continents. This confirms that all metadata columns (Scores, Tech Usage, and Identifiers) 
    have successfully mapped into the master schema.
    """)

    ################################
    #      1: Raw data tables      #
    ################################
    # Get randomized samples (1 row each for 10 target countries)
    snapshots = get_pisa_grid_samples(df, rows_per_year=10)
    years = sorted(snapshots.keys())

    # 4x2 Grid Implementation
    for row_idx in range(2): 
        cols = st.columns(4)
        for col_idx in range(4):
            year_idx = (row_idx * 4) + col_idx
            if year_idx < len(years):
                year = years[year_idx]
                with cols[col_idx]:
                    st.markdown(f"#### 📅 {year} Dataset")
                    
                    # Displaying ALL columns with a height of 160 (approx 3.5 rows)
                    # This forces a vertical scrollbar for the remaining 7 countries
                    st.dataframe(
                        snapshots[year],
                        hide_index=True,
                        use_container_width=True,
                        height=160
                    )

    ################################
    #  2: Raw Data Distributions   #
    ################################
    # Filter data
    df_bench = get_benchmark_comparison_data(df)
    
    # DICTIONARY MAP: User-friendly labels -> Actual DataFrame Columns
    subject_map = {
        'Math (Learning Efficiency)': 'Learning_Efficiency_Score',
        'Reading Proficiency': 'Reading_Proficiency_Score',
        'Science Proficiency': 'Science_Proficiency_Score'
    }

    st.header("Phase 2: Distribution Trends (Box & Whisker)")
    selected_label = st.selectbox("Select Subject for Distribution:", list(subject_map.keys()))
    selected_sub = subject_map[selected_label] # Gets the actual column name

    # Row 1: USA (Top Row)
    row1_col = st.columns(1)[0]
    with row1_col:
        fig_usa = px.box(df_bench[df_bench['Country'] == 'USA'], 
                         x='Year', y=selected_sub, points="all",
                         title=f"USA {selected_label} Distribution Over Time",
                         color_discrete_sequence=['#EF553B'])
        st.plotly_chart(fig_usa, use_container_width=True)

    # Rows 2 & 3: The other 4 countries (2x2 grid)
    countries_left = ['JPN', 'DEU', 'ARG', 'JOR']
    grid_cols = st.columns(2)

    for i, country in enumerate(countries_left):
        col_idx = i % 2
        with grid_cols[col_idx]:
            fig_bench = px.box(df_bench[df_bench['Country'] == country], 
                               x='Year', y=selected_sub, points="all",
                               title=f"{country} {selected_label} Distribution",
                               color_discrete_sequence=['#00CC96'])
            st.plotly_chart(fig_bench, use_container_width=True)

    ################################
    #  3: Longitudinal Tech Trends #
    ################################
    st.header("Phase 3: Longitudinal Subject Convergence")
    st.write("Comparing Math, Science, and Reading trends within each benchmark country. Shaded regions indicate the projected margin of influence from Internet Penetration.")

    # Define strict colors so subjects are instantly recognizable across all plots
    subject_colors = {
        'Math (Learning Efficiency)': '#EF553B',  # Red
        'Science Proficiency': '#00CC96',         # Green
        'Reading Proficiency': '#636EFA'          # Blue
    }

    bench_list = ['USA', 'JPN', 'DEU', 'ARG', 'JOR']

    # To keep the UI clean and prevent endless scrolling, we put the 5 countries into Streamlit Tabs
    country_tabs = st.tabs([f"🌍 {c}" for c in bench_list])

    # Iterate through the countries and generate a plot for each tab
    for i, country in enumerate(bench_list):
        with country_tabs[i]:
            fig_line = go.Figure()
            df_c = df_bench[df_bench['Country'] == country].sort_values('Year')
            
            # Now iterate through the 3 subjects to plot them on the SAME graph
            for label, sub_col in subject_map.items():
                
                # 1. Calculate Shaded Region (Tech Benefit Variance)
                tech_variance = df_c['INTERNET_PENETRATION'].fillna(0) / 500
                upper_bound = df_c[sub_col] * (1 + tech_variance)
                lower_bound = df_c[sub_col] * (1 - tech_variance)
                
                # 2. Add the 'Shade' trace for this specific subject
                fig_line.add_trace(go.Scatter(
                    x=pd.concat([df_c['Year'], df_c['Year'][::-1]]),
                    y=pd.concat([upper_bound, lower_bound[::-1]]),
                    fill='toself',
                    fillcolor=subject_colors[label],
                    opacity=0.15,
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f"{label} Tech Range",
                    showlegend=False
                ))
                
                # 3. Add the Main Score Line for this specific subject
                fig_line.add_trace(go.Scatter(
                    x=df_c['Year'], 
                    y=df_c[sub_col],
                    name=label,
                    line=dict(color=subject_colors[label], width=3),
                    mode='lines+markers'
                ))

            # Style the layout for this country's specific graph
            fig_line.update_layout(
                title=f"Cognitive Domain Divergence in {country} (2000-2022)",
                xaxis_title="Year",
                yaxis_title="PISA Score",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_line, use_container_width=True)

# ---------------------------------------------------------------------
# REMAINING TABS (Keep your existing logic here)
# ---------------------------------------------------------------------
with tab2:
    st.header("Phase 2: Subject-Specific Divergence")
    # ... (Your existing Tab 2 logic)

with tab3:
    st.header("Phase 3: Digital Literacy & Correlation")
    # ... (Your existing Tab 3 logic)

with tab4:
    st.header("Phase 4: Theoretical Sandbox")
    # ... (Your existing Tab 4 logic)