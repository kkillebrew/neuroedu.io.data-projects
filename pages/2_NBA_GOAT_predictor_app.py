"""
=============================================================================
MODULE: pages/2_nba_goat_app.py (NBA GOAT Predictor UI)
AUTHOR: Kyle W. Killebrew, PhD
VERSION: 1.0 (Data Science Micro-Frontend Hub)
DESCRIPTION: 
    The Streamlit frontend (View) for the NBA GOAT Predictor spoke. 
    Renders interactive Plotly charts, handles custom sidebar routing, 
    and fetches cached data from the nba_goat_loader backend.
=============================================================================
"""

import sys
import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import scipy.stats as stats

# -------------------------------------------------------------------
# SYSTEM PATH ROUTING
# -------------------------------------------------------------------
# This allows the 'pages' directory to see and import from the 'loaders' directory.
# MATLAB Analogy: Equivalent to `addpath('../loaders')` to make backend functions accessible.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loaders.NBA_GOAT_predictor_loader import (
    PLAYERS, get_player_colors, load_and_filter_raw_data, 
    calculate_career_baselines, get_awards_hardware, 
    analyze_longevity_vs_peak, run_scoring_segment_analysis,
    get_era_adjusted_stats
)

# -------------------------------------------------------------------
# UI CONFIGURATION & CUSTOM SIDEBAR (The "View")
# -------------------------------------------------------------------
st.set_page_config(page_title="NBA GOAT Predictor | Neuro-Edu", page_icon="🏀", layout="wide")

# --- CUSTOM CSS FOR SIDEBAR BUTTON ---
st.markdown("""
    <style>
    .return-gate {
        background-color: #0f172a; color: white !important; padding: 12px;
        border-radius: 8px; text-align: center; font-weight: bold; 
        text-decoration: none; display: block; font-size: 1rem;
        transition: background-color 0.3s ease; border: 1px solid #334155;
    }
    .return-gate:hover { background-color: #1e293b; }
    </style>
""", unsafe_allow_html=True)

# --- UNIFIED SIDEBAR ---
# MATLAB Analogy: Building static UI elements in App Designer that exist globally on the left panel.
with st.sidebar:
    # 1. Hide default Streamlit navigation
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] {display: none !important;}
        </style>
    """, unsafe_allow_html=True)

    # 2. Return to Career Hub
    st.markdown("""
        <div style="padding-bottom: 1rem;">
            <a href="https://data-projects.neuro-edu.io" style="text-decoration: none; color: #1f77b4; font-weight: bold;">
                &larr; Return to Data-Projects Hub
            </a>
        </div>
    """, unsafe_allow_html=True)

    # 3. Data Projects Navigation
    st.divider()
    st.subheader("🧭 Data Projects")
    st.page_link("data_projects_app.py", label="Data Hub Home", icon="🏠")
    st.page_link("pages/1_oil_predictor_app.py", label="Macro Oil Predictor", icon="🛢️")
    # Note: Activated the NBA link since this is the actual file! 
    st.page_link("pages/2_NBA_GOAT_predictor_app.py", label="NBA GOAT Predictor", icon="🏀")
    st.markdown("💻 Tech in Education *(Coming Soon)*")
    
    # 4. Professional Presence
    st.divider()
    st.subheader("🌐 Presence")
    st.markdown("🔬 [ORCID Profile](https://orcid.org/0000-0002-9662-9844)")
    st.markdown("📈 [Google Scholar](https://scholar.google.com/citations?user=y-2G-voAAAAJ&hl=en)")
    st.markdown("💼 [LinkedIn Profile](https://www.linkedin.com/in/kylewkillebrew/)")
    st.markdown("💻 [GitHub Profile](https://github.com/kkillebrew)")

    st.divider()
    st.caption("Data Science Portfolio | 2026")

# -------------------------------------------------------------------
# DATA CACHING & INITIALIZATION
# -------------------------------------------------------------------
# MATLAB Analogy: Using @st.cache_data is like saving variables to a `.mat` workspace 
# so we don't have to rerun the heavy data processing functions every time the user 
# clicks a filter or changes a tab in the UI. It keeps the web app extremely fast.
@st.cache_data
def load_all_dashboard_data():
    df_goat = load_and_filter_raw_data()
    df_career, df_clutch = calculate_career_baselines(df_goat)
    df_awards = get_awards_hardware()
    df_longevity = analyze_longevity_vs_peak(df_goat)
    bin_pct, significant_findings = run_scoring_segment_analysis(df_goat)
    df_era = get_era_adjusted_stats(df_goat)
    colors = get_player_colors()
    return df_goat, df_career, df_clutch, df_awards, df_longevity, bin_pct, significant_findings, df_era, colors

# Display a loading spinner while the backend fetches data
with st.spinner("Crunching historical NBA game logs..."):
    df_goat, df_career, df_clutch, df_awards, df_longevity, bin_pct, sig_findings, df_era, player_colors = load_all_dashboard_data()

# -------------------------------------------------------------------
# MAIN APP LAYOUT (Interactive Controls on Main Page)
# -------------------------------------------------------------------
st.title("🏀 The Ultimate NBA GOAT Analyzer")
st.write("An exploratory data science dashboard comparing peak dominance, longevity, and statistical consistency of 10 legendary players.")
st.divider()

# Interactive Filter: Allows users to subset which players they want to compare
# MATLAB Analogy: Connecting a drop-down UI Component callback to update data.
selected_players = st.multiselect("Select Players to Compare:", PLAYERS, default=PLAYERS)

# Create layout tabs for clean MVC separation in the UI
tab1, tab2, tab3, tab4 = st.tabs(["📊 Career Baselines", "🏆 Hardware & Clutch", "📈 Consistency & Variance", "⏳ Longevity vs Peak"])

# --- TAB 1: Baseline Stats ---
with tab1:
    st.subheader("Real Career Averages (Grouped by Stat)")
    # Filter the dataframe based on the user's multiselect input
    filtered_career = df_career[df_career['Player'].isin(selected_players)]
    df_melt_career = filtered_career.melt(id_vars=['Player'], value_vars=['PTS', 'TRB', 'AST'], var_name='Stat', value_name='Value')
    
    # Plotly Express automatically builds interactive grouped bar charts
    # MATLAB Analogy: bar(categorical(x), y, 'grouped')
    fig1 = px.bar(df_melt_career, x='Stat', y='Value', color='Player', barmode='group', color_discrete_map=player_colors)
    st.plotly_chart(fig1, use_container_width=True)

    st.divider()
    st.subheader("Era-Adjusted Dominance (Z-Scores vs Peers)")
    st.markdown("Raw stats can be deceiving. Here we adjust for the 'Pace' of different eras by standardizing stats (Z-scores) against a player's contemporaries. *Bubble size indicates the pace of their era.*")
    
    # Filter based on multiselect
    filtered_era = df_era[df_era['Player'].isin(selected_players)]
    
    # Create the scatter plot (MATLAB Analogy: scatter() with varying marker sizes)
    fig_era = px.scatter(
        filtered_era, x='Scoring_Z_Score', y='Rebound_Z_Score', 
        text='Player', size='Era_Pace', color='Player',
        color_discrete_map=player_colors # Keeping our strict colors!
    )
    
    # Formatting the quadrant lines and labels
    fig_era.update_traces(textposition='top center', marker=dict(line=dict(width=1, color='Black')))
    fig_era.add_hline(y=1.5, line_dash="dot", line_color="gray")
    fig_era.add_vline(x=1.5, line_dash="dot", line_color="gray")
    fig_era.update_layout(
        xaxis_title="Scoring Dominance (Std Devs above era average)",
        yaxis_title="Rebounding Dominance (Std Devs above era average)",
        showlegend=False
    )
    st.plotly_chart(fig_era, use_container_width=True)

# --- TAB 2: Hardware & Clutch ---
with tab2:
    # Use columns to position graphs side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("The Trophy Cabinet")
        filtered_awards = df_awards[df_awards['Player'].isin(selected_players)]
        df_melt_awards = filtered_awards.melt(id_vars=['Player'], value_vars=['MVPs', 'Rings', 'Finals_MVPs', 'DPOY'], var_name='Award', value_name='Count')
        fig2 = px.bar(df_melt_awards, x='Award', y='Count', color='Player', barmode='group', color_discrete_map=player_colors)
        st.plotly_chart(fig2, use_container_width=True)
        
    with col2:
        st.subheader("The Clutch Factor (Reg Season vs Playoffs)")
        filtered_clutch = df_clutch[df_clutch['Player'].isin(selected_players)]
        df_melt_clutch = filtered_clutch.melt(id_vars=['Player'], value_vars=['Reg_Season_PTS', 'Finals_PTS'], var_name='Context', value_name='PPG')
        fig3 = px.bar(df_melt_clutch, x='Context', y='PPG', color='Player', barmode='group', color_discrete_map=player_colors)
        st.plotly_chart(fig3, use_container_width=True)

# --- TAB 3: Consistency & Variance ---
with tab3:
    st.subheader("Smoothed Scoring Distributions (KDE)")
    # Using Graph Objects (go.Figure) for fine-grained control over mathematical overlays
    fig_dist = go.Figure()
    x_range = np.linspace(0, 100, 300)
    for p in selected_players:
        pts = df_goat[df_goat['Player'] == p]['points'].dropna()
        if len(pts) > 1:
            # Kernel Density Estimation (MATLAB Analogy: ksdensity(pts))
            kde = stats.gaussian_kde(pts)
            # Add a smoothed line trace to the figure
            fig_dist.add_trace(go.Scatter(x=x_range, y=kde(x_range), mode='lines', name=p, line=dict(color=player_colors[p], width=2.5)))
    fig_dist.update_layout(xaxis_title="Points in a Single Game", yaxis_title="Probability Density", hovermode="x unified")
    st.plotly_chart(fig_dist, use_container_width=True)
    
    st.divider()
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Scoring Segment Breakdown")
        filtered_bin = bin_pct[bin_pct.index.isin(selected_players)]
        df_melt_bins = filtered_bin.reset_index().melt(id_vars='Player', var_name='Points Range', value_name='Percentage')
        
        # We use a sequential palette (Plasma) here because the ranges are ordered (<10 to 50+)
        fig_bins = px.bar(df_melt_bins, x='Player', y='Percentage', color='Points Range', color_discrete_sequence=px.colors.sequential.Plasma, text_auto='.1f')
        fig_bins.update_layout(yaxis_title="Percentage of Career Games (%)", barmode='stack')
        st.plotly_chart(fig_bins, use_container_width=True)
        
    with col2:
        st.subheader("🔬 Statistically Significant Discoveries")
        st.markdown("*(Chi-Square with Bonferroni correction, p < 0.05)*")
        
        # Iterate through the top statistical discoveries generated by our backend model
        for f in [x for x in sig_findings if x['Player'] in selected_players][:5]:
            # Use st.success for visual highlight boxes
            st.success(f"**{f['Player']}** had significantly **{f['Direction']}** games scoring {f['Bin']} pts. ({f['Player_%']:.1f}% vs {f['Rest_%']:.1f}%)")

# --- TAB 4: Longevity vs Peak ---
with tab4:
    st.subheader("Absolute Totals vs. Per-Season Accumulation")
    filtered_longevity = df_longevity[df_longevity['Player'].isin(selected_players)]
    
    # Facet grid instantly splits data into synchronized subplots (MATLAB Analogy: tiledlayout or subplot loops)
    fig_long = px.bar(filtered_longevity, x='Player', y='Value', color='Player',
                      facet_row='Measurement', facet_col='Stat',
                      color_discrete_map=player_colors)
    
    # We must explicitly decouple the Y-axes because total 'Points' dwarf 'Assists' visually
    fig_long.update_yaxes(matches=None)
    fig_long.update_layout(showlegend=False, height=700)
    st.plotly_chart(fig_long, use_container_width=True)