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
    get_era_adjusted_stats, get_radar_scaled_stats, get_dumbbell_longevity_peak
)

# --- GLOBAL PLOTLY CONFIG (Mobile Scroll Lock) ---
PLOTLY_CONFIG = {
    'scrollZoom': False, 
    'displayModeBar': False, # Hides the messy floating toolbar
    'staticPlot': False
}

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
    df_radar = get_radar_scaled_stats(df_career)
    df_dumbbell = get_dumbbell_longevity_peak(df_goat)
    colors = get_player_colors()
    return df_goat, df_career, df_clutch, df_awards, df_longevity, bin_pct, significant_findings, df_era, df_radar, df_dumbbell, colors

# Display a loading spinner while the backend fetches data
with st.spinner("Crunching historical NBA game logs..."):
    df_goat, df_career, df_clutch, df_awards, df_longevity, bin_pct, sig_findings, df_era, df_radar, df_dumbbell, player_colors = load_all_dashboard_data()

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
    st.subheader("The 'Shape' of Greatness (Radar Charts)")
    st.markdown("""
        How do these players visually compare across major statistical categories? 
        *Stats are min-max scaled (0-100) relative to the highest performer. (Note: A higher TOV score means more turnovers, which is negative).*
    """)
    
    # Side-by-side columns for the Radars
    radar_col1, radar_col2 = st.columns(2)
    
    # 1. Offensive Radar
    with radar_col1:
        fig_off = go.Figure()
        cat_off = ['PTS', 'AST', 'PLUS_MINUS', 'FG_PCT']
        cat_off_loop = cat_off + [cat_off[0]] # Close the loop
        
        for p in selected_players:
            player_data = df_radar[df_radar['Player'] == p].iloc[0]
            values = player_data[cat_off].tolist() + [player_data[cat_off[0]]]
            fig_off.add_trace(go.Scatterpolar(r=values, theta=cat_off_loop, name=p, fill='toself', line=dict(color=player_colors[p], width=2), opacity=0.5))
        fig_off.update_layout(title="Offensive Shape", polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=450)
        st.plotly_chart(fig_off, use_container_width=True, config=PLOTLY_CONFIG)

    # 2. Defensive Radar
    with radar_col2:
        fig_def = go.Figure()
        cat_def = ['TRB', 'STL', 'BLK', 'TOV']
        cat_def_loop = cat_def + [cat_def[0]]
        
        for p in selected_players:
            player_data = df_radar[df_radar['Player'] == p].iloc[0]
            values = player_data[cat_def].tolist() + [player_data[cat_def[0]]]
            fig_def.add_trace(go.Scatterpolar(r=values, theta=cat_def_loop, name=p, fill='toself', line=dict(color=player_colors[p], width=2), opacity=0.5))
        fig_def.update_layout(title="Defensive Shape", polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, height=450)
        st.plotly_chart(fig_def, use_container_width=True, config=PLOTLY_CONFIG)

    st.divider()
    st.subheader("Era-Adjusted Dominance (Z-Scores vs Peers)")
    st.markdown("Adjusted for era pacing. **Bubble size indicates Pace.** We have exponentially scaled the bubble size so the massive pacing differences between eras (e.g., Wilt's 1960s vs Kobe's 2000s) are drastically apparent.")    
    
    # Filter based on multiselect
    filtered_era = df_era[df_era['Player'].isin(selected_players)]
    
    # Create 3 columns for our scatter plots!
    z_col1, z_col2, z_col3 = st.columns(3)

    # Create the scatter plots (MATLAB Analogy: scatter() with varying marker sizes)
    def create_z_scatter(x_col, y_col, x_label, y_label, title):
        fig = px.scatter(
            filtered_era, x=x_col, y=y_col, text='Player', 
            size='Pace_Bubble_Size', size_max=45, # HUGE bubbles for the fast eras
            size_max=30, # Shrunk down by 1/3rd from 45!
            color='Player', color_discrete_map=player_colors, title=title
        )
        fig.update_traces(textposition='top center', marker=dict(line=dict(width=1, color='Black')))
        fig.add_hline(y=1.5, line_dash="dot", line_color="gray")
        fig.add_vline(x=1.5, line_dash="dot", line_color="gray")
        fig.update_layout(xaxis_title=x_label, yaxis_title=y_label, showlegend=False, height=500)
        return fig

    with z_col1:
        fig_z1 = create_z_scatter('Scoring_Z', 'Rebound_Z', 'Scoring Dominance', 'Rebounding Dominance', 'Overall Dominance')
        st.plotly_chart(fig_z1, use_container_width=True, config=PLOTLY_CONFIG)
        
    with z_col2:
        fig_z2 = create_z_scatter('Scoring_Z', 'Assist_Z', 'Scoring Dominance', 'Playmaking Dominance', 'Offensive Dominance')
        st.plotly_chart(fig_z2, use_container_width=True, config=PLOTLY_CONFIG)
        
    with z_col3:
        fig_z3 = create_z_scatter('Rebound_Z', 'Defense_Z', 'Rebounding Dominance', 'STL+BLK Dominance', 'Defensive Dominance')
        st.plotly_chart(fig_z3, use_container_width=True, config=PLOTLY_CONFIG)

# --- TAB 2: Hardware & Clutch ---
with tab2:
    # Use columns to position graphs side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("The Trophy Cabinet")
        filtered_awards = df_awards[df_awards['Player'].isin(selected_players)]
        df_melt_awards = filtered_awards.melt(id_vars=['Player'], value_vars=['MVPs', 'Rings', 'Finals_MVPs', 'DPOY'], var_name='Award', value_name='Count')
        fig2 = px.bar(df_melt_awards, x='Award', y='Count', color='Player', barmode='group', color_discrete_map=player_colors)
        st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)
        
    with col2:
        st.subheader("The Clutch Factor (Reg Season vs Playoffs)")
        filtered_clutch = df_clutch[df_clutch['Player'].isin(selected_players)]
        df_melt_clutch = filtered_clutch.melt(id_vars=['Player'], value_vars=['Reg_Season_PTS', 'Finals_PTS'], var_name='Context', value_name='PPG')
        fig3 = px.bar(df_melt_clutch, x='Context', y='PPG', color='Player', barmode='group', color_discrete_map=player_colors)
        st.plotly_chart(fig3, use_container_width=True, config=PLOTLY_CONFIG)

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
    st.plotly_chart(fig_dist, use_container_width=True, config=PLOTLY_CONFIG)
    
    st.divider()
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Scoring Segment Breakdown")
        filtered_bin = bin_pct[bin_pct.index.isin(selected_players)]
        df_melt_bins = filtered_bin.reset_index().melt(id_vars='Player', var_name='Points Range', value_name='Percentage')
        
        # We use a sequential palette (Plasma) here because the ranges are ordered (<10 to 50+)
        fig_bins = px.bar(df_melt_bins, x='Player', y='Percentage', color='Points Range', color_discrete_sequence=px.colors.sequential.Plasma, text_auto='.1f')
        fig_bins.update_layout(yaxis_title="Percentage of Career Games (%)", barmode='stack')
        st.plotly_chart(fig_bins, use_container_width=True, config=PLOTLY_CONFIG)
        
    with col2:
        st.subheader("🔬 Statistically Significant Discoveries")
        st.markdown("*(Chi-Square with Bonferroni correction, p < 0.05)*")
        
        # Iterate through the top statistical discoveries generated by our backend model
        for f in [x for x in sig_findings if x['Player'] in selected_players][:5]:
            # Use st.success for visual highlight boxes
            st.success(f"**{f['Player']}** had significantly **{f['Direction']}** games scoring {f['Bin']} pts. ({f['Player_%']:.1f}% vs {f['Rest_%']:.1f}%)")

# --- TAB 4: Longevity vs Peak ---
with tab4:
    st.subheader("Peak Dominance vs. Career Longevity (Dumbbell Plot)")
    st.markdown("""
        Does extreme peak performance sacrifice longevity? 
        **Peak** (Points Per Game) is connected to **Longevity** (Total Career Points). 
        Both metrics are scaled 0-100 relative to the top performer for direct comparison.
    """)
    
    filtered_dumb = df_dumbbell[df_dumbbell['Player'].isin(selected_players)]
    
    fig_dumb = go.Figure()
    
    for i, row in filtered_dumb.iterrows():
        p = row['Player']
        
        # 1. The Connector Line
        fig_dumb.add_trace(go.Scatter(
            x=[row['Peak_Score'], row['Longevity_Score']], y=[p, p],
            mode='lines', line=dict(color='rgba(150, 150, 150, 0.5)', width=3), showlegend=False
        ))
        
        # 2. The Peak Dot (Circle)
        fig_dumb.add_trace(go.Scatter(
            x=[row['Peak_Score']], y=[p], mode='markers',
            marker=dict(color=player_colors[p], size=16, symbol='circle'), 
            name=f"{p} (Peak)", showlegend=False,
            hovertemplate=f"<b>{p}</b><br>Peak PPG Index: %{{x:.1f}}<extra></extra>"
        ))
        
        # 3. The Longevity Dot (Diamond)
        fig_dumb.add_trace(go.Scatter(
            x=[row['Longevity_Score']], y=[p], mode='markers',
            marker=dict(color=player_colors[p], size=16, symbol='diamond'), 
            name=f"{p} (Longevity)", showlegend=False,
            hovertemplate=f"<b>{p}</b><br>Longevity Total Index: %{{x:.1f}}<extra></extra>"
        ))
    
    # Custom Legend (Dummy traces just so the user knows what the shapes mean)
    fig_dumb.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='gray', size=12, symbol='circle'), name='Peak (PPG)'))
    fig_dumb.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='gray', size=12, symbol='diamond'), name='Longevity (Total Points)'))

    fig_dumb.update_layout(
        xaxis_title="Relative Score Index (0-100)",
        yaxis_title="", height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_dumb, use_container_width=True, config=PLOTLY_CONFIG)