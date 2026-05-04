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
    get_era_adjusted_stats, get_radar_scaled_stats, get_dumbbell_longevity_peak,
    calculate_hardware_score, get_google_trends, get_mvp_shares, get_league_trends, 
    get_civic_awards, get_philanthropy_data, calculate_cultural_impact_score
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
    df_scored, df_hw_melted = calculate_hardware_score(df_awards)
    df_longevity = analyze_longevity_vs_peak(df_goat)
    bin_pct, significant_findings = run_scoring_segment_analysis(df_goat)
    df_era = get_era_adjusted_stats(df_goat)
    df_radar = get_radar_scaled_stats(df_career)
    df_dumbbell = get_dumbbell_longevity_peak(df_goat)
    df_google = get_google_trends()
    df_mvp = get_mvp_shares()
    df_trends = get_league_trends()
    df_civic = get_civic_awards()
    df_phil = get_philanthropy_data()
    df_impact_score = calculate_cultural_impact_score(df_goat, df_mvp, df_google, df_civic, df_phil)

    colors = get_player_colors()
    return df_goat, df_career, df_clutch, df_awards, df_scored, df_hw_melted, df_longevity, bin_pct, significant_findings, 
        df_era, df_radar, df_dumbbell, df_google, df_mvp, df_trends, df_civic, df_phil, df_impact_score, colors

# Display a loading spinner while the backend fetches data
with st.spinner("Crunching historical NBA game logs..."):
    (df_goat, df_career, df_clutch, df_awards, df_scored, df_hw_melted, 
     df_longevity, bin_pct, sig_findings, df_era, df_radar, df_dumbbell, 
     df_google, df_mvp, df_trends, df_civic, df_phil, df_impact_score, 
     player_colors) = load_all_dashboard_data()

# Now, we define the sidebar right after the spinner block finishes
st.sidebar.title("Dashboard Controls")

# Define the default 10
DEFAULT_10 = [
    "Michael Jordan", "LeBron James", "Magic Johnson", "Stephen Curry", 
    "Shaquille O'Neal", "Kareem Abdul-Jabbar", "Kobe Bryant", 
    "Bill Russell", "Wilt Chamberlain", "Nikola Jokic"
]

all_available_players = sorted(df_goat['Player'].unique().tolist())

selected_players = st.sidebar.multiselect(
    "Select Players to Compare:",
    options=all_available_players,
    default=DEFAULT_10
)

# -------------------------------------------------------------------
# MAIN APP LAYOUT (Interactive Controls on Main Page)
# -------------------------------------------------------------------
st.title("🏀 The Ultimate NBA GOAT Analyzer")
st.write("An exploratory data science dashboard comparing peak dominance, longevity, and statistical consistency of 10 legendary players.")
st.divider()

# Interactive Filter: Allows users to subset which players they want to compare
# MATLAB Analogy: Connecting a drop-down UI Component callback to update data.

# --- SIDEBAR CONTROLS ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/889/889442.png", width=100) # Optional basketball icon
st.sidebar.title("Dashboard Controls")

# 1. Define the original 10 to act as our default startup view
DEFAULT_10 = [
"Michael Jordan", "LeBron James", "Magic Johnson", "Stephen Curry", 
"Shaquille O'Neal", "Kareem Abdul-Jabbar", "Kobe Bryant", 
"Bill Russell", "Wilt Chamberlain", "Nikola Jokic"
]

# 2. Extract the full list of 50 players dynamically from the dataset and sort alphabetically
all_available_players = sorted(df_goat['Player'].unique().tolist())

# 3. The Dynamic Dropdown
selected_players = st.sidebar.multiselect(
    "Select Players to Compare:",
    options=all_available_players,
    default=DEFAULT_10
    )

if not selected_players:
    st.error("Please select at least one player from the sidebar to view the analytics.")
    st.stop()

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


# --- TAB 2: Hardware & The Trophy Predictor ---
with tab2:
    st.header("The Hardware Algorithm")
    st.markdown("""
        Who is the GOAT based *purely* on resume? We have assigned objective weights to every major NBA accolade. 
        * Weights: Rings (10), MVP/Finals MVP (8), DPOY (5), Scoring Title (4), ROTY (3), All-NBA/All-Defense/Clutch (2).
    """)
    
    # Filter the datasets based on user sidebar selection
    filtered_melted = df_hw_melted[df_hw_melted['Player'].isin(selected_players)]
    filtered_scored = df_scored[df_scored['Player'].isin(selected_players)]
    
    # 1. The Stacked Bar Chart
    # We order the X-axis by the players with the highest total score!
    fig_hw = px.bar(
        filtered_melted, 
        x='Player', 
        y='Weighted_Points', 
        color='Award',
        text='Count', # Shows the raw number of awards inside the colored blocks
        title="Weighted Career Accolades",
        category_orders={"Player": filtered_scored['Player'].tolist()} # Sorts X-axis highest to lowest
    )
    
    fig_hw.update_traces(textposition='inside', textfont_color='white')
    fig_hw.update_layout(yaxis_title="Total Hardware Score", xaxis_title="", barmode='stack', height=600)
    st.plotly_chart(fig_hw, use_container_width=True, config=PLOTLY_CONFIG)
    
    # 2. Show the raw dataset below for transparency
    st.divider()
    st.subheader("The Raw Trophy Cabinet")
    # Clean up the dataframe before showing it
    display_cols = ['Player', 'Total_Hardware_Score', 'Rings', 'MVPs', 'Finals_MVPs', 'All_NBA', 'All_Defense', 'Scoring_Titles', 'DPOY', 'ROTY', 'Clutch_POY']
    st.dataframe(filtered_scored[display_cols].set_index('Player'), use_container_width=True)

# --- TAB 3: Consistency & Variance ---
with tab3:

    # ---------------------------------------------------------
    # 1. ERA-ADJUSTED DOMINANCE (Moved from Tab 1)
    # ---------------------------------------------------------
    st.divider()
    st.subheader("1. Era-Adjusted Dominance (Z-Scores vs Peers)")
    st.markdown("Adjusted for era pacing.")    
    st.markdown("*Pace is calculated by the average points scored per game across the entire NBA during that player's active years. Faster eras (like the 1960s) have higher averages, which naturally inflated raw stats compared to slower, defensive eras (like the 2000s). Think Wilt's 1960s vs Kobe's 2000s.*")

    # Filter based on multiselect
    filtered_era = df_era[df_era['Player'].isin(selected_players)]
    
    # Create 3 columns for our scatter plots!
    z_col1, z_col2, z_col3 = st.columns(3)

    # Create the scatter plots (MATLAB Analogy: scatter() with varying marker sizes)
    def create_z_scatter(x_col, y_col, x_label, y_label, title):
        fig = px.scatter(
            filtered_era, x=x_col, y=y_col, text='Player', 
            size='Pace_Bubble_Size', 
            size_max=25, # Shrunk down by 1/3rd from 45!
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

    # ---------------------------------------------------------
    # 2. THE CLUTCH FACTOR (Playoff Elevation)
    # ---------------------------------------------------------
    st.divider()
    st.subheader("2. The Clutch Factor (Playoff Elevation)")
    st.markdown("""
        How do players perform when the pressure is highest? **Players above the diagonal dashed line elevate their game in the playoffs.** Players below the line shrink under pressure. Use the dropdown to explore different facets of the game.
        *Note: Players from the 1960s (Wilt, Russell) did not have the 3-point line and will appear at 0%.*
    """)
    
    filtered_clutch = df_clutch[df_clutch['Player'].isin(selected_players)].copy()
    
    # --- The Dynamic Helper Function (Now with plot_height!) ---
    def create_clutch_scatter(stat_key, title, is_percentage=False, plot_height=450):
        reg_col = f"Regular_Season_{stat_key}"
        play_col = f"Playoffs_{stat_key}"
        
        # Fallback empty chart if a column somehow goes missing
        if reg_col not in filtered_clutch.columns or play_col not in filtered_clutch.columns:
            return go.Figure()
            
        fig = px.scatter(filtered_clutch, x=reg_col, y=play_col, text='Player', color='Player', color_discrete_map=player_colors, title=title)
        
        # Calculate the perfect 45-degree baseline
        max_val = max(filtered_clutch[reg_col].max(), filtered_clutch[play_col].max())
        min_val = min(filtered_clutch[reg_col].min(), filtered_clutch[play_col].min())
        
        # Add 10% visual padding
        padding = (max_val - min_val) * 0.1 if max_val != min_val else 1
        max_val += padding
        min_val -= padding
        
        fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="gray", dash="dash"))
        
        fig.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=1, color='Black')))
        
        # >>> Apply dynamic plot_height here! <<<
        fig.update_layout(xaxis_title=f"Reg Season {title}", yaxis_title=f"Playoffs {title}", showlegend=False, height=plot_height)
        
        if is_percentage:
            fig.update_layout(xaxis_tickformat=".1%", yaxis_tickformat=".1%")
            
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        return fig

    # --- 3-Column Row Layout ---
    clutch_col1, clutch_col2, clutch_col3 = st.columns(3)
    
    with clutch_col1:
        stat_options = {
            "Points Scored": "PTS",
            "Overall Impact (+/-)": "PLUS_MINUS",
            "Defense (Steals + Blocks)": "DEF",
            "Total Rebounds": "TRB"
        }
        # The dropdown sits at the top of column 1
        selected_stat_label = st.selectbox("Select Metric:", options=list(stat_options.keys()))
        stat_key = stat_options[selected_stat_label]
        
        # We shrink this specific plot to 360px so it perfectly compensates for the dropdown menu!
        st.plotly_chart(create_clutch_scatter(stat_key, selected_stat_label, plot_height=360), use_container_width=True, config=PLOTLY_CONFIG)

    with clutch_col2:
        # Full 450px height
        st.plotly_chart(create_clutch_scatter('TS_PCT', 'True Shooting %', is_percentage=True, plot_height=450), use_container_width=True, config=PLOTLY_CONFIG)

    with clutch_col3:
        # Full 450px height
        st.plotly_chart(create_clutch_scatter('3PT_PCT', '3-Point %', is_percentage=True, plot_height=450), use_container_width=True, config=PLOTLY_CONFIG)

    # ---------------------------------------------------------
    # 3. STATISTICAL DISTRIBUTIONS & ANOMALIES
    # ---------------------------------------------------------
    st.divider()
    st.subheader("3. Scoring Distributions & Consistency (Violin Plots)")
    st.markdown("Unlike a single average number, this shows the *shape* of a player's entire career. A wider bulge means they consistently scored in that range. A long, thin tail points to rare, explosive games.")
    
    filtered_goat = df_goat[df_goat['Player'].isin(selected_players)]
    
    # Plotly Violin plot serves as a beautiful, continuous KDE (Kernel Density Estimate)
    fig_violin = px.violin(
        filtered_goat, y="points", x="Player", color="Player", 
        box=True, # Adds a mini box-plot inside the violin
        points="all", # Shows all individual game dots softly in the background
        color_discrete_map=player_colors
    )
    fig_violin.update_layout(yaxis_title="Points Scored in a Single Game", xaxis_title="", showlegend=False, height=600)
    st.plotly_chart(fig_violin, use_container_width=True, config=PLOTLY_CONFIG)

    # ---------------------------------------------------------
    # 4. STATISTICAL ANOMALIES (Chi-Square Findings)
    # ---------------------------------------------------------
    st.divider()
    st.subheader("4. Significant Statistical Anomalies")
    st.markdown("""
        *Using a Bonferroni-corrected Chi-Square analysis, we can mathematically prove if a player scores in a specific range at a statistically significant higher rate than their peers.*
    """)
    
    # Filter the significant findings to only show players currently selected in the sidebar
    relevant_findings = [f for f in sig_findings if f['Player'] in selected_players]
    
    if relevant_findings:
        for finding in relevant_findings:
            # Format the text so it reads like a clean, data science insight
            player = finding['Player']
            bin_range = finding['Bin']
            p_val = finding['P_Val']
            p_pct = finding['Player_%']
            r_pct = finding['Rest_%']
            
            st.success(
                f"**{player}** scores **{bin_range} points** in {p_pct:.1f}% of his games. "
                f"The rest of the GOAT group only does this {r_pct:.1f}% of the time. "
                f"*(p < {p_val:.5f})*"
            )
    else:
        st.info("No statistically significant anomalies detected for the currently selected players in comparison to the broader group.")

# --- TAB 4: Longevity vs Peak ---
with tab4:
    st.header("Cultural Impact & The Eye Test")
    
    # ---------------------------------------------------------
    # 1. Cultural Zeitgeist (Google Trends)
    # ---------------------------------------------------------
    st.subheader("1. The Cultural Zeitgeist (Google Trends)")
    st.markdown("""
        *Relative Search Volume over the last 5 years.* Who controls the modern news cycle? Peaks often align with Finals runs, trades, documentary releases, or MVP races.
    """)
    
    if not df_google.empty:
        valid_cols = ['date'] + [p for p in selected_players if p in df_google.columns]
        if len(valid_cols) > 1:
            df_g_filtered = df_google[valid_cols]
            df_g_melted = df_g_filtered.melt(id_vars='date', var_name='Player', value_name='Search Volume')
            fig_google = px.line(df_g_melted, x='date', y='Search Volume', color='Player', color_discrete_map=player_colors)
            fig_google.update_traces(line=dict(width=2.5))
            fig_google.update_layout(xaxis_title="Date", yaxis_title="Relative Search Volume (0-100)", height=450)
            st.plotly_chart(fig_google, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("No Google Trends data found for the currently selected players.")
    
    # ---------------------------------------------------------
    # 2. The Eye Test: MVP Voting Shares
    # ---------------------------------------------------------
    st.divider()
    st.subheader("2. MVP Voting Shares (The Peak 'Eye Test')")
    st.markdown("""
        *MVP Voting Share = (Points Won / Total Possible Points).* MVP *shares* reveal how heavily a player dominated the league's consciousness year-over-year, even if they didn't officially win the trophy.
    """)
    
    if not df_mvp.empty:
        df_mvp_filtered = df_mvp[df_mvp['Player'].isin(selected_players)]
        if not df_mvp_filtered.empty:
            fig_mvp = px.area(
                df_mvp_filtered.sort_values(by=['Year']), 
                x='Year', y='Share', color='Player', 
                color_discrete_map=player_colors, line_group='Player'
            )
            fig_mvp.update_layout(xaxis_title="Year", yaxis_title="MVP Voting Share", height=500)
            fig_mvp.update_traces(opacity=0.6)
            st.plotly_chart(fig_mvp, use_container_width=True, config=PLOTLY_CONFIG)
        else:
             st.info("No MVP voting data found for the currently selected players.")

    # ---------------------------------------------------------
    # 3. Character & Leadership (Civic Awards)
    # ---------------------------------------------------------
    st.divider()
    st.subheader("3. Character & Leadership (The Civic Matrix)")
    st.markdown("""
        *The Locker Room Impact.* This matrix tracks the four major NBA civic/leadership awards: the J. Walter Kennedy Citizenship Award, NBA Sportsmanship Award, Twyman-Stokes Teammate of the Year, and the Kareem Abdul-Jabbar Social Justice Champion Award.
    """)
    
    if not df_civic.empty:
        df_civic_filtered = df_civic[df_civic['Player'].isin(selected_players)]
        if not df_civic_filtered.empty:
            civic_melted = df_civic_filtered.melt(id_vars='Player', var_name='Award', value_name='Won')
            civic_melted['Status'] = civic_melted['Won'].apply(lambda x: '🏆 Won' if isinstance(x, (int, float)) and x > 0 else '—')
            
            fig_civic = px.density_heatmap(
                civic_melted, x='Award', y='Player', z='Won',
                color_continuous_scale=['#f4f4f4', '#FFD700'], 
                text_auto=True
            )
            fig_civic.update_traces(text=civic_melted['Status'], texttemplate="%{text}")
            fig_civic.update_layout(height=400, yaxis_title="", xaxis_title="", coloraxis_showscale=False)
            st.plotly_chart(fig_civic, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("None of the currently selected players have won these specific civic awards.")

    # ---------------------------------------------------------
    # 4. Real World Philanthropy (NLP Extracted)
    # ---------------------------------------------------------
    st.divider()
    st.subheader("4. Real World Philanthropy (NLP Text Mining)")
    st.markdown("""
        *The Philanthropic Footprint.* We used an NLP algorithm to scan Wikipedia for verified charitable actions, foundations, and community scholarships. The larger the footprint, the more documented social impact the player has generated off the court.
    """)

    if not df_phil.empty:
        df_phil_filtered = df_phil[df_phil['Player'].isin(selected_players)]
        if not df_phil_filtered.empty:
            fig_phil = px.bar(
                df_phil_filtered.sort_values(by='Philanthropic_Footprint', ascending=True),
                x='Philanthropic_Footprint', y='Player', orientation='h',
                color='Player', color_discrete_map=player_colors,
                text='Philanthropic_Footprint'
            )
            fig_phil.update_layout(xaxis_title="Verified Philanthropic Sentences", yaxis_title="", height=350, showlegend=False)
            st.plotly_chart(fig_phil, use_container_width=True, config=PLOTLY_CONFIG)
            
            # Display random sentence highlights for the top selected player!
            st.markdown("##### 🔍 Philanthropy Highlights")
            top_phil_player = df_phil_filtered.sort_values(by='Philanthropic_Footprint', ascending=False).iloc[0]
            
            if top_phil_player['Philanthropic_Footprint'] > 0:
                st.success(f"**{top_phil_player['Player']}'s Community Impact:**")
                sentence_cols = [c for c in df_phil.columns if 'Sentence_' in c]
                valid_sentences = top_phil_player[sentence_cols].dropna().tolist()
                
                for idx, sentence in enumerate(valid_sentences[:3]): 
                    st.write(f"*{idx+1}. {sentence}*")
        else:
             st.info("No philanthropy data found for the currently selected players.")

    # ---------------------------------------------------------
    # 5. The Master Cultural Impact Score
    # ---------------------------------------------------------
    st.divider()
    st.subheader("5. Overall Cultural Impact Score")
    st.markdown("""
        *The definitive off-court metric.* By Min-Max normalizing all four datasets (MVP Shares, Google Trends, Civic Awards, and Philanthropy) into a shared dimensional space, we can calculate a singular weighted score out of 100 representing a player's absolute cultural gravity.
    """)
    
    if not df_impact_score.empty:
        df_impact_filtered = df_impact_score[df_impact_score['Player'].isin(selected_players)]
        
        fig_impact = px.bar(
            df_impact_filtered.sort_values(by='Cultural_Impact_Score', ascending=True),
            x='Cultural_Impact_Score', y='Player', orientation='h',
            color='Player', color_discrete_map=player_colors,
            text='Cultural_Impact_Score'
        )
        fig_impact.update_traces(texttemplate='%{text}', textposition='outside')
        fig_impact.update_layout(xaxis_title="Weighted Cultural Impact Score (0-100)", yaxis_title="", height=400, showlegend=False)
        fig_impact.update_xaxes(range=[0, 105]) 
        st.plotly_chart(fig_impact, use_container_width=True, config=PLOTLY_CONFIG)

    # ---------------------------------------------------------
    # Methodology Expander
    # ---------------------------------------------------------
    with st.expander("📊 Methodology & Data Sources"):
        st.markdown("""
        **How was this off-court data collected?**
        * **Google Trends (The Popularity Metric):** A custom Python script connected to the `pytrends` API to pull 5-year Relative Search Volume (RSV), batched and anchored mathematically to a single baseline for all 50 players.
        * **MVP Voting Shares (The Eye Test):** Pandas `read_html` was used to web-scrape 68 years of historical MVP voting tables from *Basketball-Reference*, tracking peak dominance even in years the award wasn't won.
        * **Civic Awards (Character):** Web-scraped directly from Wikipedia tables tracking the NBA's four major character awards, with aggressive text cleaning to remove Wikipedia footnotes and symbols.
        * **Philanthropy (Real World Impact):** A Natural Language Processing (NLP) pipeline parsed every paragraph of the players' biographies using the `nltk` tokenizer. A Dual-Filter system cross-referenced sentences against a Positive Dictionary (e.g., *'charity'*, *'scholarship'*) and a Negative Dictionary (e.g., *'contract'*, *'lawsuit'*) to isolate verified social impact. 
        """)

    