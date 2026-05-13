# =====================================================================
# STREAMLIT APP: pages/4_tech_in_edu_app.py (The View)
# =====================================================================
import streamlit as st
import pandas as pd  # <--- ADD THIS LINE
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
    get_pisa_grid_samples,
    get_benchmark_comparison_data,
    get_micro_cloud_data   # <--- ADDED THIS
)
from data_projects_sidebar import apply_global_settings, render_sidebar

# --- SETTINGS & SIDEBAR ---
st.set_page_config(layout="wide") # Required for a clean 4-column grid
apply_global_settings("Neuro-Edu | The EdTech Paradox")
render_sidebar()

# --- DATA INGESTION ---
# ADDED os.path.abspath to make the path strictly absolute regardless of where Streamlit is launched
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'documents'))

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
    # Filter macro data
    df_bench = get_benchmark_comparison_data(df)
    bench_list = ['USA', 'JPN', 'DEU', 'ARG', 'JOR']
    
    # Load and sample micro data for the cloud
    df_micro = get_micro_cloud_data(base_dir, target_countries=bench_list, sample_per_group=500)
    
    subject_map = {
        'Math (Learning Efficiency)': 'Learning_Efficiency_Score',
        'Reading Proficiency': 'Reading_Proficiency_Score',
        'Science Proficiency': 'Science_Proficiency_Score'
    }

    st.header("Phase 2: Distribution Trends & Data Clouds")
    st.write("Comparing the macroscopic statistical spread (left) against the raw student population cloud (right).")
    
    selected_label = st.selectbox("Select Subject for Distribution:", list(subject_map.keys()))
    selected_sub = subject_map[selected_label] 

    country_colors = {
        'USA': '#EF553B', 'JPN': '#636EFA', 'DEU': '#00CC96', 
        'ARG': '#AB63FA', 'JOR': '#FFA15A'
    }

    # --- SIDE BY SIDE LAYOUT ---
    col_box, col_cloud = st.columns(2)

    # LEFT COLUMN: The Grouped Box Plot
    with col_box:
        fig_box = px.box(
            df_bench, 
            x='Year', y=selected_sub, color='Country',
            points=False, # Turned off the butterfly effect here so it doesn't clash with the cloud next to it
            color_discrete_map=country_colors,
            title=f"Statistical Spread: {selected_label}"
        )
        fig_box.update_layout(
            xaxis_title="Year", yaxis_title="PISA Score", 
            boxmode="group", legend_title="Nation", height=600
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # RIGHT COLUMN: The Strip Plot (Data Cloud)
    with col_cloud:
        if not df_micro.empty and selected_sub in df_micro.columns:
            fig_cloud = px.strip(
                df_micro, 
                x='Year', y=selected_sub, color='Country',
                color_discrete_map=country_colors,
                stripmode='group',
                title=f"Raw Population Cloud: {selected_label}"
            )
            # The 'Cloud' effect: Tiny dots, high transparency, wide horizontal jitter
            fig_cloud.update_traces(marker=dict(size=3, opacity=0.35), jitter=0.8)
            fig_cloud.update_layout(
                xaxis_title="Year", yaxis_title="", # Keep y-axis clean since it matches the left
                legend_title="Nation", height=600
            )
            st.plotly_chart(fig_cloud, use_container_width=True)
        else:
            st.warning("Micro-data files not found or still processing. Cannot render Data Cloud.")

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

    #########################################################
    #  Phase 4: Factor Correlation & Impact Analysis        #
    #########################################################
    st.markdown("---")
    st.header("Phase 4: Global Factor & Impact Analysis")
    st.write("Explore how socio-economic factors and technology access impact student performance globally.")

    # --- UI: DOUBLE DROPDOWN ---
    col_sel1, col_sel2 = st.columns(2)

    with col_sel1:
        factor_map = {
            'GDP per Capita (Wealth)': 'GDP_PER_CAPITA',
            'Internet Penetration (%)': 'INTERNET_PENETRATION',
            'Student-Teacher Ratio': 'Student_Teacher_Ratio',
            'ICT Use: Entertainment': 'ICT_Entertainment',
            'ICT Use: School/Academic': 'ICT_School_Use',
            'Knowledge Gap (Inequity)': 'Knowledge_Gap'
        }
        selected_factor_label = st.selectbox("Select Independent Factor (X-Axis):", list(factor_map.keys()))
        selected_factor = factor_map[selected_factor_label]

    with col_sel2:
        subject_map = {
            'Math (Learning Efficiency)': 'Learning_Efficiency_Score',
            'Reading Proficiency': 'Reading_Proficiency_Score',
            'Science Proficiency': 'Science_Proficiency_Score'
        }
        selected_sub_label = st.selectbox("Select Dependent Subject (Y-Axis):", list(subject_map.keys()))
        selected_sub = subject_map[selected_sub_label]

    # --- UI: COUNTRY MULTI-SELECT ---
    # Get all unique countries from the dataset and sort them alphabetically
    all_countries = sorted(df['Country'].dropna().unique().tolist())
    
    # Define your standard 5 countries here
    default_targets = ['USA', 'JPN', 'DEU', 'ARG', 'JOR'] 
    
    # Safety check: Ensure our defaults actually exist in the data to prevent Streamlit errors
    valid_defaults = [c for c in default_targets if c in all_countries]

    selected_countries = st.multiselect(
        "Filter by Country (Add or remove to compare):",
        options=all_countries,
        default=valid_defaults
    )

    # --- DATA PREP ---
    # 1. Drop rows where either the selected factor or the selected score is missing
    df_plot = df.dropna(subset=[selected_factor, selected_sub]).copy()
    
    # 2. Filter the dataset to ONLY include the countries chosen in the multiselect box
    df_plot = df_plot[df_plot['Country'].isin(selected_countries)]

    # Create the two columns for our side-by-side plots
    col_scatter, col_bar = st.columns(2)

    # --- COLUMN 1: CORRELATION & TRAJECTORY PLOT ---
    with col_scatter:
        if not df_plot.empty:
            # Sort chronologically so the trajectory lines draw forward in time
            df_plot = df_plot.sort_values(by=['Country', 'Year'])
            
            # Add a UI toggle to let the user choose the view
            show_trajectory = st.checkbox("Show Country Trajectories (Connect dots over time)")

            if show_trajectory:
                # TRAJECTORY VIEW: Connected Scatter Plot
                fig_scatter = px.line(
                    df_plot,
                    x=selected_factor,
                    y=selected_sub,
                    color='Country',       # Each country gets a distinct colored line
                    line_group='Country',  # Connects dots belonging to the same country
                    markers=True,          # Keeps the dots visible
                    hover_data=['Year'],  # Shows the year when you hover over a dot
                    title=f"Time Trajectories: {selected_factor_label} vs {selected_sub_label.split(' ')[0]}"
                )
            else:
                # GLOBAL TREND VIEW: Original Scatter with OLS line
                fig_scatter = px.scatter(
                    df_plot,
                    x=selected_factor,
                    y=selected_sub,
                    color='Year',          
                    hover_data=['Country'], 
                    trendline='ols',        
                    trendline_color_override="red",
                    title=f"Global Trend: {selected_factor_label} vs {selected_sub_label.split(' ')[0]}"
                )
                
            fig_scatter.update_layout(height=500, xaxis_title=selected_factor_label, yaxis_title="PISA Score")
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("Insufficient data to plot correlation.")

    # --- COLUMN 2: BINNED BAR CHART ---
    with col_bar:
        if not df_plot.empty:
            try:
                # pd.qcut automatically finds the 33% and 66% percentiles to make 3 perfectly even buckets
                df_plot['Factor_Bin'] = pd.qcut(df_plot[selected_factor], q=3, labels=['Low', 'Medium', 'High'])
                
                # Calculate the grand average score for each bucket
                df_bar = df_plot.groupby('Factor_Bin', observed=True)[selected_sub].mean().reset_index()

                fig_bar = px.bar(
                    df_bar,
                    x='Factor_Bin',
                    y=selected_sub,
                    color='Factor_Bin',
                    color_discrete_map={'Low': '#EF553B', 'Medium': '#FFA15A', 'High': '#00CC96'},
                    text_auto='.1f', # Prints the exact score on top of the bar
                    title=f"Average Score by {selected_factor_label} Tier"
                )
                
                # Zoom the Y-Axis: Because PISA scores are between 300-600, starting at 0 hides the differences.
                y_min = df_bar[selected_sub].min() * 0.90
                y_max = df_bar[selected_sub].max() * 1.05
                
                fig_bar.update_layout(
                    xaxis_title=f"{selected_factor_label} Tier",
                    yaxis_title="Average PISA Score",
                    showlegend=False,
                    height=500,
                    yaxis=dict(range=[y_min, y_max]) # Applies the zoom
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
            except Exception as e:
                # qcut will fail if too many values are perfectly identical (e.g., if many countries have exactly 0)
                st.warning(f"Not enough variance in {selected_factor_label} to calculate Low/Medium/High tiers.")

# ---------------------------------------------------------------------
# TAB 2: Tab 2: How Tech Effects Populations Differently 
# ---------------------------------------------------------------------
with tab2:
    st.header("How Tech Effects Populations Differently ")
    st.markdown("""
    **Why do we see stronger effects in certain subjects?** Historical data suggests that internet access strongly facilitates self-teaching in logic-based subjects like **Math and Science**, where concepts can be broken down step-by-step via video tutorials. However, the internet often promotes "skimming" behavior, which does not necessarily improve the deep cognitive stamina required for **Reading Comprehension**.
    
    **The Leapfrog Effect:** Does technology help poor nations more than rich ones? By splitting our global data into three wealth tiers, we can observe the marginal utility of the internet.
    """)

    st.markdown("---")

    # --- DATA PREP: BUCKETING GDP ---
    # Using your exact column names to prevent KeyErrors
    df_tab2 = df.dropna(subset=['GDP_PER_CAPITA', 'INTERNET_PENETRATION']).copy()
    
    # Create 3 distinct GDP Tiers (Poor, Moderate, Wealthy) based on statistical percentiles
    try:
        df_tab2['Wealth_Tier'] = pd.qcut(
            df_tab2['GDP_PER_CAPITA'], 
            q=3, 
            labels=['Lower Income', 'Middle Income', 'High Income']
        )
    except Exception as e:
        st.error("Not enough variance in GDP data to create wealth tiers. Check your master dataset.")

    # --- UI: SUBJECT SELECTOR ---
    subject_map_t2 = {
        'Math Scores': 'Math_Score',          
        'Reading Scores': 'Reading_Score',
        'Science Scores': 'Science_Score'
    }
    
    selected_sub_label_t2 = st.selectbox("Select Subject to Analyze:", list(subject_map_t2.keys()))
    selected_sub_t2 = subject_map_t2[selected_sub_label_t2]

    # Drop rows where the specific subject score is missing
    df_tab2_clean = df_tab2.dropna(subset=[selected_sub_t2])

    # --- COLUMN 1: THE FACETED PLOT ---
    if not df_tab2_clean.empty:
        fig_facet = px.scatter(
            df_tab2_clean,
            x='INTERNET_PENETRATION',
            y=selected_sub_t2,
            facet_col='Wealth_Tier',      # Splits the graph into 3 side-by-side panels
            color='Year',                # Using the exact spelling from your previous debug output
            hover_data=['Country'],
            trendline='ols',              # Draws the line of best fit for EACH panel
            trendline_color_override="red",
            title=f"Impact of Internet on {selected_sub_label_t2.split(' ')[0]} by National Wealth Tier",
            labels={'INTERNET_PENETRATION': 'Internet Penetration (%)', selected_sub_t2: 'PISA Score'}
        )
        
        # Make the layout look clean and uniform
        fig_facet.update_layout(height=500)
        # Ensure the Y-axis isn't locked to 0 so we can see the data spread clearly
        fig_facet.update_yaxes(matches=None, showticklabels=True) 
        
        st.plotly_chart(fig_facet, use_container_width=True)
    else:
        st.warning("Insufficient data to plot this relationship.")

    # --- DYNAMIC ANALYTICAL INSIGHT BOX ---
    st.info("""
    **How to read this chart:** Look at the red trendlines in each panel. 
    * If the line is **steeper** in the 'Lower Income' tier, it means a 10% increase in internet access creates a massive jump in scores for developing nations. 
    * If the line is **flat** in the 'High Income' tier, it indicates diminishing returns—adding more internet to an already wealthy country doesn't move the needle much.
    """)

with tab3:
    st.header("Phase 3: Digital Literacy & Correlation")
    # ... (Your existing Tab 3 logic)

with tab4:
    st.header("Phase 4: Theoretical Sandbox")
    # ... (Your existing Tab 4 logic)