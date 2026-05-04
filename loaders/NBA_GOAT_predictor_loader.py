"""
=============================================================================
MODULE: loaders/nba_goat_loader.py (NBA GOAT Backend Loader)
AUTHOR: Kyle W. Killebrew, PhD
VERSION: 1.0 (Data Science Micro-Frontend Hub)
DESCRIPTION: 
    Backend data processing logic for the NBA GOAT Predictor. Handles data 
    ingestion, Pandas aggregations, feature engineering, and Chi-Square 
    statistical modeling. Strictly follows MVC architecture.
=============================================================================
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import streamlit as st

# -------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# -------------------------------------------------------------------

# --- (Must be outside functions to be importable) ---
PLAYERS = [
    "Michael Jordan", "LeBron James", "Magic Johnson", "Stephen Curry", 
    "Shaquille O'Neal", "Kareem Abdul-Jabbar", "Kobe Bryant", 
    "Bill Russell", "Wilt Chamberlain", "Nikola Jokic"
]

def get_player_colors(df_goat):
    """
    Maintains our strict color consistency rule for the UI layer.
    
    MATLAB Analogy: 
    This is equivalent to creating a custom colormap (e.g., customMap = lines(10)) 
    and ensuring every subsequent plot uses the exact same color-to-entity mapping 
    so visual identification remains consistent across figures.
    """
    import plotly.express as px
    # Get EVERY unique player from the dataset, not just the top 10
    all_unique_players = sorted(df_goat['Player'].unique().tolist())
    
    # Use a large qualitative palette (like Alphabet or Light24) to avoid repeats
    palette = px.colors.qualitative.Alphabet 
    
    # Map every player to a color using a dictionary comprehension
    return {player: palette[i % len(palette)] for i, player in enumerate(all_unique_players)}

# -------------------------------------------------------------------
# DATA INGESTION (The "Model")
# -------------------------------------------------------------------
# UPDATE THIS FUNCTION IN: loaders/nba_goat_loader.py

def load_and_filter_raw_data():
    """
    Loads our pre-shrunk top 50 CSV and strictly filters it to our 10 active candidates.
    Returns a memory-efficient DataFrame of individual game logs.
    """
    try:
        # Dynamically find the 'documents' folder relative to this loader script
        # __file__ is loaders/nba_goat_loader.py
        # os.path.dirname(__file__) is the 'loaders' directory
        # os.path.dirname(os.path.dirname(__file__)) is the root repo directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        player_stats_path = os.path.join(base_dir, 'documents', 'goat_data_extended.csv')
        
        # No 'usecols' bouncer. Since the CSV is already small, this guarantees 
        # EVERY column (including turnovers and steals) makes it into the dataframe.
        df_all = pd.read_csv(player_stats_path)

        df_all['Player'] = df_all['firstName'] + " " + df_all['lastName']
        df_all['Year'] = pd.to_datetime(df_all['gameDate']).dt.year

        # Create the stl_blk value using steals and blocks 
        df_all['stl_blk'] = df_all['steals'].fillna(0) + df_all['blocks'].fillna(0)
        
        # MATLAB Analogy: groupsummary(df, 'Year', {'mean', 'std'})
        # We calculate the average and standard deviation of points and rebounds for EVERY year in history
        league_yearly = df_all.groupby('Year').agg({
            'points': ['mean', 'std'],
            'reboundsTotal': ['mean', 'std'],
            'assists': ['mean', 'std'],
            'stl_blk': ['mean', 'std']
        }).reset_index()
        
        league_yearly.columns = [
            'Year', 'pts_mean', 'pts_std', 'reb_mean', 'reb_std', 
            'ast_mean', 'ast_std', 'stl_blk_mean', 'stl_blk_std'
        ]

        # Merge the historical yearly averages directly to all 50 players
        df_goat = pd.merge(df_all, league_yearly, on='Year', how='left')
        
        # Z-Score Formula: (Player Score - League Average) / League Standard Deviation
        df_goat['pts_z'] = np.where(df_goat['pts_std'] > 0, (df_goat['points'] - df_goat['pts_mean']) / df_goat['pts_std'], 0)
        df_goat['reb_z'] = np.where(df_goat['reb_std'] > 0, (df_goat['reboundsTotal'] - df_goat['reb_mean']) / df_goat['reb_std'], 0)
        df_goat['ast_z'] = np.where(df_goat['ast_std'] > 0, (df_goat['assists'] - df_goat['ast_mean']) / df_goat['ast_std'], 0)
        df_goat['def_z'] = np.where(df_goat['stl_blk_std'] > 0, (df_goat['stl_blk'] - df_goat['stl_blk_mean']) / df_goat['stl_blk_std'], 0)
        
        # Extract Year for Longevity calculations
        df_goat['Year'] = pd.to_datetime(df_goat['gameDate']).dt.year
        return df_goat
        
    except Exception as e:
        # >>> FIX: Updated the error message name here! <<<
        raise RuntimeError(f"Failed to load local dataset. Ensure goat_data_extended.csv is in the documents/ folder. Error: {e}")

# -------------------------------------------------------------------
# FEATURE ENGINEERING & AGGREGATION
# -------------------------------------------------------------------
def calculate_career_baselines(df_goat):
    """
    Calculates true career averages and Reg Season vs Playoff splits.
    
    MATLAB Analogy:
    Pandas `groupby().mean()` is the exact equivalent of MATLAB's `groupsummary(df, 'Player', 'mean')`.
    It automatically aggregates the dataset based on unique categories.
    """
    # 1. Base Averages
    # Add the new fields to the groupby!
    cols_to_avg = [
        'points', 'reboundsTotal', 'assists', 'blocks', 
        'steals', 'turnovers', 'plusMinusPoints', 
        'fieldGoalsMade', 'fieldGoalsAttempted'
    ]
    df_career = df_goat.groupby('Player')[cols_to_avg].mean().reset_index()

    # Calculate Field Goal Percentage
    df_career['FG_PCT'] = np.where(
        df_career['fieldGoalsAttempted'] > 0, 
        df_career['fieldGoalsMade'] / df_career['fieldGoalsAttempted'], 
        0
    )

    # Rename columns for cleaner UI presentation (fillna prevents errors for 1960s players)
    df_career.rename(columns={
        'points': 'PTS', 'reboundsTotal': 'TRB', 'assists': 'AST', 
        'blocks': 'BLK', 'steals': 'STL', 'turnovers': 'TOV',
        'plusMinusPoints': 'PLUS_MINUS'
    }, inplace=True)
    df_career.fillna(0, inplace=True)
    
    # 2. Clutch Factor (Playoffs vs Regular Season)
    # We group by two variables here, then use `.unstack()` to pivot the 'gameType' into separate columns
    # 2. Clutch Factor (Playoffs vs Regular Season)
    clutch_agg = df_goat.groupby(['Player', 'gameType']).agg({
        'points': ['mean', 'sum'],
        'plusMinusPoints': 'mean',
        'stl_blk': 'mean',
        'reboundsTotal': 'mean',
        'fieldGoalsAttempted': 'sum',
        'threePointersMade': 'sum',
        'threePointersAttempted': 'sum',
        'freeThrowsAttempted': 'sum'
    }).reset_index()
    
    # Flatten the messy multi-level columns
    clutch_agg.columns = [
        'Player', 'gameType', 'PTS', 'PTS_sum', 'PLUS_MINUS', 
        'DEF', 'TRB', 'FGA_sum', '3PM_sum', '3PA_sum', 'FTA_sum'
    ]
    
    # Mathematically accurate True Shooting & 3PT%
    clutch_agg['TS_PCT'] = clutch_agg['PTS_sum'] / (2 * (clutch_agg['FGA_sum'] + 0.44 * clutch_agg['FTA_sum']))
    clutch_agg['3PT_PCT'] = clutch_agg['3PM_sum'] / clutch_agg['3PA_sum']
    
    # Fill NAs (e.g., Wilt/Russell didn't have 3-pointers, so we default to 0 to avoid crashes)
    clutch_agg.fillna({'TS_PCT': 0, '3PT_PCT': 0}, inplace=True)
    
    # Pivot the data so "Regular Season" and "Playoffs" become their own column headers
    df_clutch = clutch_agg.pivot(index='Player', columns='gameType', values=['PTS', 'PLUS_MINUS', 'DEF', 'TRB', 'TS_PCT', '3PT_PCT'])
    
    # Flatten the pivoted headers (e.g., turns into "Regular_Season_PTS")
    df_clutch.columns = [f"{col[1].replace(' ', '_')}_{col[0]}" for col in df_clutch.columns]
    df_clutch.reset_index(inplace=True)
    
    return df_career, df_clutch

def get_awards_hardware():
    """
    Dynamically loads the hardware data from our player_awards.csv file.
    This prevents crashes if the PLAYERS list changes.
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        awards_path = os.path.join(base_dir, 'documents', 'player_awards.csv')
        
        # Load the CSV dynamically
        df_awards = pd.read_csv(awards_path)
        
        # CRITICAL SAFEGUARD: If you left any cells blank in Excel, 
        # this turns them into 0s so the math doesn't crash!
        df_awards.fillna(0, inplace=True)
        
        return df_awards
        
    except FileNotFoundError:
        raise RuntimeError("Could not find player_awards.csv in the documents folder!")

def calculate_hardware_score(df_awards):
    """
    Applies mathematical weights to hardware to generate an objective GOAT score.
    Returns the total scores AND a 'melted' dataset perfect for Plotly stacked bars.
    """
    # The GOAT Algorithm Weights! 
    # (We can easily tweak these later in Canvas if you want to change the importance)
    weights = {
        "Rings": 10,
        "MVPs": 9,
        "Finals_MVPs": 8,
        "DPOY": 5,
        "Scoring_Titles": 4,
        "ROTY": 3,
        "Clutch_POY": 1,
        "All_NBA": 2,
        "All_Defense": 1
    }
    
    # Calculate Total Score
    df_scored = df_awards.copy()
    df_scored['Total_Hardware_Score'] = 0
    for col, weight in weights.items():
        df_scored['Total_Hardware_Score'] += df_scored[col] * weight
        
    # Sort from highest score to lowest for the UI
    df_scored = df_scored.sort_values('Total_Hardware_Score', ascending=False)
    
    # MATLAB Analogy: reshape/melt. We flatten the data so Plotly can easily stack it by Award Type.
    df_melted = df_awards.melt(id_vars=['Player'], value_vars=list(weights.keys()), var_name='Award', value_name='Count')
    
    # Calculate the actual points contributed by each award
    df_melted['Weighted_Points'] = df_melted.apply(lambda row: row['Count'] * weights[row['Award']], axis=1)
    
    # Filter out 0s so our chart tooltips are super clean
    df_melted = df_melted[df_melted['Weighted_Points'] > 0]
    
    return df_scored, df_melted

def get_era_adjusted_stats(df_goat):
    """
    Calculates the real Era-Adjusted Dominance (Z-Scores) by averaging
    the game-by-game Z-scores we calculated during ingestion.
    """
    # Group the new Z-scores
    # We use 'pts_mean' (the average points scored by an average NBA player that year) as a proxy for Era Pace
    df_era = df_goat.groupby('Player')[['pts_z', 'reb_z', 'ast_z', 'def_z', 'pts_mean']].mean().reset_index()
    
    # Rename columns so they instantly link to our Plotly UI
    df_era.rename(columns={
        'pts_z': 'Scoring_Z', 'reb_z': 'Rebound_Z', 
        'ast_z': 'Assist_Z', 'def_z': 'Defense_Z',
        'pts_mean': 'Era_Pace'
    }, inplace=True)

    min_pace = df_era['Era_Pace'].min()
    max_pace = df_era['Era_Pace'].max()
    
    # Formula: Scaled = Min_Size + (Raw - Min_Raw) * (Max_Size - Min_Size) / (Max_Raw - Min_Raw)
    if max_pace > min_pace: # Safeguard against divide-by-zero
        df_era['Pace_Bubble_Size'] = 10 + ((df_era['Era_Pace'] - min_pace) * (40 - 10) / (max_pace - min_pace))
    else:
        df_era['Pace_Bubble_Size'] = 25 # Fallback if all eras are identical

    return df_era

def analyze_longevity_vs_peak(df_goat):
    """
    Calculates absolute totals and per-season accumulation rates.
    """
    # Calculate absolute lifetime totals
    df_totals = df_goat.groupby('Player')[['points', 'reboundsTotal', 'assists']].sum().reset_index()
    
    # Count unique years played to determine longevity
    years_played = df_goat.groupby('Player')['Year'].nunique().reset_index()
    years_played.rename(columns={'Year': 'Seasons'}, inplace=True)
    
    # MATLAB Analogy: Using `join()` or `innerjoin()` to merge two tables based on a shared key ('Player')
    df_totals = pd.merge(df_totals, years_played, on='Player')
    
    # Per-season calculations (Peak vs Sustained capability)
    df_totals['PTS_per_season'] = df_totals['points'] / df_totals['Seasons']
    df_totals['TRB_per_season'] = df_totals['reboundsTotal'] / df_totals['Seasons']
    df_totals['AST_per_season'] = df_totals['assists'] / df_totals['Seasons']
    
    # Melt the dataframe for our Facet Grid UI
    # MATLAB Analogy: The `stack()` function, pivoting "wide" data into "long" data.
    df_totals_melt = df_totals.melt(
        id_vars=['Player', 'Seasons'],
        value_vars=['points', 'reboundsTotal', 'assists', 'PTS_per_season', 'TRB_per_season', 'AST_per_season'],
        var_name='Raw_Metric', value_name='Value'
    )
    
    # Standardize names for cleaner chart labels
    df_totals_melt['Stat'] = df_totals_melt['Raw_Metric'].map({
        'points': 'Points', 'PTS_per_season': 'Points',
        'reboundsTotal': 'Rebounds', 'TRB_per_season': 'Rebounds',
        'assists': 'Assists', 'AST_per_season': 'Assists'
    })
    
    # Categorize whether the metric is an absolute total or a per-season rate
    df_totals_melt['Measurement'] = df_totals_melt['Raw_Metric'].apply(
        lambda x: 'Per Season Average' if 'per_season' in x else 'Absolute Career Total'
    )
    return df_totals_melt

# -------------------------------------------------------------------
# STATISTICAL MODELING
# -------------------------------------------------------------------
def run_scoring_segment_analysis(df_goat):
    """
    Runs Chi-Square analysis with Bonferroni correction on scoring bins.
    Returns the percentage dataframe and a list of significant findings.
    
    MATLAB Analogy: 
    This is equivalent to using `discretize()` to group data into bins, 
    and then running `crosstab()` and `chi2gof()` to test for statistical independence.
    """
    # 1. Bin the data (Discretization)
    bins = [0, 10, 20, 30, 40, 50, 150] 
    labels = ['<10', '10-19', '20-29', '30-39', '40-49', '50+']
    df_goat['Score_Bin'] = pd.cut(df_goat['points'], bins=bins, labels=labels, right=False)
    
    # 2. Calculate percentages of games in each bin per player
    bin_counts = df_goat.groupby(['Player', 'Score_Bin'], observed=True).size().unstack(fill_value=0)
    bin_pct = bin_counts.div(bin_counts.sum(axis=1), axis=0) * 100
    
    # 3. Chi-Square Testing Setup
    alpha = 0.05
    num_tests = len(PLAYERS) * len(labels)
    # Bonferroni correction prevents false positives due to multiple comparisons
    adjusted_alpha = alpha / num_tests 
    
    significant_findings = []
    
    # Run a 1-vs-Rest Chi-Square test for every player across every scoring bin
    for player in PLAYERS:
        player_games = df_goat[df_goat['Player'] == player]
        rest_games = df_goat[df_goat['Player'] != player]
        
        for bin_label in labels:
            # Construct a 2x2 Contingency Table
            player_in = (player_games['Score_Bin'] == bin_label).sum()
            player_out = len(player_games) - player_in
            rest_in = (rest_games['Score_Bin'] == bin_label).sum()
            rest_out = len(rest_games) - rest_in
            
            obs = np.array([[player_in, player_out], [rest_in, rest_out]])
            
            if obs.sum() > 0:
                # stats.chi2_contingency calculates the p-value
                chi2, p_val, dof, expected = stats.chi2_contingency(obs)
                
                # Check significance against our strict Bonferroni-adjusted alpha
                if p_val < adjusted_alpha:
                    p_pct = (player_in / len(player_games)) * 100
                    r_pct = (rest_in / len(rest_games)) * 100
                    direction = "MORE" if p_pct > r_pct else "LESS"
                    
                    significant_findings.append({
                        "Player": player, "Bin": bin_label, "Direction": direction, 
                        "P_Val": p_val, "Player_%": p_pct, "Rest_%": r_pct
                    })
                    
    # Sort by the magnitude of the difference so the most extreme anomalies rise to the top
    significant_findings.sort(key=lambda x: abs(x['Player_%'] - x['Rest_%']), reverse=True)
    return bin_pct, significant_findings

def get_radar_scaled_stats(df_career):
    """
    Min-Max scales baseline stats from 0 to 100 relative to the best performer in this cohort.
    MATLAB Analogy: mapminmax() or normalizing vectors to [0, 1].
    """
    stats_to_scale = ['PTS', 'AST', 'PLUS_MINUS', 'FG_PCT', 'TRB', 'STL', 'BLK', 'TOV']
    df_radar = df_career[['Player'] + stats_to_scale].copy()
    
    for col in stats_to_scale:
        max_val = df_radar[col].max()
        # Scale to 100. If max_val is 0 (safeguard), return 0.
        df_radar[col] = df_radar[col].apply(lambda x: (x / max_val * 100) if max_val > 0 else 0)
    return df_radar

def get_dumbbell_longevity_peak(df_goat):
    """
    Calculates 0-100 scaled scores for Peak (PPG) vs Longevity (Total Points).
    """
    df_peak = df_goat.groupby('Player')['points'].mean().reset_index().rename(columns={'points': 'Peak_PPG'})
    df_long = df_goat.groupby('Player')['points'].sum().reset_index().rename(columns={'points': 'Total_PTS'})
    df_merged = pd.merge(df_peak, df_long, on='Player')

    # Normalize both variables to a 0-100 scale so they can share an X-axis!
    df_merged['Peak_Score'] = (df_merged['Peak_PPG'] / df_merged['Peak_PPG'].max()) * 100
    df_merged['Longevity_Score'] = (df_merged['Total_PTS'] / df_merged['Total_PTS'].max()) * 100
    
    # Sort by Longevity to make the dumbbell chart look like a clean staircase
    df_merged = df_merged.sort_values('Longevity_Score', ascending=True)
    return df_merged

def get_google_trends():
    """Loads the 5-year Relative Search Volume (RSV) data."""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return pd.read_csv(os.path.join(base_dir, 'documents', 'real_google_trends.csv')).fillna(0)
    except: return pd.DataFrame()

def get_mvp_shares():
    """Loads career-summed MVP voting shares from Basketball-Reference."""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return pd.read_csv(os.path.join(base_dir, 'documents', 'real_mvp_shares.csv')).fillna(0)
    except: return pd.DataFrame()

def get_civic_awards():
    """
    Loads the Civic & Leadership awards matrix scraped from Wikipedia.
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Ensure this filename matches exactly what you downloaded from Colab
        civic_path = os.path.join(base_dir, 'documents', 'real_man_of_the_year.csv')
        df_civic = pd.read_csv(civic_path)
        return df_civic
    except Exception as e:
        st.error(f"Error loading Civic Awards: {e}")
        return pd.DataFrame()

def get_philanthropy_data():
    """
    Loads the Philanthropy footprint and sentence matrix.
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        phil_path = os.path.join(base_dir, 'documents', 'social_impact_scores_with_sentences.csv')
        df_phil = pd.read_csv(phil_path)
        return df_phil
    except Exception as e:
        print(f"Error loading Philanthropy data: {e}")
        return pd.DataFrame()

def calculate_cultural_impact_score(df_goat, df_mvp, df_trends, df_civic, df_phil):
    """
    Merges all Phase 4 datasets, applies Min-Max normalization, 
    and calculates a weighted out-of-100 Cultural Impact Score.
    """
    if any(df.empty for df in [df_goat, df_mvp, df_trends, df_civic, df_phil]):
        return pd.DataFrame()

    # 1. MVP (Zeitgeist Dominance)
    df_mvp_grouped = df_mvp.groupby('Player')['Share'].sum().reset_index()
    df_mvp_grouped.rename(columns={'Share': 'Total_MVP_Shares'}, inplace=True)

    # 2. Trends (Modern Relevance)
    df_trends_mean = df_trends.drop(columns=['date', 'Unnamed: 0'], errors='ignore').mean().reset_index()
    df_trends_mean.columns = ['Player', 'Mean_Search_Volume']

    # 3. Civic Awards (Character)
    award_cols = [col for col in df_civic.columns if col not in ['Player', 'Award', 'Won']]
    # If the civic matrix is pivoted, we just sum the numeric columns
    df_civic_copy = df_civic.copy()
    numeric_cols = df_civic_copy.select_dtypes(include='number').columns
    df_civic_copy['Civic_Hardware_Total'] = df_civic_copy[numeric_cols].sum(axis=1)
    df_civic_grouped = df_civic_copy[['Player', 'Civic_Hardware_Total']]

    # 4. Philanthropy (Real World Impact)
    df_phil_grouped = df_phil[['Player', 'Philanthropic_Footprint']]

    # 5. Master Merge (Anchored to our 50 players)
    df_master = df_goat[['Player']].drop_duplicates()
    df_master = pd.merge(df_master, df_mvp_grouped, on='Player', how='left')
    df_master = pd.merge(df_master, df_trends_mean, on='Player', how='left')
    df_master = pd.merge(df_master, df_civic_grouped, on='Player', how='left')
    df_master = pd.merge(df_master, df_phil_grouped, on='Player', how='left').fillna(0)

    # 6. Min-Max Normalization
    metrics = ['Total_MVP_Shares', 'Mean_Search_Volume', 'Civic_Hardware_Total', 'Philanthropic_Footprint']
    for col in metrics:
        min_val = df_master[col].min()
        max_val = df_master[col].max()
        if max_val - min_val == 0:
            df_master[f'{col}_Norm'] = 0.0
        else:
            df_master[f'{col}_Norm'] = (df_master[col] - min_val) / (max_val - min_val)

    # 7. The Weighting Schema
    WEIGHTS = {
        'Total_MVP_Shares_Norm': 0.35,     
        'Mean_Search_Volume_Norm': 0.20,   
        'Philanthropic_Footprint_Norm': 0.22, 
        'Civic_Hardware_Total_Norm': 0.23  
    }

    df_master['Cultural_Impact_Score'] = (
        (df_master['Total_MVP_Shares_Norm'] * WEIGHTS['Total_MVP_Shares_Norm']) +
        (df_master['Mean_Search_Volume_Norm'] * WEIGHTS['Mean_Search_Volume_Norm']) +
        (df_master['Philanthropic_Footprint_Norm'] * WEIGHTS['Philanthropic_Footprint_Norm']) +
        (df_master['Civic_Hardware_Total_Norm'] * WEIGHTS['Civic_Hardware_Total_Norm'])
    ) * 100

    df_master['Cultural_Impact_Score'] = df_master['Cultural_Impact_Score'].round(1)
    df_master = df_master.sort_values(by='Cultural_Impact_Score', ascending=False)
    
    return df_master

@st.cache_data
def load_all_dashboard_data():
    """
    The Master Wrapper: Calls all individual loaders and aggregators.
    Returns a tuple of every dataset needed for the dashboard.
    """
    # 1. Base Stats & Era Data
    df_goat = load_and_filter_raw_data() # Your existing function
    df_career = calculate_career_baselines(df_goat)
    df_era = get_era_adjusted_stats(df_goat)
    df_radar = get_radar_scaled_stats(df_career)
    
    # 2. Advanced Analysis (Tabs 1-3)
    df_clutch = run_clutch_analysis(df_goat) 
    df_awards = get_awards_hardware()
    df_scored, df_melted = run_scoring_segment_analysis(df_goat)
    df_longevity, bin_pct, significant_findings = analyze_longevity_vs_peak(df_goat)
    df_dumbbell = get_dumbbell_longevity_peak(df_goat)
    
    # 3. Cultural & Civic Data (Tab 4)
    df_google = get_google_trends()
    df_mvp = get_mvp_shares()
    df_civic = get_civic_awards()
    df_phil = get_philanthropy_data()
    
    # 4. The Master Impact Score (Calculated dynamically)
    df_impact_score = calculate_cultural_impact_score(df_goat, df_mvp, df_google, df_civic, df_phil)
    
    # 5. UI Helpers
    colors = get_player_colors(df_goat)
    
    # The order here MUST match the unpacking order in your app.py!
    return (
        df_goat, df_career, df_clutch, df_awards, df_scored, df_melted, 
        df_longevity, bin_pct, significant_findings, 
        df_era, df_radar, df_dumbbell, 
        df_google, df_mvp, df_civic, df_phil, 
        df_impact_score, colors
    )
