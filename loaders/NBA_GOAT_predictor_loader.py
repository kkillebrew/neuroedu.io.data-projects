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

# -------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# -------------------------------------------------------------------
# CRITICAL: This is the exact variable your UI is looking for!
# Define our target cohort. In MATLAB, this would be a string array or cell array.
PLAYERS = [
    "Michael Jordan", "LeBron James", "Magic Johnson", "Stephen Curry", 
    "Shaquille O'Neal", "Kareem Abdul-Jabbar", "Kobe Bryant", 
    "Bill Russell", "Wilt Chamberlain", "Nikola Jokic"
]

def get_player_colors():
    """
    Maintains our strict color consistency rule for the UI layer.
    
    MATLAB Analogy: 
    This is equivalent to creating a custom colormap (e.g., customMap = lines(10)) 
    and ensuring every subsequent plot uses the exact same color-to-entity mapping 
    so visual identification remains consistent across figures.
    """
    import plotly.express as px
    palette = px.colors.qualitative.Bold
    # Creates a dictionary mapping player name to a specific hex color
    return {player: palette[i % len(palette)] for i, player in enumerate(PLAYERS)}

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
        
        # MATLAB Analogy: groupsummary(df, 'Year', {'mean', 'std'})
        # We calculate the average and standard deviation of points and rebounds for EVERY year in history
        league_yearly = df_all.groupby('Year').agg({
            'points': ['mean', 'std'],
            'reboundsTotal': ['mean', 'std']
        }).reset_index()
        
        # Flatten the multi-level columns created by agg() so they are easy to use
        league_yearly.columns = ['Year', 'pts_mean', 'pts_std', 'reb_mean', 'reb_std']

        # Logical Indexing: We filter the 50 players down to just the 10 active ones in PLAYERS
        df_goat = df_all[df_all['Player'].isin(PLAYERS)].copy()

        # MATLAB Analogy: innerjoin()
        # Merge the historical yearly averages back to our specific GOAT players' game logs
        df_goat = pd.merge(df_goat, league_yearly, on='Year', how='left')
        
        # Z-Score Formula: (Player Score - League Average) / League Standard Deviation
        df_goat['pts_z'] = (df_goat['points'] - df_goat['pts_mean']) / df_goat['pts_std']
        df_goat['reb_z'] = (df_goat['reboundsTotal'] - df_goat['reb_mean']) / df_goat['reb_std']
        
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
    df_career = df_goat.groupby('Player')[['points', 'reboundsTotal', 'assists', 'blocks', 'steals', 'turnovers']].mean().reset_index()
    # Rename columns for cleaner UI presentation (fillna prevents errors for 1960s players)
    df_career.rename(columns={'points': 'PTS', 'reboundsTotal': 'TRB', 'assists': 'AST', 'blocks': 'BLK', 'steals': 'STL', 'turnovers': 'TOV'}, inplace=True)
    df_career.fillna(0, inplace=True)
    
    # 2. Clutch Factor (Playoffs vs Regular Season)
    # We group by two variables here, then use `.unstack()` to pivot the 'gameType' into separate columns
    df_clutch = df_goat.groupby(['Player', 'gameType'])['points'].mean().unstack().reset_index()
    if 'Regular Season' in df_clutch.columns and 'Playoffs' in df_clutch.columns:
        df_clutch.rename(columns={'Regular Season': 'Reg_Season_PTS', 'Playoffs': 'Finals_PTS'}, inplace=True)
    
    return df_career, df_clutch

def get_awards_hardware():
    """
    Returns the hardcoded awards DataFrame (as this requires external scraping not present in our CSV).
    
    MATLAB Analogy: 
    Creating a structured table from scratch using `table(array1, array2, ...)`.
    """
    return pd.DataFrame({
        "Player": PLAYERS,
        "MVPs": [5, 4, 3, 2, 1, 6, 1, 5, 4, 3],
        "Rings": [6, 4, 5, 4, 4, 6, 5, 11, 2, 1],
        "Finals_MVPs": [6, 4, 3, 1, 3, 2, 2, 0, 1, 1], 
        "DPOY": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
    })

def get_era_adjusted_stats(df_goat):
    """
    Calculates the real Era-Adjusted Dominance (Z-Scores) by averaging
    the game-by-game Z-scores we calculated during ingestion.
    """
    # We use 'pts_mean' (the average points scored by an average NBA player that year) as a proxy for Era Pace
    df_era = df_goat.groupby('Player')[['pts_z', 'reb_z', 'pts_mean']].mean().reset_index()
    
    # Rename columns so they instantly link to our Plotly UI
    df_era.rename(columns={
        'pts_z': 'Scoring_Z_Score',
        'reb_z': 'Rebound_Z_Score',
        'pts_mean': 'Era_Pace'
    }, inplace=True)

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
    stats_to_scale = ['PTS', 'TRB', 'AST', 'BLK', 'STL']
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