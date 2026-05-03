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
# -------------------------------------------------------------------
# DATA INGESTION (The "Model")
# -------------------------------------------------------------------
def load_and_filter_raw_data():
    """
    Downloads the massive Kaggle dataset and strictly filters it to our 10 candidates.
    Returns a memory-efficient DataFrame of individual game logs.
    
    MATLAB Analogy: 
    This process mimics using `readtable()` with specific `'ReadVariableNames'` 
    and `'ReadRowNames'` to save memory, followed by logical indexing 
    (e.g., `data(ismember(data.Player, players), :)`) to subset the data.
    """
    try:
        # Note for DigitalOcean Deployment: If you don't want to download 1GB every time 
        # your Docker container spins up, you can pre-download this CSV and place it in 
        # your 'documents/' folder, then change this path to point locally.
        path = kagglehub.dataset_download("eoinamoore/historical-nba-data-and-player-box-scores")
        player_stats_path = os.path.join(path, 'PlayerStatistics.csv')
        
        # We explicitly load ONLY the columns we need to prevent memory exhaustion (RAM optimization)
        cols_to_load = ['firstName', 'lastName', 'gameType', 'gameDate', 'points', 'reboundsTotal', 'assists', 'blocks']
        df_all = pd.read_csv(player_stats_path, usecols=cols_to_load)
        
        # Combine first and last name for easier matching
        df_all['Player'] = df_all['firstName'] + " " + df_all['lastName']
        
        # Logical Indexing: Keep only rows where 'Player' exists in our PLAYERS list
        df_goat = df_all[df_all['Player'].isin(PLAYERS)].copy()
        
        # Extract Year for Longevity calculations (datetime conversion)
        # MATLAB Analogy: year(datetime(df_goat.gameDate))
        df_goat['Year'] = pd.to_datetime(df_goat['gameDate']).dt.year
        return df_goat
        
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")

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
    df_career = df_goat.groupby('Player')[['points', 'reboundsTotal', 'assists', 'blocks']].mean().reset_index()
    # Rename columns for cleaner UI presentation
    df_career.rename(columns={'points': 'PTS', 'reboundsTotal': 'TRB', 'assists': 'AST', 'blocks': 'BLK'}, inplace=True)
    
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