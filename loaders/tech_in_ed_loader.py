# =====================================================================
# MODULE: loaders/tech_in_ed_loader.py (The Model)
# PURPOSE: Data loading, memory caching, and mathematical transformations.
# STRICT DECOUPLING: No Streamlit UI rendering commands permitted here.
# =====================================================================
import pandas as pd
import numpy as np
import streamlit as st
import os

# ---------------------------------------------------------------------
# PHASE 1: CACHED DATA LOADING (Memory Management)
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_edtech_master(base_dir):
    """ 
    Loads the unified master dataset into RAM from the compiled Parquet file. 
    """
    master_path = os.path.join(base_dir, 'tech_in_ed_master_dataset.parquet')
    
    if os.path.exists(master_path):
        return pd.read_parquet(master_path)
    else:
        raise FileNotFoundError(
            "EdTech master dataset not found. Ensure you have run 'scripts/tech_in_ed_etl.py' "
            "to compile the raw Colab outputs."
        )

# ---------------------------------------------------------------------
# PHASE 2: FEATURE ENGINEERING & MATH (The Calculations)
# ---------------------------------------------------------------------
def calculate_knowledge_gap(df):
    """
    Calculates the delta between curriculum requirements and human learning efficiency.
    Utilizes vectorization to handle the 22-year timeline.
    """
    df_calc = df.copy()
    
    # Ensure our required columns exist. We prioritize the composite 
    # 'Learning_Efficiency_Score' created during ETL.
    required_cols = ['Curriculum_Complexity_Index', 'Learning_Efficiency_Score']
    for col in required_cols:
        if col not in df_calc.columns:
            # If the column is missing, we initialize it as NaN to prevent crash
            df_calc[col] = np.nan

    # 1. Calculate the core gap (Complexity vs. Efficiency)
    df_calc['Knowledge_Gap'] = df_calc['Curriculum_Complexity_Index'] - df_calc['Learning_Efficiency_Score']
    
    # 2. Calculate the Velocity (First Derivative) of the Gap
    # Grouping by Country ensures the time-series continuity is preserved per-nation.
    df_calc['Gap_Velocity'] = df_calc.groupby('Country')['Knowledge_Gap'].diff().fillna(0)
    
    # 3. New: Subject-Specific Gains (Math vs. Science vs. Reading)
    # This tracks if certain cognitive domains are accelerating faster than others.
    scores = ['Math_Score', 'Reading_Score', 'Science_Score']
    for score in scores:
        if score in df_calc.columns:
            df_calc[f'{score}_Velocity'] = df_calc.groupby('Country')[score].diff().fillna(0)

    return df_calc

def calculate_correlations(df, target_country='USA'):
    """
    Isolates a country and returns the Pearson correlation matrix for core metrics.
    Useful for identifying if Internet_Penetration correlates with score shifts.
    """
    df_iso = df[df['Country'] == target_country]
    
    # expanded list to include our new timeline subjects
    cols_to_correlate = [
        'Internet_Penetration', 
        'Curriculum_Complexity_Index', 
        'Learning_Efficiency_Score',
        'Math_Score',
        'Reading_Score',
        'Science_Score',
        'Digital_Reading_Score',
        'Problem_Solving_Score',
        'Knowledge_Gap'
    ]
    
    # Filter for columns that actually exist in the final master merge
    valid_cols = [c for c in cols_to_correlate if c in df_iso.columns]
    
    # Generate matrix and fill NaNs (common in early legacy years with missing values)
    corr_matrix = df_iso[valid_cols].corr().fillna(0)
    
    return corr_matrix

def get_country_summary(df, country_code):
    """
    Returns a dictionary of key performance indicators for a specific country.
    Calculates the 'Legacy vs. Modern' improvement across the 22-year span.
    """
    country_df = df[df['Country'] == country_code].sort_values('Year')
    
    if country_df.empty:
        return None
    
    summary = {
        'start_year': int(country_df['Year'].min()),
        'end_year': int(country_df['Year'].max()),
        'avg_math': country_df['Math_Score'].mean(),
        'max_complexity': country_df['Curriculum_Complexity_Index'].max(),
        'net_gap_change': country_df['Knowledge_Gap'].iloc[-1] - country_df['Knowledge_Gap'].iloc[0] if len(country_df) > 1 else 0
    }
    
    return summary

def get_pisa_snapshots(df, rows=10):
    """
    Groups the master dataframe by Year and returns a dictionary 
    of the first N rows for each unique PISA cycle.
    """
    snapshots = {}
    # Ensure we only look at years that actually exist in the data
    unique_years = sorted(df['Year'].unique())
    
    for year in unique_years:
        # Filter and take the top N rows
        snapshots[year] = df[df['Year'] == year].head(rows)
        
    return snapshots

def get_benchmark_comparison_data(df):
    """
    Filters for the 5 longitudinal benchmark countries.
    """
    benchmarks = ['USA', 'JPN', 'DEU', 'ARG', 'JOR']
    return df[df['Country'].isin(benchmarks)].sort_values(['Country', 'Year'])