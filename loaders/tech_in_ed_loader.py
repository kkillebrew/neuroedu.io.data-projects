# =====================================================================
# MODULE: loaders/tech_in_ed_loader.py (The Model)
# PURPOSE: Data loading, memory caching, and mathematical transformations.
# STRICT DECOUPLING: No Streamlit UI rendering commands permitted here.
# =====================================================================
import pandas as pd
import numpy as np
import streamlit as st
import glob
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
    """
    df_calc = df.copy()
    
    required_cols = ['Curriculum_Complexity_Index', 'Learning_Efficiency_Score']
    for col in required_cols:
        if col not in df_calc.columns:
            df_calc[col] = np.nan

    # 1. Calculate the core gap (Complexity vs. Efficiency)
    df_calc['Knowledge_Gap'] = df_calc['Curriculum_Complexity_Index'] - df_calc['Learning_Efficiency_Score']
    
    # 2. Calculate the Velocity (First Derivative) of the Gap
    df_calc['Gap_Velocity'] = df_calc.groupby('Country')['Knowledge_Gap'].diff().fillna(0)
    
    # 3. FIXED: Use the actual mapped column names for Subject Velocities
    scores = ['Learning_Efficiency_Score', 'Reading_Proficiency_Score', 'Science_Proficiency_Score']
    for score in scores:
        if score in df_calc.columns:
            # Extract just the first word (Learning, Reading, Science) for the velocity column name
            prefix = score.split('_')[0] 
            df_calc[f'{prefix}_Velocity'] = df_calc.groupby('Country')[score].diff().fillna(0)

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

# ---------------------------------------------------------------------
# PHASE 3: UI HELPER FUNCTIONS (Filtering & Sampling)
# ---------------------------------------------------------------------
def get_pisa_grid_samples(df, rows_per_year=10):
    """
    Randomly samples one row for each of the 10 target countries per year.
    Returns a dictionary of dataframes for the 4x2 grid.
    """
    targets = ['USA', 'MEX', 'ARG', 'BRA', 'DEU', 'GBR', 'JPN', 'CHN', 'JOR', 'MAR']
    
    snapshots = {}
    unique_years = sorted(df['Year'].unique())
    
    for year in unique_years:
        year_data = df[df['Year'] == year]
        sampled_rows = year_data[year_data['Country'].isin(targets)]
        
        if len(sampled_rows) > rows_per_year:
            snapshots[year] = sampled_rows.sample(n=rows_per_year).sort_values('Country')
        else:
            snapshots[year] = sampled_rows.sort_values('Country')
            
    return snapshots

def get_benchmark_comparison_data(df):
    """
    Filters for the 5 longitudinal benchmark countries for the distribution trends.
    """
    benchmarks = ['USA', 'JPN', 'DEU', 'ARG', 'JOR']
    return df[df['Country'].isin(benchmarks)].sort_values(['Country', 'Year'])

# Load raw micro data filtered by country
def get_micro_cloud_data(base_dir, target_countries, sample_per_group=500):
    """
    Loads raw student micro-data, filters for benchmark countries, 
    and samples the points to prevent browser memory crashes during plotting.
    """
    pisa_raw_dir = os.path.join(base_dir, 'PSA_Outputs')
    micro_files = glob.glob(os.path.join(pisa_raw_dir, "PISA_*_micro.parquet"))

    if not micro_files:
        return pd.DataFrame()

    all_micro = []
    for f in micro_files:
        try:
            df_m = pd.read_parquet(f)
            # Filter for target countries
            df_m = df_m[df_m['Country'].isin(target_countries)]
            
            # Sample heavily to keep the UI snappy (e.g., 500 points per country per year)
            # 'observed=True' handles categorical groupings safely
            df_m = df_m.groupby('Country', observed=True).sample(
                n=min(len(df_m), sample_per_group), 
                replace=True, # Allows sampling even if a country has fewer rows
                random_state=42
            )
            all_micro.append(df_m)
        except Exception as e:
            print(f"Skipping micro file {f}: {e}")

    if all_micro:
        df_cloud = pd.concat(all_micro, ignore_index=True)
        # Re-map the Colab output names to our Master ETL schema names
        df_cloud = df_cloud.rename(columns={
            'Math_Score': 'Learning_Efficiency_Score',
            'Reading_Score': 'Reading_Proficiency_Score',
            'Science_Score': 'Science_Proficiency_Score'
        })
        return df_cloud
        
    return pd.DataFrame()