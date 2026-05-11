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
# MATLAB Bridge: This decorator acts like a persistent variable or a 
# memoized function. It ensures the 80+ rows (or millions in future datasets) 
# are only loaded from the hard drive once when the server boots.
@st.cache_data(show_spinner=False)
def load_edtech_master(base_dir):
    """ 
    Loads the unified master dataset into RAM from the compiled Parquet file. 
    """
    # 🎯 THE FIX: Target the newly namespaced parquet file
    master_path = os.path.join(base_dir, 'tech_in_ed_master_dataset.parquet')
    
    if os.path.exists(master_path):
        return pd.read_parquet(master_path)
    else:
        # Failsafe error that the UI will catch and display to the user
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
    Also calculates the Year-Over-Year (YoY) velocity of this gap.
    
    MATLAB Bridge: This utilizes standard vectorization (e.g., C = A - B) 
    and the diff() function for finding the discrete derivative.
    """
    # Create a safe copy in memory to avoid Pandas 'SettingWithCopyWarning'
    df_calc = df.copy()
    
    # Ensure our required columns exist before doing math
    required_cols = ['Curriculum_Complexity_Index', 'Learning_Efficiency_Score']
    for col in required_cols:
        if col not in df_calc.columns:
            raise KeyError(f"Missing required metric: {col}. Check your ETL schema.")

    # 1. Calculate the core gap (The "Cognitive Offload" requirement)
    df_calc['Knowledge_Gap'] = df_calc['Curriculum_Complexity_Index'] - df_calc['Learning_Efficiency_Score']
    
    # 2. Calculate the Velocity (First Derivative) of the Gap
    # We group by Country so a USA 2022 value doesn't subtract from a FIN 1995 value.
    df_calc['Gap_Velocity'] = df_calc.groupby('Country')['Knowledge_Gap'].diff().fillna(0)
    
    # 3. Calculate Technology Acceleration (YoY growth in Internet Penetration)
    if 'Internet_Penetration' in df_calc.columns:
        df_calc['Tech_Acceleration'] = df_calc.groupby('Country')['Internet_Penetration'].diff().fillna(0)
        
    return df_calc

def calculate_correlations(df, target_country='USA'):
    """
    Isolates a country and returns the Pearson correlation matrix for the core metrics.
    MATLAB Bridge: Equivalent to corrcoef().
    """
    # Filter for the specific country
    df_iso = df[df['Country'] == target_country]
    
    # Select only the numeric columns