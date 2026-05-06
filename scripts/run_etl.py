# scripts/run_etl.py
import pandas as pd
import os
import sys

# Ensure Python can find your loaders folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from loaders.typo_behavior_loader import (
    apply_typo_taxonomy, build_word_boundaries, apply_historical_consistency_filter
)

base_dir = os.path.join(os.path.dirname(__file__), '..', 'documents')
aalto_path = os.path.join(base_dir, 'aalto_macro.parquet')

if os.path.exists(aalto_path):
    print("Loading raw Aalto dataset...")
    df = pd.read_parquet(aalto_path)

    print("Running Phase 1 Pipeline Calculations...")
    df = apply_typo_taxonomy(df)
    df = build_word_boundaries(df)
    df = apply_historical_consistency_filter(df)

    print("Exporting production-ready UI arrays...")
    # 1. Export just the typos for Tab 1 and Tab 3
    df_typos = df[df['Is_Typo'] == True]
    df_typos.to_parquet(os.path.join(base_dir, 'aalto_typos_only.parquet'), index=False)

    # 2. Export the baseline metrics for Tab 2
    df_base = df[['Flight_Time', 'Dwell_Time']].dropna()
    df_base.to_parquet(os.path.join(base_dir, 'aalto_baseline_arrays.parquet'), index=False)
    
    print("ETL Pipeline Complete!")
else:
    print("Raw Aalto file not found.")
