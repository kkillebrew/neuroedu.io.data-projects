# scripts/run_etl.py
import pandas as pd
import os
import sys

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

    print("Exporting full processed UI array...")
    # OVERWRITE the raw file with the fully processed mathematical version
    output_path = os.path.join(base_dir, 'aalto_processed.parquet')
    df.to_parquet(output_path, index=False)
    
    print(f"ETL Pipeline Complete! Saved {len(df)} rows to {output_path}")
else:
    print("Raw Aalto file not found.")
