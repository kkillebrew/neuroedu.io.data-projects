# scripts/run_etl.py
import pandas as pd
import os
import sys
import tarfile
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from loaders.typo_behavior_loader import (
    apply_typo_taxonomy, build_word_boundaries, apply_historical_consistency_filter
)

base_dir = os.path.join(os.path.dirname(__file__), '..', 'documents')
aalto_path = os.path.join(base_dir, 'aalto_macro.parquet')

def ingest_clarkson_II(folder_path):
    """ Parses the raw tab-separated Clarkson II files. """
    dfs = []
    for file_name in os.listdir(folder_path):
        if file_name.isdigit(): # E.g., '25563'
            file_path = os.path.join(folder_path, file_name)
            # Read tab-separated values
            df = pd.read_csv(file_path, sep='\t', header=None, names=['Timestamp_Ticks', 'Action', 'Key_Code'])
            df['PARTICIPANT_ID'] = f"C2_{file_name}"
            # Convert 100-nanosecond Windows ticks to milliseconds
            df['Timestamp'] = df['Timestamp_Ticks'] / 10000.0
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def ingest_clarkson_I(tar_path):
    """ Reads raw keystrokes directly out of the compressed tar archive. """
    dfs = []
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith('.txt'): # Adjust extension if needed based on tar contents
                user_id = f"C1_{member.name.split('/')[1]}" # Extract user ID from the path
                f = tar.extractfile(member)
                if f:
                    content = f.read().decode('utf-8')
                    # Standard parsing logic goes here depending on the exact comma/colon delimiter pattern
                    # For now, we will create a placeholder dataframe structure to merge
                    pass
    return pd.DataFrame() # Replace with actual concat once internal structure is fully mapped

# ==========================================
# 1. PROCESS AALTO
# ==========================================
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

# ==========================================
# 2. PROCESS CLARKSON I & II
# ==========================================
print("Processing Clarkson Datasets...")

clarkson_ii_folder = os.path.join(base_dir, 'filtered_dataset')
clarkson_i_tar = os.path.join(base_dir, 'Clarkson-I-2014.tar.gz')

df_c2 = ingest_clarkson_II(clarkson_ii_folder)
df_c1 = ingest_clarkson_I(clarkson_i_tar)

# Combine all raw Clarkson data (currently just C2 until C1 parsing is finished)
df_clarkson_raw = pd.concat([df_c1, df_c2], ignore_index=True) if not df_c1.empty else df_c2

if not df_clarkson_raw.empty:
    # Separate and sort
    presses = df_clarkson_raw[df_clarkson_raw['Action'] == 0].copy()
    releases = df_clarkson_raw[df_clarkson_raw['Action'] == 1].copy()
    
    presses = presses.rename(columns={'Timestamp': 'Timestamp_Press'})
    releases = releases.rename(columns={'Timestamp': 'Timestamp_Release'})
    
    presses = presses.sort_values(by=['PARTICIPANT_ID', 'Key_Code', 'Timestamp_Press'])
    releases = releases.sort_values(by=['PARTICIPANT_ID', 'Key_Code', 'Timestamp_Release'])

    # Heavy ETL Merge
    print("Running merge_asof matrix operation for Clarkson...")
    df_merged = pd.merge_asof(
        presses, 
        releases[['PARTICIPANT_ID', 'Key_Code', 'Timestamp_Release']], 
        left_on='Timestamp_Press', right_on='Timestamp_Release', 
        by=['PARTICIPANT_ID', 'Key_Code'], direction='forward'
    )
    
    # Map to Master Schema
    df_merged = df_merged.rename(columns={'PARTICIPANT_ID': 'User_ID'})
    df_merged['Dataset'] = 'Clarkson'
    df_merged['Device_Type'] = 'desktop' 
    df_merged['Session_ID'] = df_merged['Dataset'] + '_' + df_merged['User_ID']
    df_merged['Dwell_Time'] = df_merged['Timestamp_Release'] - df_merged['Timestamp_Press']
    
    df_merged = df_merged.sort_values(by=['User_ID', 'Timestamp_Press'])
    df_merged['Flight_Time'] = df_merged.groupby('User_ID')['Timestamp_Press'].diff().fillna(0)
    
    # Route through Phase 1 Taxonomy
    df_merged = apply_typo_taxonomy(df_merged)
    df_merged = build_word_boundaries(df_merged)
    df_merged = apply_historical_consistency_filter(df_merged)

    # Export the finalized Parquet
    clarkson_output = os.path.join(base_dir, 'clarkson_processed.parquet')
    df_merged.to_parquet(clarkson_output, index=False)
    print(f"✅ Saved Clarkson: {len(df_merged)} rows.")

print("ETL Pipeline Complete!")
