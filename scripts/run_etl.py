# scripts/run_etl.py
import pandas as pd
import numpy as np
import os
import sys
import tarfile
import zipfile
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from loaders.typo_behavior_loader import (
    build_word_boundaries, 
    apply_typo_taxonomy, 
    calculate_raw_digraphs,
    apply_historical_consistency_filter
)


# ==========================================
# --- STRICT DIRECTORY & PATH CONFIGURATION ---
# ==========================================
base_dir = os.path.join(os.path.dirname(__file__), '..', 'documents')

# SOURCES (Read-Only: ETL must NEVER overwrite these)
aalto_source = os.path.join(base_dir, 'aalto_macro.parquet')
cmu_source = os.path.join(base_dir, 'cmu_baseline.parquet')
keyrecs_source = os.path.join(base_dir, 'keyrecs_micro.parquet') 
clarkson_1_source = os.path.join(base_dir, 'Clarkson-I-2014.tar.gz')
clarkson_2_source = os.path.join(base_dir, 'clarkson-II-2018-filtered_dataset.zip')

# OUTPUTS (The ONLY files this script is allowed to write)
aalto_out = os.path.join(base_dir, 'aalto_processed.parquet')
cmu_out = os.path.join(base_dir, 'cmu_processed.parquet')
keyrecs_out = os.path.join(base_dir, 'keyrecs_processed.parquet')
clarkson_out = os.path.join(base_dir, 'clarkson_processed.parquet')
master_out = os.path.join(base_dir, 'master_dataset.parquet')


# ==========================================
# --- HELPER FUNCTIONS ---
# ==========================================
def optimize_memory(df):
    """ Downcasts 64-bit arrays to 32-bit and converts strings to categories to slash RAM usage by 75%. """
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('Int32')
        elif df[col].dtype == 'object':
            df[col] = df[col].astype('category')
    return df

def ingest_clarkson_I(tar_path):
    dfs = []
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            if member.name.split('/')[-1].startswith('._'): continue
            if member.isfile() and member.name.endswith('.txt'):
                user_id = f"C1_{member.name.split('/')[1]}"
                f = tar.extractfile(member)
                if f:
                    content = f.read().decode('utf-8', errors='ignore').strip()
                    parsed_data = []
                    for line in content.split('\n'):
                        line_parts = line.split('\t')
                        if len(line_parts) >= 4:
                            task_id = line_parts[1]
                            events = line_parts[3].split(',')
                            for event in events:
                                parts = event.split(':')
                                if len(parts) >= 3 and parts[0].isdigit():
                                    try:
                                        action_type = 'PRESS' if parts[0] == '0' else 'RELEASE'
                                        parsed_data.append({
                                            'Timestamp_ms': float(parts[2]),
                                            'Action_Type': action_type,
                                            'Key_Code': int(parts[1]),
                                            'Task_Type': task_id
                                        })
                                    except ValueError:
                                        continue
                    if parsed_data:
                        df_user = pd.DataFrame(parsed_data, columns=['Timestamp_ms', 'Action_Type', 'Key_Code', 'Task_Type'])
                        df_user['Participant_ID'] = user_id
                        df_user['Source_Dataset'] = 'Clarkson_I'
                        dfs.append(df_user)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def ingest_clarkson_II(folder_path):
    dfs = []
    # 🛡️ THE FIX: os.walk recursively hunts through any hidden nested subfolders
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            # Clarkson files are named purely with digits (e.g., '12345')
            if file_name.isdigit():
                file_path = os.path.join(root, file_name)
                # Use delim_whitespace and skip corrupted rows with extra columns
                df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=['Timestamp_Ticks', 'Action', 'Key_Code'], on_bad_lines='skip')
                df['Participant_ID'] = f"C2_{file_name}"
                df['Timestamp_ms'] = (df['Timestamp_Ticks'] - 116444736000000000) / 10000
                df['Action_Type'] = df['Action'].map({1: 'PRESS', 0: 'RELEASE'})
                df['Source_Dataset'] = 'Clarkson_II'
                dfs.append(df)
                
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


if __name__ == "__main__":
    
    # ==========================================
    # 1. PROCESS AALTO
    # ==========================================
    if os.path.exists(aalto_source):
        print("Loading raw Aalto dataset...")
        df_aalto = pd.read_parquet(aalto_source)

        # Rename to Master Schema naming convention
        # We check for both 'Timestamp' (from Colab) and 'PRESS_TIME' (if raw)
        df_aalto = df_aalto.rename(columns={
            'PARTICIPANT_ID': 'Participant_ID',
            'Timestamp': 'Timestamp_ms',   # <--- THE CRITICAL FIX
            'PRESS_TIME': 'Timestamp_ms' 
        })
        
        # Aalto rows are already individual keypresses; we tag them as PRESS
        df_aalto['Action_Type'] = 'PRESS'
        df_aalto['Source_Dataset'] = 'Aalto'
        
        # Generate the missing Session_ID required by the taxonomy function
        df_aalto['Session_ID'] = 'Aalto_' + df_aalto['Participant_ID'].astype(str)

        print("Running Phase 1 Pipeline Calculations...")
        df_aalto = calculate_raw_digraphs(df_aalto)
        df_aalto = apply_typo_taxonomy(df_aalto)
        df_aalto = build_word_boundaries(df_aalto)
        df_aalto = apply_historical_consistency_filter(df_aalto)

        # Apply to Aalto before saving
        df_aalto = optimize_memory(df_aalto)
        df_aalto.to_parquet(aalto_out, index=False)
        print(f"✅ Saved Aalto: {len(df_aalto)} rows to {os.path.basename(aalto_out)}")
    else:
        print("Raw Aalto file not found.")

    # ==========================================
    # 2. PROCESS CLARKSON I & II
    # ==========================================
    print("Processing Clarkson Datasets...")

    # --- CLARKSON I (Reads directly from tarball) ---
    df_c1 = ingest_clarkson_I(clarkson_1_source) if os.path.exists(clarkson_1_source) else pd.DataFrame()
    
    # --- CLARKSON II (Requires extracted folder) ---
    clarkson_ii_folder = os.path.join(base_dir, 'clarkson_2_extracted') 
    if not os.path.exists(clarkson_ii_folder) and os.path.exists(clarkson_2_source):
        print(f"Extracting {clarkson_2_source}...")
        with zipfile.ZipFile(clarkson_2_source, 'r') as zip_ref:
            zip_ref.extractall(path=clarkson_ii_folder) 
            
    df_c2 = ingest_clarkson_II(clarkson_ii_folder) if os.path.exists(clarkson_ii_folder) else pd.DataFrame()

    # Combine all raw Clarkson data
    df_clarkson_raw = pd.concat([df_c1, df_c2], ignore_index=True)

    if not df_clarkson_raw.empty:
        # Separate and sort  
        presses = df_clarkson_raw[df_clarkson_raw['Action_Type'] == 'PRESS'].sort_values('Timestamp_ms')
        releases = df_clarkson_raw[df_clarkson_raw['Action_Type'] == 'RELEASE'].sort_values('Timestamp_ms')

        # Explicitly duplicate the timestamp so it survives the merge as a data column
        releases['Timestamp_ms_Release'] = releases['Timestamp_ms']

        # Heavy ETL Merge
        print("Running merge_asof matrix operation for Clarkson...")
        # merge_asof requires both dataframes to be sorted by the timestamp
        df_merged = pd.merge_asof(
            presses, 
            releases[['Participant_ID', 'Key_Code', 'Timestamp_ms', 'Timestamp_ms_Release']], 
            on='Timestamp_ms', 
            by=['Participant_ID', 'Key_Code'], 
            direction='forward'
        )
        
        # Map to Master Schema
        df_merged['Device_Type'] = 'desktop' 
        df_merged['Session_ID'] = df_merged['Source_Dataset'] + '_' + df_merged['Participant_ID'].astype(str)
        
        # Calculate Latencies using our new standardized names
        df_merged['Hold_Time_ms'] = df_merged['Timestamp_ms_Release'] - df_merged['Timestamp_ms']
        
        df_merged = df_merged.sort_values(by=['Participant_ID', 'Timestamp_ms'])
        df_merged['Flight_DD_ms'] = df_merged.groupby('Participant_ID')['Timestamp_ms'].diff().fillna(0)
        
        # Route through Phase 1 Taxonomy
        df_merged = apply_typo_taxonomy(df_merged)
        df_merged = build_word_boundaries(df_merged)
        df_merged = apply_historical_consistency_filter(df_merged)

        # Apply to Clarkson before saving
        df_merged = optimize_memory(df_merged)
        
        # Save as Combined Clarkson processed file
        df_merged.to_parquet(clarkson_out, index=False)
        print(f"✅ Saved Clarkson Combined: {len(df_merged)} rows to {os.path.basename(clarkson_out)}")

    # ==========================================
    # 3. PROCESS CMU
    # ==========================================
    # --- CMU (Seconds to Milliseconds & ID Alignment) ---
    if os.path.exists(cmu_source):
        # --- CMU SOURCE PROTECTION ---
        # We treat cmu_baseline as a fixed reference. We never overwrite it.
        print("Loading CMU Baseline Source...")
        df_cmu = pd.read_parquet(cmu_source)
        
        # 1. Convert timing columns from Seconds to Milliseconds
        timing_cols = [c for c in df_cmu.columns if any(pref in c for pref in ['H.', 'DD.', 'UD.'])]
        
        # CRITICAL FIX: Only multiply if the max value is < 15. 
        # If it's already > 15, it means it was already converted to ms in a previous run!
        if not df_cmu[timing_cols].empty and df_cmu[timing_cols].max().max() < 15:
            print("Converting CMU seconds to milliseconds...")
            df_cmu[timing_cols] = df_cmu[timing_cols] * 1000
        else:
            print("CMU already in milliseconds. Skipping conversion.")
        
        # 2. Standardize Schema
        df_cmu = df_cmu.rename(columns={'subject': 'Participant_ID'})
        df_cmu['Source_Dataset'] = 'CMU'
        df_cmu['Action_Type'] = 'DIGRAPH'
        df_cmu['Task_Type'] = 'Password'
        
        # We save to 'cmu_processed' to keep the original file untouched
        df_cmu.to_parquet(cmu_out, index=False)
        print(f"✅ Created derived CMU: {len(df_cmu)} rows to {os.path.basename(cmu_out)}")
    else:
        print(f"❌ ERROR: {cmu_source} not found. Baseline must be present.")

    # ==========================================
    # 4. PROCESS KEYRECS
    # ==========================================
    if os.path.exists(keyrecs_source):
        # --- KEYRECS SOURCE PROTECTION ---
        print("Loading KeyRecs Source...")
        df_keyrecs = pd.read_parquet(keyrecs_source)
        
        # Standardize Schema
        df_keyrecs = df_keyrecs.rename(columns={
            'participant': 'Participant_ID',
            'DD.key1.key2': 'Flight_DD_ms', 
            'UD.key1.key2': 'Flight_UD_ms'
        })
        df_keyrecs['Source_Dataset'] = 'KeyRecs'
        df_keyrecs['Action_Type'] = 'DIGRAPH'
        df_keyrecs['Task_Type'] = 'Mixed'
        
        # We save to 'keyrecs_processed' to keep the original file untouched
        df_keyrecs.to_parquet(keyrecs_out, index=False)
        print(f"✅ Created derived KeyRecs: {len(df_keyrecs)} rows to {os.path.basename(keyrecs_out)}")
    else:
        print(f"❌ ERROR: {keyrecs_source} not found.")

    # ==========================================
    # 5. BUILD THE UNIFIED MASTER DATASET
    # ==========================================
    print("Combining all datasets into Master Matrix...")
    master_dfs = []
    
    # Strictly pull only from the immutable `_processed` files we just built
    for out_path, name in [(aalto_out, 'Aalto'), (clarkson_out, 'Clarkson'), (cmu_out, 'CMU'), (keyrecs_out, 'KeyRecs')]:
        if os.path.exists(out_path):
            master_dfs.append(pd.read_parquet(out_path))
            print(f"  -> Loaded {name} for Fusion.")

    if master_dfs:
        df_master = pd.concat(master_dfs, ignore_index=True)
        
        # Map final columns
        df_master = df_master.rename(columns={'Dataset': 'Source_Dataset', 'Flight_Time': 'Flight_DD_ms'})
        df_master = df_master.loc[:, ~df_master.columns.duplicated()]
        
        if 'Participant_ID' in df_master.columns:
            df_master['Participant_ID'] = df_master['Participant_ID'].astype(str)
        if 'Session_ID' in df_master.columns:
            df_master['Session_ID'] = df_master['Session_ID'].astype(str)

        if 'Key_Code' in df_master.columns:
            is_backspace = df_master['Key_Code'].isin([8, 8.0, '8', 'Back', 'Backspace'])
            df_master['Is_Typo'] = is_backspace.shift(-1).fillna(False)
            
        float_cols = df_master.select_dtypes(include=['float64']).columns
        if not float_cols.empty:
            df_master[float_cols] = df_master[float_cols].astype('float32')
            
        obj_cols = df_master.select_dtypes(include=['object', 'str']).columns
        if not obj_cols.empty:
            df_master[obj_cols] = df_master[obj_cols].astype('category')
            
        if 'Timestamp_ms' in df_master.columns:
            df_master = df_master.sort_values(['Participant_ID', 'Timestamp_ms'])

        # =========================================================
        # 🚨 FAILSAFE DATA SANITIZER (V3) 🚨
        # =========================================================
        print("Sanitizing timing metrics...")
        
        # 1. Ensure Timestamps are standard numbers, NOT datetimes
        if 'Timestamp_ms' in df_master.columns:
            df_master['Timestamp_ms'] = pd.to_numeric(df_master['Timestamp_ms'], errors='coerce')

        # 2. Nuke the 10^20 Overflow Bug & Enforce Statistical Boundaries
        for col in ['Flight_DD_ms', 'Hold_Time_ms']:
            if col in df_master.columns:
                # Force raw float32
                df_master[col] = pd.to_numeric(df_master[col], errors='coerce').astype('float32')
                
                # Wipe out impossible negative times and outliers > 2000ms
                # Anything > 2000ms is not muscle memory anyway, so we safely drop it
                df_master.loc[df_master[col] > 2000, col] = np.nan
                df_master.loc[df_master[col] < 0, col] = np.nan

        # 3. Clean up legacy columns to prevent PyArrow mixed-type crashes
        if 'User_ID' in df_master.columns:
            df_master = df_master.drop(columns=['User_ID'])

        # 4. Export final fused dataset
        df_master.to_parquet(master_out, index=False, compression='snappy')
        print(f"✅ Master Dataset Exported! Total rows: {len(df_master)}")
    else:
        print("❌ CRITICAL: No processed files found for fusion.")
        sys.exit(1)