# scripts/run_etl.py
import pandas as pd
import os
import sys
import tarfile
import zipfile
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from loaders.typo_behavior_loader import (
    build_word_boundaries, 
    apply_typo_taxonomy, 
    calculate_raw_digraphs,  # <--- Add this
    apply_historical_consistency_filter  # <--- Add this line
)

base_dir = os.path.join(os.path.dirname(__file__), '..', 'documents')
aalto_path = os.path.join(base_dir, 'aalto_macro.parquet')

def optimize_memory(df):
    """ Downcasts 64-bit arrays to 32-bit and converts strings to categories to slash RAM usage by 75%. """
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            # Use nullable integer type in case of NaNs
            df[col] = df[col].astype('Int32')
        elif df[col].dtype == 'object':
            # Convert repetitive strings (like 'Clarkson', 'Spatial', 'desktop') into lightweight categories
            df[col] = df[col].astype('category')
    return df

def ingest_clarkson_I(tar_path):
    """ Reads raw keystrokes directly out of the compressed tar archive. """
    dfs = []
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            if member.name.split('/')[-1].startswith('._'): continue
            if member.isfile() and member.name.endswith('.txt'): # Adjust extension if needed based on tar contents
                user_id = f"C1_{member.name.split('/')[1]}" # Extract user ID from the path
                f = tar.extractfile(member)
                if f:
                    content = f.read().decode('utf-8', errors='ignore').strip()

                    parsed_data = []
                    
                    # 1. Split the file into lines (each line is a task)
                    for line in content.split('\n'):
                        line_parts = line.split('\t')
                        if len(line_parts) >= 4:
                            task_id = line_parts[1] # Metadata: 1=Pass, 2=Free, 3=Trans
                            events = line_parts[3].split(',')
                            
                            for event in events:
                                parts = event.split(':')
                                if len(parts) >= 3 and parts[0].isdigit():
                                    try:
                                        # Format: Action:KeyCode:Timestamp:Duration
                                        action_type = 'PRESS' if parts[0] == '0' else 'RELEASE'
                                        key_code = int(parts[1])
                                        timestamp_ms = float(parts[2])
                                        
                                        # Store as a list of dictionaries for easier DF creation
                                        parsed_data.append({
                                            'Timestamp_ms': timestamp_ms,
                                            'Action_Type': action_type,
                                            'Key_Code': key_code,
                                            'Task_Type': task_id
                                        })
                                    except ValueError:
                                        continue
                    
                    # 2. Convert the user's parsed array into a dataframe
                    if parsed_data:
                        # Update your df_user creation:
                        df_user = pd.DataFrame(parsed_data, columns=['Timestamp_ms', 'Action_Type', 'Key_Code', 'Task_Type'])
                        df_user['Participant_ID'] = user_id
                        df_user['Source_Dataset'] = 'Clarkson_I'
                        dfs.append(df_user)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

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
            df['Timestamp_ms'] = (df[0] - 116444736000000000) / 10000
            df['Action_Type'] = df[1].map({1: 'PRESS', 0: 'RELEASE'}) # Based on publisher: 1/0
            df['Key_Code'] = df[2]
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ==========================================
# 1. PROCESS AALTO
# ==========================================
if os.path.exists(aalto_path):
    print("Loading raw Aalto dataset...")
    df_aalto = pd.read_parquet(aalto_path)

    # Rename to Master Schema naming convention
    df_aalto = df_aalto.rename(columns={
        'PARTICIPANT_ID': 'Participant_ID',
        'PRESS_TIME': 'Timestamp_ms'
    })

    # Aalto rows are already individual keypresses; we tag them as PRESS
    df_aalto['Action_Type'] = 'PRESS'
    df_aalto['Source_Dataset'] = 'Aalto'

    # Rename raw participant columns if necessary
    if 'SUBJECT_ID' in df_aalto.columns:
        df_aalto = df_aalto.rename(columns={'SUBJECT_ID': 'User_ID'})
    elif 'PARTICIPANT_ID' in df_aalto.columns:
        df_aalto = df_aalto.rename(columns={'PARTICIPANT_ID': 'User_ID'})

    # Generate the missing Session_ID required by the taxonomy function
    df_aalto['Dataset'] = 'Aalto'
    if 'User_ID' in df_aalto.columns:
        df_aalto['Session_ID'] = df_aalto['Dataset'] + '_' + df_aalto['User_ID'].astype(str)
    else:
        df_aalto['Session_ID'] = 'Aalto_UnknownSession'

    print("Running Phase 1 Pipeline Calculations...")
    df_aalto = apply_typo_taxonomy(df_aalto)
    df_aalto = build_word_boundaries(df_aalto)
    df_aalto = apply_historical_consistency_filter(df_aalto)
    df_aalto = calculate_raw_digraphs(df_aalto)

    # Apply to Aalto before saving
    df_aalto = optimize_memory(df_aalto)
    output_path = os.path.join(base_dir, 'aalto_processed.parquet')
    df_aalto.to_parquet(output_path, index=False)

    print(f"✅ Saved Aalto: {len(df_aalto)} rows to {output_path}")
else:
    print("Raw Aalto file not found.")

# ==========================================
# 2. PROCESS CLARKSON I & II
# ==========================================
print("Processing Clarkson Datasets...")

# --- CLARKSON I (Reads directly from tarball) ---
clarkson_i_tar_path = os.path.join(base_dir, 'Clarkson-I-2014.tar.gz')

if os.path.exists(clarkson_i_tar_path):
    # Pass the .tar.gz file directly into the function!
    df_c1 = ingest_clarkson_I(clarkson_i_tar_path)
else:
    print(f"Warning: {clarkson_i_tar_path} not found.")
    df_c1 = pd.DataFrame()


# --- CLARKSON II (Requires extracted folder) ---
clarkson_ii_zip_path = os.path.join(base_dir, 'clarkson-II-2018-filtered_dataset.zip')
clarkson_ii_folder = os.path.join(base_dir, 'clarkson_2_extracted') 

if not os.path.exists(clarkson_ii_folder):
    if os.path.exists(clarkson_ii_zip_path):
        print(f"Extracting {clarkson_ii_zip_path}...")
        with zipfile.ZipFile(clarkson_ii_zip_path, 'r') as zip_ref:
            zip_ref.extractall(path=clarkson_ii_folder) 
    else:
        print(f"Warning: {clarkson_ii_zip_path} not found.")

if os.path.exists(clarkson_ii_folder):
    df_c2 = ingest_clarkson_II(clarkson_ii_folder)
else:
    df_c2 = pd.DataFrame()
# -----------------------------

# Repeat for Clarkson processing:
df_c1 = calculate_raw_digraphs(df_c1)
df_c2 = calculate_raw_digraphs(df_c2)

# Combine all raw Clarkson data
df_clarkson_raw = pd.concat([df_c1, df_c2], ignore_index=True) if not df_c1.empty else df_c2

if not df_clarkson_raw.empty:
    # Separate and sort    
    presses = df_clarkson_raw[df_clarkson_raw['Action_Type'] == 'PRESS'].copy()
    releases = df_clarkson_raw[df_clarkson_raw['Action_Type'] == 'RELEASE'].copy()

    # Heavy ETL Merge
    print("Running merge_asof matrix operation for Clarkson II...")
    
    # merge_asof requires both dataframes to be sorted by the timestamp
    presses = presses.sort_values('Timestamp_ms')
    releases = releases.sort_values('Timestamp_ms')

    # Explicitly duplicate the timestamp so it survives the merge as a data column
    releases['Timestamp_ms_Release'] = releases['Timestamp_ms']

    df_merged = pd.merge_asof(
        presses, 
        releases[['Participant_ID', 'Key_Code', 'Timestamp_ms', 'Timestamp_ms_Release']], 
        on='Timestamp_ms', 
        by=['Participant_ID', 'Key_Code'], 
        direction='forward'
    )
    
    # Map to Master Schema
    df_merged['Source_Dataset'] = 'Clarkson_II'
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
    
    # Save as Clarkson II specifically, matching our Step 6 merge list
    clarkson_output = os.path.join(base_dir, 'clarkson_ii_processed.parquet')
    df_merged.to_parquet(clarkson_output, index=False)
    
    print(f"✅ Saved Clarkson_II: {len(df_merged)} rows.")

# --- CMU (Seconds to Milliseconds & ID Alignment) ---
cmu_raw_path = os.path.join(base_dir, 'cmu_baseline.parquet')
if os.path.exists(cmu_raw_path):
    print("Standardizing CMU Dataset...")
    df_cmu = pd.read_parquet(cmu_raw_path)
    
    # 1. Convert all timing columns from Seconds to Milliseconds
    # CMU uses 'H.', 'DD.', and 'UD.' prefixes
    timing_cols = [c for c in df_cmu.columns if any(pref in c for pref in ['H.', 'DD.', 'UD.'])]
    # CRITICAL FIX: Force numeric so Pandas doesn't multiply strings
    df_cmu[timing_cols] = df_cmu[timing_cols].apply(pd.to_numeric, errors='coerce')
    df_cmu[timing_cols] = df_cmu[timing_cols] * 1000
    
    # 2. Standardize Schema
    df_cmu = df_cmu.rename(columns={'subject': 'Participant_ID'})
    df_cmu['Source_Dataset'] = 'CMU'
    df_cmu['Action_Type'] = 'DIGRAPH'
    df_cmu['Task_Type'] = 'Password'
    
    # Save it back so the merge loop sees the updated version
    df_cmu.to_parquet(cmu_raw_path)

# --- KeyRecs (Column Mapping) ---
keyrecs_raw_path = os.path.join(base_dir, 'keyrecs_micro.parquet')
if os.path.exists(keyrecs_raw_path):
    print("Standardizing KeyRecs Dataset...")
    df_keyrecs = pd.read_parquet(keyrecs_raw_path)
    
    # 3. Rename columns to Master Schema
    df_keyrecs = df_keyrecs.rename(columns={
        'participant': 'Participant_ID',
        'DD.key1.key2': 'Flight_DD_ms', 
        'UD.key1.key2': 'Flight_UD_ms'
    })
    df_keyrecs['Source_Dataset'] = 'KeyRecs'
    df_keyrecs['Action_Type'] = 'DIGRAPH'
    df_keyrecs['Task_Type'] = 'Mixed'
    
    df_keyrecs.to_parquet(keyrecs_raw_path)

print("ETL Pipeline Complete!")

# ==========================================
# 3. BUILD THE UNIFIED MASTER DATASET
# ==========================================
print("Combining all datasets into Master Matrix...")

# Load all processed files
cmu_path = os.path.join(base_dir, 'cmu_baseline.parquet')
keyrecs_path = os.path.join(base_dir, 'keyrecs_micro.parquet')
aalto_path = os.path.join(base_dir, 'aalto_processed.parquet')
clarkson_i_path = os.path.join(base_dir, 'clarkson_i_processed.parquet')
clarkson_ii_path = os.path.join(base_dir, 'clarkson_ii_processed.parquet')

master_dfs = []

# List of all standardized datasets to merge
merge_list = [
    (cmu_path, 'CMU'), 
    (keyrecs_path, 'KeyRecs'), 
    (aalto_path, 'Aalto'), 
    (clarkson_i_path, 'Clarkson_I'),
    (clarkson_ii_path, 'Clarkson_II')
]

for path, name in merge_list:
    if os.path.exists(path):
        df_temp = pd.read_parquet(path)
        master_dfs.append(df_temp)
        print(f"✅ Loaded {name} for Master Merge.")

if master_dfs:
    print("Fusing Master Matrix...")
    # Concatenate on GitHub's 7GB RAM runner
    df_master = pd.concat(master_dfs, ignore_index=True)
    
    # --- SCHEMA MAPPING FOR THE FRONTEND ---
    # Rename columns to match Streamlit expectations
    df_master = df_master.rename(columns={
        'Dataset': 'Source_Dataset',
        'Flight_Time': 'Flight_DD_ms'
    })
    
    # --- DROP DUPLICATE COLUMNS ---
    # Fixes the attribute crash by ensuring column names are strictly unique
    df_master = df_master.loc[:, ~df_master.columns.duplicated()]
    
    # Force ID columns to be purely strings to prevent PyArrow mixed-type crashes
    if 'Participant_ID' in df_master.columns:
        df_master['Participant_ID'] = df_master['Participant_ID'].astype(str)
    if 'Session_ID' in df_master.columns:
        df_master['Session_ID'] = df_master['Session_ID'].astype(str)

    # --- FAST TYPO DETECTION ---
    # Look for backspaces (Key_Code 8 or the string 'Back') and flag the preceding keystroke as a typo
    if 'Key_Code' in df_master.columns:
        is_backspace = df_master['Key_Code'].isin([8, 8.0, '8', 'Back', 'Backspace'])
        # Shift the boolean mask backwards by 1 to flag the character that caused the backspace
        df_master['Is_Typo'] = is_backspace.shift(-1).fillna(False)
        
    # --- SAFE MEMORY DOWNCASTING ---
    # Uses built-in bulk selection instead of the brittle for-loop
    float_cols = df_master.select_dtypes(include=['float64']).columns
    if not float_cols.empty:
        df_master[float_cols] = df_master[float_cols].astype('float32')
        
    obj_cols = df_master.select_dtypes(include=['object']).columns
    if not obj_cols.empty:
        df_master[obj_cols] = df_master[obj_cols].astype('category')
        
    # Sort by User and Time to maintain sequential integrity for ML training
    if 'Timestamp_ms' in df_master.columns:
        df_master = df_master.sort_values(['Participant_ID', 'Timestamp_ms'])

    # =========================================================
    # 🚨 FAILSAFE DATA SANITIZER (Add before saving Parquet) 🚨
    # =========================================================
    print("Sanitizing timing metrics...")
    
    # 1. Ensure Timestamps are standard numbers, NOT datetimes
    if 'Timestamp_ms' in df_master.columns:
        df_master['Timestamp_ms'] = pd.to_numeric(df_master['Timestamp_ms'], errors='coerce')

    # 2. Recalculate Aalto Flight Times correctly
    is_aalto = df_master['Source_Dataset'] == 'Aalto'
    if is_aalto.any() and 'Timestamp_ms' in df_master.columns:
        # Flight time is the difference between sequential keystrokes
        df_master.loc[is_aalto, 'Flight_DD_ms'] = df_master[is_aalto].groupby('Participant_ID')['Timestamp_ms'].diff()

    # 3. Nuke the 10^20 Overflow Bug & Enforce Statistical Boundaries
    for col in ['Flight_DD_ms', 'Hold_Time_ms']:
        if col in df_master.columns:
            # Force raw float32
            df_master[col] = pd.to_numeric(df_master[col], errors='coerce').astype('float32')
            
            # Wipe out the massive overflow numbers and impossible negative times
            # Anything > 2000ms is not muscle memory anyway, so we safely drop it
            df_master.loc[df_master[col] > 2000, col] = np.nan
            df_master.loc[df_master[col] < 0, col] = np.nan
    # =========================================================

    # Export the single, unified UI matrix
    master_output = os.path.join(base_dir, 'master_dataset.parquet')
    df_master.to_parquet(master_output, index=False, compression='snappy')
    print(f"✅ Master Dataset Exported! Total rows: {len(df_master)}")
