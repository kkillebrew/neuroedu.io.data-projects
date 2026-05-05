import pandas as pd
import numpy as np
import streamlit as st
import os

# =====================================================================
# MODULE: typo_behavior_loader.py (The Model / Backend)
# Strict Decoupling: No Streamlit UI rendering commands (st.write, etc.) 
# are permitted in this file. 
# =====================================================================

# ---------------------------------------------------------------------
# PHASE 1: DATA INGESTION & SCHEMA UNIFICATION
# ---------------------------------------------------------------------

# 1: aalto data loader
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_aalto(filepath="documents/aalto_macro.parquet"):
    """ Loads and maps Aalto dataset to the Master Schema. """
    if not os.path.exists(filepath):
        return pd.DataFrame()

    df = pd.read_parquet(filepath)

    # 1. Map to Master Schema Columns
    df = df.rename(columns={
        'PARTICIPANT_ID': 'User_ID'
        # Timestamp, Key_Code, Key_Char, Device_Type are already correct from Phase 0
    })

    # Aalto tracks Press time as 'Timestamp'
    df['Timestamp_Press'] = df['Timestamp']

    # 2. Fill Missing/Static Categorical Data
    df['Dataset'] = 'Aalto'
    df['Task_Type'] = 'transcription' # Aalto was mostly copying sentences
    df['User_ID'] = df['User_ID'].fillna('Unknown_User').astype(str)
    df['Session_ID'] = df['Dataset'] + '_' + df['User_ID']
    df['Device_Type'] = df['Device_Type'].fillna('unknown')

    # 3. Dynamic Calculations (Aalto lacks Release timestamps)
    # We must defensively impute Dwell Time. ~100ms is standard human baseline.
    df['Dwell_Time'] = 100.0 
    df['Timestamp_Release'] = df['Timestamp_Press'] + df['Dwell_Time']
    
    # Sort & Calculate Flight Time
    df = df.sort_values(by=['User_ID', 'Timestamp_Press'])
    df['Flight_Time'] = df.groupby('User_ID')['Timestamp_Press'].diff().fillna(0)

    # 4. Normalize Timestamps
    session_starts = df.groupby('User_ID')['Timestamp_Press'].transform('min')
    df['Delta_Milliseconds'] = df['Timestamp_Press'] - session_starts

    # 5. Initialize Phase 2 Placeholder Columns
    df['Intended_Char'] = np.nan
    df['Typed_Char'] = np.nan
    df['Is_Typo'] = False

    # 6. Final Master Schema Alignment
    master_cols = [
        'Dataset', 'Session_ID', 'User_ID', 'Device_Type', 'Task_Type', 
        'Key_Code', 'Key_Char', 'Timestamp_Press', 'Timestamp_Release', 
        'Delta_Milliseconds', 'Dwell_Time', 'Flight_Time', 
        'Intended_Char', 'Typed_Char', 'Is_Typo'
    ]
    
    return df[master_cols]

# 2: keyrecs data loader
# ---------------------------------------------------------------------
@st.cache_data(show_spinner="Loading optimized Parquet chunks into RAM...")
def load_all_datasets():
    """
    Loads the highly compressed Parquet files generated in Colab.
    MATLAB Analogy: Pre-allocating and loading data once at the start of 
    a script so loops don't constantly hit the hard drive.
    """
    # Define paths to the 'documents/' folder
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'documents'))
    
    # Placeholder paths for our compressed datasets
    cmu_path = os.path.join(base_dir, 'cmu_baseline.parquet')
    keyrecs_path = os.path.join(base_dir, 'keyrecs_micro.parquet')
    aalto_path = os.path.join(base_dir, 'aalto_macro.parquet')
    
    try:
        # df_cmu = pd.read_parquet(cmu_path)
        # df_keyrecs = pd.read_parquet(keyrecs_path)
        # df_aalto = pd.read_parquet(aalto_path)
        
        # For barebones testing before files are uploaded:
        df_cmu = pd.DataFrame({"Subject": [], "Rep": [], "H.period": []})
        df_keyrecs = pd.DataFrame({"Timestamp": [], "Key_Char": []})
        df_aalto = pd.DataFrame({"Device": [], "WPM": []})
        
    except FileNotFoundError:
        st.error("Data files missing from documents/ directory. Please run Phase 0 in Colab first.")
        return None, None, None
        
    # TODO: Schema Unification logic will go here
    return df_cmu, df_keyrecs, df_aalto

# 3: CMU data loader
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_cmu(filepath="documents/cmu_baseline.parquet"):
    """ Loads CMU Benchmark. Melts password repetitions into single keystrokes. """
    if not os.path.exists(filepath):
        return pd.DataFrame()

    df = pd.read_parquet(filepath)
    
    # 1. Define the password sequence used in the CMU dataset
    password_chars = ['.', 't', 'i', 'e', '5', 'R', 'o', 'a', 'n', 'l', 'Enter']
    
    records = []
    
    # CMU data is structured wide (1 row = 1 password attempt). 
    # We must melt it to long (1 row = 1 keystroke) to match our event-based schema.
    for _, row in df.iterrows():
        user = str(row.get('subject', 'Unknown'))
        session = f"CMU_{user}_S{row.get('sessionIndex', 1)}_R{row.get('rep', 1)}"
        
        # We simulate a timeline starting at T=0 for each repetition
        current_time = 0.0
        
        # CMU column naming convention: H.key (Hold/Dwell), DD.key1.key2 (Down-Down/Flight)
        for i, char in enumerate(password_chars):
            # Dwell time
            dwell_col = f"H.{char}" if char != '5' else "H.five"
            dwell_col = "H.Shift.r" if char == 'R' else dwell_col
            dwell = row.get(dwell_col, 0.1) * 1000  # Convert seconds to ms
            
            # Flight time (from previous key)
            flight = 0.0
            if i > 0:
                prev_char = password_chars[i-1]
                # Simplify naming lookup for demonstration
                flight = 0.2 * 1000 # Placeholder baseline for CMU flight conversion
                
            current_time += flight
            
            records.append({
                'Dataset': 'CMU',
                'Session_ID': session,
                'User_ID': user,
                'Device_Type': 'desktop',
                'Task_Type': 'password',
                'Key_Code': -1, # CMU doesn't provide explicit browser keycodes
                'Key_Char': char,
                'Timestamp_Press': current_time,
                'Timestamp_Release': current_time + dwell,
                'Delta_Milliseconds': current_time,
                'Dwell_Time': dwell,
                'Flight_Time': flight,
                'Intended_Char': char,
                'Typed_Char': char,
                'Is_Typo': False
            })
            
    master_df = pd.DataFrame(records)
    return master_df

# 4: Clarkson data loader
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_clarkson(filepath="documents/clarkson_cognitive.parquet"):
    """ Loads Clarkson Dataset. Pairs separate Press/Release rows. """
    if not os.path.exists(filepath):
        return pd.DataFrame()

    df = pd.read_parquet(filepath)

    # 1. Separate Presses and Releases
    presses = df[df['Action'] == 0].copy()
    releases = df[df['Action'] == 1].copy()
    
    # 2. Rename for clarity
    presses = presses.rename(columns={'Timestamp': 'Timestamp_Press'})
    releases = releases.rename(columns={'Timestamp': 'Timestamp_Release'})
    
    # Sort them to allow pairing
    presses = presses.sort_values(by=['PARTICIPANT_ID', 'Key_Code', 'Timestamp_Press'])
    releases = releases.sort_values(by=['PARTICIPANT_ID', 'Key_Code', 'Timestamp_Release'])

    # 3. Pair Presses with the nearest Release (using merge_asof to find the closest time)
    # This matches the release of 'Key A' with the exact press of 'Key A'
    df_merged = pd.merge_asof(
        presses, 
        releases[['PARTICIPANT_ID', 'Key_Code', 'Timestamp_Release']], 
        left_on='Timestamp_Press', 
        right_on='Timestamp_Release', 
        by=['PARTICIPANT_ID', 'Key_Code'],
        direction='forward'
    )

    # 4. Map to Master Schema Columns
    df_merged = df_merged.rename(columns={'PARTICIPANT_ID': 'User_ID'})

    # 5. Fill Missing/Static Categorical Data
    df_merged['Dataset'] = 'Clarkson'
    df_merged['Device_Type'] = 'desktop' 
    df_merged['User_ID'] = df_merged['User_ID'].fillna('Unknown_User').astype(str)
    df_merged['Session_ID'] = df_merged['Dataset'] + '_' + df_merged['User_ID']
    df_merged['Key_Char'] = np.nan
    
    if 'Task' in df_merged.columns:
        df_merged['Task_Type'] = df_merged['Task'].fillna('free_text')
    else:
        df_merged['Task_Type'] = 'free_text'

    # 6. Dynamic Calculations
    df_merged['Dwell_Time'] = df_merged['Timestamp_Release'] - df_merged['Timestamp_Press']
    
    # Resort strictly by time for Flight calculation
    df_merged = df_merged.sort_values(by=['User_ID', 'Timestamp_Press'])
    df_merged['Flight_Time'] = df_merged.groupby('User_ID')['Timestamp_Press'].diff().fillna(0)

    # 7. Normalize Timestamps
    session_starts = df_merged.groupby('User_ID')['Timestamp_Press'].transform('min')
    df_merged['Delta_Milliseconds'] = df_merged['Timestamp_Press'] - session_starts

    # 8. Missing Value Imputation
    df_merged['Dwell_Time'] = df_merged['Dwell_Time'].fillna(100.0)

    # 9. Initialize Phase 2 Placeholder Columns
    df_merged['Intended_Char'] = np.nan
    df_merged['Typed_Char'] = np.nan
    df_merged['Is_Typo'] = False

    # 10. Final Master Schema Alignment
    master_cols = [
        'Dataset', 'Session_ID', 'User_ID', 'Device_Type', 'Task_Type', 
        'Key_Code', 'Key_Char', 'Timestamp_Press', 'Timestamp_Release', 
        'Delta_Milliseconds', 'Dwell_Time', 'Flight_Time', 
        'Intended_Char', 'Typed_Char', 'Is_Typo'
    ]
    
    return df_merged[master_cols]

# ---------------------------------------------------------------------
# PHASE 1B: TYPO DETECTION ALGORITHMS
# ---------------------------------------------------------------------
def extract_backspace_footprints(df):
    """ Finds self-corrected typos by tracing the [BACKSPACE] keycode. """
    # TODO: Implement vectorized Pandas shift() logic here
    pass

def calculate_levenshtein_alignments(prompt, typed):
    """ Uses NLTK/Edit Distance to find uncorrected typos. """
    # TODO: Implement Sequence Alignment logic here
    pass

# ---------------------------------------------------------------------
# PHASE 2: MACRO-LEVEL BENCHMARKING
# ---------------------------------------------------------------------
def calculate_muscle_memory_asymptote(df_cmu):
    """ Calculates steady-state latency from the last 50 CMU reps. """
    # TODO: Phase 2 logic here
    pass

# ---------------------------------------------------------------------
# PHASE 3 & 4: TAXONOMY & FEATURE ENGINEERING
# ---------------------------------------------------------------------
def engineer_behavioral_features(df_micro):
    """ 
    Calculates Dwell/Flight latencies, Rolling Variance (Burstiness), 
    and Euclidean Keyboard Distances.
    """
    # TODO: Phase 3 & 4 logic here
    pass

# ---------------------------------------------------------------------
# PHASE 5: MACHINE LEARNING (SINGLETON INSTANCE)
# ---------------------------------------------------------------------
@st.cache_resource
def load_ml_pipeline():
    """
    Caches the Scikit-Learn Random Forest model to prevent UI lag.
    MATLAB Analogy: Compiling a function to a .mex file to bypass interpreter overhead.
    """
    # TODO: Train or load the SMOTE-balanced ML model
    model = None 
    return model
