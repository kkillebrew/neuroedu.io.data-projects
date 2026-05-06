import pandas as pd
import numpy as np
import difflib
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

    ### TEMPORARY TO TEST RAM bottleneck ###
    # df = pd.read_parquet(filepath)
    # Force pandas to only load a single column to test RAM limits
    df = pd.read_parquet(filepath, columns=['Flight_Time'])

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
@st.cache_data(show_spinner=False)
def load_keyrecs(base_dir):
    """ Loads KeyRecs Digraph data. Cannot be used for word reconstruction. """
    filepath = os.path.join(base_dir, 'keyrecs_micro.parquet')
    
    if not os.path.exists(filepath):
        return pd.DataFrame()

    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        st.error(f"⚠️ KeyRecs Parquet Corrupted. Please re-download from Colab.")
        return pd.DataFrame()

    # Rename to match Master Schema where possible
    # In this dataset, DU.key1.key1 = Dwell Time, DD.key1.key2 = Flight Time
    df = df.rename(columns={
        'participant': 'User_ID',
        'key1': 'Key_Char',
        'DU.key1.key1': 'Dwell_Time',
        'DD.key1.key2': 'Flight_Time'
    })

    df['Dataset'] = 'KeyRecs'
    df['Device_Type'] = 'desktop' 
    df['Task_Type'] = 'free_text'
    df['User_ID'] = df['User_ID'].fillna('Unknown_User').astype(str)
    df['Session_ID'] = df['Dataset'] + '_' + df['User_ID']
    if 'Key_Char' not in df.columns:
        df['Key_Char'] = np.nan

    # Convert seconds to milliseconds for our standard
    if 'Dwell_Time' in df.columns:
        df['Dwell_Time'] = df['Dwell_Time'] * 1000
    if 'Flight_Time' in df.columns:
        df['Flight_Time'] = df['Flight_Time'] * 1000

    # Fill Missing/Static Categorical Data
    df['Dataset'] = 'KeyRecs'
    df['Device_Type'] = 'desktop' 
    df['Task_Type'] = 'free_text'
    df['User_ID'] = df['User_ID'].fillna('Unknown_User').astype(str)
    df['Session_ID'] = df['Dataset'] + '_' + df['User_ID']
    df['Key_Code'] = -1
    df['Timestamp_Press'] = 0.0
    df['Timestamp_Release'] = df['Dwell_Time']
    df['Delta_Milliseconds'] = 0.0

    # Initialize Phase 2 Placeholder Columns (Will remain empty)
    df['Intended_Char'] = np.nan
    df['Typed_Char'] = np.nan
    df['Is_Typo'] = False

    master_cols = [
        'Dataset', 'Session_ID', 'User_ID', 'Device_Type', 'Task_Type', 
        'Key_Code', 'Key_Char', 'Timestamp_Press', 'Timestamp_Release', 
        'Delta_Milliseconds', 'Dwell_Time', 'Flight_Time', 
        'Intended_Char', 'Typed_Char', 'Is_Typo'
    ]
    
    # Only return columns that successfully mapped
    available_cols = [col for col in master_cols if col in df.columns]
    return df[available_cols]

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

# 5: Call and cache our four functions
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
    
    # Check if files exist (prevents crash if Colab step hasn't been completed for all files yet)
    if not os.path.exists(os.path.join(base_dir, 'keyrecs_micro.parquet')):
        st.error("Data files missing from documents/ directory. Please run Phase 0 in Colab first.")
        return None, None, None, None
        
    # --- Apply Phase 1A Loading the Data ---
    df_keyrecs = load_keyrecs(base_dir)
    df_aalto = load_aalto(base_dir)
    df_cmu = load_cmu(base_dir)
    df_clarkson = load_clarkson(base_dir)

    # --- Apply Phase 1B Typo Detection (Backspace Footprints) ---
    # KeyRecs remains excluded as it is a macro-digraph dataset.
    df_clarkson = apply_typo_taxonomy(df_clarkson)
    # df_aalto = apply_typo_taxonomy(df_aalto)
    
    # --- Apply Phase 1C Word Boundaries ---
    df_clarkson = build_word_boundaries(df_clarkson)
    # df_aalto = build_word_boundaries(df_aalto)
    
    # --- Apply Phase 1C Levenshtein Anomaly Detection ---
    df_clarkson = flag_levenshtein_anomalies(df_clarkson)
    # df_aalto = flag_levenshtein_anomalies(df_aalto)
    
    # --- Apply Phase 1C Historical Consistency Filter ---
    df_clarkson = apply_historical_consistency_filter(df_clarkson)
    # df_aalto = apply_historical_consistency_filter(df_aalto)

    df_cmu = calculate_muscle_memory_decay(df_cmu)

    return df_cmu, df_keyrecs, df_aalto, df_clarkson

# ---------------------------------------------------------------------
# PHASE 1B: TYPO DETECTION ALGORITHMS
# ---------------------------------------------------------------------
def apply_typo_taxonomy(df):
    """
    Scans the DataFrame for Backspace Footprints (Immediate Self-Corrections).
    Tags the error, the intention, and categorizes it via qwerty_mapper.
    """
    if df is None or df.empty:
        return df

    # 1. Create shifted columns to look ahead/behind in time
    # Grouping by Session_ID ensures we don't accidentally link keystrokes from two different users
    df['Next_Key'] = df.groupby('Session_ID')['Key_Code'].shift(-1)
    df['Next_Next_Char'] = df.groupby('Session_ID')['Key_Char'].shift(-2)

    # 2. Define the Backspace Footprint Mask
    # An error occurred if the CURRENT keystroke is immediately followed by a Backspace (Key_Code 8)
    error_mask = (df['Next_Key'] == 8)

    # 3. Apply the tags
    df.loc[error_mask, 'Is_Typo'] = True
    df.loc[error_mask, 'Typed_Char'] = df.loc[error_mask, 'Key_Char']
    df.loc[error_mask, 'Intended_Char'] = df.loc[error_mask, 'Next_Next_Char']

    # 4. Apply the Taxonomy Classification (Category A vs B vs C)
    def categorize_row(row):
        if row['Is_Typo'] and pd.notna(row['Intended_Char']) and pd.notna(row['Typed_Char']):
            return classify_typo(row['Intended_Char'], row['Typed_Char'])
        return 'None'

    # Create the new column for the taxonomy label
    df['Typo_Category'] = df.apply(categorize_row, axis=1)

    # 5. Cleanup temporary columns to save memory
    df = df.drop(columns=['Next_Key', 'Next_Next_Char'])
    
    return df

# ---------------------------------------------------------------------
# PHASE 1C: WORD RECONSTRUCTION & HISTORICAL CONSISTENCY
# ---------------------------------------------------------------------

def build_word_boundaries(df):
    """
    Groups individual keystrokes into discrete word blocks.
    Triggers a new Word_ID every time [Space] or [Enter] is pressed.
    """
    if df is None or df.empty:
        return df

    # Create a boolean mask for delimiters (Space = 32, Enter = 13)
    delimiters = df['Key_Code'].isin([32, 13])
    
    # cumsum() adds 1 to the Word_ID every time it hits a True value in the delimiter mask.
    # Grouping by Session_ID ensures Word 1 for User A doesn't bleed into User B.
    df['Word_ID'] = df.groupby('Session_ID')[delimiters].cumsum().ffill().fillna(0).astype(int)

    return df

def flag_levenshtein_anomalies(df, dictionary_fallback=None):
    """
    Reconstructs keystrokes into words and uses Sequence Alignment 
    to find Uncorrected Typos (Category A, B, C) that lack a Backspace footprint.
    """
    if df is None or df.empty or 'Word_ID' not in df.columns:
        return df

    # 1. Reconstruct the Typed Word for each Word_ID block
    # We drop NaNs (like Shift keys) and join the characters into a single string
    reconstructed = df.dropna(subset=['Key_Char']).groupby(['Session_ID', 'Word_ID'])['Key_Char'].apply(lambda x: ''.join(x.astype(str))).reset_index()
    reconstructed = reconstructed.rename(columns={'Key_Char': 'Submitted_Word'})

    # 2. Identify the Expected Word
    # Note: For Transcription tasks, you will map the prompt text here.
    # For Free Text, we assume 'dictionary_fallback' is a spellcheck lookup.
    # For now, we will create a placeholder 'Expected_Word' column to build the architecture.
    reconstructed['Expected_Word'] = reconstructed['Submitted_Word'] # Placeholder: replace with actual expected text logic later

    # 3. Calculate Sequence Alignment for mismatches
    mismatch_mask = reconstructed['Expected_Word'] != reconstructed['Submitted_Word']
    mismatches = reconstructed[mismatch_mask].copy()

    # 4. Find the exact character index of the typo using difflib
    error_records = []
    for _, row in mismatches.iterrows():
        expected = row['Expected_Word']
        submitted = row['Submitted_Word']
        
        # SequenceMatcher finds the exact diff operations
        matcher = difflib.SequenceMatcher(None, expected, submitted)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in ('replace', 'insert', 'delete'):
                error_records.append({
                    'Session_ID': row['Session_ID'],
                    'Word_ID': row['Word_ID'],
                    'Typo_Index': j1, # The index in the submitted string
                    'Intended_Char': expected[i1:i2] if tag != 'insert' else '',
                    'Typed_Char': submitted[j1:j2] if tag != 'delete' else ''
                })

    # 5. Map the errors back to the original microscopic keystroke DataFrame
    if error_records:
        error_df = pd.DataFrame(error_records)
        
        # We merge based on Session, Word_ID, and use cumcount to match the character index
        df['Char_Index'] = df.groupby(['Session_ID', 'Word_ID']).cumcount()
        
        df = pd.merge(df, error_df, left_on=['Session_ID', 'Word_ID', 'Char_Index'], right_on=['Session_ID', 'Word_ID', 'Typo_Index'], how='left')
        
        # Update the master columns
        new_errors = df['Typo_Index'].notna()
        df.loc[new_errors, 'Is_Typo'] = True
        # Coalesce the intended/typed chars so we don't overwrite Backspace footprints
        df['Intended_Char'] = df['Intended_Char_y'].combine_first(df['Intended_Char_x'])
        df['Typed_Char'] = df['Typed_Char_y'].combine_first(df['Typed_Char_x'])
        
        # Drop temporary merge columns
        df = df.drop(columns=['Char_Index', 'Typo_Index', 'Intended_Char_x', 'Intended_Char_y', 'Typed_Char_x', 'Typed_Char_y'])

    return df

def apply_historical_consistency_filter(df, consistency_threshold=0.8):
    """
    Identifies 'Recall Errors' (Misspellings) vs 'Motor Errors' (Typos).
    If a user makes the exact same error for a specific word > 80% of the time,
    it is flagged as a competence error so the ML model can ignore it.
    """
    if df is None or df.empty or 'Word_ID' not in df.columns:
        return df

    # 1. Reconstruct words to evaluate consistency
    words_df = df.dropna(subset=['Key_Char']).groupby(['User_ID', 'Session_ID', 'Word_ID'])['Key_Char'].apply(lambda x: ''.join(x.astype(str))).reset_index()
    words_df = words_df.rename(columns={'Key_Char': 'Submitted_Word'})
    
    # Placeholder for the prompt text (Expected_Word)
    words_df['Expected_Word'] = words_df['Submitted_Word'] 

    # 2. Count how many times the user made a specific exact error
    error_counts = words_df.groupby(['User_ID', 'Expected_Word', 'Submitted_Word']).size().reset_index(name='Specific_Error_Count')

    # 3. Count total times the user attempted the intended word
    total_attempts = words_df.groupby(['User_ID', 'Expected_Word']).size().reset_index(name='Total_Attempts')

    # 4. Merge to calculate the ratio
    consistency_df = pd.merge(error_counts, total_attempts, on=['User_ID', 'Expected_Word'])
    consistency_df['Error_Ratio'] = consistency_df['Specific_Error_Count'] / consistency_df['Total_Attempts']

    # 5. Create the filter mask (True = Consistent Misspelling/Recall Error)
    # We only care if it's actually a mismatch. If Expected == Submitted, it's not an error.
    consistency_df['Is_Recall_Error'] = (consistency_df['Expected_Word'] != consistency_df['Submitted_Word']) & (consistency_df['Error_Ratio'] >= consistency_threshold)
    
    # 6. Merge the flag back into the word dataframe, then back to the microscopic keystrokes
    words_df = pd.merge(words_df, consistency_df[['User_ID', 'Expected_Word', 'Submitted_Word', 'Is_Recall_Error']], on=['User_ID', 'Expected_Word', 'Submitted_Word'], how='left')
    
    df = pd.merge(df, words_df[['Session_ID', 'Word_ID', 'Is_Recall_Error']], on=['Session_ID', 'Word_ID'], how='left')
    
    # Clean up NaNs
    df['Is_Recall_Error'] = df['Is_Recall_Error'].fillna(False)

    return df

# ---------------------------------------------------------------------
# PHASE 2: MACRO-LEVEL BENCHMARKING
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def calculate_muscle_memory_decay(df_cmu):
    """ 
    Calculates the universal muscle memory decay curve from the CMU dataset.
    Averages flight times across all users to find the steady-state latency.
    """
    if df_cmu is None or df_cmu.empty:
        return pd.DataFrame()

    # 1. Isolate the Flight Time columns
    dd_cols = [col for col in df_cmu.columns if col.startswith('DD.')]
    
    if not dd_cols:
        return pd.DataFrame()

    # 2. Calculate the average flight time for the entire password attempt
    # MATLAB Analogy: mean(matrix, 2) to get the row-wise average
    df_cmu['Avg_Flight_Time'] = df_cmu[dd_cols].mean(axis=1)

    # 3. Create a continuous X-Axis (Attempt Number 1 through 400)
    # CMU users did 8 sessions of 50 reps. 
    if 'sessionIndex' in df_cmu.columns and 'rep' in df_cmu.columns:
        df_cmu['Attempt_Number'] = ((df_cmu['sessionIndex'] - 1) * 50) + df_cmu['rep']
    else:
        # Fallback just in case the column names vary slightly
        df_cmu['Attempt_Number'] = df_cmu.groupby('subject').cumcount() + 1

    # 4. Collapse the 51 users into a single, universal average line
    decay_curve = df_cmu.groupby('Attempt_Number')['Avg_Flight_Time'].mean().reset_index()

    return decay_curve

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
