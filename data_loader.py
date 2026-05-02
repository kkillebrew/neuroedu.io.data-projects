# ==============================================================================
# THE MODEL LAYER: data_loader.py
# ==============================================================================
# This module acts purely as the data ingestion and transformation layer.
# It contains NO execution logic or UI rendering.
# 
# MATLAB Analogy: This is a library of custom functions that utilize `webread` 
# and `timetable` methods (like `synchronize` and `fillmissing`) to return 
# clean datasets (structs/tables) to the main AppDesigner UI.
# ==============================================================================

import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime
import time
import warnings

def fetch_real_oil_data(start_year=1976, end_year=datetime.datetime.now().year):
    """
    Fetches core macroeconomic oil data from the FRED APIs.
    
    Data Transformations:
    1. Timeline Alignment: Concatenates disparate timeseries into a unified DataFrame.
    2. Imputation: Uses linear time interpolation for missing monthly data.
    3. Inflation Adjustment: Calculates real price using the CPI ratio:
       $P_{real} = P_{nominal} \\times \\frac{CPI_{current}}{CPI_{historical}}$
    """
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    
    print(f"Fetching real data from FRED ({start_year} to {end_year})...")
    
    # --------------------------------------------------------------------------
    # 1. FETCH DATA FROM APIs (The "Real" Datasets)
    # --------------------------------------------------------------------------
    tickers = {
        'Nominal_Oil_Price': 'MCOILWTICO',    # WTI Crude Oil Price (Monthly)
        'CPI': 'CPIAUCSL',                    # Consumer Price Index
        'Nominal_Gas_Price': 'APU000074714',  # Retail Gasoline Price (Per Gallon)
        'US_Oil_Production': 'IPG211111CN',   # US Crude Production Index
        'Extraction_Cost_Index': 'WPU0561'    # PPI: Drilling Oil & Gas Wells
    }
    
    df_list = []
    error_msg = None
    
    try:
        for column_name, ticker in tickers.items():
            # CRITICAL: 1.5s delay to avoid HTTP 429 (Rate Limit) from FRED
            time.sleep(1.5)
            series = web.DataReader(ticker, 'fred', start_date, end_date)
            series.columns = [column_name]
            df_list.append(series)
            
    except Exception as e:
        error_msg = f"Failed on '{column_name}': {str(e)}"
        print(f"API Error: {error_msg}")
        return pd.DataFrame(), error_msg 

    # --------------------------------------------------------------------------
    # 2. MERGE & ALIGN TIMELINES
    # --------------------------------------------------------------------------
    # Align Timelines (MATLAB: synchronize(T1, T2, 'outer'))
    df = pd.concat(df_list, axis=1)
    
    # --------------------------------------------------------------------------
    # 3. HANDLE MISSING DATA (Imputation)
    # --------------------------------------------------------------------------
    # Handle Missing Data (MATLAB: fillmissing(T, 'linear'))
    df = df.interpolate(method='time')
    df = df.bfill()
    
    df = df.reset_index()
    df = df.rename(columns={'DATE': 'Date'})
    df['Year'] = df['Date'].dt.year

    # --------------------------------------------------------------------------
    # 4. CALCULATE REAL PRICES (Adjusted for Inflation)
    # --------------------------------------------------------------------------
    current_cpi = df['CPI'].iloc[-1] 
    df['Real_Oil_Price'] = df['Nominal_Oil_Price'] * (current_cpi / df['CPI'])
    df['Real_Gas_Price'] = df['Nominal_Gas_Price'] * (current_cpi / df['CPI'])

    # --------------------------------------------------------------------------
    # 5. INJECT GEOPOLITICAL EVENT FLAGS
    # --------------------------------------------------------------------------
    # Inject Hardcoded Geopolitical Event Flags (1 = Active Conflict, 0 = Peacetime)
    df['War_Conflict_Flag'] = 0 
    
    conflicts = [
        ('1979-01-01', '1980-12-31'), # Iranian Revolution / Energy Crisis
        ('1980-09-01', '1988-08-20'), # Iran-Iraq War
        ('1990-08-02', '1991-02-28'), # Gulf War
        ('2003-03-20', '2011-12-18'), # Iraq War
        ('2011-02-15', '2011-10-23'), # Libyan Civil War 
        ('2022-02-24', '2026-12-31')  # Russia-Ukraine War (Ongoing)
    ]
    
    for start, end in conflicts:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        mask = (df['Date'] >= start_dt) & (df['Date'] <= end_dt)
        df.loc[mask, 'War_Conflict_Flag'] = 1

    return df, None


def fetch_ripple_data(start_year=1976):
    """
    Fetches downstream socioeconomic ripple effect data (Main Street indicators).
    """
    start_date = f"{start_year}-01-01"
    end_date = f"{datetime.datetime.now().year}-12-31"
    
    tickers = {
        'Price_Eggs': 'APU0000708111',      # Avg Price: Eggs, Grade A, Large (per Dozen)
        'Electricity_CPI': 'CUSR0000SEHF01',# CPI: Electricity
        'Hourly_Wage': 'AHETPI'             # Avg Hourly Earnings of Production Employees
    }
    
    df_list = []
    for col, ticker in tickers.items():
        time.sleep(1.5) # Polite API delay to prevent FRED blocks
        try:
            s = web.DataReader(ticker, 'fred', start_date, end_date)
            s.columns = [col]
            df_list.append(s)
        except Exception:
            pass # Silently pass secondary data failures so the main app doesn't crash
            
    if df_list:
        df_ripple = pd.concat(df_list, axis=1).interpolate(method='time').bfill().reset_index()
        df_ripple = df_ripple.rename(columns={'DATE': 'Date'})
        return df_ripple
        
    return pd.DataFrame()

if __name__ == "__main__":
    # Test block to verify everything runs locally before Streamlit uses it
    print("Testing the Data Loader Module with Real Data...")
    test_data, err = fetch_real_oil_data(start_year=1976) 
    if not test_data.empty:
        print("\nMain Data successfully loaded and merged!")
        print(test_data[['Date', 'Real_Oil_Price', 'Nominal_Gas_Price', 'US_Oil_Production', 'Extraction_Cost_Index']].tail()) 
        
        print("\nTesting Ripple Data...")
        test_ripple = fetch_ripple_data(start_year=1976)
        if not test_ripple.empty:
            print("Ripple Data successfully loaded!")
            print(test_ripple.tail())
    else:
        print(f"\nFailed to load data. Error: {err}")