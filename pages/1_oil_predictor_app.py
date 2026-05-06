# ==============================================================================
# THE VIEW LAYER: app.py
# ==============================================================================
# This module acts exclusively as the Streamlit User Interface.
# MATLAB Analogy: This is the equivalent of an AppDesigner (.mlapp) file 
# defining the UIComponents (UIAxes, UISliders) and their callbacks.
# ==============================================================================

import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- PATH CONFIGURATION ---
# This tells the script to look one folder up to find the 'loaders' directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders.oil_predictor_loader import fetch_real_oil_data, fetch_ripple_data

from career_hub_sidebar import apply_global_settings, render_sidebar

########################################
#        APPLY GLOBAL SETTINGS         #
########################################
apply_global_settings("Neuro-Edu | What affects gas prices?")

########################################
# RENDER THE SIDEBAR FOR DATA-PROJECTS #
########################################
render_sidebar()

# ------------------------------------------------------------------------------
# DATA INGESTION & CACHING
# ------------------------------------------------------------------------------
@st.cache_data(ttl=3600) 
def prepare_main_data():
    result = fetch_real_oil_data(start_year=1976)
    if isinstance(result, tuple):
        df_raw, error_msg = result
    else:
        df_raw, error_msg = result, "Unknown API Error"
    
    if df_raw.empty:
        return pd.DataFrame(), error_msg 
        
    # Define Target Variable for ML: Y = 1 if (P_t - P_{t-1}) / P_{t-1} > 0.05
    df_raw['Price_Spike'] = (df_raw['Real_Oil_Price'].pct_change() > 0.05).astype(int)
    df_raw = df_raw.fillna(0) 
    return df_raw, None

@st.cache_data(ttl=3600)
def prepare_ripple_data(min_yr):
    return fetch_ripple_data(start_year=min_yr)

with st.spinner("Fetching macro data from FRED APIs (Please allow ~10 seconds)..."):
    df, api_error = prepare_main_data()

if df.empty:
    st.error("🚨 **Error:** Failed to load data from the FRED API.")
    st.warning(f"**Details:** {api_error}")
    st.stop()

# ------------------------------------------------------------------------------
# DASHBOARD CONTROLS: MAIN PAGE
# ------------------------------------------------------------------------------
st.title("🛢️ 100 Years of Macro Oil Trends")
st.markdown("""
Welcome to the interactive Macro Oil Explorer. This dashboard uses live API data from 
the **Federal Reserve Economic Data (FRED)** to model relationships between 
geopolitics, extraction costs, and crude prices.
""")

st.markdown("### 🎛️ Analysis Controls")

# Set up columns so the slider and checkbox sit nicely next to each other
control_col1, control_col2 = st.columns([2, 1])

min_year = int(df['Year'].min())
max_year = int(df['Year'].max())

with control_col1:
    selected_years = st.slider("Select Year Range:", min_value=min_year, max_value=max_year, value=(1980, 2024))

with control_col2:
    st.write("") # Adds a tiny bit of vertical padding to align with the slider
    st.write("")
    adjust_inflation = st.checkbox("Adjust for Inflation (Real Price)", value=True)

# Apply the filters
mask = (df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])
df_filtered = df.loc[mask]

oil_metric = 'Real_Oil_Price' if adjust_inflation else 'Nominal_Oil_Price'
gas_metric = 'Real_Gas_Price' if adjust_inflation else 'Nominal_Gas_Price'


# ------------------------------------------------------------------------------
# VISUALIZATION: HISTORICAL TIMELINE (Tufte Aesthetics Applied)
# ------------------------------------------------------------------------------
st.subheader("Historical Timeline")

fig, ax1 = plt.subplots(figsize=(12, 5))
color1 = '#1f77b4' 
ax1.plot(df_filtered['Date'], df_filtered[oil_metric], color=color1, linewidth=2, label='WTI Crude')
ax1.set_ylabel('Oil Price ($/bbl)', color=color1, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color1)

# Highlight conflict zones
conflict_starts = df_filtered[(df_filtered['War_Conflict_Flag'] == 1) & (df_filtered['War_Conflict_Flag'].shift(1, fill_value=0) == 0)]['Date']
conflict_ends = df_filtered[(df_filtered['War_Conflict_Flag'] == 1) & (df_filtered['War_Conflict_Flag'].shift(-1, fill_value=0) == 0)]['Date']
for start, end in zip(conflict_starts, conflict_ends):
    ax1.axvspan(start, end, color='red', alpha=0.1)

ax2 = ax1.twinx()  
color2 = '#7f8c8d' # A more subdued gray for the secondary axis
ax2.plot(df_filtered['Date'], df_filtered[gas_metric], color=color2, linewidth=1.5, alpha=0.8, label='Retail Gas')
ax2.set_ylabel('Gas Price ($/gal)', color=color2, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color2)

# Tufte Principles: High data-to-ink ratio. Remove top spines, use minimal grid.
sns.despine(ax=ax1, top=True, right=False)
sns.despine(ax=ax2, top=True, left=False)
ax1.grid(True, linestyle=':', alpha=0.4) 

fig.tight_layout()
st.pyplot(fig) 

# ------------------------------------------------------------------------------
# STATISTICAL MODELING: OLS REGRESSION
# ------------------------------------------------------------------------------
st.markdown("---")
st.subheader("Statistical Drivers (OLS Regression)")
st.markdown("""
This model estimates the linear relationship between our macroeconomic features ($X$) and the price of oil ($Y$). 
The mathematical formulation is: 
$$Y = \\beta_0 + \\beta_1(Production) + \\beta_2(Cost) + \\beta_3(War) + \\beta_4(CPI) + \epsilon$$
""")

col1, col2 = st.columns([1, 2])
features = ['US_Oil_Production', 'Extraction_Cost_Index', 'War_Conflict_Flag', 'CPI']

with col1:
    X = df_filtered[features]
    X = sm.add_constant(X)
    y = df_filtered[oil_metric]
    model = sm.OLS(y, X).fit()
    
    st.metric(
        label="Model $R^2$", 
        value=f"{model.rsquared:.2f}", 
        help="Proportion of variance explained by the model."
    )
    st.metric(
        label="War Premium Impact ($\hat{\\beta_3}$)", 
        value=f"${model.params.get('War_Conflict_Flag', 0):.2f}/bbl", 
        delta="Added during conflict",
        help="Estimated price increase per barrel (bbl) uniquely attributed to active major geopolitical conflicts."
    )

with col2:
    st.write("**Regression Coefficients (The $\\beta$ values):**")
    coef_df = pd.DataFrame({"Coefficient": model.params, "P-Value": model.pvalues}).round(4)
    coef_df = coef_df.rename(index={
        'const': 'Baseline (Intercept)', 'US_Oil_Production': 'US Oil Production',
        'Extraction_Cost_Index': 'Extraction Cost', 'War_Conflict_Flag': 'Active Geopolitical Conflict',
        'CPI': 'Consumer Price Index (Inflation)'
    })
    st.dataframe(coef_df, use_container_width=True)

# ------------------------------------------------------------------------------
# EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Exploratory Data Analysis")
st.write("A deeper dive into the relationships and distributions of our macroeconomic factors.")

tab1, tab2, tab3 = st.tabs(["Price Volatility", "Production vs. Price", "Correlation Heatmap"])

with tab1:
    st.write("### Distribution of Month-over-Month Price Changes")
    st.write("This histogram shows how often oil prices jump or drop month-to-month. The red dashed line marks the 5% threshold we use to define a 'Price Spike' for our Machine Learning model.")
    fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
    pct_changes = df_filtered['Real_Oil_Price'].pct_change().dropna() * 100
    sns.histplot(pct_changes, bins=50, kde=True, ax=ax_hist, color='#34495e')
    ax_hist.axvline(5, color='#e74c3c', linestyle='dashed', linewidth=2, label='5% Spike Threshold')
    ax_hist.legend()
    sns.despine(ax=ax_hist)
    st.pyplot(fig_hist)

with tab2:
    st.write("### US Production vs. Real Oil Price")
    st.write("Does more production mean cheaper oil? This scatterplot visualizes their relationship, colored by periods of active conflict.")
    fig_scat, ax_scat = plt.subplots(figsize=(10, 4))
    sns.scatterplot(data=df_filtered, x='US_Oil_Production', y=oil_metric, hue='War_Conflict_Flag', palette='coolwarm', ax=ax_scat, alpha=0.8, s=60)
    
    # Custom Legend
    handles, labels = ax_scat.get_legend_handles_labels()
    ax_scat.legend(handles, ['Peacetime', 'Active Conflict'], title='Geopolitical Status')
    sns.despine(ax=ax_scat)
    st.pyplot(fig_scat)

with tab3:
    st.write("### Macroeconomic Correlation Heatmap")
    st.write("Values closer to 1 (red) indicate a strong positive correlation, while values closer to -1 (blue) indicate a strong negative correlation.")
    fig_corr, ax_corr = plt.subplots(figsize=(8, 5))
    corr_cols = ['Real_Oil_Price', 'Real_Gas_Price', 'US_Oil_Production', 'Extraction_Cost_Index', 'CPI', 'War_Conflict_Flag']
    corr_labels = ['Oil Price', 'Gas Price', 'US Production', 'Extraction Cost', 'Inflation (CPI)', 'War/Conflict']
    corr_matrix = df_filtered[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt=".2f", ax=ax_corr, xticklabels=corr_labels, yticklabels=corr_labels)
    st.pyplot(fig_corr)

# ------------------------------------------------------------------------------
# PREDICTIVE MODELING: RANDOM FOREST
# ------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🤖 Predictive Modeling: Nonlinear Risk")
st.markdown("""
Unlike OLS, Random Forests can capture nonlinear interactions by ensembling decision trees. 
We use this to predict the probability of a high-volatility event ($>5\\%$ jump).
""")

X_ml = df_filtered[features] 
y_ml = df_filtered['Price_Spike']

if len(df_filtered) > 50:
    # MATLAB: cvpartition() equivalent
    X_train, X_test, y_train, y_test = train_test_split(X_ml, y_ml, test_size=0.2, random_state=42)
    
    # MATLAB: fitcensemble() equivalent
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, rf_model.predict(X_test))
    
    st.caption(f"*Test Accuracy: {accuracy*100:.1f}%*")
    st.markdown("### What-If Scenario Builder")
    
    scol1, scol2, scol3, scol4 = st.columns(4)
    with scol1: 
        sim_prod = st.slider("US Prod (Index)", 50, 200, 100, 5,
            help="Measures the relative volume of crude oil extracted in the US.")
    with scol2: 
        sim_cost = st.slider("Extraction Cost Index", 50, 400, 200, 10,
            help="Producer Price Index (PPI) for drilling oil and gas wells.")
    with scol3: 
        sim_cpi = st.slider("Inflation (CPI)", 50, 400, int(df_filtered['CPI'].mean()), 10,
            help="Consumer Price Index. Represents the overall inflation in the broader economy.")
    with scol4: 
        sim_war = st.selectbox("Active Major Conflict?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No",
            help="Simulates the presence of a major global conflict involving key oil-producing nations.")
    
    scenario_df = pd.DataFrame([[sim_prod, sim_cost, sim_war, sim_cpi]], columns=features)
    spike_prob = rf_model.predict_proba(scenario_df)[0][1] 
    
    st.write("") 
    if spike_prob > 0.5: st.error(f"🚨 High Risk: {spike_prob*100:.1f}% probability of a price spike.")
    elif spike_prob > 0.25: st.warning(f"⚠️ Moderate Risk: {spike_prob*100:.1f}% probability of a price spike.")
    else: st.success(f"✅ Stable Outlook: {spike_prob*100:.1f}% probability of a price spike.")
else:
    st.warning("Expand year range to train model.")

# ------------------------------------------------------------------------------
# DOWNSTREAM SOCIOECONOMIC RIPPLE EFFECTS
# ------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🌊 Downstream Ripple Effects")
st.write("When foundational energy costs spike, the shockwaves travel through the entire economy. How does this affect daily life and political stability?")

df_ripple = prepare_ripple_data(min_year)

if not df_ripple.empty:
    df_ripple = df_ripple[(df_ripple['Date'].dt.year >= selected_years[0]) & (df_ripple['Date'].dt.year <= selected_years[1])]
    df_merged = pd.merge(df_filtered, df_ripple, on='Date', how='left')
    
    rtab1, rtab2 = st.tabs(["🛒 Consumer Goods", "🗳️ Political Flips"])
    
    with rtab1:
        st.write("### Oil Price vs. The Price of Eggs")
        st.write("Energy is required to run farms, manufacture fertilizer, and ship food. Notice how the price of groceries often reacts to energy shocks.")
        
        fig_rip, ax_rip1 = plt.subplots(figsize=(10, 4))
        ax_rip1.plot(df_merged['Date'], df_merged['Real_Oil_Price'], color='#34495e', label='Real Oil Price', linewidth=2)
        ax_rip1.set_ylabel('Real Oil Price ($)', color='#34495e', fontweight='bold')
        
        ax_rip2 = ax_rip1.twinx()
        ax_rip2.plot(df_merged['Date'], df_merged['Price_Eggs'], color='#27ae60', alpha=0.8, label='Price of Eggs', linestyle='-')
        ax_rip2.set_ylabel('Eggs ($/Dozen)', color='#27ae60', fontweight='bold')
        
        sns.despine(ax=ax_rip1, right=False)
        ax_rip1.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig_rip)

    with rtab2:
        st.write("### Do Oil Spikes Correlate with Presidential Flips?")
        st.write("The chart below overlays major >5% monthly oil price spikes (orange dots) against years where the US Presidency flipped to the opposing party.")
        
        fig_pol, ax_pol = plt.subplots(figsize=(10, 4))
        ax_pol.plot(df_merged['Date'], df_merged['Real_Oil_Price'], color='gray', alpha=0.4, label='Real Oil Price')
        
        spikes = df_merged[df_merged['Price_Spike'] == 1]
        ax_pol.scatter(spikes['Date'], spikes['Real_Oil_Price'], color='#e67e22', s=40, zorder=5, label='>5% Monthly Spike')
        
        flips = [('1977-01-20', 'Dem'), ('1981-01-20', 'Rep'), ('1993-01-20', 'Dem'), 
                 ('2001-01-20', 'Rep'), ('2009-01-20', 'Dem'), ('2017-01-20', 'Rep'), ('2021-01-20', 'Dem')]
        
        added_dem = False
        added_rep = False
        
        for date, party in flips:
            dt = pd.to_datetime(date)
            if df_merged['Date'].min() <= dt <= df_merged['Date'].max():
                c = '#2980b9' if party == 'Dem' else '#c0392b'
                lbl = ""
                if party == 'Dem' and not added_dem:
                    lbl = "Flip to Democrat"
                    added_dem = True
                elif party == 'Rep' and not added_rep:
                    lbl = "Flip to Republican"
                    added_rep = True
                    
                ax_pol.axvline(dt, color=c, linestyle=':', linewidth=2, label=lbl, alpha=0.7)
        
        ax_pol.set_ylabel('Real Oil Price ($)')
        ax_pol.legend(loc='upper left', bbox_to_anchor=(1, 1))
        sns.despine(ax=ax_pol)
        fig_pol.tight_layout()
        st.pyplot(fig_pol)
        
        st.caption("While correlation doesn't equal causation, voters often hold incumbent administrations accountable for energy-driven inflation and gas prices.")