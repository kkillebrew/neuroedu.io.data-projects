"""
=============================================================================
MODULE: pages/components/dna_evidence_demo.py
AUTHOR: Kyle W. Killebrew, PhD
DESCRIPTION: 
    Custom Streamlit component for Genomic Variance and Admixture.
    Demonstrates pipeline structures for tools like `admix` and `myvariant`,
    rendering interactive Admixture profiles and Polygenic Risk Scores (PRS).
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Standard UI Lock for Plotly
PLOTLY_CONFIG = {'scrollZoom': False, 'displayModeBar': False, 'staticPlot': False}

@st.cache_data
def load_authentic_genomic_data():
    """
    Gracefully loads the pre-processed Parquet files from the documents folder.
    MATLAB Analogy: Equivalent to a robust load('documents/genomic_data.mat')
    with an embedded try/catch block.
    """
    # System Pathing: Navigate up 3 directories to reach the Root /documents/ folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    doc_dir = os.path.join(root_dir, "documents")
    
    # Establish the color map for Admixture consistency
    regions = ["Northwestern European", "Aegean / Mediterranean", "Baltic", "Indigenous Americas", "Sub-Saharan African", "Unassigned"]
    colors = ["#1E3A8A", "#10B981", "#38BDF8", "#F59E0B", "#8B5CF6", "#64748B"]
    color_map = dict(zip(regions, colors))
    
    try:
        # Load the ultra-fast Parquet binaries
        pie_data = pd.read_parquet(os.path.join(doc_dir, "dna_pie.parquet"))
        chrom_df = pd.read_parquet(os.path.join(doc_dir, "dna_chromosomes.parquet"))
        traits_data = pd.read_parquet(os.path.join(doc_dir, "dna_traits.parquet"))
        return pie_data, chrom_df, color_map, traits_data
        
    except FileNotFoundError:
        # Graceful Failure: Return empty structures to trigger the UI warning safely
        return pd.DataFrame(), pd.DataFrame(), color_map, pd.DataFrame()

def render_dna_evidence():
    pie_data, chrom_df, color_map, traits_data = load_authentic_genomic_data()
    
    if traits_data.empty:
        st.warning("⚠️ Genomic Parquet files not found in the `documents/` directory. Please run the Colab Database Pipeline and upload the binaries to view this module.")
        return

    # =================================================================
    # TOP SECTION: ADMIXTURE & CLINICAL CHROMOSOME MAPPING
    # =================================================================
    st.subheader("Global Admixture & Physical Chromosomal Loci")
    st.write("Using predictive algorithms against the 1000 Genomes Project (1KG) reference populations, we calculate the Maximum Likelihood of your biogeographical ancestry. The map on the right displays the exact physical base-pair loci of your clinically significant genes queried from the ClinVar database.")
    
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        # Pie Chart (Unchanged - Now fed by real 1KG data)
        fig_pie = px.pie(
            pie_data, names="Region", values="Percentage", 
            hole=0.4, color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_pie.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.5),
            font=dict(color="#F8FAFC")
        )
        st.plotly_chart(fig_pie, use_container_width=True, config=PLOTLY_CONFIG)

    with col2:
        # [DELTA] Updated Chromosome Map to plot exact database physical loci
        fig_chrom = go.Figure()
        
        # We draw the 23 chromosomes as a backdrop
        chrom_list = [str(i) for i in range(22, 0, -1)] + ['X']
        for c in chrom_list:
            fig_chrom.add_trace(go.Scatter(
                x=[0, 160], y=[c, c], mode="lines", 
                line=dict(color="#334155", width=10), showlegend=False, hoverinfo="skip"
            ))
            
        # We plot the real clinical SNPs as bright markers on top of the chromosomes
        fig_chrom.add_trace(go.Scatter(
            x=chrom_df['Position_Mb'], y=chrom_df['Chromosome'],
            mode="markers+text",
            marker=dict(size=12, color="#38BDF8", line=dict(width=2, color="#F8FAFC")),
            text=chrom_df['Gene'],
            textposition="top center",
            hovertemplate="<b>%{text}</b><br>Chromosome %{y}<br>Locus: %{x:.2f} Mb<extra></extra>",
            showlegend=False
        ))

        fig_chrom.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Chromosomal Position (Megabases)",
            yaxis_title="Chromosome Number",
            yaxis=dict(categoryorder='array', categoryarray=chrom_list),
            margin=dict(l=0, r=0, t=10, b=0),
            height=500,
            font=dict(color="#F8FAFC")
        )
        st.plotly_chart(fig_chrom, use_container_width=True, config=PLOTLY_CONFIG)

    st.divider()

    # =================================================================
    # MIDDLE SECTION: POLYGENIC RISK & TRAITS (DIVERGING BARS)
    # =================================================================
    st.subheader("Polygenic Trait Modeling & Risk Scores (PRS)")
    st.write("By querying APIs like `myvariant.info` and utilizing GWAS summary statistics, we aggregate the effect sizes of thousands of Single Nucleotide Polymorphisms (SNPs). This diverging chart maps personal genetic predispositions as Z-Scores (Standard Deviations) away from the global population average (0).")

    # Diverging Bar Logic: Color based on positive or negative Z-Score
    traits_data['Color'] = np.where(traits_data['Z_Score'] > 0, '#38BDF8', '#1E3A8A')
    
    fig_prs = go.Figure()

    fig_prs.add_trace(go.Bar(
        x=traits_data['Z_Score'],
        y=traits_data['Trait'],
        orientation='h',
        marker_color=traits_data['Color'],
        text=traits_data['Z_Score'].round(2),
        textposition='outside',
        customdata=traits_data['Bio_Note'],
        hovertemplate="<b>%{y}</b><br>Z-Score: %{x}<br><i>%{customdata}</i><extra></extra>"
    ))

    # Add a prominent centerline for the Population Average
    fig_prs.add_vline(x=0, line_width=2, line_color="#F8FAFC", opacity=0.7, line_dash="dash")

    fig_prs.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Standard Deviations from Population Mean (Z-Score)",
        yaxis_title="",
        yaxis=dict(autorange="reversed"), # Reads top-to-bottom
        margin=dict(l=0, r=0, t=30, b=20),
        height=450,
        font=dict(color="#F8FAFC")
    )
    
    # Annotations to explain the X-axis visually
    fig_prs.add_annotation(x=-1.5, y=-0.8, text="← Lower Expression/Risk", showarrow=False, font=dict(color="#94A3B8"))
    fig_prs.add_annotation(x=1.5, y=-0.8, text="Higher Expression/Risk →", showarrow=False, font=dict(color="#94A3B8"))

    st.plotly_chart(fig_prs, use_container_width=True, config=PLOTLY_CONFIG)