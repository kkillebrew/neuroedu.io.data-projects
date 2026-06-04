"""
=============================================================================
MODULE: pages/components/migration_map_demo.py
AUTHOR: Kyle W. Killebrew, PhD
DESCRIPTION: 
    Custom Streamlit component rendering a 3D interactive Plotly globe.
    Plots the geographic migration vectors for four distinct lineages using
    a highly customized, low-contrast UI palette.
=============================================================================
"""

import streamlit as st
import plotly.graph_objects as go

def render_migration_map():
    # --- 1. GEOGRAPHIC DATA REPOSITORIES ---
    # MATLAB Analogy: Think of these as structured arrays containing geographic 
    # waypoints that we will plot sequentially as line vectors.
    
    lineages = {
        "Killebrew": {
            "color": "#38BDF8", # Sky Blue Accent
            "nodes": [
                {"city": "Cornwall, UK", "lat": 50.2660, "lon": -5.0527},
                {"city": "Isle of Wight, VA", "lat": 36.9057, "lon": -76.7022},
                {"city": "Tarboro, NC", "lat": 35.8979, "lon": -77.5358},
                {"city": "Clarksville, TN", "lat": 36.5298, "lon": -87.3595},
                {"city": "McAlester, OK", "lat": 34.9334, "lon": -95.7697},
                {"city": "Las Vegas, NV", "lat": 36.1699, "lon": -115.1398}
            ]
        },
        "Rasmussen": {
            "color": "#4ADE80", # Green
            "nodes": [
                {"city": "Copenhagen, Denmark", "lat": 55.6761, "lon": 12.5683},
                {"city": "Hamburg, Germany", "lat": 53.5511, "lon": 9.9937},
                {"city": "New York, NY", "lat": 40.7128, "lon": -74.0060},
                {"city": "Montreal, QC", "lat": 45.5017, "lon": -73.5673},
                {"city": "Detroit, MI", "lat": 42.3314, "lon": -83.0458},
                {"city": "Chicago, IL", "lat": 41.8781, "lon": -87.6298},
                {"city": "Wyoming, NE", "lat": 40.6322, "lon": -95.8453},
                {"city": "Ephraim, UT", "lat": 39.3602, "lon": -111.5866},
                {"city": "Monroe, UT", "lat": 38.6270, "lon": -112.1220},
                {"city": "Las Vegas, NV", "lat": 36.1699, "lon": -115.1398}
            ]
        },
        "Vanderhoop": {
            "color": "#EF4444", # Red
            "nodes": [
                {"city": "Paramaribo, Suriname", "lat": 5.8520, "lon": -55.2038},
                {"city": "Gay Head (Aquinnah), MA", "lat": 41.3368, "lon": -70.8316},
                {"city": "Las Vegas, NV", "lat": 36.1699, "lon": -115.1398}
            ]
        },
        "Lieber": {
            "color": "#FACC15", # Yellow
            "nodes": [
                {"city": "Bonn, Germany", "lat": 50.7374, "lon": 7.0982},
                {"city": "Hofgeismar, Germany", "lat": 51.4947, "lon": 9.3828},
                {"city": "Orlando, FL", "lat": 28.5383, "lon": -81.3792},
                {"city": "Las Vegas, NV", "lat": 36.1699, "lon": -115.1398}
            ]
        }
    }

    fig = go.Figure()

    # --- 2. VECTOR OVERLAYS ---
    # Iterate through each lineage and add a spatial trace
    for branch_name, data in lineages.items():
        lats = [node["lat"] for node in data["nodes"]]
        lons = [node["lon"] for node in data["nodes"]]
        cities = [node["city"] for node in data["nodes"]]
        
        fig.add_trace(go.Scattergeo(
            locationmode="ISO-3",
            lat=lats,
            lon=lons,
            mode="lines+markers",
            line=dict(width=3, color=data["color"]),
            marker=dict(size=6, color=data["color"], line=dict(width=1, color="white")),
            name=branch_name,
            text=cities,
            hoverinfo="text+name"
        ))

    # --- 3. GLOBE & CAMERA CONFIGURATION ---
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)', # Transparent to let Streamlit background through
        plot_bgcolor='rgba(0,0,0,0)',
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#F8FAFC")
        ),
        geo=dict(
            projection_type="orthographic", # Forces the 3D globe render
            showcoastlines=True,
            coastlinecolor="#334155",
            showland=True,
            landcolor="#1E293B",        # Muted slate landmasses
            showocean=True,
            oceancolor="#0F172A",       # Deep slate oceans (matches sidebar)
            showlakes=True,
            lakecolor="#0F172A",
            showcountries=True,
            countrycolor="#334155",
            projection_rotation=dict(lon=-60, lat=25, roll=0) # Centers over the Atlantic
        )
    )

    # --- 4. RENDER IN STREAMLIT ---
    # Strict adherence to the Architecture Guide: No floating toolbars, no scroll-zooming.
    # The globe remains rotatable by clicking and dragging.
    PLOTLY_CONFIG = {'scrollZoom': False, 'displayModeBar': False, 'staticPlot': False}
    
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)