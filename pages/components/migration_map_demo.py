"""
=============================================================================
MODULE: pages/components/migration_map_demo.py
AUTHOR: Kyle W. Killebrew, PhD
DESCRIPTION: 
    Custom Streamlit component rendering a 3D interactive Plotly globe.
    Plots the geographic migration vectors for six distinct lineages using
    a mathematical color gradient that fades to white at the origin.
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
            "base_color": (0, 0, 240), # Blue
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
            "base_color": (160, 32, 240), # Purple
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
            "base_color": (0, 240, 0), # Green
            "nodes": [
                {"city": "Paramaribo, Suriname", "lat": 5.8520, "lon": -55.2038},
                {"city": "Gay Head (Aquinnah), MA", "lat": 41.3368, "lon": -70.8316},
                {"city": "Las Vegas, NV", "lat": 36.1699, "lon": -115.1398}
            ]
        },
        "Lieber": {
            "base_color": (240, 240, 0), # Yellow
            "nodes": [
                {"city": "Bonn, Germany", "lat": 50.7374, "lon": 7.0982},
                {"city": "Hofgeismar, Germany", "lat": 51.4947, "lon": 9.3828},
                {"city": "Orlando, FL", "lat": 28.5383, "lon": -81.3792},
                {"city": "Las Vegas, NV", "lat": 36.1699, "lon": -115.1398}
            ]
        },
        "Buzunis": {
            "base_color": (240, 0, 0), # Red
            "nodes": [
                {"city": "Levidion, Greece", "lat": 37.6811, "lon": 22.2968},
                {"city": "Vanguard, SK, Canada", "lat": 49.9167, "lon": -107.0333},
                {"city": "Winnipeg, MB, Canada", "lat": 49.8951, "lon": -97.1384},
                {"city": "Las Vegas, NV", "lat": 36.1699, "lon": -115.1398}
            ]
        },
        "Ginakes": {
            "base_color": (240, 128, 0), # Orange
            "nodes": [
                {"city": "Greece (General)", "lat": 39.0742, "lon": 21.8243}, 
                {"city": "Fargo, ND", "lat": 46.8772, "lon": -96.7898},
                {"city": "Winnipeg, MB, Canada", "lat": 49.8951, "lon": -97.1384},
                {"city": "Las Vegas, NV", "lat": 36.1699, "lon": -115.1398}
            ]
        }
    }

    fig = go.Figure()

    # --- 2. VECTOR OVERLAYS WITH MATHEMATICAL GRADIENTS ---
    for branch_name, data in lineages.items():
        nodes = data["nodes"]
        bc = data["base_color"]
        max_steps = len(nodes) - 1

        # We add a hidden dummy trace strictly to populate the legend with the solid base color
        fig.add_trace(go.Scattergeo(
            lon=[None], lat=[None],
            mode="markers",
            marker=dict(size=10, color=f"rgb({bc[0]},{bc[1]},{bc[2]})"),
            name=branch_name,
            showlegend=True
        ))

        # MATLAB Analogy: Iterating through a struct array to plot discrete line segments 
        # so we can dynamically shift the RGB values as it gets closer to Vegas
        for i in range(max_steps):
            start_node = nodes[i]
            end_node = nodes[i+1]

            # Calculate color for the start of this segment
            t_start = i / max_steps
            r1 = int(bc[0] + (240 - bc[0]) * t_start)
            g1 = int(bc[1] + (240 - bc[1]) * t_start)
            b1 = int(bc[2] + (240 - bc[2]) * t_start)
            c_start = f"rgb({r1},{g1},{b1})"

            # Calculate color for the end of this segment
            t_end = (i + 1) / max_steps
            r2 = int(bc[0] + (240 - bc[0]) * t_end)
            g2 = int(bc[1] + (240 - bc[1]) * t_end)
            b2 = int(bc[2] + (240 - bc[2]) * t_end)
            c_end = f"rgb({r2},{g2},{b2})"

            fig.add_trace(go.Scattergeo(
                lon=[start_node["lon"], end_node["lon"]],
                lat=[start_node["lat"], end_node["lat"]],
                mode="lines+markers",
                line=dict(width=3, color=c_start), # Line holds the starting color
                marker=dict(
                    size=[6, 6],
                    color=[c_start, c_end], # Plotly accepts an array to color the two dots individually!
                    line=dict(width=1, color="white")
                ),
                text=[start_node["city"], end_node["city"]],
                hoverinfo="text+name",
                name=branch_name,
                showlegend=False
            ))

    # --- 3. GLOBE & CAMERA CONFIGURATION ---
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        height=700,
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
            projection_type="orthographic",
            showcoastlines=True,
            coastlinecolor="#334155",
            showland=True,
            landcolor="#1E293B",        
            showocean=True,
            oceancolor="#0F172A",       
            showlakes=True,
            lakecolor="#0F172A",
            showcountries=True,
            countrycolor="#475569",
            showsubunits=True,          # Show States / Provinces
            subunitcolor="#334155",
            showrivers=True,            # Show Major Rivers
            rivercolor="#0F172A",
            resolution=50,              # Higher detail boundary mapping
            projection_rotation=dict(lon=-80, lat=35, roll=0) 
        )
    )

    # --- 4. RENDER IN STREAMLIT ---
    # scrollZoom is now True, allowing you to wheel-zoom in to see states/provinces
    PLOTLY_CONFIG = {'scrollZoom': True, 'displayModeBar': False, 'staticPlot': False}
    
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)