"""
=============================================================================
MODULE: pages/components/migration_map_demo.py
AUTHOR: Kyle W. Killebrew, PhD
DESCRIPTION: 
    Custom Streamlit component rendering a 3D interactive Plotly globe.
    Plots the geographic migration vectors for all branches using
    data fetched from the family_tree_loader.
=============================================================================
"""

import streamlit as st
import plotly.graph_objects as go
from loaders.family_tree_loader import get_migration_data

def render_migration_map():
    # 1. Fetch data from central repository
    lineages = get_migration_data()

    fig = go.Figure()

    # 2. Vector Overlays with Fading Gradients & Spatial Jitter
    for branch_idx, (branch_name, data) in enumerate(lineages.items()):
        nodes = data["nodes"]
        bc = data["base_color"]
        max_steps = len(nodes) - 1

        # [DELTA] Calculate a deterministic micro-offset to fan out overlapping lines (e.g., terminating in Vegas)
        # This shifts each branch slightly along a radial axis so they don't stack on top of each other
        lat_offset = (branch_idx % 3 - 1) * 0.45
        lon_offset = (branch_idx * 0.35) - 0.7

        # --- CUSTOM LEGEND HANDLING ---
        if branch_name == "Robinson_Impson":
            legend_html = (
                "<span style='color:rgb(255, 105, 180)'>●</span><br>"
                "<span style='color:rgb(0, 255, 255)'>●</span> Side Trees<br>"
                "<span style='color:rgb(205, 127, 50)'>●</span>"
            )
            fig.add_trace(go.Scattergeo(
                lon=[None], lat=[None], mode="markers",
                marker=dict(size=1, color="rgba(0,0,0,0)"), name=legend_html, showlegend=True
            ))
        elif branch_name in ["Vanderhoop_Salisbury", "Vanderhoop_Diamond"]:
            pass
        else:
            fig.add_trace(go.Scattergeo(
                lon=[None], lat=[None], mode="markers",
                marker=dict(size=10, color=f"rgb({bc[0]},{bc[1]},{bc[2]})"),
                name=branch_name.replace("_", "/"), showlegend=True
            ))

        for i in range(max_steps):
            start_node = nodes[i]
            end_node = nodes[i+1]

            # [DELTA] Apply the jitter ONLY if it's the final converging node, or apply slightly across the board
            # For simplicity and clean visuals, we apply it to all coordinates for this specific branch trace
            s_lat = start_node["lat"] + lat_offset
            s_lon = start_node["lon"] + lon_offset
            e_lat = end_node["lat"] + lat_offset
            e_lon = end_node["lon"] + lon_offset

            t_start = i / max_steps
            c_start = f"rgb({int(bc[0] + (240 - bc[0]) * t_start)},{int(bc[1] + (240 - bc[1]) * t_start)},{int(bc[2] + (240 - bc[2]) * t_start)})"
            
            t_end = (i + 1) / max_steps
            c_end = f"rgb({int(bc[0] + (240 - bc[0]) * t_end)},{int(bc[1] + (240 - bc[1]) * t_end)},{int(bc[2] + (240 - bc[2]) * t_end)})"

            hover_text_start = f"<b>{start_node['city']}</b><br>──────────<br>"
            for p in start_node['people']: hover_text_start += f"<b>{p['name']}</b> ({p['years']})<br>{p['desc']}<br><br>"

            hover_text_end = f"<b>{end_node['city']}</b><br>──────────<br>"
            for p in end_node['people']: hover_text_end += f"<b>{p['name']}</b> ({p['years']})<br>{p['desc']}<br><br>"

            fig.add_trace(go.Scattergeo(
                lon=[s_lon, e_lon], lat=[s_lat, e_lat], # [DELTA] Using the offset coordinates
                mode="lines+markers", line=dict(width=3, color=c_start), 
                marker=dict(size=[8, 8], color=[c_start, c_end], line=dict(width=1, color="#F8FAFC")),
                text=[hover_text_start, hover_text_end], hovertemplate="%{text}<extra></extra>",
                name=branch_name, showlegend=False
            ))

    # 3. Globe & Camera Configuration
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
            showsubunits=True,          # States / Provinces
            subunitcolor="#334155",
            showrivers=True,            # Major Rivers
            rivercolor="#0F172A",
            resolution=50,              # Higher detail boundary mapping
            projection_rotation=dict(lon=-80, lat=35, roll=0) 
        )
    )

    # 4. Render in Streamlit
    PLOTLY_CONFIG = {'scrollZoom': True, 'displayModeBar': False, 'staticPlot': False}
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)