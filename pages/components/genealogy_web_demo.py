"""
=============================================================================
MODULE: pages/components/genealogy_web_demo.py
AUTHOR: Kyle W. Killebrew, PhD
DESCRIPTION: 
    Custom Streamlit component to render an interactive dual-view Genealogy Web. 
    Left Panel: Force-directed gravity cluster (Procedurally generated).
    Right Panel: Generational Pedigree Tree.
=============================================================================
"""

import streamlit.components.v1 as components
import json
import streamlit as st
import plotly.graph_objects as go # <--- NEW IMPORT
from loaders.family_tree_loader import get_family_tree_data

# Standard UI Lock for Plotly
PLOTLY_CONFIG = {'scrollZoom': False, 'displayModeBar': False, 'staticPlot': False}

def render_genealogy_web():
    
    # 1. Fetch abstracted data from our Python loader
    graph_data, _ = get_family_tree_data() # We drop tree_data as we are using Plotly now
    
    # 2. Serialize dict to JSON string for JavaScript consumption
    graph_data_json = json.dumps(graph_data)

    # [DELTA] Removed the split-panel HTML. Added warm-up ticks and adjusted physics.
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body { margin: 0; background-color: #0F172A; font-family: 'Inter', sans-serif; overflow: hidden; }
            .container { display: flex; width: 100%; height: 750px; }
            .panel { flex: 1; position: relative; height: 100%; }
            .panel-title { 
                position: absolute; top: 15px; left: 20px; color: #94A3B8; 
                font-size: 14px; font-weight: bold; letter-spacing: 1px; text-transform: uppercase;
            }
            #tooltip {
                position: absolute; text-align: left; padding: 12px; font-size: 12px;
                background: rgba(15, 23, 42, 0.98); color: #F8FAFC; border: 1px solid #475569;
                border-radius: 6px; opacity: 0; box-shadow: 0 4px 10px rgba(0,0,0,0.5);
                transition: opacity 0.2s; width: 250px; z-index: 10;
                pointer-events: auto;
            }
            #tooltip .bio-box {
                margin-top: 8px; padding-top: 8px; border-top: 1px dashed #334155;
                font-size: 11px; color: #CBD5E1; max-height: 150px; overflow-y: auto;
            }
            .node { cursor: pointer; stroke: #0F172A; stroke-width: 1.5px; transition: stroke-width 0.2s; }
            .node:hover { stroke: #F8FAFC; stroke-width: 4px; }
            .link-main { fill: none; stroke: #475569; }
            .link-marriage { fill: none; stroke: #94A3B8; stroke-dasharray: 6,6; }
            .link-leaf { fill: none; stroke: #475569; }
            .link-inlaw { fill: none; stroke: #64748B; stroke-dasharray: 4,4; }
        </style>
    </head>
    <body>
        <div id="tooltip"></div>
        <div class="container">
            <div class="panel">
                <div class="panel-title">Procedural Ancestral Cluster</div>
                <svg id="viz-force" width="100%" height="100%"></svg>
            </div>
        </div>

        <script>
            const graphData = ___GRAPH_DATA___;
            let maxSteps = { K: 0, V: 0, R: 0, L: 0, B: 0, A: 0 };
            
            graphData.nodes.forEach(d => {
                if (d.lateral === 0 && maxSteps[d.branch] !== undefined) maxSteps[d.branch] = Math.max(maxSteps[d.branch], d.steps);
            });

            function calcRadius(d) {
                const ROOT_SIZE = 25;
                if (d.id === "Kyle Killebrew" || d.name === "Kyle Killebrew") return ROOT_SIZE;
                if (d.inLaw || d.isSpouseLine) return Math.max(2, (ROOT_SIZE * (1 - 0.10 * (d.anchorStep || d.steps))) * 0.4); 
                return Math.max(2, (ROOT_SIZE * (1 - 0.10 * (d.steps + d.lateral))) * (d.lateral === 0 ? 1 : Math.pow(0.33, d.lateral)));
            }

            function calcOpacity(d) { return d.lateral === 0 ? 1.0 : Math.max(0.1, 1.0 - (0.2 * d.lateral)); }
            
            function calcColor(d) {
                if (d.id === "Kyle Killebrew" || d.name === "Kyle Killebrew") return "rgb(240, 240, 240)";
                if (d.inLaw || d.isSpouseLine) {
                    if (d.branch === "K" || d.branch === "R") return "rgb(0, 255, 255)";
                    if (d.branch === "B" || d.branch === "A") return "rgb(255, 105, 180)";
                    if (d.branch === "V" || d.branch === "L") return "rgb(205, 127, 50)";
                    return "rgb(255, 215, 0)";
                }
                let t = Math.min(1, Math.max(0, (d.steps + d.lateral) / (maxSteps[d.branch] || 1)));
                let r = 240, g = 240, b = 240; 
                if (d.branch === "K") { r = g = Math.round(240 - 240 * t); } 
                else if (d.branch === "R") { r = Math.round(240 - 80 * t); g = Math.round(240 - 208 * t); } 
                else if (d.branch === "V") { r = b = Math.round(240 - 240 * t); } 
                else if (d.branch === "L") { b = Math.round(240 - 240 * t); } 
                else if (d.branch === "B") { g = b = Math.round(240 - 240 * t); } 
                else if (d.branch === "A") { g = Math.round(240 - 112 * t); b = Math.round(240 - 240 * t); }
                return `rgb(${r}, ${g}, ${b})`;
            }

            function renderTooltip(e, d) {
                let stat = d.desc;
                if(d.inLaw && !d.desc) stat = "In-Law (Spouse)";
                let htmlStr = `<strong>${d.name || d.id}</strong><br/><span style="color:#94A3B8">${stat}</span>`;
                if (d.bio) htmlStr += `<div class="bio-box">${d.bio}</div>`;
                d3.select("#tooltip").html(htmlStr).style("left", (e.pageX + 15) + "px").style("top", (e.pageY - 28) + "px")
                  .transition().duration(200).style("opacity", 1).style("pointer-events", "auto");
            }
            
            d3.select("body").on("click", (e) => {
                if(!e.target.closest('.node') && !e.target.closest('#tooltip')) {
                    d3.select("#tooltip").transition().duration(200).style("opacity", 0).style("pointer-events", "none");
                }
            });

            const fSvg = d3.select("#viz-force");
            const width = fSvg.node().getBoundingClientRect().width;
            const height = 750;
            fSvg.attr("viewBox", [-width / 2, -height / 2, width, height]);

            // [DELTA] Adjusted Physics Engine for massive clusters
            const simulation = d3.forceSimulation(graphData.nodes)
                .force("link", d3.forceLink(graphData.links).id(d => d.id)
                    .distance(d => {
                        let isSameBranch = (d.source.branch === d.target.branch);
                        let rootStep = d.target.steps || 1;
                        let dist = isSameBranch ? 10 : 35; // Tighter if same family line
                        return Math.max(5, dist * Math.pow(0.85, rootStep)); // Exponential decay
                    })
                    .strength(d => (d.source.branch === d.target.branch) ? 1.5 : 0.5) // Higher attraction for family
                )
                .force("charge", d3.forceManyBody().strength(d => {
                    let rootStep = d.lateral > 0 ? (d.steps + d.lateral) : d.steps;
                    let repulsion = -350 * Math.pow(0.55, rootStep); // Less pushing on edges
                    return Math.min(-5, repulsion); 
                }))
                .force("collide", d3.forceCollide().radius(d => calcRadius(d) + 4).iterations(3))
                .force("x", d3.forceX())
                .force("y", d3.forceY());

            const fLink = fSvg.append("g").selectAll("line").data(graphData.links).join("line")
                .attr("class", d => `link-${d.type}`)
                .attr("stroke-width", d => Math.max(1, calcRadius(d.target) * 0.2))
                .attr("stroke-opacity", d => calcOpacity(d.target));

            const fNode = fSvg.append("g").selectAll("g").data(graphData.nodes).join("g")
                .call(d3.drag()
                    .on("start", (e,d) => { if(!e.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
                    .on("drag", (e,d) => { d.fx=e.x; d.fy=e.y; })
                    .on("end", (e,d) => { if(!e.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; })
                );

            fNode.append("circle").attr("class", "node").attr("r", d => calcRadius(d))
                 .attr("fill", d => calcColor(d)).attr("opacity", d => calcOpacity(d))
                 .on("click", (e,d) => { e.stopPropagation(); renderTooltip(e, d); });

            // [DELTA] Pre-compute ticks to instantly stabilize the jumbled map on render!
            for (let i = 0; i < 200; ++i) simulation.tick();

            simulation.on("tick", () => {
                fLink.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
                     .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
                fNode.attr("transform", d => `translate(${d.x},${d.y})`);
            });
        </script>
    </body>
    </html>
    """
    
    html_code = html_template.replace("___GRAPH_DATA___", graph_data_json)
    
    # Render the D3 Web
    components.html(html_code, height=750)
    
    st.divider()
    
    # =================================================================
    # [DELTA] 3. PROCEDURAL SANKEY GENERATION (Python/Plotly)
    # =================================================================
    st.subheader("Ancestral Sankey Flow")
    st.write("Visualizing the direct genetic flow backwards through the generations.")
    
    # Filter only direct ancestors (Drop siblings/in-laws to prevent Sankey chaos)
    direct_nodes = [n for n in graph_data["nodes"] if n["lateral"] == 0 and not n["inLaw"]]
    
    if not direct_nodes:
        st.warning("Insufficient data to render Sankey.")
        return
        
    # Map GEDCOM IDs to integers for Plotly
    node_mapping = {n["id"]: idx for idx, n in enumerate(direct_nodes)}
    
    # Map colors to match the web
    color_map = {
        "K": "#1E3A8A", # Deep Blue
        "R": "#7E22CE", # Deep Purple
        "V": "#15803D", # Deep Green
        "L": "#A16207", # Deep Yellow/Bronze
        "B": "#B91C1C", # Deep Red
        "A": "#C2410C", # Deep Orange
        "M": "#F8FAFC"  # White (User)
    }
    
    sankey_colors = [color_map.get(n["branch"], "#64748B") for n in direct_nodes]
    sankey_labels = [n["name"] for n in direct_nodes]
    
    source = []
    target = []
    value = []
    
    for link in graph_data["links"]:
        # Only include links where BOTH nodes are in our direct ancestors list
        if link["source"] in node_mapping and link["target"] in node_mapping:
            # In a Reverse Family Tree, we flow FROM child (target) TO parent (source)
            source.append(node_mapping[link["target"]])
            target.append(node_mapping[link["source"]])
            value.append(1) # Equal weighting per ancestor
            
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 20, thickness = 20,
            line = dict(color = "#0F172A", width = 0.5),
            label = sankey_labels,
            color = sankey_colors
        ),
        link = dict(
            source = source, target = target, value = value,
            color = "rgba(148, 163, 184, 0.3)" # Transparent Slate for links
        )
    )])

    fig.update_layout(
        font=dict(size=10, color="#F8FAFC"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=600,
        margin=dict(l=0, r=0, t=10, b=10)
    )
    
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)