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
from loaders.family_tree_loader import get_family_tree_data

def render_genealogy_web():
    
    # 1. Fetch abstracted data from our Python loader
    graph_data, tree_data = get_family_tree_data()
    
    # 2. Serialize dicts to JSON strings for JavaScript consumption
    graph_data_json = json.dumps(graph_data)
    tree_data_json = json.dumps(tree_data)

    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body { margin: 0; background-color: #0F172A; font-family: 'Inter', sans-serif; overflow: hidden; }
            .container { display: flex; width: 100%; height: 700px; }
            .panel { flex: 1; position: relative; height: 100%; }
            .right-panel { border-left: 2px solid #1E293B; }
            .panel-title { 
                position: absolute; top: 15px; left: 20px; color: #94A3B8; 
                font-size: 14px; font-weight: bold; letter-spacing: 1px; text-transform: uppercase;
            }
            
            /* RICH TOOLTIP UPDATES */
            #tooltip {
                position: absolute; text-align: left; padding: 12px; font-size: 12px;
                background: rgba(15, 23, 42, 0.98); color: #F8FAFC; border: 1px solid #475569;
                border-radius: 6px; opacity: 0; box-shadow: 0 4px 10px rgba(0,0,0,0.5);
                transition: opacity 0.2s; width: 250px; z-index: 10;
                pointer-events: auto; /* Allows clicking links inside the tooltip */
            }
            #tooltip a { color: #38BDF8; text-decoration: none; }
            #tooltip a:hover { text-decoration: underline; }
            #tooltip .bio-box {
                margin-top: 8px; padding-top: 8px; border-top: 1px dashed #334155;
                font-size: 11px; color: #CBD5E1; max-height: 150px; overflow-y: auto;
            }

            .node { cursor: pointer; stroke: #0F172A; stroke-width: 1.5px; transition: stroke-width 0.2s; }
            .node:hover { stroke: #F8FAFC; stroke-width: 4px; }
            
            /* FORCE WEB LINK STYLES: Handled dynamically by D3 */
            .link-main { fill: none; stroke: #475569; }
            .link-marriage { fill: none; stroke: #94A3B8; stroke-dasharray: 6,6; }
            .link-leaf { fill: none; stroke: #475569; }
            .link-inlaw { fill: none; stroke: #64748B; stroke-dasharray: 4,4; }
            .link-spouse_main { fill: none; stroke: #64748B; stroke-opacity: 0.5; stroke-width: 1.5px; }
            
            /* TIMELINE TREE LINK STYLES */
            .tree-link { fill: none; stroke: #334155; }
            .tree-link.inlaw { stroke-dasharray: 4,4; }
            
            .grid-line { stroke: #1E293B; stroke-width: 1px; stroke-dasharray: 4 4; }
            .grid-label { fill: #64748B; font-size: 10px; font-family: monospace; }
        </style>
    </head>
    <body>
        <div id="tooltip"></div>
        <div class="container">
            <div class="panel">
                <div class="panel-title">Procedural Ancestral Cluster</div>
                <svg id="viz-force" width="100%" height="100%"></svg>
            </div>
            <div class="panel right-panel">
                <div class="panel-title">Generational Family Tree</div>
                <svg id="viz-tree" width="100%" height="100%"></svg>
            </div>
        </div>

        <script>
            // ==========================================
            // 1. DATA HYDRATION (Injected via Python)
            // ==========================================
            const graphData = ___GRAPH_DATA___;
            const treeData = ___TREE_DATA___;

            // ==========================================
            // 2. PROCEDURAL ALGORITHMS (Color & Size)
            // ==========================================
            let maxSteps = { K: 0, V: 0, R: 0, L: 0, B: 0, A: 0 };
            
            graphData.nodes.forEach(d => {
                if (d.lateral === 0 && maxSteps[d.branch] !== undefined) {
                    maxSteps[d.branch] = Math.max(maxSteps[d.branch], d.steps);
                }
            });

            function calcRadius(d) {
                const ROOT_SIZE = 25;
                if (d.id === "Kyle Killebrew" || d.name === "Kyle Killebrew") return ROOT_SIZE;

                if (d.inLaw || d.isSpouseLine) {
                    let anchor = d.anchorStep || d.steps;
                    let spouseSize = ROOT_SIZE * (1 - 0.10 * anchor);
                    if (d.lateral > 0 && !d.isSpouseLine) spouseSize *= 0.3333; 
                    return Math.max(2, spouseSize * 0.4); 
                }

                if (d.lateral === 0) {
                    let sz = ROOT_SIZE * (1 - 0.10 * d.steps);
                    return Math.max(2, sz);
                } else {
                    let rootStep = d.steps + d.lateral; 
                    let rootSize = ROOT_SIZE * (1 - 0.10 * rootStep);
                    let siblingSize = rootSize * 0.3333;
                    
                    if (d.lateral === 1) return Math.max(2, siblingSize);
                    let factor = Math.pow(0.95, d.lateral - 1);
                    return Math.max(2, siblingSize * factor);
                }
            }

            function calcOpacity(d) {
                if (d.lateral === 0) return 1.0;
                return Math.max(0.1, 1.0 - (0.2 * d.lateral));
            }

            function calcLinkWidth(d) {
                if (d.type === "marriage" || (d.target && d.target.inLaw)) return 2;
                let targetNode = d.target.data ? d.target.data : d.target;
                let targetRadius = calcRadius(targetNode);
                return Math.max(1, targetRadius * 0.2); 
            }

            function calcColor(d) {
                // 1. Force the root node to always be pure white
                if (d.id === "Kyle Killebrew" || d.name === "Kyle Killebrew") {
                    return "rgb(240, 240, 240)";
                }

                // 2. Assign the static spousal accent colors
                if (d.inLaw || d.isSpouseLine) {
                    if (d.branch === "K" || d.branch === "R") return "rgb(0, 255, 255)"; // Cyan (Dad)
                    if (d.branch === "B" || d.branch === "A") return "rgb(255, 105, 180)"; // Hot Pink (Step-Mom)
                    if (d.branch === "V" || d.branch === "L") return "rgb(205, 127, 50)"; // Bright Bronze (Bio Mom)
                    return "rgb(255, 215, 0)"; // Fallback Gold
                }

                // 3. Dynamic Generation Fading (White to Full Saturation)
                let mainSteps = d.steps + d.lateral; 
                let mMax = maxSteps[d.branch] || 1;
                let t = Math.min(1, Math.max(0, mainSteps / mMax));
                
                let r = 240, g = 240, b = 240; 

                if (d.branch === "K") { r = Math.round(240 - 240 * t); g = Math.round(240 - 240 * t); } 
                else if (d.branch === "R") { r = Math.round(240 - 80 * t); g = Math.round(240 - 208 * t); } 
                else if (d.branch === "V") { r = Math.round(240 - 240 * t); b = Math.round(240 - 240 * t); } 
                else if (d.branch === "L") { b = Math.round(240 - 240 * t); } 
                else if (d.branch === "B") { g = Math.round(240 - 240 * t); b = Math.round(240 - 240 * t); } 
                else if (d.branch === "A") { g = Math.round(240 - 112 * t); b = Math.round(240 - 240 * t); }

                return `rgb(${r}, ${g}, ${b})`;
            }

            // Centralized Tooltip Rendering Function
            function renderTooltip(e, d, isTree = false) {
                const tooltip = d3.select("#tooltip");
                let data = isTree ? d.data : d;
                
                let name = data.name || data.id;
                let stat = data.desc;
                if(data.inLaw && !data.desc) stat = "In-Law (Spouse)";
                
                let htmlStr = `<strong>${name}</strong><br/><span style="color:#94A3B8">${stat}</span>`;
                
                // If a rich bio string exists in the JSON, build the scrolling text box
                if (data.bio) {
                    htmlStr += `<div class="bio-box">${data.bio}</div>`;
                }

                tooltip.html(htmlStr)
                       .style("left", (e.pageX + 15) + "px")
                       .style("top", (e.pageY - 28) + "px");
                
                // Instead of instantly hiding on mouseout of circle, we let the tooltip stay open 
                // briefly so they can mouse onto it and click links.
                tooltip.transition().duration(200).style("opacity", 1).style("pointer-events", "auto");
            }
            
            // Global click listener to hide tooltip when clicking elsewhere
            d3.select("body").on("click", (e) => {
                if(!e.target.closest('.node') && !e.target.closest('#tooltip')) {
                    d3.select("#tooltip").transition().duration(200).style("opacity", 0).style("pointer-events", "none");
                }
            });

            // ==========================================
            // 3. LEFT PANEL: FORCE WEB
            // ==========================================
            const fSvg = d3.select("#viz-force");
            const width = fSvg.node().getBoundingClientRect().width;
            const height = 700;

            fSvg.attr("viewBox", [-width / 2, -height / 2, width, height]);

            const simulation = d3.forceSimulation(graphData.nodes)
                .force("link", d3.forceLink(graphData.links).id(d => d.id)
                    .distance(d => {
                        let targetNode = d.target.data ? d.target.data : d.target;
                        if (d.type === "spouse_main") return 8;

                        let rootStep = targetNode.lateral > 0 ? (targetNode.steps + targetNode.lateral) : targetNode.steps;
                        let shrinkFactor = Math.max(0.2, 1 - 0.10 * rootStep);
                        
                        if (d.type === "marriage" || d.type === "inlaw") return 15; 
                        if (d.type === "leaf") return 25 * shrinkFactor; 
                        return 65 * shrinkFactor;
                    })
                    .strength(d => {
                        if (d.type === "marriage") return 0.1; 
                        if (d.type === "leaf" || d.type === "inlaw" || d.type === "spouse_main") return 2; 
                        return 1;
                    }) 
                )
                .force("charge", d3.forceManyBody().strength(d => {
                    let rootStep = d.lateral > 0 ? (d.steps + d.lateral) : d.steps;
                    let shrinkFactor = Math.max(0.2, 1 - 0.10 * rootStep);
                    
                    if (d.inLaw || d.isSpouseLine) return -5;
                    if (d.lateral > 0) return -15 * shrinkFactor; 
                    return -200 * shrinkFactor; 
                }))
                .force("collide", d3.forceCollide().radius(d => calcRadius(d) + 5).iterations(2))
                .force("x", d3.forceX())
                .force("y", d3.forceY());

            const fLink = fSvg.append("g").selectAll("line").data(graphData.links).join("line")
                .attr("class", d => `link-${d.type}`);

            const fNode = fSvg.append("g").selectAll("g").data(graphData.nodes).join("g")
                .call(d3.drag()
                    .on("start", (e,d) => { if(!e.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
                    .on("drag", (e,d) => { d.fx=e.x; d.fy=e.y; })
                    .on("end", (e,d) => { if(!e.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; })
                );

            fNode.append("circle")
                .attr("class", "node")
                .attr("r", d => calcRadius(d))
                .attr("fill", d => calcColor(d))
                .attr("opacity", d => calcOpacity(d))
                // Changed from mouseover to click so tooltip links can be engaged
                .on("click", (e,d) => {
                    e.stopPropagation(); 
                    renderTooltip(e, d, false);
                });

            simulation.on("tick", () => {
                fLink.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
                     .attr("x2", d => d.target.x).attr("y2", d => d.target.y)
                     .attr("stroke-width", d => calcLinkWidth(d))
                     .attr("stroke-opacity", d => calcOpacity(d.target));
                fNode.attr("transform", d => `translate(${d.x},${d.y})`);
            });

            // --- Dimension & Color Legend ---
            const legend = fSvg.append("g").attr("transform", `translate(${-width/2 + 20}, ${height/2 - 170})`);
            legend.append("text").attr("fill", "#94A3B8").attr("font-size", "12px").attr("y", -10).text("LINEAGE COLORS:");
            
            const defs = fSvg.append("defs");
            function buildGrad(id, c1, c2) {
                let g = defs.append("linearGradient").attr("id", id);
                g.append("stop").attr("offset", "0%").attr("stop-color", c1);
                g.append("stop").attr("offset", "100%").attr("stop-color", c2);
            }
            buildGrad("k-grad", "rgb(240,240,240)", "rgb(0,0,240)");
            buildGrad("r-grad", "rgb(240,240,240)", "rgb(160,32,240)");
            buildGrad("v-grad", "rgb(240,240,240)", "rgb(0,240,0)");
            buildGrad("l-grad", "rgb(240,240,240)", "rgb(240,240,0)");
            buildGrad("b-grad", "rgb(240,240,240)", "rgb(240,0,0)");
            buildGrad("a-grad", "rgb(240,240,240)", "rgb(240,128,0)");

            const labels = ["Killebrew", "Rasmussen", "Vanderhoop", "Lieber", "Buzunis", "Ginakes"];
            const grads = ["url(#k-grad)", "url(#r-grad)", "url(#v-grad)", "url(#l-grad)", "url(#b-grad)", "url(#a-grad)"];
            
            labels.forEach((l, i) => {
                legend.append("rect").attr("y", i*20).attr("width", 50).attr("height", 10).style("fill", grads[i]);
                legend.append("text").attr("x", 60).attr("y", i*20 + 9).attr("fill", "#64748B").attr("font-size", "10px").text(l);
            });
            legend.append("rect").attr("x", 0).attr("y", 125).attr("width", 10).attr("height", 10).style("fill", "rgb(0, 255, 255)");
            legend.append("text").attr("x", 15).attr("y", 134).attr("fill", "#64748B").attr("font-size", "10px").text("Dad's In-Laws");
            legend.append("rect").attr("x", 85).attr("y", 125).attr("width", 10).attr("height", 10).style("fill", "rgb(205, 127, 50)");
            legend.append("text").attr("x", 100).attr("y", 134).attr("fill", "#64748B").attr("font-size", "10px").text("Mom's In-Laws");
            legend.append("rect").attr("x", 0).attr("y", 140).attr("width", 10).attr("height", 10).style("fill", "rgb(255, 105, 180)");
            legend.append("text").attr("x", 15).attr("y", 149).attr("fill", "#64748B").attr("font-size", "10px").text("Step-Mom's In-Laws");

            // ==========================================
            // 4. RIGHT PANEL: GENERATIONAL TREE
            // ==========================================
            const tSvg = d3.select("#viz-tree");
            const tWidth = tSvg.node().getBoundingClientRect().width;
            
            const tScale = 0.65;
            const tRad = d => calcRadius(d) * tScale;
            const root = d3.hierarchy(treeData);
            
            d3.tree().size([tWidth - 60, height - 100])
                .separation((a, b) => {
                    let isSpouse = a.data.inLaw || b.data.inLaw;
                    let baseSep = a.parent === b.parent ? 0.75 : 2.5;
                    return isSpouse ? 0.25 : baseSep; 
                })(root);
            
            const base_y = 120;
            const gen_gap = 70;
            let minStep = 0; let maxStep = 0;

            root.each(d => { 
                d.y = base_y + (d.data.steps * gen_gap); 
                if (d.data.lateral > 0 && !d.data.inLaw && !d.data.isSpouseLine && d.parent) {
                    let lateralSiblings = d.parent.children.filter(c => !c.data.inLaw && !c.data.isSpouseLine);
                    if (lateralSiblings.length > 1) {
                        let idx = lateralSiblings.indexOf(d);
                        d.y += (idx % 2 === 0) ? -15 : 15; 
                    }
                }
                minStep = Math.min(minStep, d.data.steps);
                maxStep = Math.max(maxStep, d.data.steps);
            });

            root.each(d => {
                if (d.data.inLaw && !d.data.isSpouseLine && d.parent) {
                    d.x = d.parent.x + tRad(d.parent.data) + tRad(d.data) + 2;
                    d.y = d.parent.y; 
                }
            });

            const genMarks = d3.range(minStep, maxStep + 1);
            tSvg.selectAll(".grid-line").data(genMarks).enter().append("line")
                .attr("class", "grid-line").attr("x1", 30).attr("x2", tWidth - 10)
                .attr("y1", d => base_y + (d * gen_gap)).attr("y2", d => base_y + (d * gen_gap));
                
            tSvg.selectAll(".grid-label").data(genMarks).enter().append("text")
                .attr("class", "grid-label").attr("x", 5).attr("y", d => base_y + (d * gen_gap) - 5)
                .text(d => d === 0 ? "Baseline" : (d < 0 ? `Gen ${Math.abs(d)} (Next)` : `Gen ${d}`));

            const treeGroup = tSvg.append("g").attr("transform", "translate(30, 20)");
            
            treeGroup.selectAll(".tree-link").data(root.links()).enter().append("path")
                .attr("class", d => {
                    if (d.target.data.inLaw || d.target.data.isSpouseLine) return "tree-link inlaw";
                    if (d.target.data.lateral > 0) return "tree-link leaf";
                    return "tree-link main";
                })
                .attr("d", d3.linkVertical().x(d => d.x).y(d => d.y))
                .attr("stroke-width", d => {
                    if (d.target.data.inLaw && !d.target.data.isSpouseLine) return 0;
                    return Math.max(1.5, (calcRadius(d.target.data) * 0.2) * tScale);
                })
                .attr("stroke-opacity", d => (d.target.data.inLaw && !d.target.data.isSpouseLine) ? 0 : 0.8);

            const tNode = treeGroup.selectAll(".tree-node").data(root.descendants()).enter().append("g")
                .attr("transform", d => `translate(${d.x},${d.y})`);

            tNode.append("circle")
                .attr("class", "node")
                .attr("r", d => tRad(d.data))
                .attr("fill", d => calcColor(d.data))
                .on("click", (e,d) => {
                    e.stopPropagation();
                    renderTooltip(e, d, true);
                });

        </script>
    </body>
    </html>
    """
    
    # 3. Dynamic Python string injection!
    html_code = html_template.replace("___GRAPH_DATA___", graph_data_json).replace("___TREE_DATA___", tree_data_json)
    
    components.html(html_code, height=720)