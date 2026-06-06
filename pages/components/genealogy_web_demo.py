"""
=============================================================================
MODULE: pages/components/genealogy_web_demo.py
AUTHOR: Kyle W. Killebrew, PhD
DESCRIPTION: 
    Custom Streamlit component to render an interactive dual-view Genealogy Web. 
    Left Panel: Force-directed gravity cluster (Procedurally generated).
    Right Panel: Chronological Pedigree Tree mapped to historical years.
=============================================================================
"""

import streamlit.components.v1 as components

def render_genealogy_web():
    html_code = """
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
            #tooltip {
                position: absolute; text-align: center; padding: 10px; font-size: 12px;
                background: rgba(30, 41, 59, 0.95); color: #F8FAFC; border: 1px solid #475569;
                border-radius: 6px; pointer-events: none; opacity: 0; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                transition: opacity 0.2s; max-width: 200px; z-index: 10;
            }
            .node { cursor: pointer; stroke: #0F172A; stroke-width: 1.5px; transition: stroke-width 0.2s; }
            .node:hover { stroke: #F8FAFC; stroke-width: 4px; }
            
            /* FORCE WEB LINK STYLES */
            .link-main { fill: none; stroke: #475569; stroke-opacity: 0.6; stroke-width: 5px; }
            .link-marriage { fill: none; stroke: #94A3B8; stroke-opacity: 0.4; stroke-width: 3px; stroke-dasharray: 6,6; }
            .link-leaf { fill: none; stroke: #475569; stroke-opacity: 0.8; stroke-width: 1.5px; }
            .link-inlaw { fill: none; stroke: #64748B; stroke-opacity: 0.6; stroke-width: 1.5px; stroke-dasharray: 4,4; }
            
            /* TIMELINE TREE LINK STYLES */
            .tree-link { fill: none; stroke: #334155; }
            .tree-link.main { stroke-width: 5px; stroke-opacity: 0.6; }
            .tree-link.leaf { stroke-width: 1.5px; stroke-opacity: 0.8; }
            .tree-link.inlaw { stroke-width: 1.5px; stroke-opacity: 0.6; stroke-dasharray: 4,4; }
            
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
                <div class="panel-title">Chronological Pedigree</div>
                <svg id="viz-tree" width="100%" height="100%"></svg>
            </div>
        </div>

        <script>
            // ==========================================
            // 1. PROCEDURAL DATA ARCHITECTURE
            // ==========================================
            
            const graphData = {
                nodes: [
                    { id: "Kyle Killebrew", branch: "M", steps: 0, lateral: 0, desc: "1990 - Present" },
                    { id: "Eric Killebrew", branch: "K", steps: 1, lateral: 0, desc: "1961 - Present" },
                    { id: "Christina Vanderhoop", branch: "V", steps: 1, lateral: 0, desc: "Mother" },
                    { id: "Robert Killebrew", branch: "K", steps: 2, lateral: 0, desc: "1930 - 2017" },
                    { id: "Bonnie Rasmussen", branch: "R", steps: 2, lateral: 0, desc: "1934 - 2020" },
                    { id: "John O. Vanderhoop", branch: "V", steps: 2, lateral: 0, desc: "1934 - 2022" },
                    { id: "Waltrud M. Lieber", branch: "L", steps: 2, lateral: 0, desc: "1934 - 2022" },
                    { id: "William H. K.", branch: "K", steps: 3, lateral: 0, desc: "1898 - 1970" },
                    { id: "Daniel Boone K.", branch: "K", steps: 4, lateral: 0, desc: "1860 - 1939" },
                    { id: "George W. K.", branch: "K", steps: 5, lateral: 0, desc: "1812 - 1871" },
                    { id: "Whitfield K.", branch: "K", steps: 6, lateral: 0, desc: "1793 - 1859" },
                    { id: "Joseph K.", branch: "K", steps: 7, lateral: 0, desc: "1753 - 1824" },
                    { id: "Francis K.", branch: "K", steps: 8, lateral: 0, desc: "1619 - 1673" },
                    { id: "Kelly K.", branch: "K", steps: 1, lateral: 1 },
                    { id: "Stephen K.", branch: "K", steps: 1, lateral: 1 },
                    { id: "Suzie K.", branch: "K", steps: 1, lateral: 1 },
                    { id: "Tony K.", branch: "K", steps: 1, lateral: 1 },
                    { id: "Keri K.", branch: "K", steps: 1, lateral: 1 },
                    { id: "Sheri K.", branch: "K", steps: 1, lateral: 1 },
                    { id: "Ron K.", branch: "K", steps: 2, lateral: 1 },
                    { id: "Urma K.", branch: "K", steps: 2, lateral: 1 },
                    { id: "Clinton Rasmussen", branch: "R", steps: 3, lateral: 0, desc: "1904 - 1979" },
                    { id: "James A. R.", branch: "R", steps: 4, lateral: 0, desc: "1877 - 1965" },
                    { id: "Rasmus J. R.", branch: "R", steps: 5, lateral: 0, desc: "1842 - 1920" },
                    { id: "Jens Rasmussen", branch: "R", steps: 6, lateral: 0, desc: "1810 - 1888" },
                    { id: "Richard R.", branch: "R", steps: 2, lateral: 1 },
                    { id: "Bettie R.", branch: "R", steps: 2, lateral: 1 },
                    { id: "Rhett R.", branch: "R", steps: 2, lateral: 1 },
                    { id: "Oranell", branch: "R", steps: 2, lateral: 1, inLaw: true },
                    { id: "James", branch: "R", steps: 1, lateral: 2 },
                    { id: "Rosemary", branch: "R", steps: 1, lateral: 2 },
                    { id: "Ruth", branch: "R", steps: 1, lateral: 2 },
                    { id: "Karen", branch: "R", steps: 1, lateral: 2 },
                    { id: "Bob", branch: "R", steps: 2, lateral: 1, inLaw: true },
                    { id: "Michelle", branch: "R", steps: 1, lateral: 2 },
                    { id: "Leonard V.", branch: "V", steps: 3, lateral: 0, desc: "1895 - 1989" },
                    { id: "Edwin DeVries V.", branch: "V", steps: 4, lateral: 0, desc: "1848 - 1923" },
                    { id: "William A. V.", branch: "V", steps: 5, lateral: 0, desc: "~1816 - 1893" },
                    { id: "Beulah Salisbury", branch: "V", steps: 5, lateral: 0, desc: "1814 - 1892" },
                    { id: "Johnny Vanderhoop", branch: "V", steps: 1, lateral: 1 },
                    { id: "Marie Emilie Ibe", branch: "L", steps: 3, lateral: 0, desc: "Unknown"},
                    { id: "G. Heinrich L. Lieber", branch: "L", steps: 3, lateral: 0, desc: "Unknown"},
                    { id: "Manfred Lieber", branch: "L", steps: 2, lateral: 1 }
                ],
                links: [
                    { source: "Eric Killebrew", target: "Kyle Killebrew", type: "main" },
                    { source: "Christina Vanderhoop", target: "Kyle Killebrew", type: "main" },
                    { source: "Robert Killebrew", target: "Eric Killebrew", type: "main" },
                    { source: "Bonnie Rasmussen", target: "Eric Killebrew", type: "main" },
                    { source: "John O. Vanderhoop", target: "Christina Vanderhoop", type: "main" },
                    { source: "Waltrud M. Lieber", target: "Christina Vanderhoop", type: "main" },
                    { source: "Eric Killebrew", target: "Christina Vanderhoop", type: "marriage" },
                    { source: "Robert Killebrew", target: "Bonnie Rasmussen", type: "marriage" },
                    { source: "John O. Vanderhoop", target: "Waltrud M. Lieber", type: "marriage" },
                    { source: "William H. K.", target: "Robert Killebrew", type: "main" },
                    { source: "Daniel Boone K.", target: "William H. K.", type: "main" },
                    { source: "George W. K.", target: "Daniel Boone K.", type: "main" },
                    { source: "Whitfield K.", target: "George W. K.", type: "main" },
                    { source: "Joseph K.", target: "Whitfield K.", type: "main" },
                    { source: "Francis K.", target: "Joseph K.", type: "main" },
                    { source: "William H. K.", target: "Ron K.", type: "leaf" },
                    { source: "William H. K.", target: "Urma K.", type: "leaf" },
                    { source: "Robert Killebrew", target: "Kelly K.", type: "leaf" },
                    { source: "Robert Killebrew", target: "Stephen K.", type: "leaf" },
                    { source: "Robert Killebrew", target: "Suzie K.", type: "leaf" },
                    { source: "Robert Killebrew", target: "Tony K.", type: "leaf" },
                    { source: "Robert Killebrew", target: "Keri K.", type: "leaf" },
                    { source: "Robert Killebrew", target: "Sheri K.", type: "leaf" },
                    { source: "Clinton Rasmussen", target: "Bonnie Rasmussen", type: "main" },
                    { source: "James A. R.", target: "Clinton Rasmussen", type: "main" },
                    { source: "Rasmus J. R.", target: "James A. R.", type: "main" },
                    { source: "Jens Rasmussen", target: "Rasmus J. R.", type: "main" },
                    { source: "Clinton Rasmussen", target: "Richard R.", type: "leaf" },
                    { source: "Clinton Rasmussen", target: "Bettie R.", type: "leaf" },
                    { source: "Clinton Rasmussen", target: "Rhett R.", type: "leaf" },
                    { source: "Richard R.", target: "Oranell", type: "inlaw" },
                    { source: "Richard R.", target: "James", type: "leaf" },
                    { source: "Richard R.", target: "Rosemary", type: "leaf" },
                    { source: "Richard R.", target: "Ruth", type: "leaf" },
                    { source: "Richard R.", target: "Karen", type: "leaf" },
                    { source: "Bettie R.", target: "Bob", type: "inlaw" },
                    { source: "Bettie R.", target: "Michelle", type: "leaf" },
                    { source: "Leonard V.", target: "John O. Vanderhoop", type: "main" },
                    { source: "Edwin DeVries V.", target: "Leonard V.", type: "main" },
                    { source: "William A. V.", target: "Edwin DeVries V.", type: "main" },
                    { source: "Beulah Salisbury", target: "Edwin DeVries V.", type: "main" },
                    { source: "John O. Vanderhoop", target: "Johnny Vanderhoop", type: "leaf" },
                    { source: "Marie Emilie Ibe", target: "Waltrud M. Lieber", type: "main"},
                    { source: "G. Heinrich L. Lieber", target: "Waltrud M. Lieber", type: "main"},
                    { source: "Marie Emilie Ibe", target: "Manfred Lieber", type: "leaf"}
                ]
            };

            // HYDRATED TIMELINE TREE
            // Siblings/Laterals have been nested under their branching ancestors to render correctly
            const treeData = {
                name: "Kyle Killebrew", year: 1990, branch: "M", steps: 0, lateral: 0, desc: "Present",
                children: [
                    {
                        name: "Eric Killebrew", year: 1961, branch: "K", steps: 1, lateral: 0, desc: "Father",
                        children: [
                            { name: "Robert Killebrew", year: 1930, branch: "K", steps: 2, lateral: 0, children: [
                                { name: "William H. K.", year: 1898, branch: "K", steps: 3, lateral: 0, children: [
                                    { name: "Daniel Boone K.", year: 1860, branch: "K", steps: 4, lateral: 0, children: [
                                        { name: "George W. K.", year: 1812, branch: "K", steps: 5, lateral: 0, children: [
                                            { name: "Whitfield K.", year: 1793, branch: "K", steps: 6, lateral: 0, children: [
                                                { name: "Joseph K.", year: 1753, branch: "K", steps: 7, lateral: 0, children: [
                                                    { name: "Francis K.", year: 1619, branch: "K", steps: 8, lateral: 0 }
                                                ]}
                                            ]}
                                        ]}
                                    ]},
                                    { name: "Ron K.", year: 1928, branch: "K", steps: 2, lateral: 1 },
                                    { name: "Urma K.", year: 1932, branch: "K", steps: 2, lateral: 1 }
                                ]},
                                { name: "Kelly K.", year: 1955, branch: "K", steps: 1, lateral: 1 },
                                { name: "Stephen K.", year: 1957, branch: "K", steps: 1, lateral: 1 },
                                { name: "Suzie K.", year: 1959, branch: "K", steps: 1, lateral: 1 },
                                { name: "Tony K.", year: 1963, branch: "K", steps: 1, lateral: 1 },
                                { name: "Keri K.", year: 1965, branch: "K", steps: 1, lateral: 1 },
                                { name: "Sheri K.", year: 1967, branch: "K", steps: 1, lateral: 1 }
                            ]},
                            { name: "Bonnie Rasmussen", year: 1934, branch: "R", steps: 2, lateral: 0, children: [
                                { name: "Clinton R.", year: 1904, branch: "R", steps: 3, lateral: 0, children: [
                                    { name: "James A. R.", year: 1877, branch: "R", steps: 4, lateral: 0, children: [
                                        { name: "Rasmus J. R.", year: 1842, branch: "R", steps: 5, lateral: 0, children: [
                                            { name: "Jens Rasmussen", year: 1810, branch: "R", steps: 6, lateral: 0 }
                                        ]}
                                    ]},
                                    { name: "Richard R.", year: 1932, branch: "R", steps: 2, lateral: 1, children: [
                                        { name: "Oranell", year: 1932, branch: "R", steps: 2, lateral: 1, inLaw: true },
                                        { name: "James", year: 1955, branch: "R", steps: 1, lateral: 2 },
                                        { name: "Rosemary", year: 1957, branch: "R", steps: 1, lateral: 2 },
                                        { name: "Ruth", year: 1959, branch: "R", steps: 1, lateral: 2 },
                                        { name: "Karen", year: 1961, branch: "R", steps: 1, lateral: 2 }
                                    ]},
                                    { name: "Bettie R.", year: 1936, branch: "R", steps: 2, lateral: 1, children: [
                                        { name: "Bob", year: 1936, branch: "R", steps: 2, lateral: 1, inLaw: true },
                                        { name: "Michelle", year: 1960, branch: "R", steps: 1, lateral: 2 }
                                    ]},
                                    { name: "Rhett R.", year: 1938, branch: "R", steps: 2, lateral: 1 }
                                ]}
                            ]}
                        ]
                    },
                    {
                        name: "Christina Vanderhoop", year: 1961, branch: "V", steps: 1, lateral: 0, desc: "Mother",
                        children: [
                            { name: "John O. Vanderhoop", year: 1934, branch: "V", steps: 2, lateral: 0, children: [
                                { name: "Leonard V.", year: 1895, branch: "V", steps: 3, lateral: 0, children: [
                                    { name: "Edwin DeVries V.", year: 1848, branch: "V", steps: 4, lateral: 0, children: [
                                        { name: "William A. V.", year: 1816, branch: "V", steps: 5, lateral: 0 },
                                        { name: "Beulah Salisbury", year: 1814, branch: "V", steps: 5, lateral: 0 }
                                    ]},
                                    { name: "Johnny Vanderhoop", year: 1936, branch: "V", steps: 1, lateral: 1 }
                                ]}
                            ]},
                            { name: "Waltrud M. Lieber", year: 1934, branch: "L", steps: 2, lateral: 0, children: [
                                { name: "Marie Emilie Ibe", year: 1900, branch: "L", steps: 3, lateral: 0, children: [
                                    { name: "Manfred Lieber", year: 1932, branch: "L", steps: 2, lateral: 1 }
                                ]},
                                { name: "G. Heinrich L. Lieber", year: 1900, branch: "L", steps: 3, lateral: 0 }
                            ]}
                        ]
                    }
                ]
            };

            // ==========================================
            // 2. PROCEDURAL ALGORITHMS (Color & Size)
            // ==========================================
            
            let maxSteps = { K: 0, V: 0, R: 0, L: 0 };
            let maxLateral = 0;
            
            graphData.nodes.forEach(d => {
                if (d.lateral === 0 && maxSteps[d.branch] !== undefined) {
                    maxSteps[d.branch] = Math.max(maxSteps[d.branch], d.steps);
                }
                if (d.lateral > maxLateral) maxLateral = d.lateral;
            });
            maxLateral = Math.max(3, maxLateral); 

            function calcRadius(d) {
                const ROOT_SIZE = 25;
                if (d.inLaw) return ROOT_SIZE * 0.05;

                if (d.lateral === 0) {
                    let sz = ROOT_SIZE * (1 - 0.05 * d.steps);
                    return Math.max(1, sz);
                } else {
                    let branchRootStep = d.steps + d.lateral;
                    let branchRootSize = ROOT_SIZE * (1 - 0.05 * branchRootStep);
                    let startSize = branchRootSize / 3;
                    let factor = 1 - 0.10 * (d.lateral - 1);
                    return Math.max(1, startSize * factor);
                }
            }

            function calcColor(d) {
                if (d.inLaw) return "rgb(0,0,0)";

                let mainSteps = d.steps + d.lateral; 
                let mMax = maxSteps[d.branch] || 1;
                
                let t = (mainSteps <= 1) ? 0 : (mainSteps - 1) / Math.max(1, mMax - 1);
                let r = 240, g = 0, b = 240; 

                if (d.branch === "K") {
                    r = Math.round(220 + (0 - 220) * t);
                    b = 240;
                } else if (d.branch === "V") {
                    r = 240;
                    b = Math.round(220 + (0 - 220) * t);
                } else if (d.branch === "R") {
                    r = Math.round(220 + (0 - 220) * t);
                    g = Math.round(0 + (240 - 0) * t);
                    b = Math.round(240 + (0 - 240) * t);
                } else if (d.branch === "L") {
                    r = 240;
                    g = Math.round(0 + (240 - 0) * t);
                    b = Math.round(220 + (0 - 220) * t);
                }

                if (d.lateral > 0) {
                    let wT = d.lateral / maxLateral; 
                    r = Math.round(r + (240 - r) * wT);
                    g = Math.round(g + (240 - g) * wT);
                    b = Math.round(b + (240 - b) * wT);
                }
                return `rgb(${r}, ${g}, ${b})`;
            }

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
                        if (d.type === "leaf" || d.type === "inlaw") return 20; 
                        return 80;
                    })
                    .strength(d => d.type === "marriage" ? 0.001 : 1) 
                )
                .force("charge", d3.forceManyBody().strength(d => d.inLaw ? -15 : -300))
                .force("collide", d3.forceCollide().radius(d => calcRadius(d) + 5).iterations(2))
                .force("x", d3.forceX())
                .force("y", d3.forceY());

            const fLink = fSvg.append("g").selectAll("line").data(graphData.links).join("line")
                .attr("class", d => `link-${d.type}`);
            
            const tooltip = d3.select("#tooltip");

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
                .on("mouseover", (e,d) => {
                    tooltip.transition().duration(200).style("opacity", 1);
                    let stat = d.lateral === 0 ? "Direct Ancestor" : "Lateral Relative";
                    if(d.inLaw) stat = "In-Law";
                    tooltip.html(`<strong>${d.id}</strong><br/>${d.desc || stat}<br/><span style="color:#94A3B8; font-size:10px;">Size: ${Math.round(calcRadius(d))}px | RGB: ${calcColor(d)}</span>`)
                        .style("left", (e.pageX + 15) + "px").style("top", (e.pageY - 28) + "px");
                })
                .on("mouseout", () => tooltip.transition().duration(500).style("opacity", 0));

            simulation.on("tick", () => {
                fLink.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
                     .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
                fNode.attr("transform", d => `translate(${d.x},${d.y})`);
            });

            // ==========================================
            // 4. RIGHT PANEL: TIMELINE
            // ==========================================
            const tSvg = d3.select("#viz-tree");
            const tWidth = tSvg.node().getBoundingClientRect().width;
            
            const yScale = d3.scaleLinear().domain([2010, 1600]).range([height - 60, 60]);
            const yearMarks = [1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000];
            
            tSvg.selectAll(".grid-line").data(yearMarks).enter().append("line")
                .attr("class", "grid-line").attr("x1", 30).attr("x2", tWidth - 10)
                .attr("y1", d => yScale(d)).attr("y2", d => yScale(d));
                
            tSvg.selectAll(".grid-label").data(yearMarks).enter().append("text")
                .attr("class", "grid-label").attr("x", 5).attr("y", d => yScale(d) + 3).text(d => d);

            const root = d3.hierarchy(treeData);
            d3.tree().size([tWidth - 60, height])(root);
            root.each(d => d.y = yScale(d.data.year));

            const treeGroup = tSvg.append("g").attr("transform", "translate(30,0)");
            
            // Apply the procedural Link class logic
            treeGroup.selectAll(".tree-link").data(root.links()).enter().append("path")
                .attr("class", d => {
                    if (d.target.data.inLaw) return "tree-link inlaw";
                    if (d.target.data.lateral > 0) return "tree-link leaf";
                    return "tree-link main";
                })
                .attr("d", d3.linkVertical().x(d => d.x).y(d => d.y));

            const tNode = treeGroup.selectAll(".tree-node").data(root.descendants()).enter().append("g")
                .attr("transform", d => `translate(${d.x},${d.y})`);

            // Apply procedural Node class logic
            tNode.append("circle")
                .attr("class", "node")
                .attr("r", d => calcRadius(d.data))
                .attr("fill", d => calcColor(d.data))
                .on("mouseover", (e,d) => {
                    tooltip.transition().duration(200).style("opacity", 1);
                    tooltip.html(`<strong>${d.data.name}</strong><br/>${d.data.year} | ${d.data.desc || ""}`)
                        .style("left", (e.pageX + 15) + "px").style("top", (e.pageY - 28) + "px");
                })
                .on("mouseout", () => tooltip.transition().duration(500).style("opacity", 0));

        </script>
    </body>
    </html>
    """
    components.html(html_code, height=720)