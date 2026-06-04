"""
=============================================================================
MODULE: pages/components/genealogy_web_demo.py
AUTHOR: Kyle W. Killebrew, PhD
DESCRIPTION: 
    Custom Streamlit component to render an interactive dual-view Genealogy Web. 
    Left Panel: Force-directed gravity cluster.
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
            .node:hover { stroke: #F8FAFC; stroke-width: 3px; }
            .link { fill: none; stroke: #475569; stroke-opacity: 0.5; stroke-width: 2px; }
            .tree-link { fill: none; stroke: #334155; stroke-width: 1.5px; }
            .label { fill: #F8FAFC; font-size: 10px; font-weight: 500; pointer-events: none; text-anchor: middle; }
            .grid-line { stroke: #1E293B; stroke-width: 1px; stroke-dasharray: 4 4; }
            .grid-label { fill: #64748B; font-size: 10px; font-family: monospace; }
        </style>
    </head>
    <body>
        <div id="tooltip"></div>
        <div class="container">
            <div class="panel">
                <div class="panel-title">Ancestral Cluster</div>
                <svg id="viz-force" width="100%" height="100%"></svg>
            </div>
            <div class="panel right-panel">
                <div class="panel-title">Chronological Pedigree</div>
                <svg id="viz-tree" width="100%" height="100%"></svg>
            </div>
        </div>

        <script>
            // ==========================================
            // 1. DATA ARCHITECTURE
            // ==========================================
            
            // Palette: Center(Purple), Killebrew(Blue), Vanderhoop(Red), Rasmussen(Green), Lieber(Yellow)
            const color = d3.scaleOrdinal().domain([0, 1, 2, 3, 4])
                .range(["#A855F7", "#38BDF8", "#EF4444", "#4ADE80", "#FACC15"]);

            const graphData = {
                nodes: [
                    { id: "Kyle Killebrew", group: 0, radius: 25, desc: "1990 - Present" },
                    { id: "Eric Killebrew", group: 1, radius: 18, desc: "1961 - Present" },
                    { id: "Christina Vanderhoop", group: 2, radius: 18, desc: "Mother" },
                    { id: "Robert Killebrew", group: 1, radius: 15, desc: "1930 - 2017" },
                    { id: "Bonnie Rasmussen", group: 3, radius: 15, desc: "1934 - 2020" },
                    { id: "John O. Vanderhoop", group: 2, radius: 15, desc: "1934 - 2022" },
                    { id: "Waltrud M. Lieber", group: 4, radius: 15, desc: "1934 - 2022" },
                    { id: "William H. K.", group: 1, radius: 12, desc: "1898 - 1970" },
                    { id: "Daniel Boone K.", group: 1, radius: 12, desc: "1860 - 1939" },
                    { id: "George W. K.", group: 1, radius: 12, desc: "1812 - 1871" },
                    { id: "Whitfield K.", group: 1, radius: 12, desc: "1793 - 1859" },
                    { id: "Joseph K.", group: 1, radius: 12, desc: "1753 - 1824" },
                    { id: "Francis K.", group: 1, radius: 14, desc: "1619 - 1673" },
                    { id: "Clinton Rasmussen", group: 3, radius: 12, desc: "1904 - 1979" },
                    { id: "James A. R.", group: 3, radius: 12, desc: "1877 - 1965" },
                    { id: "Rasmus J. R.", group: 3, radius: 12, desc: "1842 - 1920" },
                    { id: "Jens Rasmussen", group: 3, radius: 14, desc: "1810 - 1888" },
                    { id: "Leonard V.", group: 2, radius: 12, desc: "1895 - 1989" },
                    { id: "Edwin DeVries V.", group: 2, radius: 12, desc: "1848 - 1923" },
                    { id: "William A. V.", group: 2, radius: 14, desc: "~1816 - 1893" },
                    { id: "Beulah Salisbury", group: 2, radius: 14, desc: "1814 - 1892" },
                    { id: "Marie Emilie Ibe", group: 4, radius: 12, desc: "Unknown"},
                    { id: "G. Heinrich L. Lieber", group: 4, radius: 12, desc: "Unknown"}
                ],
                links: [
                    { source: "Kyle Killebrew", target: "Eric Killebrew" },
                    { source: "Kyle Killebrew", target: "Christina Vanderhoop" },
                    { source: "Eric Killebrew", target: "Robert Killebrew" },
                    { source: "Eric Killebrew", target: "Bonnie Rasmussen" },
                    { source: "Christina Vanderhoop", target: "John O. Vanderhoop" },
                    { source: "Christina Vanderhoop", target: "Waltrud M. Lieber" },
                    { source: "Robert Killebrew", target: "William H. K." },
                    { source: "William H. K.", target: "Daniel Boone K." },
                    { source: "Daniel Boone K.", target: "George W. K." },
                    { source: "George W. K.", target: "Whitfield K." },
                    { source: "Whitfield K.", target: "Joseph K." },
                    { source: "Joseph K.", target: "Francis K." },
                    { source: "Bonnie Rasmussen", target: "Clinton Rasmussen" },
                    { source: "Clinton Rasmussen", target: "James A. R." },
                    { source: "James A. R.", target: "Rasmus J. R." },
                    { source: "Rasmus J. R.", target: "Jens Rasmussen" },
                    { source: "John O. Vanderhoop", target: "Leonard V." },
                    { source: "Leonard V.", target: "Edwin DeVries V." },
                    { source: "Edwin DeVries V.", target: "William A. V." },
                    { source: "Edwin DeVries V.", target: "Beulah Salisbury" },
                    { source: "Waltrud M. Lieber", target: "Marie Emilie Ibe"},
                    { source: "Waltrud M. Lieber", target: "G. Heinrich L. Lieber"}
                ]
            };

            // Hierarchical Pedigree Data (Children = Ancestors moving upward)
            // Eric is placed first to lock the father's line to the left.
            const treeData = {
                name: "Kyle Killebrew", year: 1990, group: 0, desc: "Present",
                children: [
                    {
                        name: "Eric Killebrew", year: 1961, group: 1, desc: "Father",
                        children: [
                            { name: "Robert Killebrew", year: 1930, group: 1, desc: "Grandfather", children: [
                                { name: "William H. K.", year: 1898, group: 1, children: [
                                    { name: "Daniel Boone K.", year: 1860, group: 1, children: [
                                        { name: "George W. K.", year: 1812, group: 1, children: [
                                            { name: "Whitfield K.", year: 1793, group: 1, children: [
                                                { name: "Joseph K.", year: 1753, group: 1, children: [
                                                    { name: "Francis K.", year: 1619, group: 1 }
                                                ]}
                                            ]}
                                        ]}
                                    ]}
                                ]}
                            ]},
                            { name: "Bonnie Rasmussen", year: 1934, group: 3, desc: "Grandmother", children: [
                                { name: "Clinton R.", year: 1904, group: 3, children: [
                                    { name: "James A. R.", year: 1877, group: 3, children: [
                                        { name: "Rasmus J. R.", year: 1842, group: 3, children: [
                                            { name: "Jens Rasmussen", year: 1810, group: 3 }
                                        ]}
                                    ]}
                                ]}
                            ]}
                        ]
                    },
                    {
                        name: "Christina Vanderhoop", year: 1961, group: 2, desc: "Mother",
                        children: [
                            { name: "John O. Vanderhoop", year: 1934, group: 2, desc: "Grandfather", children: [
                                { name: "Leonard V.", year: 1895, group: 2, children: [
                                    { name: "Edwin DeVries V.", year: 1848, group: 2, children: [
                                        { name: "William A. V.", year: 1816, group: 2 },
                                        { name: "Beulah Salisbury", year: 1814, group: 2 }
                                    ]}
                                ]}
                            ]},
                            { name: "Gaby Lieber", year: 1934, group: 4, desc: "Grandmother", children: [
                                { name: "Marie Emilie Ibe", year: 1900, group: 4 },
                                { name: "G. Heinrich L. Lieber", year: 1900, group: 4 }
                            ]}
                        ]
                    }
                ]
            };

            // Tooltip Logic
            const tooltip = d3.select("#tooltip");
            function showTooltip(event, name, desc) {
                tooltip.transition().duration(200).style("opacity", 1);
                tooltip.html(`<strong>${name}</strong><br/>${desc || ""}`)
                    .style("left", (event.pageX + 15) + "px")
                    .style("top", (event.pageY - 28) + "px");
            }
            function hideTooltip() {
                tooltip.transition().duration(500).style("opacity", 0);
            }

            // ==========================================
            // 2. LEFT PANEL: FORCE WEB
            // ==========================================
            const fSvg = d3.select("#viz-force");
            const width = fSvg.node().getBoundingClientRect().width;
            const height = 700;

            fSvg.attr("viewBox", [-width / 2, -height / 2, width, height]);

            const simulation = d3.forceSimulation(graphData.nodes)
                .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(60))
                .force("charge", d3.forceManyBody().strength(-200))
                .force("collide", d3.forceCollide().radius(d => d.radius + 10).iterations(2))
                .force("x", d3.forceX())
                .force("y", d3.forceY());

            const fLink = fSvg.append("g").selectAll("line").data(graphData.links).join("line").attr("class", "link");
            
            const fNode = fSvg.append("g").selectAll("g").data(graphData.nodes).join("g")
                .call(d3.drag()
                    .on("start", (e,d) => { if(!e.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
                    .on("drag", (e,d) => { d.fx=e.x; d.fy=e.y; })
                    .on("end", (e,d) => { if(!e.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; })
                );

            fNode.append("circle")
                .attr("class", "node").attr("r", d => d.radius).attr("fill", d => color(d.group))
                .on("mouseover", (e,d) => showTooltip(e, d.id, d.desc))
                .on("mouseout", hideTooltip);

            fNode.append("text")
                .attr("class", "label").attr("dy", d => d.radius + 12).text(d => d.id);

            simulation.on("tick", () => {
                fLink.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
                     .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
                fNode.attr("transform", d => `translate(${d.x},${d.y})`);
            });

            // ==========================================
            // 3. RIGHT PANEL: PEDIGREE TIMELINE
            // ==========================================
            const tSvg = d3.select("#viz-tree");
            const tWidth = tSvg.node().getBoundingClientRect().width;
            
            // Map the Y-axis to Time (Bottom = 2010, Top = 1600)
            const yScale = d3.scaleLinear().domain([2010, 1600]).range([height - 60, 60]);
            
            // Draw Timeline Grid
            const yearMarks = [1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000];
            tSvg.selectAll(".grid-line").data(yearMarks).enter().append("line")
                .attr("class", "grid-line")
                .attr("x1", 30).attr("x2", tWidth - 10)
                .attr("y1", d => yScale(d)).attr("y2", d => yScale(d));
                
            tSvg.selectAll(".grid-label").data(yearMarks).enter().append("text")
                .attr("class", "grid-label")
                .attr("x", 5).attr("y", d => yScale(d) + 3).text(d => d);

            // Compute Tree Layout (for X-coordinates only)
            const root = d3.hierarchy(treeData);
            d3.tree().size([tWidth - 60, height])(root);

            // Override the Y-coordinates strictly based on birth year
            root.each(d => d.y = yScale(d.data.year));

            const treeGroup = tSvg.append("g").attr("transform", "translate(30,0)");

            // Draw curved links from parent to child
            treeGroup.selectAll(".tree-link").data(root.links()).enter().append("path")
                .attr("class", "tree-link")
                .attr("d", d3.linkVertical().x(d => d.x).y(d => d.y));

            // Draw Nodes
            const tNode = treeGroup.selectAll(".tree-node").data(root.descendants()).enter().append("g")
                .attr("transform", d => `translate(${d.x},${d.y})`);

            tNode.append("circle")
                .attr("class", "node").attr("r", 7).attr("fill", d => color(d.data.group))
                .on("mouseover", (e,d) => showTooltip(e, d.data.name, `${d.data.year} | ${d.data.desc || ""}`))
                .on("mouseout", hideTooltip);

            // Stagger text alternatively to prevent overlap
            tNode.append("text")
                .attr("class", "label")
                .attr("dy", d => (d.depth % 2 === 0) ? -12 : 18)
                .text(d => d.data.name);

        </script>
    </body>
    </html>
    """
    components.html(html_code, height=720)