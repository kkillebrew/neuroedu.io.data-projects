"""
=============================================================================
MODULE: pages/components/genealogy_web_demo.py
AUTHOR: Kyle W. Killebrew, PhD
DESCRIPTION: 
    Custom Streamlit component to render an interactive force-directed 
    Genealogy Web. Uses D3.js to handle node physics, dragging, and hover 
    events independent of the Streamlit Python backend.
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
            #tooltip {
                position: absolute; text-align: center; padding: 10px; font-size: 12px;
                background: rgba(30, 41, 59, 0.9); color: #F8FAFC; border: 1px solid #475569;
                border-radius: 6px; pointer-events: none; opacity: 0; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                transition: opacity 0.2s; max-width: 200px;
            }
            .node { cursor: pointer; stroke: #0F172A; stroke-width: 1.5px; }
            .link { fill: none; stroke: #475569; stroke-opacity: 0.6; }
            .label { fill: #F8FAFC; font-size: 10px; font-weight: bold; pointer-events: none; text-anchor: middle; }
        </style>
    </head>
    <body>
        <div id="tooltip"></div>
        <svg id="viz" width="100%" height="700"></svg>

        <script>
            // --- 1. GENEALOGICAL DATASET ---
            const graph = {
                nodes: [
                    // Root
                    { id: "Kyle Killebrew", group: 0, radius: 25, desc: "1990 - Present\\nLas Vegas, NV" },
                    
                    // Gen 2 (Parents)
                    { id: "Eric Killebrew", group: 1, radius: 18, desc: "1961 - Present" },
                    { id: "Christina Vanderhoop", group: 2, radius: 18, desc: "Mother" },
                    
                    // Gen 3 (Grandparents)
                    { id: "Robert Killebrew", group: 1, radius: 15, desc: "1930 - 2017" },
                    { id: "Bonnie Rasmussen", group: 3, radius: 15, desc: "1934 - 2020" },
                    { id: "John O. Vanderhoop", group: 2, radius: 15, desc: "1934 - 2022\\nMajor, USAF" },
                    { id: "Gaby Lieber", group: 4, radius: 15, desc: "1934 - 2022\\nGermany" },

                    // Gen 4+ (Killebrew Line)
                    { id: "William H. Killebrew", group: 1, radius: 12, desc: "1898 - 1970" },
                    { id: "Daniel Boone K.", group: 1, radius: 12, desc: "1860 - 1939" },
                    { id: "George W. Killebrew", group: 1, radius: 12, desc: "1812 - 1871" },
                    { id: "Whitfield Killebrew", group: 1, radius: 12, desc: "1793 - 1859" },
                    { id: "Joseph Killebrew", group: 1, radius: 12, desc: "1753 - 1824" },
                    { id: "Francis Killebrew", group: 1, radius: 14, desc: "1619 - 1673\\nCornwall, England" },

                    // Gen 4+ (Rasmussen Line)
                    { id: "Clinton Rasmussen", group: 3, radius: 12, desc: "1904 - 1979" },
                    { id: "James A. Rasmussen", group: 3, radius: 12, desc: "1877 - 1965" },
                    { id: "Rasmus J. Rasmusson", group: 3, radius: 12, desc: "1842 - 1920" },
                    { id: "Jens Rasmussen", group: 3, radius: 14, desc: "1810 - 1888\\nDenmark to Utah" },

                    // Gen 4+ (Vanderhoop Line)
                    { id: "Leonard Vanderhoop", group: 2, radius: 12, desc: "1895 - 1989" },
                    { id: "Edwin DeVries V.", group: 2, radius: 12, desc: "1848 - 1923" },
                    { id: "William A. Vanderhoop", group: 2, radius: 14, desc: "~1816 - 1893\\nSuriname Immigrant" },
                    { id: "Beulah Salisbury", group: 2, radius: 14, desc: "1814 - 1892\\nPrincess of Aquinnah" }
                ],
                links: [
                    { source: "Kyle Killebrew", target: "Eric Killebrew" },
                    { source: "Kyle Killebrew", target: "Christina Vanderhoop" },
                    
                    { source: "Eric Killebrew", target: "Robert Killebrew" },
                    { source: "Eric Killebrew", target: "Bonnie Rasmussen" },
                    
                    { source: "Christina Vanderhoop", target: "John O. Vanderhoop" },
                    { source: "Christina Vanderhoop", target: "Gaby Lieber" },

                    // Killebrew Path
                    { source: "Robert Killebrew", target: "William H. Killebrew" },
                    { source: "William H. Killebrew", target: "Daniel Boone K." },
                    { source: "Daniel Boone K.", target: "George W. Killebrew" },
                    { source: "George W. Killebrew", target: "Whitfield Killebrew" },
                    { source: "Whitfield Killebrew", target: "Joseph Killebrew" },
                    { source: "Joseph Killebrew", target: "Francis Killebrew" },

                    // Rasmussen Path
                    { source: "Bonnie Rasmussen", target: "Clinton Rasmussen" },
                    { source: "Clinton Rasmussen", target: "James A. Rasmussen" },
                    { source: "James A. Rasmussen", target: "Rasmus J. Rasmusson" },
                    { source: "Rasmus J. Rasmusson", target: "Jens Rasmussen" },

                    // Vanderhoop Path
                    { source: "John O. Vanderhoop", target: "Leonard Vanderhoop" },
                    { source: "Leonard Vanderhoop", target: "Edwin DeVries V." },
                    { source: "Edwin DeVries V.", target: "William A. Vanderhoop" },
                    { source: "Edwin DeVries V.", target: "Beulah Salisbury" }
                ]
            };

            // --- 2. D3 PHYSICS ENGINE SETUP ---
            const width = document.getElementById('viz').clientWidth;
            const height = 700;

            const svg = d3.select("#viz")
                .attr("viewBox", [-width / 2, -height / 2, width, height]);

            // Branch Colors: Center(Purple), Killebrew(Blue), Vanderhoop(Red), Rasmussen(Green), Lieber(Yellow)
            const color = d3.scaleOrdinal()
                .domain([0, 1, 2, 3, 4])
                .range(["#A855F7", "#38BDF8", "#EF4444", "#4ADE80", "#FACC15"]);

            const simulation = d3.forceSimulation(graph.nodes)
                .force("link", d3.forceLink(graph.links).id(d => d.id).distance(60))
                .force("charge", d3.forceManyBody().strength(-300))
                .force("collide", d3.forceCollide().radius(d => d.radius + 5).iterations(2))
                .force("x", d3.forceX())
                .force("y", d3.forceY());

            // --- 3. RENDERING ---
            const link = svg.append("g")
                .selectAll("line")
                .data(graph.links)
                .join("line")
                .attr("class", "link")
                .attr("stroke-width", 2);

            const nodeGroup = svg.append("g")
                .selectAll("g")
                .data(graph.nodes)
                .join("g")
                .call(drag(simulation));

            nodeGroup.append("circle")
                .attr("class", "node")
                .attr("r", d => d.radius)
                .attr("fill", d => color(d.group))
                .on("mouseover", showTooltip)
                .on("mouseout", hideTooltip);

            nodeGroup.append("text")
                .attr("class", "label")
                .attr("dy", d => d.radius + 12)
                .text(d => d.id);

            // --- 4. PHYSICS TICK UPDATES ---
            simulation.on("tick", () => {
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);

                nodeGroup.attr("transform", d => `translate(${d.x},${d.y})`);
            });

            // --- 5. INTERACTIVITY FUNCTIONS ---
            const tooltip = d3.select("#tooltip");

            function showTooltip(event, d) {
                tooltip.transition().duration(200).style("opacity", 1);
                tooltip.html(`<strong>${d.id}</strong><br/>${d.desc.replace("\\n", "<br/>")}`)
                    .style("left", (event.pageX + 15) + "px")
                    .style("top", (event.pageY - 28) + "px");
                d3.select(this).style("stroke", "#F8FAFC").style("stroke-width", "3px");
            }

            function hideTooltip(event, d) {
                tooltip.transition().duration(500).style("opacity", 0);
                d3.select(this).style("stroke", "#0F172A").style("stroke-width", "1.5px");
            }

            function drag(simulation) {
                function dragstarted(event) {
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    event.subject.fx = event.subject.x;
                    event.subject.fy = event.subject.y;
                }
                function dragged(event) {
                    event.subject.fx = event.x;
                    event.subject.fy = event.y;
                }
                function dragended(event) {
                    if (!event.active) simulation.alphaTarget(0);
                    event.subject.fx = null;
                    event.subject.fy = null;
                }
                return d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended);
            }
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=720)