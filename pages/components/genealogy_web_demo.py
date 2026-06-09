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
            // 1. PROCEDURAL DATA ARCHITECTURE
            // ==========================================
            
            const graphData = {
                nodes: [
                    // --- CORE / IMMEDIATE FAMILY ---
                    { id: "Kyle Killebrew", branch: "M", steps: 0, lateral: 0, desc: "You (Present)" },
                    { id: "Eric Killebrew", branch: "K", steps: 1, lateral: 0, desc: "Parent" },
                    { id: "Christina Vanderhoop", branch: "V", steps: 1, lateral: 0, desc: "Parent" },
                    
                    // FIX: Removed inLaw: true so she acts as a Primary Root Branch
                    { id: "Antonia Constance Buzunis", branch: "B", steps: 1, lateral: 0, desc: "Step Mother" },
                    
                    { id: "Andrea Nicole Killebrew", branch: "K", steps: 0, lateral: 1, desc: "Sister / Brother" },
                    { id: "Eric Scott Killebrew, Jr.", branch: "K", steps: 0, lateral: 1, desc: "Sister / Brother" },

                    // --- GRANDPARENTS ---
                    { id: "Robert Killebrew", branch: "K", steps: 2, lateral: 0, desc: "Grandparent (1930 - 2017)" },
                    { id: "Bonnie Rasmussen", branch: "R", steps: 2, lateral: 0, desc: "Grandparent (1934 - 2020)" },
                    { id: "John O. Vanderhoop", branch: "V", steps: 2, lateral: 0, desc: "Grandparent (1934 - 2022)" },
                    { id: "Waltrud M. Lieber", branch: "L", steps: 2, lateral: 0, desc: "Grandparent (1934 - 2022)" },
                    { id: "Peter Buzunis", branch: "B", steps: 2, lateral: 0, desc: "Grandparent (1917 - 2007)" },
                    { id: "Anastasia Ginakes", branch: "A", steps: 2, lateral: 0, desc: "Grandparent (1925 - 2018)" },
                    
                    // --- KILLEBREW BRANCH (K) ---
                    { id: "William H. K.", branch: "K", steps: 3, lateral: 0, desc: "Great Grandparent (1898 - 1970)" },
                    { id: "Mary Esther Robinson", branch: "K", steps: 3, inLaw: true, anchorStep: 3, desc: "Great Grandmother (1902-1981)" },
                    { id: "James Wesley Robinson", branch: "K", steps: 4, lateral: 1, isSpouseLine: true, anchorStep: 3, desc: "2x Great Grandparent (1847-1916)" },
                    { id: "Jane Jincey Impson", branch: "K", steps: 4, lateral: 1, isSpouseLine: true, anchorStep: 3, desc: "2x Great Grandparent (1862-1940)" },
                    { id: "Niel C. Robinson", branch: "K", steps: 5, lateral: 1, isSpouseLine: true, anchorStep: 3, desc: "3x Great Grandparent (d. 1864)" },
                    { id: "Huldah Jennie Wood", branch: "K", steps: 5, lateral: 1, isSpouseLine: true, anchorStep: 3, desc: "3x Great Grandparent (1830-1880)" },
                    { id: "Neal Clark Robeson", branch: "K", steps: 6, lateral: 1, isSpouseLine: true, anchorStep: 3, desc: "4x Great Grandparent (1785-1836)" },
                    { id: "Ileyvina Robinson", branch: "K", steps: 6, lateral: 1, isSpouseLine: true, anchorStep: 3, desc: "4x Great Grandparent (1788-1868)" },
                    { id: "Neal Clark Robeson Sr.", branch: "K", steps: 7, lateral: 1, isSpouseLine: true, anchorStep: 3, desc: "5x Great Grandparent (1760-1841)" },
                    { id: "Josiah Impson", branch: "K", steps: 5, lateral: 1, isSpouseLine: true, anchorStep: 3, desc: "3x Great Grandparent (1824-1896)" },
                    { id: "Isaac Impson", branch: "K", steps: 6, lateral: 1, isSpouseLine: true, anchorStep: 3, desc: "4x Great Grandparent (1800-1899)" },
                    { id: "John Adam Josiah Impson", branch: "K", steps: 7, lateral: 1, isSpouseLine: true, anchorStep: 3, desc: "5x Great Grandparent (1745-1833)" },
                    { id: "John Adam Impson", branch: "K", steps: 8, lateral: 1, isSpouseLine: true, anchorStep: 3, desc: "6x Great Grandparent (1718-?)" },
                    { id: "William John Impson", branch: "K", steps: 9, lateral: 1, isSpouseLine: true, anchorStep: 3, desc: "7x Great Grandparent (1700-?)" },

                    { id: "Daniel Boone K.", branch: "K", steps: 4, lateral: 0, desc: "Great Great Grandparent (1860 - 1939)" },
                    { id: "George W. K.", branch: "K", steps: 5, lateral: 0, desc: "3x Great Grandparent (1812 - 1871)" },
                    { id: "Whitfield K.", branch: "K", steps: 6, lateral: 0, desc: "4x Great Grandparent (1793 - 1859)" },
                    { id: "Joseph K.", branch: "K", steps: 7, lateral: 0, desc: "5x Great Grandparent (1753 - 1824)" },
                    { id: "Francis K.", branch: "K", steps: 8, lateral: 0, desc: "6x Great Grandparent (1619 - 1673)" },
                    
                    { id: "Ron K.", branch: "K", steps: 2, lateral: 1, desc: "Great Aunt / Uncle" },
                    { id: "Urma K.", branch: "K", steps: 2, lateral: 1, desc: "Great Aunt / Uncle" },
                    { id: "Kelly K.", branch: "K", steps: 1, lateral: 1, desc: "Aunt / Uncle" },
                    { id: "Stephen K.", branch: "K", steps: 1, lateral: 1, desc: "Aunt / Uncle" },
                    { id: "Iman Killebrew", branch: "K", steps: 1, lateral: 1, inLaw: true },
                    { id: "Alec K.", branch: "K", steps: 0, lateral: 2, desc: "First Cousin" },
                    { id: "Gillan K.", branch: "K", steps: 0, lateral: 2, desc: "First Cousin" },
                    { id: "Jayce K.", branch: "K", steps: 0, lateral: 2, desc: "First Cousin" },
                    { id: "Suzie K.", branch: "K", steps: 1, lateral: 1, desc: "Aunt / Uncle" },
                    { id: "Steve Gerphy", branch: "K", steps: 1, lateral: 1, inLaw: true },
                    { id: "Matt Gerphy", branch: "K", steps: 0, lateral: 2, desc: "First Cousin" },
                    { id: "Francesca", branch: "K", steps: 0, lateral: 2, inLaw: true },
                    { id: "Amelia Gerphy", branch: "K", steps: -1, lateral: 3, desc: "First Cousin Once Removed" },
                    { id: "Riley Gerphy", branch: "K", steps: -1, lateral: 3, desc: "First Cousin Once Removed" },
                    { id: "Oliver Gerphy", branch: "K", steps: -1, lateral: 3, desc: "First Cousin Once Removed" },
                    { id: "Chris Gerphy", branch: "K", steps: 0, lateral: 2, desc: "First Cousin" },
                    { id: "Tony K.", branch: "K", steps: 1, lateral: 1, desc: "Aunt / Uncle" },
                    { id: "Martha", branch: "K", steps: 1, lateral: 1, inLaw: true },
                    { id: "Will K.", branch: "K", steps: 0, lateral: 2, desc: "First Cousin" },
                    { id: "Ben K.", branch: "K", steps: 0, lateral: 2, desc: "First Cousin" },
                    { id: "Keri K.", branch: "K", steps: 1, lateral: 1, desc: "Aunt / Uncle" },
                    { id: "Stacy", branch: "K", steps: 1, lateral: 1, inLaw: true },
                    { id: "Sheri K.", branch: "K", steps: 1, lateral: 1, desc: "Aunt / Uncle" },
                    { id: "Ray Snisky", branch: "K", steps: 1, lateral: 1, inLaw: true },
                    { id: "Ellie Snisky", branch: "K", steps: 0, lateral: 2, desc: "First Cousin" },
                    { id: "Ava Snisky", branch: "K", steps: 0, lateral: 2, desc: "First Cousin" },

                    // --- RASMUSSEN BRANCH (R) ---
                    { id: "Clinton Rasmussen", branch: "R", steps: 3, lateral: 0, desc: "Great Grandparent (1904 - 1979)" },
                    { id: "James A. R.", branch: "R", steps: 4, lateral: 0, desc: "Great Great Grandparent (1877 - 1965)" },
                    { id: "Rasmus J. R.", branch: "R", steps: 5, lateral: 0, desc: "3x Great Grandparent (1842 - 1920)" },
                    { id: "Jens Rasmussen", branch: "R", steps: 6, lateral: 0, desc: "4x Great Grandparent (1810 - 1888)" },
                    { id: "Richard R.", branch: "R", steps: 2, lateral: 1, desc: "Great Aunt / Uncle" },
                    { id: "Bettie R.", branch: "R", steps: 2, lateral: 1, desc: "Great Aunt / Uncle" },
                    { id: "Rhett R.", branch: "R", steps: 2, lateral: 1, desc: "Great Aunt / Uncle" },
                    { id: "Oranell", branch: "R", steps: 2, lateral: 1, inLaw: true },
                    { id: "James", branch: "R", steps: 1, lateral: 2, desc: "First Cousin Once Removed" },
                    { id: "Rosemary", branch: "R", steps: 1, lateral: 2, desc: "First Cousin Once Removed" },
                    { id: "Ruth", branch: "R", steps: 1, lateral: 2, desc: "First Cousin Once Removed" },
                    { id: "Karen", branch: "R", steps: 1, lateral: 2, desc: "First Cousin Once Removed" },
                    { id: "Bob", branch: "R", steps: 2, lateral: 1, inLaw: true },
                    { id: "Michelle", branch: "R", steps: 1, lateral: 2, desc: "First Cousin Once Removed" },

                    // --- VANDERHOOP BRANCH (V) ---
                    { id: "Leonard V.", branch: "V", steps: 3, lateral: 0, desc: "Great Grandparent (1895 - 1989)" },
                    { id: "Edwin DeVries V.", branch: "V", steps: 4, lateral: 0, desc: "Great Great Grandparent (1848 - 1923)" },
                    { id: "William A. V.", branch: "V", steps: 5, lateral: 0, desc: "3x Great Grandparent (~1816 - 1893)" },
                    { id: "Beulah Salisbury", branch: "V", steps: 5, lateral: 0, inLaw: true },
                    { id: "Johnny Vanderhoop", branch: "V", steps: 1, lateral: 1, desc: "Aunt / Uncle" },
                    { id: "Leonard Jr. V.", branch: "V", steps: 2, lateral: 1, desc: "Great Aunt / Uncle" },
                    { id: "William (Billy) V.", branch: "V", steps: 2, lateral: 1, desc: "Great Aunt / Uncle" },
                    { id: "Edmund V.", branch: "V", steps: 2, lateral: 1, desc: "Great Aunt / Uncle" },
                    { id: "Margery V.", branch: "V", steps: 2, lateral: 1, desc: "Great Aunt / Uncle" },
                    { id: "Paul V.", branch: "V", steps: 1, lateral: 2, desc: "First Cousin Once Removed" },
                    { id: "Polly V.", branch: "V", steps: 1, lateral: 2, desc: "First Cousin Once Removed" },
                    { id: "Paul's Wife", branch: "V", steps: 1, lateral: 2, inLaw: true },
                    { id: "Maushup V.", branch: "V", steps: 0, lateral: 3, desc: "Second Cousin" },
                    { id: "Nashawn V.", branch: "V", steps: 0, lateral: 3, desc: "Second Cousin" },

                    // --- LIEBER BRANCH (L) ---
                    { id: "G. Heinrich L. Lieber", branch: "L", steps: 3, lateral: 0, desc: "Great Grandparent"},
                    { id: "Marie Emilie Ibe", branch: "L", steps: 3, lateral: 0, inLaw: true },
                    { id: "Manfred Lieber", branch: "L", steps: 2, lateral: 1, desc: "Great Aunt / Uncle" },

                    // --- BUZUNIS BRANCH (B) ---
                    { id: "George Constantine Buzunis", branch: "B", steps: 4, lateral: 0, desc: "Great Great Grandparent (1846 - 1912)" },
                    { id: "Elenis 'Helen' Georgakopoulis", branch: "B", steps: 4, lateral: 0, desc: "Great Great Grandparent (1867 - 1912)", inLaw: true },
                    { id: "Theodore Buzunis", branch: "B", steps: 3, lateral: 0, desc: "Great Grandparent (1885 - 1978)" },
                    { id: "Constantina Colilos", branch: "B", steps: 3, lateral: 0, desc: "Great Grandparent (1887 - 1963)", inLaw: true },
                    
                    { id: "William Buzunis", branch: "B", steps: 3, lateral: 1, desc: "Great Grand Aunt / Uncle" },
                    { id: "Alexander Buzunis", branch: "B", steps: 3, lateral: 1, desc: "Great Grand Aunt / Uncle" },
                    { id: "Nickolas Buzunis", branch: "B", steps: 3, lateral: 1, desc: "Great Grand Aunt / Uncle" },
                    { id: "Christopher Buzunis", branch: "B", steps: 3, lateral: 1, desc: "Great Grand Aunt / Uncle" },
                    
                    { id: "Helen Buzunis", branch: "B", steps: 2, lateral: 1, desc: "Great Aunt / Uncle (1911 - 1971)" },
                    { id: "James Effos", branch: "B", steps: 2, lateral: 1, inLaw: true },
                    { id: "George Buzunis", branch: "B", steps: 2, lateral: 1, desc: "Great Aunt / Uncle (1914 - 1988)" },
                    { id: "Christina Buzunis", branch: "B", steps: 2, lateral: 1, desc: "Great Aunt / Uncle (1915 - 2005)" },

                    { id: "Teddy Buzunis", branch: "B", steps: 1, lateral: 1, desc: "Aunt / Uncle" },
                    { id: "Victoria Buzunis", branch: "B", steps: 0, lateral: 2, desc: "First Cousin" },
                    { id: "Demo Buzunis", branch: "B", steps: 1, lateral: 1, desc: "Aunt / Uncle" },
                    { id: "Andrea Buzunis", branch: "B", steps: 1, lateral: 1, desc: "Aunt / Uncle" },
                    { id: "Timothy Buzunis", branch: "B", steps: 1, lateral: 1, desc: "Aunt / Uncle" },
                    { id: "Kerry", branch: "B", steps: 1, lateral: 1, inLaw: true },

                    // --- GINAKES BRANCH (A) ---
                    { id: "Unknown Ginakes", branch: "A", steps: 4, lateral: 0, desc: "Great Great Grandparent" },
                    { id: "Andrew Demetrius Ginakes", branch: "A", steps: 3, lateral: 0, desc: "Great Grandparent (1895 - 1967)" },
                    { id: "Arcondo A. Boosalis", branch: "A", steps: 3, lateral: 0, desc: "Great Grandparent (1895 - 1984)", inLaw: true },
                    
                    { id: "Constantine D Ginakes", branch: "A", steps: 3, lateral: 1, desc: "Great Grand Aunt / Uncle (1908 - 1965)" },
                    { id: "Anastasia Mirras", branch: "A", steps: 3, lateral: 1, inLaw: true },
                    { id: "Nicholas Ginakes", branch: "A", steps: 3, lateral: 1, desc: "Great Grand Aunt / Uncle" },
                    { id: "John Ginakes", branch: "A", steps: 3, lateral: 1, desc: "Great Grand Aunt / Uncle" },

                    { id: "Desmos Andrew Ginakes", branch: "A", steps: 2, lateral: 1, desc: "Great Aunt / Uncle (1923 - 1996)" },
                    { id: "Mary Ginakes", branch: "A", steps: 2, lateral: 1, desc: "Great Aunt / Uncle (1928 - 2013)" }
                ],
                links: [
                    // Main Ancestral Lines
                    { source: "Eric Killebrew", target: "Kyle Killebrew", type: "main" },
                    { source: "Christina Vanderhoop", target: "Kyle Killebrew", type: "main" },
                    { source: "Robert Killebrew", target: "Eric Killebrew", type: "main" },
                    { source: "Bonnie Rasmussen", target: "Eric Killebrew", type: "main" },
                    { source: "John O. Vanderhoop", target: "Christina Vanderhoop", type: "main" },
                    { source: "Waltrud M. Lieber", target: "Christina Vanderhoop", type: "main" },
                    
                    { source: "Peter Buzunis", target: "Antonia Constance Buzunis", type: "main" },
                    { source: "Anastasia Ginakes", target: "Antonia Constance Buzunis", type: "main" },
                    { source: "Theodore Buzunis", target: "Peter Buzunis", type: "main" },
                    { source: "George Constantine Buzunis", target: "Theodore Buzunis", type: "main" },
                    
                    { source: "Andrew Demetrius Ginakes", target: "Anastasia Ginakes", type: "main" },
                    { source: "Unknown Ginakes", target: "Andrew Demetrius Ginakes", type: "main" },

                    // Marriages & Step-Relations
                    { source: "Eric Killebrew", target: "Christina Vanderhoop", type: "marriage" },
                    { source: "Eric Killebrew", target: "Antonia Constance Buzunis", type: "marriage" },
                    { source: "Robert Killebrew", target: "Bonnie Rasmussen", type: "marriage" },
                    { source: "John O. Vanderhoop", target: "Waltrud M. Lieber", type: "marriage" },
                    
                    { source: "Peter Buzunis", target: "Anastasia Ginakes", type: "marriage" },
                    { source: "Theodore Buzunis", target: "Constantina Colilos", type: "marriage" },
                    { source: "George Constantine Buzunis", target: "Elenis 'Helen' Georgakopoulis", type: "marriage" },
                    { source: "Andrew Demetrius Ginakes", target: "Arcondo A. Boosalis", type: "marriage" },
                    { source: "Constantine D Ginakes", target: "Anastasia Mirras", type: "marriage" },
                    
                    { source: "Antonia Constance Buzunis", target: "Kyle Killebrew", type: "inlaw" },
                    
                    // Siblings & Relatives (Killebrew/Rasmussen/Vanderhoop/Lieber)
                    { source: "Eric Killebrew", target: "Andrea Nicole Killebrew", type: "leaf" },
                    { source: "Eric Killebrew", target: "Eric Scott Killebrew, Jr.", type: "leaf" },
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
                    { source: "Stephen K.", target: "Iman Killebrew", type: "inlaw" },
                    { source: "Stephen K.", target: "Alec K.", type: "leaf" },
                    { source: "Stephen K.", target: "Gillan K.", type: "leaf" },
                    { source: "Stephen K.", target: "Jayce K.", type: "leaf" },
                    { source: "Robert Killebrew", target: "Suzie K.", type: "leaf" },
                    { source: "Suzie K.", target: "Steve Gerphy", type: "inlaw" },
                    { source: "Suzie K.", target: "Matt Gerphy", type: "leaf" },
                    { source: "Suzie K.", target: "Chris Gerphy", type: "leaf" },
                    { source: "Matt Gerphy", target: "Francesca", type: "inlaw" },
                    { source: "Matt Gerphy", target: "Amelia Gerphy", type: "leaf" },
                    { source: "Matt Gerphy", target: "Riley Gerphy", type: "leaf" },
                    { source: "Matt Gerphy", target: "Oliver Gerphy", type: "leaf" },
                    { source: "Robert Killebrew", target: "Tony K.", type: "leaf" },
                    { source: "Tony K.", target: "Martha", type: "inlaw" },
                    { source: "Tony K.", target: "Will K.", type: "leaf" },
                    { source: "Tony K.", target: "Ben K.", type: "leaf" },
                    { source: "Robert Killebrew", target: "Keri K.", type: "leaf" },
                    { source: "Keri K.", target: "Stacy", type: "inlaw" },
                    { source: "Robert Killebrew", target: "Sheri K.", type: "leaf" },
                    { source: "Sheri K.", target: "Ray Snisky", type: "inlaw" },
                    { source: "Sheri K.", target: "Ellie Snisky", type: "leaf" },
                    { source: "Sheri K.", target: "Ava Snisky", type: "leaf" },
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
                    { source: "William A. V.", target: "Beulah Salisbury", type: "inlaw" },
                    { source: "John O. Vanderhoop", target: "Johnny Vanderhoop", type: "leaf" },
                    { source: "Leonard V.", target: "Leonard Jr. V.", type: "leaf" },
                    { source: "Leonard V.", target: "William (Billy) V.", type: "leaf" },
                    { source: "Leonard V.", target: "Edmund V.", type: "leaf" },
                    { source: "Leonard V.", target: "Margery V.", type: "leaf" },
                    { source: "Margery V.", target: "Paul V.", type: "leaf" },
                    { source: "Margery V.", target: "Polly V.", type: "leaf" },
                    { source: "Paul V.", target: "Paul's Wife", type: "inlaw" },
                    { source: "Paul V.", target: "Maushup V.", type: "leaf" },
                    { source: "Paul V.", target: "Nashawn V.", type: "leaf" },
                    { source: "G. Heinrich L. Lieber", target: "Waltrud M. Lieber", type: "main"},
                    { source: "G. Heinrich L. Lieber", target: "Marie Emilie Ibe", type: "inlaw" },
                    { source: "G. Heinrich L. Lieber", target: "Manfred Lieber", type: "leaf"},
                    { source: "William H. K.", target: "Mary Esther Robinson", type: "marriage" },
                    { source: "Mary Esther Robinson", target: "James Wesley Robinson", type: "spouse_main" },
                    { source: "Mary Esther Robinson", target: "Jane Jincey Impson", type: "spouse_main" },
                    { source: "James Wesley Robinson", target: "Niel C. Robinson", type: "spouse_main" },
                    { source: "James Wesley Robinson", target: "Huldah Jennie Wood", type: "spouse_main" },
                    { source: "Niel C. Robinson", target: "Neal Clark Robeson", type: "spouse_main" },
                    { source: "Niel C. Robinson", target: "Ileyvina Robinson", type: "spouse_main" },
                    { source: "Neal Clark Robeson", target: "Neal Clark Robeson Sr.", type: "spouse_main" },
                    { source: "Jane Jincey Impson", target: "Josiah Impson", type: "spouse_main" },
                    { source: "Josiah Impson", target: "Isaac Impson", type: "spouse_main" },
                    { source: "Isaac Impson", target: "John Adam Josiah Impson", type: "spouse_main" },
                    { source: "John Adam Josiah Impson", target: "John Adam Impson", type: "spouse_main" },
                    { source: "John Adam Impson", target: "William John Impson", type: "spouse_main" },

                    // Siblings & Relatives (Buzunis / Ginakes)
                    { source: "George Constantine Buzunis", target: "William Buzunis", type: "leaf" },
                    { source: "George Constantine Buzunis", target: "Alexander Buzunis", type: "leaf" },
                    { source: "George Constantine Buzunis", target: "Nickolas Buzunis", type: "leaf" },
                    { source: "George Constantine Buzunis", target: "Christopher Buzunis", type: "leaf" },
                    
                    { source: "Theodore Buzunis", target: "Helen Buzunis", type: "leaf" },
                    { source: "Theodore Buzunis", target: "George Buzunis", type: "leaf" },
                    { source: "Theodore Buzunis", target: "Christina Buzunis", type: "leaf" },
                    { source: "Helen Buzunis", target: "James Effos", type: "inlaw" },
                    
                    { source: "Peter Buzunis", target: "Teddy Buzunis", type: "leaf" },
                    { source: "Peter Buzunis", target: "Demo Buzunis", type: "leaf" },
                    { source: "Peter Buzunis", target: "Andrea Buzunis", type: "leaf" },
                    { source: "Peter Buzunis", target: "Timothy Buzunis", type: "leaf" },
                    { source: "Teddy Buzunis", target: "Victoria Buzunis", type: "leaf" },
                    { source: "Timothy Buzunis", target: "Kerry", type: "inlaw" },

                    { source: "Unknown Ginakes", target: "Constantine D Ginakes", type: "leaf" },
                    { source: "Unknown Ginakes", target: "Nicholas Ginakes", type: "leaf" },
                    { source: "Unknown Ginakes", target: "John Ginakes", type: "leaf" },
                    
                    { source: "Andrew Demetrius Ginakes", target: "Desmos Andrew Ginakes", type: "leaf" },
                    { source: "Andrew Demetrius Ginakes", target: "Mary Ginakes", type: "leaf" }
                ]
            };

            const treeData = {
                name: "Kyle Killebrew", year: 1990, branch: "M", steps: 0, lateral: 0, desc: "You (Present)",
                children: [
                    {
                        name: "Eric Killebrew", year: 1961, branch: "K", steps: 1, lateral: 0, desc: "Parent",
                        children: [
                            { name: "Robert Killebrew", year: 1930, branch: "K", steps: 2, lateral: 0, desc: "Grandparent", children: [
                                { name: "William H. K.", year: 1898, branch: "K", steps: 3, lateral: 0, desc: "Great Grandparent", children: [
                                    { name: "Mary Esther Robinson", year: 1902, branch: "K", steps: 3, inLaw: true, anchorStep: 3, desc: "Great Grandmother", children: [
                                        { name: "James Wesley Robinson", year: 1847, branch: "K", steps: 4, lateral: 1, isSpouseLine: true, anchorStep: 3, desc: "2x Great Grandparent", children: [
                                            { name: "Niel C. Robinson", year: 1820, branch: "K", steps: 5, lateral: 1, isSpouseLine: true, anchorStep: 3, desc: "3x Great Grandparent", children: [
                                                { name: "Neal Clark Robeson", year: 1785, branch: "K", steps: 6, lateral: 1, isSpouseLine: true, anchorStep: 3, desc: "4x Great Grandparent", children: [
                                                    { name: "Neal Clark Robeson Sr.", year: 1760, branch: "K", steps: 7, lateral: 1, isSpouseLine: true, anchorStep: 3, desc: "5x Great Grandparent" }
                                                ]},
                                                { name: "Ileyvina Robinson", year: 1788, branch: "K", steps: 6, lateral: 1, isSpouseLine: true, anchorStep: 3, desc: "4x Great Grandparent" }
                                            ]},
                                            { name: "Huldah Jennie Wood", year: 1830, branch: "K", steps: 5, lateral: 1, isSpouseLine: true, anchorStep: 3, desc: "3x Great Grandparent" }
                                        ]},
                                        { name: "Jane Jincey Impson", year: 1862, branch: "K", steps: 4, lateral: 1, isSpouseLine: true, anchorStep: 3, desc: "2x Great Grandparent", children: [
                                            { name: "Josiah Impson", year: 1824, branch: "K", steps: 5, lateral: 1, isSpouseLine: true, anchorStep: 3, desc: "3x Great Grandparent", children: [
                                                { name: "Isaac Impson", year: 1800, branch: "K", steps: 6, lateral: 1, isSpouseLine: true, anchorStep: 3, desc: "4x Great Grandparent", children: [
                                                    { name: "John Adam Josiah Impson", year: 1745, branch: "K", steps: 7, lateral: 1, isSpouseLine: true, anchorStep: 3, desc: "5x Great Grandparent", children: [
                                                        { name: "John Adam Impson", year: 1718, branch: "K", steps: 8, lateral: 1, isSpouseLine: true, anchorStep: 3, desc: "6x Great Grandparent", children: [
                                                            { name: "William John Impson", year: 1700, branch: "K", steps: 9, lateral: 1, isSpouseLine: true, anchorStep: 3, desc: "7x Great Grandparent" }
                                                        ]}
                                                    ]}
                                                ]}
                                            ]}
                                        ]}
                                    ]},
                                    { name: "Daniel Boone K.", year: 1860, branch: "K", steps: 4, lateral: 0, desc: "Great Great Grandparent", children: [
                                        { name: "George W. K.", year: 1812, branch: "K", steps: 5, lateral: 0, desc: "3x Great Grandparent", children: [
                                            { name: "Whitfield K.", year: 1793, branch: "K", steps: 6, lateral: 0, desc: "4x Great Grandparent", children: [
                                                { name: "Joseph K.", year: 1753, branch: "K", steps: 7, lateral: 0, desc: "5x Great Grandparent", children: [
                                                    { name: "Francis K.", year: 1619, branch: "K", steps: 8, lateral: 0, desc: "6x Great Grandparent" }
                                                ]}
                                            ]}
                                        ]}
                                    ]},
                                    { name: "Ron K.", year: 1928, branch: "K", steps: 2, lateral: 1, desc: "Great Aunt / Uncle" },
                                    { name: "Urma K.", year: 1932, branch: "K", steps: 2, lateral: 1, desc: "Great Aunt / Uncle" }
                                ]},
                                { name: "Kelly K.", year: 1955, branch: "K", steps: 1, lateral: 1, desc: "Aunt / Uncle" },
                                { name: "Stephen K.", year: 1957, branch: "K", steps: 1, lateral: 1, desc: "Aunt / Uncle", children: [
                                    { name: "Iman Killebrew", year: 1957, branch: "K", steps: 1, lateral: 1, inLaw: true },
                                    { name: "Alec K.", year: 1985, branch: "K", steps: 0, lateral: 2, desc: "First Cousin" },
                                    { name: "Gillan K.", year: 1987, branch: "K", steps: 0, lateral: 2, desc: "First Cousin" },
                                    { name: "Jayce K.", year: 1989, branch: "K", steps: 0, lateral: 2, desc: "First Cousin" }
                                ]},
                                { name: "Suzie K.", year: 1959, branch: "K", steps: 1, lateral: 1, desc: "Aunt / Uncle", children: [
                                    { name: "Steve Gerphy", year: 1959, branch: "K", steps: 1, lateral: 1, inLaw: true },
                                    { name: "Matt Gerphy", year: 1985, branch: "K", steps: 0, lateral: 2, desc: "First Cousin", children: [
                                        { name: "Francesca", year: 1985, branch: "K", steps: 0, lateral: 2, inLaw: true },
                                        { name: "Amelia Gerphy", year: 2010, branch: "K", steps: -1, lateral: 3, desc: "First Cousin Once Removed" },
                                        { name: "Riley Gerphy", year: 2012, branch: "K", steps: -1, lateral: 3, desc: "First Cousin Once Removed" },
                                        { name: "Oliver Gerphy", year: 2014, branch: "K", steps: -1, lateral: 3, desc: "First Cousin Once Removed" }
                                    ]},
                                    { name: "Chris Gerphy", year: 1988, branch: "K", steps: 0, lateral: 2, desc: "First Cousin" }
                                ]},
                                { name: "Tony K.", year: 1963, branch: "K", steps: 1, lateral: 1, desc: "Aunt / Uncle", children: [
                                    { name: "Martha", year: 1963, branch: "K", steps: 1, lateral: 1, inLaw: true },
                                    { name: "Will K.", year: 1990, branch: "K", steps: 0, lateral: 2, desc: "First Cousin" },
                                    { name: "Ben K.", year: 1992, branch: "K", steps: 0, lateral: 2, desc: "First Cousin" }
                                ]},
                                { name: "Keri K.", year: 1965, branch: "K", steps: 1, lateral: 1, desc: "Aunt / Uncle", children: [
                                    { name: "Stacy", year: 1965, branch: "K", steps: 1, lateral: 1, inLaw: true }
                                ]},
                                { name: "Sheri K.", year: 1967, branch: "K", steps: 1, lateral: 1, desc: "Aunt / Uncle", children: [
                                    { name: "Ray Snisky", year: 1965, branch: "K", steps: 1, lateral: 1, inLaw: true },
                                    { name: "Ellie Snisky", year: 1995, branch: "K", steps: 0, lateral: 2, desc: "First Cousin" },
                                    { name: "Ava Snisky", year: 1998, branch: "K", steps: 0, lateral: 2, desc: "First Cousin" }
                                ]}
                            ]},
                            { name: "Bonnie Rasmussen", year: 1934, branch: "R", steps: 2, lateral: 0, desc: "Grandparent", children: [
                                { name: "Clinton R.", year: 1904, branch: "R", steps: 3, lateral: 0, desc: "Great Grandparent", children: [
                                    { name: "James A. R.", year: 1877, branch: "R", steps: 4, lateral: 0, desc: "Great Great Grandparent", children: [
                                        { name: "Rasmus J. R.", year: 1842, branch: "R", steps: 5, lateral: 0, desc: "3x Great Grandparent", children: [
                                            { name: "Jens Rasmussen", year: 1810, branch: "R", steps: 6, lateral: 0, desc: "4x Great Grandparent" }
                                        ]}
                                    ]},
                                    { name: "Richard R.", year: 1932, branch: "R", steps: 2, lateral: 1, desc: "Great Aunt / Uncle", children: [
                                        { name: "Oranell", year: 1932, branch: "R", steps: 2, lateral: 1, inLaw: true },
                                        { name: "James", year: 1955, branch: "R", steps: 1, lateral: 2, desc: "First Cousin Once Removed" },
                                        { name: "Rosemary", year: 1957, branch: "R", steps: 1, lateral: 2, desc: "First Cousin Once Removed" },
                                        { name: "Ruth", year: 1959, branch: "R", steps: 1, lateral: 2, desc: "First Cousin Once Removed" },
                                        { name: "Karen", year: 1961, branch: "R", steps: 1, lateral: 2, desc: "First Cousin Once Removed" }
                                    ]},
                                    { name: "Bettie R.", year: 1936, branch: "R", steps: 2, lateral: 1, desc: "Great Aunt / Uncle", children: [
                                        { name: "Bob", year: 1936, branch: "R", steps: 2, lateral: 1, inLaw: true },
                                        { name: "Michelle", year: 1960, branch: "R", steps: 1, lateral: 2, desc: "First Cousin Once Removed" }
                                    ]},
                                    { name: "Rhett R.", year: 1938, branch: "R", steps: 2, lateral: 1, desc: "Great Aunt / Uncle" }
                                ]}
                            ]},
                            
                            // FIX: Removed inLaw: true so she acts as a Primary Root Branch
                            { name: "Antonia Constance Buzunis", year: 1964, branch: "B", steps: 1, lateral: 0, desc: "Step Mother", children: [
                                { name: "Peter Buzunis", year: 1917, branch: "B", steps: 2, lateral: 0, desc: "Grandparent", children: [
                                    { name: "Theodore Buzunis", year: 1885, branch: "B", steps: 3, lateral: 0, desc: "Great Grandparent", children: [
                                        { name: "George Constantine Buzunis", year: 1846, branch: "B", steps: 4, lateral: 0, desc: "Great Great Grandparent", children: [
                                            { name: "Elenis 'Helen' Georgakopoulis", year: 1867, branch: "B", steps: 4, lateral: 0, inLaw: true },
                                            { name: "William Buzunis", year: 1880, branch: "B", steps: 3, lateral: 1, desc: "Great Grand Aunt / Uncle" },
                                            { name: "Alexander Buzunis", year: 1882, branch: "B", steps: 3, lateral: 1, desc: "Great Grand Aunt / Uncle" },
                                            { name: "Nickolas Buzunis", year: 1884, branch: "B", steps: 3, lateral: 1, desc: "Great Grand Aunt / Uncle" },
                                            { name: "Christopher Buzunis", year: 1886, branch: "B", steps: 3, lateral: 1, desc: "Great Grand Aunt / Uncle" }
                                        ]},
                                        { name: "Constantina Colilos", year: 1887, branch: "B", steps: 3, lateral: 0, inLaw: true },
                                        { name: "Helen Buzunis", year: 1911, branch: "B", steps: 2, lateral: 1, desc: "Great Aunt / Uncle", children: [
                                            { name: "James Effos", year: 1894, branch: "B", steps: 2, lateral: 1, inLaw: true }
                                        ]},
                                        { name: "George Buzunis", year: 1914, branch: "B", steps: 2, lateral: 1, desc: "Great Aunt / Uncle" },
                                        { name: "Christina Buzunis", year: 1915, branch: "B", steps: 2, lateral: 1, desc: "Great Aunt / Uncle" }
                                    ]},
                                    { name: "Teddy Buzunis", year: 1960, branch: "B", steps: 1, lateral: 1, desc: "Aunt / Uncle", children: [
                                        { name: "Victoria Buzunis", year: 1990, branch: "B", steps: 0, lateral: 2, desc: "First Cousin" }
                                    ]},
                                    { name: "Demo Buzunis", year: 1962, branch: "B", steps: 1, lateral: 1, desc: "Aunt / Uncle" },
                                    { name: "Andrea Buzunis", year: 1968, branch: "B", steps: 1, lateral: 1, desc: "Aunt / Uncle" },
                                    { name: "Timothy Buzunis", year: 1970, branch: "B", steps: 1, lateral: 1, desc: "Aunt / Uncle", children: [
                                        { name: "Kerry", year: 1970, branch: "B", steps: 1, lateral: 1, inLaw: true }
                                    ]}
                                ]},
                                { name: "Anastasia Ginakes", year: 1925, branch: "A", steps: 2, lateral: 0, desc: "Grandparent", children: [
                                    { name: "Andrew Demetrius Ginakes", year: 1895, branch: "A", steps: 3, lateral: 0, desc: "Great Grandparent", children: [
                                        { name: "Unknown Ginakes", year: 1865, branch: "A", steps: 4, lateral: 0, desc: "Great Great Grandparent", children: [
                                            { name: "Constantine D Ginakes", year: 1908, branch: "A", steps: 3, lateral: 1, desc: "Great Grand Aunt / Uncle", children: [
                                                { name: "Anastasia Mirras", year: 1910, branch: "A", steps: 3, lateral: 1, inLaw: true }
                                            ]},
                                            { name: "Nicholas Ginakes", year: 1890, branch: "A", steps: 3, lateral: 1, desc: "Great Grand Aunt / Uncle" },
                                            { name: "John Ginakes", year: 1892, branch: "A", steps: 3, lateral: 1, desc: "Great Grand Aunt / Uncle" }
                                        ]},
                                        { name: "Arcondo A. Boosalis", year: 1895, branch: "A", steps: 3, lateral: 0, inLaw: true },
                                        { name: "Desmos Andrew Ginakes", year: 1923, branch: "A", steps: 2, lateral: 1, desc: "Great Aunt / Uncle" },
                                        { name: "Mary Ginakes", year: 1928, branch: "A", steps: 2, lateral: 1, desc: "Great Aunt / Uncle" }
                                    ]}
                                ]}
                            ]},
                            { name: "Andrea Nicole Killebrew", year: 1985, branch: "K", steps: 0, lateral: 1, desc: "Sister / Brother" },
                            { name: "Eric Scott Killebrew, Jr.", year: 1988, branch: "K", steps: 0, lateral: 1, desc: "Sister / Brother" }
                        ]
                    },
                    {
                        name: "Christina Vanderhoop", year: 1961, branch: "V", steps: 1, lateral: 0, desc: "Mother",
                        children: [
                            { name: "John O. Vanderhoop", year: 1934, branch: "V", steps: 2, lateral: 0, desc: "Grandparent", children: [
                                { name: "Leonard V.", year: 1895, branch: "V", steps: 3, lateral: 0, desc: "Great Grandparent", children: [
                                    { name: "Edwin DeVries V.", year: 1848, branch: "V", steps: 4, lateral: 0, desc: "Great Great Grandparent", children: [
                                        { name: "William A. V.", year: 1816, branch: "V", steps: 5, lateral: 0, desc: "3x Great Grandparent", children: [
                                            { name: "Beulah Salisbury", year: 1814, branch: "V", steps: 5, lateral: 0, inLaw: true }
                                        ]}
                                    ]},
                                    { name: "Leonard Jr. V.", year: 1925, branch: "V", steps: 2, lateral: 1, desc: "Great Aunt / Uncle" },
                                    { name: "William (Billy) V.", year: 1928, branch: "V", steps: 2, lateral: 1, desc: "Great Aunt / Uncle" },
                                    { name: "Edmund V.", year: 1930, branch: "V", steps: 2, lateral: 1, desc: "Great Aunt / Uncle" },
                                    { name: "Margery V.", year: 1932, branch: "V", steps: 2, lateral: 1, desc: "Great Aunt / Uncle", children: [
                                        { name: "Paul V.", year: 1960, branch: "V", steps: 1, lateral: 2, desc: "First Cousin Once Removed", children: [
                                            { name: "Paul's Wife", year: 1960, branch: "V", steps: 1, lateral: 2, inLaw: true },
                                            { name: "Maushup V.", year: 1985, branch: "V", steps: 0, lateral: 3, desc: "Second Cousin" },
                                            { name: "Nashawn V.", year: 1987, branch: "V", steps: 0, lateral: 3, desc: "Second Cousin" }
                                        ]},
                                        { name: "Polly V.", year: 1962, branch: "V", steps: 1, lateral: 2, desc: "First Cousin Once Removed" }
                                    ]}
                                ]},
                                { name: "Johnny Vanderhoop", year: 1936, branch: "V", steps: 1, lateral: 1, desc: "Aunt / Uncle" }
                            ]},
                            { name: "Waltrud M. Lieber", year: 1934, branch: "L", steps: 2, lateral: 0, desc: "Grandparent", children: [
                                { name: "G. Heinrich L. Lieber", year: 1900, branch: "L", steps: 3, lateral: 0, desc: "Great Grandparent", children: [
                                    { name: "Marie Emilie Ibe", year: 1900, branch: "L", steps: 3, lateral: 0, inLaw: true },
                                    { name: "Manfred Lieber", year: 1932, branch: "L", steps: 2, lateral: 1, desc: "Great Aunt / Uncle" }
                                ]}
                            ]}
                        ]
                    }
                ]
            };

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

                // Spouses AND their ancestral lines are locked to 40% of their anchor generation
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
                // NEW: Branch-specific Spouse Colors!
                if (d.inLaw || d.isSpouseLine) {
                    if (d.branch === "K" || d.branch === "R") return "rgb(0, 255, 255)"; // Cyan (Dad)
                    if (d.branch === "B" || d.branch === "A") return "rgb(255, 105, 180)"; // Hot Pink (Step-Mom)
                    if (d.branch === "V" || d.branch === "L") return "rgb(205, 127, 50)"; // Bright Bronze (Bio Mom)
                    return "rgb(255, 215, 0)"; // Fallback
                }

                let mainSteps = d.steps + d.lateral; 
                let mMax = maxSteps[d.branch] || 1;
                let t = (mainSteps <= 1) ? 0 : (mainSteps - 1) / Math.max(1, mMax - 1);
                let r = 240, g = 0, b = 240; 

                if (d.branch === "K") { r = Math.round(220 + (0 - 220) * t); b = 240; } 
                else if (d.branch === "V") { r = 240; b = Math.round(220 + (0 - 220) * t); } 
                else if (d.branch === "R") { r = Math.round(220 + (0 - 220) * t); g = Math.round(0 + (240 - 0) * t); b = Math.round(240 + (0 - 240) * t); } 
                else if (d.branch === "L") { r = 240; g = Math.round(0 + (240 - 0) * t); b = Math.round(220 + (0 - 220) * t); } 
                else if (d.branch === "B") { r = 240; g = Math.round(220 + (0 - 220) * t); b = Math.round(220 + (0 - 220) * t); } 
                else if (d.branch === "A") { r = 240; g = Math.round(220 + (128 - 220) * t); b = Math.round(220 + (0 - 220) * t); }

                return `rgb(${r}, ${g}, ${b})`;
            }

            // ==========================================
            // 3. LEFT PANEL: FORCE WEB
            // ==========================================
            const fSvg = d3.select("#viz-force");
            const width = fSvg.node().getBoundingClientRect().width;
            const height = 700;

            fSvg.attr("viewBox", [-width / 2, -height / 2, width, height]);

            // Physics Engine 
            const simulation = d3.forceSimulation(graphData.nodes)
                .force("link", d3.forceLink(graphData.links).id(d => d.id)
                    .distance(d => {
                        let targetNode = d.target.data ? d.target.data : d.target;

                        // NEW: Ultra-short links to keep offshoot spouse branches clustered tightly
                        if (d.type === "spouse_main") return 8;

                        let rootStep = targetNode.lateral > 0 ? (targetNode.steps + targetNode.lateral) : targetNode.steps;
                        let shrinkFactor = Math.max(0.2, 1 - 0.10 * rootStep);
                        
                        if (d.type === "marriage" || d.type === "inlaw") return 15; 
                        if (d.type === "leaf") return 25 * shrinkFactor; 
                        
                        // Main branches start longer (65) and shrink exactly 10% per generation
                        return 65 * shrinkFactor;
                    })
                    .strength(d => {
                        if (d.type === "marriage") return 0.1; 
                        if (d.type === "leaf" || d.type === "inlaw") return 2; 
                        return 1;
                    }) 
                )
                .force("charge", d3.forceManyBody().strength(d => {
                    let rootStep = d.lateral > 0 ? (d.steps + d.lateral) : d.steps;
                    let shrinkFactor = Math.max(0.2, 1 - 0.10 * rootStep);
                    
                    if (d.inLaw) return -5;
                    if (d.lateral > 0) return -15 * shrinkFactor; 
                    return -200 * shrinkFactor; 
                }))
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
                .attr("opacity", d => calcOpacity(d))
                .on("mouseover", (e,d) => {
                    tooltip.transition().duration(200).style("opacity", 1);
                    let stat = d.desc;
                    if(d.inLaw && !d.desc) stat = "In-Law (Spouse)";
                    tooltip.html(`<strong>${d.id}</strong><br/>${stat}`)
                        .style("left", (e.pageX + 15) + "px").style("top", (e.pageY - 28) + "px");
                })
                .on("mouseout", () => tooltip.transition().duration(500).style("opacity", 0));

            simulation.on("tick", () => {
                fLink.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
                     .attr("x2", d => d.target.x).attr("y2", d => d.target.y)
                     .attr("stroke-width", d => calcLinkWidth(d))
                     .attr("stroke-opacity", d => calcOpacity(d.target));
                // FIX: Reverted to standard JS single-brace interpolation
                fNode.attr("transform", d => `translate(${d.x},${d.y})`);
            });

            // --- Dimension & Color Legend ---
            const legend = fSvg.append("g").attr("transform", `translate(${-width/2 + 20}, ${height/2 - 170})`);
            legend.append("text").attr("fill", "#94A3B8").attr("font-size", "12px").attr("y", -10).text("DIMENSIONAL RULES:");
            
            const defs = fSvg.append("defs");
            function buildGrad(id, c1, c2) {
                let g = defs.append("linearGradient").attr("id", id);
                g.append("stop").attr("offset", "0%").attr("stop-color", c1);
                g.append("stop").attr("offset", "100%").attr("stop-color", c2);
            }
            // All gradients start perfectly at White (240,240,240) and end at their exact target RGB
            buildGrad("k-grad", "rgb(240,240,240)", "rgb(0,0,240)");     // Blue
            buildGrad("r-grad", "rgb(240,240,240)", "rgb(160,32,240)");  // Purple
            buildGrad("v-grad", "rgb(240,240,240)", "rgb(0,240,0)");     // Green
            buildGrad("l-grad", "rgb(240,240,240)", "rgb(240,240,0)");   // Yellow
            buildGrad("b-grad", "rgb(240,240,240)", "rgb(240,0,0)");     // Red
            buildGrad("a-grad", "rgb(240,240,240)", "rgb(240,128,0)");   // Orange

            const labels = ["Killebrew (Dad's Father)", "Rasmussen (Dad's Mother)", "Vanderhoop (Mom's Father)", "Lieber (Mom's Mother)", "Buzunis (Step Mom's Father)", "Ginakes (Step Mom's Mother)"];
            const grads = ["url(#k-grad)", "url(#r-grad)", "url(#v-grad)", "url(#l-grad)", "url(#b-grad)", "url(#a-grad)"];
            
            labels.forEach((l, i) => {
                legend.append("rect").attr("y", i*20).attr("width", 50).attr("height", 10).style("fill", grads[i]);
                legend.append("text").attr("x", 60).attr("y", i*20 + 9).attr("fill", "#64748B").attr("font-size", "10px").text(l);
            });
            
            // Render the Spousal Color Map
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
            
            // 1. Scale Down Elements to Prevent Overlap (35% smaller)
            const tScale = 0.65;
            const tRad = d => calcRadius(d) * tScale;
            
            const root = d3.hierarchy(treeData);
            
            /// 2. Dynamic Separation with In-Law compression & Sibling X-Axis Reduction
            d3.tree().size([tWidth - 60, height - 100])
                .separation((a, b) => {
                    // Spouses will be snapped manually, keep their footprint minimal
                    let isSpouse = a.data.inLaw || b.data.inLaw;
                    
                    // X-AXIS REDUCTION: Shrink distance between children (siblings) by ~50% (0.75 instead of 1.5)
                    let baseSep = a.parent === b.parent ? 0.75 : 2.5;
                    return isSpouse ? 0.25 : baseSep; 
                })(root);
            
            // --- Custom Generational Y-Positioning ---
            const base_y = 120; // Starting Y position for Kyle
            const gen_gap = 70; // Pixel distance between generations
            
            let minStep = 0;
            let maxStep = 0;

            // 3. First Pass: Apply Y-Coordinates, Zipper Staggering, and Min/Max
            root.each(d => { 
                d.y = base_y + (d.data.steps * gen_gap); 
                
                // ZIPPER ALGORITHM: Offset lateral descendants (siblings/cousins) vertically to pack them tighter
                if (d.data.lateral > 0 && !d.data.inLaw && d.parent) {
                    let lateralSiblings = d.parent.children.filter(c => !c.data.inLaw);
                    if (lateralSiblings.length > 1) {
                        let idx = lateralSiblings.indexOf(d);
                        d.y += (idx % 2 === 0) ? -15 : 15; // Alternate up and down by 15px
                    }
                }

                minStep = Math.min(minStep, d.data.steps);
                maxStep = Math.max(maxStep, d.data.steps);
            });

            // 4. Second Pass: Spousal Abutment 
            root.each(d => {
                // Snap the spouse, but let the spouse's ancestors flow naturally!
                if (d.data.inLaw && !d.data.isSpouseLine && d.parent) {
                    d.x = d.parent.x + tRad(d.parent.data) + tRad(d.data) + 2;
                    d.y = d.parent.y; 
                }
            });

            // --- Draw Generational Gridlines ---
            const genMarks = d3.range(minStep, maxStep + 1);
            
            tSvg.selectAll(".grid-line").data(genMarks).enter().append("line")
                .attr("class", "grid-line").attr("x1", 30).attr("x2", tWidth - 10)
                .attr("y1", d => base_y + (d * gen_gap))
                .attr("y2", d => base_y + (d * gen_gap));
                
            tSvg.selectAll(".grid-label").data(genMarks).enter().append("text")
                .attr("class", "grid-label").attr("x", 5)
                .attr("y", d => base_y + (d * gen_gap) - 5)
                .text(d => {
                    if (d === 0) return "Baseline";
                    if (d < 0) return `Gen ${Math.abs(d)} (Next)`;
                    return `Gen ${d}`;
                });

            // --- Draw Nodes and Links ---
            const treeGroup = tSvg.append("g").attr("transform", "translate(30, 20)");
            
            treeGroup.selectAll(".tree-link").data(root.links()).enter().append("path")
                .attr("class", d => {
                    if (d.target.data.inLaw) return "tree-link inlaw";
                    if (d.target.data.lateral > 0) return "tree-link leaf";
                    return "tree-link main";
                })
                .attr("d", d3.linkVertical().x(d => d.x).y(d => d.y))
                .attr("stroke-width", d => {
                    // SPOSUAL LINK OVERRIDE: 0 thickness for the actual marriage, but normal thickness for their ancestors
                    if (d.target.data.inLaw && !d.target.data.isSpouseLine) return 0;
                    return Math.max(1.5, (calcRadius(d.target.data) * 0.2) * tScale);
                })
                .attr("stroke-opacity", d => (d.target.data.inLaw && !d.target.data.isSpouseLine) ? 0 : 0.8);

            const tNode = treeGroup.selectAll(".tree-node").data(root.descendants()).enter().append("g")
                .attr("transform", d => `translate(${d.x},${d.y})`);

            tNode.append("circle")
                .attr("class", "node")
                .attr("r", d => tRad(d.data)) // Use the scaled radius
                .attr("fill", d => calcColor(d.data))
                .on("mouseover", (e,d) => {
                    tooltip.transition().duration(200).style("opacity", 1);
                    let stat = d.data.desc;
                    if(d.data.inLaw && !d.data.desc) stat = "In-Law (Spouse)";
                    tooltip.html(`<strong>${d.data.name}</strong><br/>${stat}`)
                        .style("left", (e.pageX + 15) + "px").style("top", (e.pageY - 28) + "px");
                })
                .on("mouseout", () => tooltip.transition().duration(500).style("opacity", 0));
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=720)