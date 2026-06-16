"""
=============================================================================
MODULE: loaders/family_tree_loader.py
AUTHOR: Kyle W. Killebrew, PhD
DESCRIPTION: 
    Data extraction and hydration script for the Genealogy Web & Migration Map.
    Dynamically fetches and parses GEDCOM data from a secure GitHub repository
    to procedurally generate the UI nodes and hierarchical trees.
=============================================================================
"""
import os
import re
import requests
import streamlit as st

# =====================================================================
# 1. DYNAMIC GEDCOM PARSER & GRAPH GENERATOR
# =====================================================================

@st.cache_data(max_entries=2, ttl=3600)
def fetch_and_parse_gedcom():
    """
    MATLAB Equivalent: webread() -> regexpi()
    Fetches the raw .ged file securely and parses it into relational dictionaries.
    """
    # 1. Secure Token Retrieval (Mirrors academic_research_loader.py)
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        try:
            github_token = st.secrets["GITHUB_TOKEN"]
        except Exception:
            pass
            
    url = "https://raw.githubusercontent.com/kkillebrew/neuroedu.io.data-projects/refs/heads/main/documents/Kyle%20Killebrew%20family%20tree.ged"
    headers = {'Authorization': f'token {github_token}'} if github_token else {}
    
    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code != 200:
            return {}, {}
    except Exception as e:
        print(f"GEDCOM Fetch Error: {e}")
        return {}, {}
        
    lines = res.text.split('\n')
    
    indis, fams = {}, {}
    curr_id, curr_type, prev_tag = None, None, None
    
    # 2. Text Parsing Engine
    for line in lines:
        parts = line.strip().split(' ', 2)
        if len(parts) < 2: continue
        lvl, tag = parts[0], parts[1]
        val = parts[2] if len(parts) > 2 else ""
        
        if lvl == '0':
            if tag.startswith('@I'):
                curr_id, curr_type = tag, 'INDI'
                indis[curr_id] = {'name': 'Unknown', 'birth': '', 'death': '', 'bio': '', 'fams': [], 'famc': []}
            elif tag.startswith('@F'):
                curr_id, curr_type = tag, 'FAM'
                fams[curr_id] = {'husb': None, 'wife': None, 'chil': []}
            else:
                curr_type = None
        elif curr_type == 'INDI':
            if lvl == '1': prev_tag = tag
            if tag == 'NAME': 
                # Clean up the standard GEDCOM surname slashes
                indis[curr_id]['name'] = val.replace('/', '').replace('\\', '').strip()
            elif lvl == '2' and tag == 'DATE':
                year_match = re.search(r'\b\d{4}\b', val)
                if year_match:
                    if prev_tag == 'BIRT': indis[curr_id]['birth'] = year_match.group(0)
                    if prev_tag == 'DEAT': indis[curr_id]['death'] = year_match.group(0)
            elif tag == 'NOTE': indis[curr_id]['bio'] = val
            elif tag == 'CONT' and prev_tag == 'NOTE': indis[curr_id]['bio'] += " " + val
            elif tag == 'FAMS': indis[curr_id]['fams'].append(val)
            elif tag == 'FAMC': indis[curr_id]['famc'].append(val)
        elif curr_type == 'FAM':
            if tag == 'HUSB': fams[curr_id]['husb'] = val
            elif tag == 'WIFE': fams[curr_id]['wife'] = val
            elif tag == 'CHIL': fams[curr_id]['chil'].append(val)
            
    return indis, fams

def get_family_tree_data():
    """
    Translates the parsed GEDCOM dictionaries into the specific graph DataFrames
    required by the D3.js UI component using a Breadth-First Search (BFS).
    """
    indis, fams = fetch_and_parse_gedcom()
    
    # Graceful failure if GitHub fetch drops
    if not indis:
        return {"nodes": [], "links": []}, {"name": "Data Unavailable"}

    # =================================================================
    # 1. EXPANDED DYNAMIC COLOR MAPPING
    # =================================================================
    def assign_branch(name, current_branch):
        name_lower = name.lower()
        
        # Core Identity Overrides (Locks the center of the gravity graph)
        if "kyle william" in name_lower: return "M"
        if "eric scott" in name_lower: return "K"
        if "antonia constance" in name_lower: return "B"
        
        # Procedural Surname Catch-alls for Extended Families
        if any(sub in name_lower for sub in ["killebrew", "robinson", "impson", "moore"]): return "K"
        if any(sub in name_lower for sub in ["rasmussen", "rasmusson", "gregerson", "gregersen"]): return "R"
        if any(sub in name_lower for sub in ["vanderhoop", "cleggett", "salisbury", "diamond", "madison", "smalley"]): return "V"
        if "lieber" in name_lower: return "L"
        if "buzunis" in name_lower: return "B"
        if any(sub in name_lower for sub in ["ginakes", "giannakis", "boosalis", "effos"]): return "A"
        
        return current_branch

    # Locate the Root Node dynamically
    root_id = next((i_id for i_id, d in indis.items() if "Kyle William" in d['name']), None)
    
    nodes, raw_links, visited = [], [], set()
    queue = []
    
    if root_id:
        queue.append((root_id, "M", 0, 0, False))

    # =================================================================
    # 2. BUILD THE FORCE GRAPH DATA (Left Panel BFS)
    # =================================================================
    while queue:
        curr_id, branch, steps, lateral, in_law = queue.pop(0)
        
        # Guardrail: Prevent infinite recursion and cap the depth
        if curr_id in visited or steps > 8 or lateral > 2: continue
        visited.add(curr_id)
        
        indi = indis[curr_id]
        name = indi['name']
        branch = assign_branch(name, branch)
        
        by, dy = indi['birth'], indi['death']
        years = f"({by} - {dy})" if by and dy else f"({by})" if by else f"(d. {dy})" if dy else ""
        
        if lateral == 0:
            desc = {0: "You (Present)", 1: "Parent", 2: "Grandparent", 3: "Great Grandparent"}.get(steps, f"{steps-2}x Great Grandparent")
        else:
            desc = {0: "Sibling", 1: "Aunt / Uncle", 2: "Great Aunt / Uncle"}.get(steps, "Extended Family")
        if in_law: desc = "Spouse / In-Law"
            
        nodes.append({
            "id": curr_id, # Strict GEDCOM ID to prevent duplicate name crashes
            "name": name, "branch": branch, "steps": steps, 
            "lateral": lateral, "desc": f"{desc} {years}".strip(), 
            "bio": indi['bio'], "inLaw": in_law, "isSpouseLine": in_law, "anchorStep": steps
        })
        
        if lateral == 0:
            for famc_id in indi['famc']:
                fam = fams.get(famc_id)
                if not fam: continue
                
                for p_key in ['husb', 'wife']:
                    p_id = fam.get(p_key)
                    if p_id:
                        queue.append((p_id, branch, steps + 1, 0, False))
                        raw_links.append({"source": p_id, "target": curr_id, "type": "main"})
                        
                for chil_id in fam['chil']:
                    if chil_id != curr_id and chil_id not in visited:
                        queue.append((chil_id, branch, steps, lateral + 1, False))
                        parent_src = fam['husb'] if fam['husb'] else fam['wife']
                        if parent_src:
                            raw_links.append({"source": parent_src, "target": chil_id, "type": "leaf"})

        for fams_id in indi['fams']:
            fam = fams.get(fams_id)
            if not fam: continue
            
            spouse_id = fam['wife'] if fam['husb'] == curr_id else fam['husb']
            if spouse_id and spouse_id not in visited:
                queue.append((spouse_id, branch, steps, lateral, True))
                raw_links.append({"source": curr_id, "target": spouse_id, "type": "marriage"})

    # D3 Safe-Link Culling Filter
    valid_node_ids = {n["id"] for n in nodes}
    links = [l for l in raw_links if l["source"] in valid_node_ids and l["target"] in valid_node_ids]
    
    graph_data = {"nodes": nodes, "links": links}

    # =================================================================
    # 3. BUILD THE HIERARCHICAL TREE DATA (Right Panel Recursive)
    # =================================================================
    def build_pedigree(curr_id, current_step=0, current_branch="M", tree_visited=None):
        if tree_visited is None: tree_visited = set()
        
        # Protect against recursive loops
        if current_step > 5 or curr_id in tree_visited: return None
        tree_visited.add(curr_id)
        
        indi = indis.get(curr_id)
        if not indi: return None
        
        b = assign_branch(indi['name'], current_branch)
        
        node = {
            "name": indi['name'],
            "steps": current_step,
            "lateral": 0,
            "inLaw": False,
            "isSpouseLine": False,
            "branch": b
        }
        
        children = []
        for famc_id in indi['famc']:
            fam = fams.get(famc_id)
            if fam:
                if fam.get('husb'): 
                    h_node = build_pedigree(fam['husb'], current_step + 1, b, set(tree_visited))
                    if h_node: children.append(h_node)
                if fam.get('wife'):
                    w_node = build_pedigree(fam['wife'], current_step + 1, b, set(tree_visited))
                    if w_node: children.append(w_node)
                    
        if children:
            node["children"] = children
            
        return node
        
    tree_data = build_pedigree(root_id) if root_id else {"name": "Root Not Found"}

    return graph_data, tree_data

def get_migration_data():
    """
    Returns the grouped geographic migration paths.
    Each branch is an array of 'nodes' (waypoints) containing arrays of 'people'.
    """
    return {
        "Killebrew": {
            "base_color": (0, 0, 240), # Blue
            "nodes": [
                {"city": "Cornwall, England", "lat": 50.2660, "lon": -5.0527, "people": [
                    {"name": "Francis Killebrew", "years": "1619-1673", "desc": "Original immigrant from Cornwall to Virginia."}
                ]},
                {"city": "Westmoreland, VA", "lat": 38.1065, "lon": -76.8152, "people": [
                    {"name": "Francis Killebrew", "years": "d. 1673", "desc": "Settled and passed away here."}
                ]},
                {"city": "Tarboro, NC", "lat": 35.8979, "lon": -77.5358, "people": [
                    {"name": "Joseph Buckner Killebrew", "years": "1753-1824", "desc": "Born in Tarboro, migrated to Tennessee."}
                ]},
                {"city": "Clarksville / Montgomery Co., TN", "lat": 36.5298, "lon": -87.3595, "people": [
                    {"name": "Whitfield Killebrew", "years": "1793-1859", "desc": "Buried at Osburn-Killebrew Cemetery."},
                    {"name": "George Washington Killebrew", "years": "1812-1871", "desc": "Buried at Osburn-Killebrew Cemetery."},
                    {"name": "William Henry Killebrew", "years": "1898-1970", "desc": "Born here before migrating west."}
                ]},
                {"city": "Christian County, KY", "lat": 36.8682, "lon": -87.4913, "people": [
                    {"name": "Daniel Boone Killebrew", "years": "1860-1939", "desc": "Born here."}
                ]},
                {"city": "McAlester / Hartshorne, OK", "lat": 34.9334, "lon": -95.7697, "people": [
                    {"name": "Daniel Boone Killebrew", "years": "d. 1939", "desc": "Passed away in McAlester."},
                    {"name": "Robert 'Bob' Killebrew", "years": "1930-2017", "desc": "Born in Hartshorne."}
                ]},
                {"city": "Las Vegas, NV", "lat": 36.1699, "lon": -115.1398, "people": [
                    {"name": "Robert Killebrew", "years": "d. 2017", "desc": "Final resting place."},
                    {"name": "Eric Scott Killebrew Sr.", "years": "1961-Present", "desc": "Current resident."},
                    {"name": "Kyle William Killebrew", "years": "1990-Present", "desc": "Current resident."}
                ]}
            ]
        },
        "Robinson_Impson": {
            "base_color": (0, 255, 255), # Cyan (Spouse Branch of Killebrew)
            "nodes": [
                {"city": "Shelby County, IN", "lat": 39.5215, "lon": -85.7686, "people": [
                    {"name": "Neal Clark Robeson Sr.", "years": "1760-1841", "desc": "Early Indiana Territory settler."}
                ]},
                {"city": "Fannagusha Creek, MS", "lat": 32.8809, "lon": -90.0460, "people": [
                    {"name": "Isaac Impson", "years": "1800-1899", "desc": "English descent, married into Choctaw tribe."},
                    {"name": "Josiah Impson", "years": "b. 1824", "desc": "Born on a 160-acre farm here."}
                ]},
                {"city": "Sullivan County, IN", "lat": 38.9567, "lon": -87.4093, "people": [
                    {"name": "James Wesley Robinson", "years": "b. 1847", "desc": "Born in Carlisle."}
                ]},
                {"city": "Roseville, AR", "lat": 35.3948, "lon": -93.8242, "people": [
                    {"name": "Niel C. Robinson", "years": "d. 1864", "desc": "<b>Military:</b> Union Soldier (Kansas 2nd Cavalry). Died of wounds received in action."}
                ]},
                {"city": "Jumbo, OK", "lat": 34.3986, "lon": -95.6961, "people": [
                    {"name": "Josiah Impson", "years": "d. 1896", "desc": "Survived the 1833 'Trail of Tears'. Passed away in what is now a ghost town."}
                ]},
                {"city": "Hartshorne, OK", "lat": 34.8468, "lon": -95.5566, "people": [
                    {"name": "James Wesley Robinson", "years": "d. 1916", "desc": "Civil War Veteran, passed away here."},
                    {"name": "Mary Esther Robinson", "years": "1902-1981", "desc": "Born here, eventually married William H. Killebrew."}
                ]}
            ]
        },
        "Rasmussen": {
            "base_color": (160, 32, 240), # Purple
            "nodes": [
                {"city": "Copenhagen, Denmark", "lat": 55.6761, "lon": 12.5683, "people": [
                    {"name": "Jens Rasmussen", "years": "b. 1810", "desc": "Weaver. Emigrated from here in 1866 on the steamship Aurora."},
                    {"name": "Rasmus Jensen Rasmussen", "years": "b. 1842", "desc": "Born in Denmark."}
                ]},
                {"city": "Wyoming, NE", "lat": 40.6322, "lon": -95.8453, "people": [
                    {"name": "Maren Jorgensen Rasmussen", "years": "1866", "desc": "Arrived via riverboat. The sick were not allowed on board."}
                ]},
                {"city": "Platte River, WY", "lat": 42.8501, "lon": -106.3005, "people": [
                    {"name": "Maren Jorgensen Rasmussen", "years": "d. 1866", "desc": "<b>Tragedy:</b> Died of cholera en route to Utah. Buried somewhere along the Platte River."}
                ]},
                {"city": "Ephraim, UT", "lat": 39.3602, "lon": -111.5866, "people": [
                    {"name": "Jens Rasmussen", "years": "d. 1888", "desc": "Settled here. Served in Utah Territorial Militia."},
                    {"name": "Rasmus Jensen Rasmussen", "years": "d. 1920", "desc": "Served in Utah Territorial Militia during the Black Hawk War."}
                ]},
                {"city": "Monroe, UT", "lat": 38.6270, "lon": -112.1220, "people": [
                    {"name": "Clinton Rasmussen", "years": "1904-1979", "desc": "Passed away here."},
                    {"name": "Bonnie Rasmussen", "years": "1934-2020", "desc": "Born here."}
                ]},
                {"city": "Las Vegas, NV", "lat": 36.1699, "lon": -115.1398, "people": [
                    {"name": "Bonnie Rasmussen", "years": "d. 2020", "desc": "Final resting place."}
                ]}
            ]
        },
        "Vanderhoop": {
            "base_color": (0, 240, 0), # Green
            "nodes": [
                {"city": "Champagne, France", "lat": 48.9567, "lon": 4.3631, "people": [
                    {"name": "Francois Adriend l'Espoir", "years": "Unknown", "desc": "Original patriarch. Migrated to Amsterdam."}
                ]},
                {"city": "Amsterdam, Netherlands", "lat": 52.3676, "lon": 4.9041, "people": [
                    {"name": "Baron Cornelis Jacobus Vanderhoop", "years": "1640-1689", "desc": "Born here, eventually migrated to Batavia, Indonesia."},
                    {"name": "Baron Adriaan vanDel Vanderhoop Sr.", "years": "1778-1854", "desc": "Lived in Amsterdam and Santpoort Estate."}
                ]},
                {"city": "Batavia, Indonesia", "lat": -6.2000, "lon": 106.8166, "people": [
                    {"name": "Baron Francois Adrien Vanderhoop", "years": "1675-1741", "desc": "Born in Batavia, Dutch East Indies."}
                ]},
                {"city": "The Hague, Netherlands", "lat": 52.0705, "lon": 4.3007, "people": [
                    {"name": "Baron Francois Adrien Vanderhoop", "years": "d. 1741", "desc": "Returned from Indonesia, passed away here."},
                    {"name": "Maria Hartley", "years": "1682-?", "desc": "Married Baron Francois Adrien."},
                    {"name": "Baron Adriaan Vanderhoop I", "years": "1701-1767", "desc": "Born in 's-Gravenhage (The Hague)."},
                    {"name": "Joan Cornelis Vanderhoop", "years": "1742-1825", "desc": "Lived entire life in The Hague."}
                ]},
                {"city": "Paramaribo, Suriname", "lat": 5.8520, "lon": -55.2038, "people": [
                    {"name": "William A. Vanderhoop", "years": "b. 1816", "desc": "Dutch Surinamese immigrant."}
                ]},
                {"city": "Gay Head (Aquinnah), MA", "lat": 41.3368, "lon": -70.8316, "people": [
                    {"name": "William A. Vanderhoop", "years": "d. 1893", "desc": "First Vanderhoop on Martha's Vineyard. Built the homestead."},
                    {"name": "Cummings Bray Vanderhoop", "years": "1853-1886", "desc": "Born and lived here."},
                    {"name": "Edwin Devries Vanderhoop", "years": "1848-1923", "desc": "<b>Military:</b> Union Navy gunboat Maheska (Civil War)."},
                    {"name": "David F. Vanderhoop", "years": "Unknown", "desc": "<b>Military:</b> WWI Veteran."},
                    {"name": "Arthur Herbert Vanderhoop", "years": "1877-?", "desc": "Born here."},
                    {"name": "Baroness Mary Ann Cleggett", "years": "1860-1924", "desc": "Born in PA, lived and passed away here."},
                    {"name": "William Diamond Vanderhoop, Sr.", "years": "1890-?", "desc": "Air Force member."},
                    {"name": "Helen Edith (Vanderhoop) Manning", "years": "1919-?", "desc": "Authored 'Moshup's Footsteps'."},
                    {"name": "June Manning", "years": "Unknown", "desc": "Aquinnah native and family historian."},
                    {"name": "William D. 'Buddy' Vanderhoop Jr.", "years": "1927-2004", "desc": "Legendary Aquinnah fisherman."}
                ]},
                {"city": "Las Vegas, NV", "lat": 36.1699, "lon": -115.1398, "people": [
                    {"name": "John O. Vanderhoop", "years": "1934-2022", "desc": "<b>Military:</b> USAF Major (Vietnam/Thailand/Germany). Bronze Star."}
                ]}
            ]
        },
        "Vanderhoop_Salisbury": {
            "base_color": (205, 127, 50), # Bright Bronze (Mom's In-Laws)
            "nodes": [
                {"city": "Gay Head (Aquinnah), MA", "lat": 41.3368, "lon": -70.8316, "people": [
                    {"name": "John Salisbury", "years": "~1792-~1870", "desc": "Wampanoag Tribal Member."},
                    {"name": "Naomi Occouch Salisbury", "years": "1788-?", "desc": "Wampanoag Tribal Member."},
                    {"name": "Beulah Salisbury", "years": "1814-1892", "desc": "'Princess of Aquinnah'. Hid escaped slaves under a false barn floor (Underground Railroad)."}
                ]},
                {"city": "Las Vegas, NV", "lat": 36.1699, "lon": -115.1398, "people": [
                    {"name": "Christina Vanderhoop", "years": "1961-Present", "desc": "Current resident."}
                ]}
            ]
        },
        "Vanderhoop_Diamond": {
            "base_color": (205, 127, 50), # Bright Bronze (Mom's In-Laws)
            "nodes": [
                {"city": "New York, NY", "lat": 40.7128, "lon": -74.0060, "people": [
                    {"name": "James Diamond", "years": "b. 1820", "desc": "Born in New York."},
                    {"name": "Samuel Smalley", "years": "1829-1893", "desc": "Born in NY, later moved to Gay Head."},
                    {"name": "John Smalley", "years": "Unknown", "desc": "Lived in New York."}
                ]},
                {"city": "Gay Head (Aquinnah), MA", "lat": 41.3368, "lon": -70.8316, "people": [
                    {"name": "James Diamond", "years": "Moved 1870", "desc": "Relocated from New York."},
                    {"name": "Abiah Manning", "years": "1821-1906", "desc": "Married James Diamond in Chilmark."},
                    {"name": "Julia Bassett Smalley", "years": "1838-1888", "desc": "Owned cottage at 19 Dukes County Ave."},
                    {"name": "Amos Peters Smalley", "years": "1877-1961", "desc": "Legendary Gay Head harpooner credited with killing a white whale."},
                    {"name": "Leander Bassett", "years": "1810-1879", "desc": "From Edgartown."},
                    {"name": "Huldah Jeffers", "years": "1807-1879", "desc": "Aquinnah Wampanoag native."},
                    {"name": "Rosetta Ellis Diamond", "years": "1862-1922", "desc": "Born and lived here."},
                    {"name": "Durwood Delmond Diamond", "years": "1878-1947", "desc": "Born and passed away here."},
                    {"name": "Baroness Elsie Ester (Diamond) Vanderhoop", "years": "1899-1936", "desc": "Invested Posthumously as Baroness Consort van der Hoop of the Netherlands."}
                ]},
                {"city": "St. Louis, MI", "lat": 43.4092, "lon": -84.6067, "people": [
                    {"name": "Durwood Delmond Diamond", "years": "1923", "desc": "Served in the Military here."}
                ]},
                {"city": "Las Vegas, NV", "lat": 36.1699, "lon": -115.1398, "people": [
                    {"name": "Christina Vanderhoop", "years": "1961-Present", "desc": "Current resident."}
                ]}
            ]
        },
        "Vanderhoop_Matriarchs": {
            "base_color": (205, 127, 50), # Bright Bronze (Mom's In-Laws)
            "nodes": [
                {"city": "Pennsylvania, USA", "lat": 41.2033, "lon": -77.1945, "people": [
                    {"name": "Baroness Mary Ann Cleggett", "years": "b. 1860", "desc": "Born in Pennsylvania."}
                ]},
                {"city": "Providence, RI", "lat": 41.8240, "lon": -71.4128, "people": [
                    {"name": "Anne Madison", "years": "Unknown", "desc": "Lived in Providence before migrating."}
                ]},
                {"city": "Edgartown, MA", "lat": 41.3888, "lon": -70.5133, "people": [
                    {"name": "Anne Madison", "years": "Unknown", "desc": "Moved through Edgartown on her way to Gay Head."}
                ]},
                {"city": "Gay Head (Aquinnah), MA", "lat": 41.3368, "lon": -70.8316, "people": [
                    {"name": "Baroness Mary Ann Cleggett", "years": "d. 1924", "desc": "Married Edwin Devries Vanderhoop."},
                    {"name": "Anne Madison", "years": "Unknown", "desc": "Married William Diamond Vanderhoop Sr."}
                ]}
            ]
        },
        "Buzunis": {
            "base_color": (240, 0, 0), # Red
            "nodes": [
                {"city": "Levidion, Greece", "lat": 37.6811, "lon": 22.2968, "people": [
                    {"name": "George Constantine Buzunis", "years": "1846-1912", "desc": "Lived entire life here."}
                ]},
                {"city": "Tripoli, Greece", "lat": 37.5113, "lon": 22.3737, "people": [
                    {"name": "Theodore Buzunis", "years": "b. 1885", "desc": "Born here before migrating to Canada."}
                ]},
                {"city": "Vanguard, SK, Canada", "lat": 49.9167, "lon": -107.0333, "people": [
                    {"name": "Peter Buzunis", "years": "b. 1917", "desc": "Born here."}
                ]},
                {"city": "Winnipeg, MB, Canada", "lat": 49.8951, "lon": -97.1384, "people": [
                    {"name": "Theodore Buzunis", "years": "d. 1978", "desc": "Passed away here."},
                    {"name": "Peter Buzunis", "years": "d. 2007", "desc": "Passed away here."}
                ]},
                {"city": "Las Vegas, NV", "lat": 36.1699, "lon": -115.1398, "people": [
                    {"name": "Antonia Constance Buzunis", "years": "1964-Present", "desc": "Current resident."}
                ]}
            ]
        },
        "Ginakes": {
            "base_color": (240, 128, 0), # Orange
            "nodes": [
                {"city": "Greece (General)", "lat": 39.0742, "lon": 21.8243, "people": [
                    {"name": "Desmos Giannakis", "years": "b. 1865", "desc": "Lived in Greece."}
                ]},
                {"city": "Fargo, ND", "lat": 46.8772, "lon": -96.7898, "people": [
                    {"name": "Andrew Demetrius Ginakes", "years": "d. 1967", "desc": "Migrated from Greece and passed away here."}
                ]},
                {"city": "Winnipeg, MB, Canada", "lat": 49.8951, "lon": -97.1384, "people": [
                    {"name": "Anastasia Ginakes", "years": "d. 2018", "desc": "Passed away here."}
                ]},
                {"city": "Las Vegas, NV", "lat": 36.1699, "lon": -115.1398, "people": [
                    {"name": "Antonia Constance Buzunis", "years": "1964-Present", "desc": "Current resident."}
                ]}
            ]
        }
    }