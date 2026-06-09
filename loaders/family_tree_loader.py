"""
=============================================================================
MODULE: loaders/family_tree_loader.py
AUTHOR: Kyle W. Killebrew, PhD
DESCRIPTION: 
    Data extraction and hydration script for the Genealogy Web & Migration Map.
    This acts similar to a .mat data file, isolating the raw nested dictionaries
    and node arrays from the visualization logic. Contains rich bio data.
=============================================================================
"""

def get_family_tree_data():
    """
    Returns the graphData (nodes, links) and treeData (hierarchical) dictionaries
    for injection into the D3.js visualization.
    """
    
    graph_data = {
        "nodes": [
            # --- CORE / IMMEDIATE FAMILY ---
            {"id": "Kyle Killebrew", "branch": "M", "steps": 0, "lateral": 0, "desc": "You (Present)", "bio": "1990-Present; Las Vegas, NV"},
            {"id": "Eric Killebrew", "branch": "K", "steps": 1, "lateral": 0, "desc": "Parent", "bio": "1961-Present; Las Vegas, NV"},
            {"id": "Christina Vanderhoop", "branch": "V", "steps": 1, "lateral": 0, "desc": "Parent"},
            {"id": "Antonia Constance Buzunis", "branch": "B", "steps": 1, "lateral": 0, "desc": "Step Mother", "bio": "1964-Present; Las Vegas, NV"},
            {"id": "Andrea Nicole Killebrew", "branch": "K", "steps": 0, "lateral": 1, "desc": "Sister / Brother"},
            {"id": "Eric Scott Killebrew, Jr.", "branch": "K", "steps": 0, "lateral": 1, "desc": "Sister / Brother"},

            # --- GRANDPARENTS ---
            {"id": "Robert Killebrew", "branch": "K", "steps": 2, "lateral": 0, "desc": "Grandparent (1930 - 2017)", "bio": "Hartshorne, OK; Las Vegas, NV"},
            {"id": "Bonnie Rasmussen", "branch": "R", "steps": 2, "lateral": 0, "desc": "Grandparent (1934 - 2020)", "bio": "Monroe, UT"},
            {"id": "John O. Vanderhoop", "branch": "V", "steps": 2, "lateral": 0, "desc": "Grandparent (1934 - 2022)", "bio": "<b>Born:</b> July 15, 1934, Gay Head, MA.<br>Proud Wampanoag tribe member. <b>Military:</b> Retired Major USAF (Germany, Thailand, Vietnam). Bronze Star.<br><a href='#' target='_blank'>[View Obituary]</a>"},
            {"id": "Waltrud M. Lieber", "branch": "L", "steps": 2, "lateral": 0, "desc": "Grandparent (1934 - 2022)", "bio": "Met John in Hofgeismar, Germany in 1961."},
            {"id": "Peter Buzunis", "branch": "B", "steps": 2, "lateral": 0, "desc": "Grandparent (1917 - 2007)", "bio": "Vanguard, SK, Canada; Winnipeg, MB."},
            {"id": "Anastasia Ginakes", "branch": "A", "steps": 2, "lateral": 0, "desc": "Grandparent (1925 - 2018)", "bio": "Fargo, ND; Winnipeg, MB."},
            
            # --- KILLEBREW BRANCH (K) ---
            {"id": "William H. K.", "branch": "K", "steps": 3, "lateral": 0, "desc": "Great Grandparent (1898 - 1970)"},
            {"id": "Mary Esther Robinson", "branch": "K", "steps": 3, "inLaw": True, "anchorStep": 3, "desc": "Great Grandmother (1902-1981)"},
            {"id": "Daniel Boone K.", "branch": "K", "steps": 4, "lateral": 0, "desc": "Great Great Grandparent (1860 - 1939)"},
            {"id": "George W. K.", "branch": "K", "steps": 5, "lateral": 0, "desc": "3x Great Grandparent (1812 - 1871)", "bio": "Buried: Osburn-Killebrew Cemetery."},
            {"id": "Whitfield K.", "branch": "K", "steps": 6, "lateral": 0, "desc": "4x Great Grandparent (1793 - 1859)"},
            {"id": "Joseph K.", "branch": "K", "steps": 7, "lateral": 0, "desc": "5x Great Grandparent (1753 - 1824)"},
            {"id": "Francis K.", "branch": "K", "steps": 8, "lateral": 0, "desc": "6x Great Grandparent (1619 - 1673)"},
            
            # --- ROBINSON / IMPSON SPOUSAL OFFSHOOTS ---
            {"id": "James Wesley Robinson", "branch": "K", "steps": 4, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "2x Great Grandparent (1847-1916)", "bio": "Civil War Veteran. Married Jane 'Jincey' Impson, became Choctaw citizen."},
            {"id": "Jane Jincey Impson", "branch": "K", "steps": 4, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "2x Great Grandparent (1862-1940)"},
            {"id": "Niel C. Robinson", "branch": "K", "steps": 5, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "3x Great Grandparent (d. 1864)", "bio": "<b>Military:</b> Union Soldier, Kansas 2nd Cavalry. Died of wounds in action at Roseville, AR."},
            {"id": "Huldah Jennie Wood", "branch": "K", "steps": 5, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "3x Great Grandparent (1830-1880)"},
            {"id": "Neal Clark Robeson", "branch": "K", "steps": 6, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "4x Great Grandparent (1785-1836)"},
            {"id": "Ileyvina Robinson", "branch": "K", "steps": 6, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "4x Great Grandparent (1788-1868)"},
            {"id": "Neal Clark Robeson Sr.", "branch": "K", "steps": 7, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "5x Great Grandparent (1760-1841)"},
            {"id": "Josiah Impson", "branch": "K", "steps": 5, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "3x Great Grandparent (1824-1896)", "bio": "Survived the 1833 'Trail of Tears' to the Choctaw Nation."},
            {"id": "Isaac Impson", "branch": "K", "steps": 6, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "4x Great Grandparent (1800-1899)", "bio": "English descent. Married into the Choctaw tribe."},
            {"id": "John Adam Josiah Impson", "branch": "K", "steps": 7, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "5x Great Grandparent (1745-1833)"},
            {"id": "John Adam Impson", "branch": "K", "steps": 8, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "6x Great Grandparent (1718-?)"},
            {"id": "William John Impson", "branch": "K", "steps": 9, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "7x Great Grandparent (1700-?)"},

            # Killebrew Laterals
            {"id": "Ron K.", "branch": "K", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle"},
            {"id": "Urma K.", "branch": "K", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle"},
            {"id": "Kelly K.", "branch": "K", "steps": 1, "lateral": 1, "desc": "Aunt / Uncle"},
            {"id": "Stephen K.", "branch": "K", "steps": 1, "lateral": 1, "desc": "Aunt / Uncle"},
            {"id": "Iman Killebrew", "branch": "K", "steps": 1, "lateral": 1, "inLaw": True},
            {"id": "Alec K.", "branch": "K", "steps": 0, "lateral": 2, "desc": "First Cousin"},
            {"id": "Gillan K.", "branch": "K", "steps": 0, "lateral": 2, "desc": "First Cousin"},
            {"id": "Jayce K.", "branch": "K", "steps": 0, "lateral": 2, "desc": "First Cousin"},
            {"id": "Suzie K.", "branch": "K", "steps": 1, "lateral": 1, "desc": "Aunt / Uncle"},
            {"id": "Steve Gerphy", "branch": "K", "steps": 1, "lateral": 1, "inLaw": True},
            {"id": "Matt Gerphy", "branch": "K", "steps": 0, "lateral": 2, "desc": "First Cousin"},
            {"id": "Francesca", "branch": "K", "steps": 0, "lateral": 2, "inLaw": True},
            {"id": "Amelia Gerphy", "branch": "K", "steps": -1, "lateral": 3, "desc": "First Cousin Once Removed"},
            {"id": "Riley Gerphy", "branch": "K", "steps": -1, "lateral": 3, "desc": "First Cousin Once Removed"},
            {"id": "Oliver Gerphy", "branch": "K", "steps": -1, "lateral": 3, "desc": "First Cousin Once Removed"},
            {"id": "Chris Gerphy", "branch": "K", "steps": 0, "lateral": 2, "desc": "First Cousin"},
            {"id": "Tony K.", "branch": "K", "steps": 1, "lateral": 1, "desc": "Aunt / Uncle"},
            {"id": "Martha", "branch": "K", "steps": 1, "lateral": 1, "inLaw": True},
            {"id": "Will K.", "branch": "K", "steps": 0, "lateral": 2, "desc": "First Cousin"},
            {"id": "Ben K.", "branch": "K", "steps": 0, "lateral": 2, "desc": "First Cousin"},
            {"id": "Keri K.", "branch": "K", "steps": 1, "lateral": 1, "desc": "Aunt / Uncle"},
            {"id": "Stacy", "branch": "K", "steps": 1, "lateral": 1, "inLaw": True},
            {"id": "Sheri K.", "branch": "K", "steps": 1, "lateral": 1, "desc": "Aunt / Uncle"},
            {"id": "Ray Snisky", "branch": "K", "steps": 1, "lateral": 1, "inLaw": True},
            {"id": "Ellie Snisky", "branch": "K", "steps": 0, "lateral": 2, "desc": "First Cousin"},
            {"id": "Ava Snisky", "branch": "K", "steps": 0, "lateral": 2, "desc": "First Cousin"},

            # --- RASMUSSEN BRANCH (R) ---
            {"id": "Clinton Rasmussen", "branch": "R", "steps": 3, "lateral": 0, "desc": "Great Grandparent (1904 - 1979)"},
            {"id": "James A. R.", "branch": "R", "steps": 4, "lateral": 0, "desc": "Great Great Grandparent (1877 - 1965)"},
            {"id": "Rasmus J. R.", "branch": "R", "steps": 5, "lateral": 0, "desc": "3x Great Grandparent (1842 - 1920)", "bio": "Served in Utah Territorial Militia during the Black Hawk War."},
            {"id": "Jens Rasmussen", "branch": "R", "steps": 6, "lateral": 0, "desc": "4x Great Grandparent (1810 - 1888)", "bio": "Immigrated from Denmark. Weaver. Wife Maren died of Cholera en route to Utah in 1866."},
            
            # Rasmussen Laterals
            {"id": "Richard R.", "branch": "R", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle"},
            {"id": "Bettie R.", "branch": "R", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle"},
            {"id": "Rhett R.", "branch": "R", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle"},
            {"id": "Oranell", "branch": "R", "steps": 2, "lateral": 1, "inLaw": True},
            {"id": "James", "branch": "R", "steps": 1, "lateral": 2, "desc": "First Cousin Once Removed"},
            {"id": "Rosemary", "branch": "R", "steps": 1, "lateral": 2, "desc": "First Cousin Once Removed"},
            {"id": "Ruth", "branch": "R", "steps": 1, "lateral": 2, "desc": "First Cousin Once Removed"},
            {"id": "Karen", "branch": "R", "steps": 1, "lateral": 2, "desc": "First Cousin Once Removed"},
            {"id": "Bob", "branch": "R", "steps": 2, "lateral": 1, "inLaw": True},
            {"id": "Michelle", "branch": "R", "steps": 1, "lateral": 2, "desc": "First Cousin Once Removed"},

            # --- VANDERHOOP BRANCH (V) ---
            {"id": "Leonard V.", "branch": "V", "steps": 3, "lateral": 0, "desc": "Great Grandparent (1895 - 1989)", "bio": "Gay Head, Dukes, MA.<br><a href='#' target='_blank'>[Wampanoag Recording: The Cranberry Hunt]</a>"},
            {"id": "Edwin DeVries V.", "branch": "V", "steps": 4, "lateral": 0, "desc": "Great Great Grandparent (1848 - 1923)", "bio": "Served in the Navy on the gunboat Maheska during the Civil War (Union)."},
            {"id": "William A. V.", "branch": "V", "steps": 5, "lateral": 0, "desc": "3x Great Grandparent (~1816 - 1893)", "bio": "Dutch Surinamese Immigrant from Paramaribo. Built the Vanderhoop homestead in Gay Head."},
            {"id": "Beulah Salisbury", "branch": "V", "steps": 5, "inLaw": True, "anchorStep": 5, "desc": "3x Great Grandmother (1814-1892)", "bio": "Known as the 'Princess of Aquinnah'. Assisted the Underground Railroad by hiding escaped slaves under a false floor in her family's barn."},
            
            # --- VANDERHOOP ANCESTRY EXPANSION ---
            {"id": "Baron Adriaan vanDel Vanderhoop Sr.", "branch": "V", "steps": 6, "lateral": 0, "desc": "4x Great Grandparent (1778-1854)", "bio": "Amsterdam, Netherlands; Santpoort Estate."},
            {"id": "Anthonia Immerentia Weveringh", "branch": "V", "steps": 6, "lateral": 0, "inLaw": True, "desc": "4x Great Grandmother (1775-1832)"},
            {"id": "Joan Cornelis Vanderhoop", "branch": "V", "steps": 7, "lateral": 0, "desc": "5x Great Grandparent (1742-1825)", "bio": "The Hague, South Holland, Netherlands."},
            {"id": "Agnes Maria Dedel", "branch": "V", "steps": 7, "lateral": 0, "inLaw": True, "desc": "5x Great Grandmother (1742-1825)"},
            {"id": "Baron Adriaan Vanderhoop I", "branch": "V", "steps": 8, "lateral": 0, "desc": "6x Great Grandparent (1701-1767)", "bio": "'s-Gravenhage; Naaldwijk, Netherlands."},
            {"id": "Susana Sophia Dedel", "branch": "V", "steps": 8, "lateral": 0, "inLaw": True, "desc": "6x Great Grandmother (1708-1796)"},
            {"id": "Baron Francois Adrien Vanderhoop", "branch": "V", "steps": 9, "lateral": 0, "desc": "7x Great Grandparent (1675-1741)", "bio": "Born in Batavia, Indonesia; died in 's-Gravenhage, Netherlands."},
            
            # --- SALISBURY SPOUSAL OFFSHOOTS ---
            {"id": "John Salisbury", "branch": "V", "steps": 6, "lateral": 1, "isSpouseLine": True, "anchorStep": 5, "desc": "4x Great Grandparent (~1792-~1870)", "bio": "Gay Head, MA."},
            {"id": "Naomi Occouch Salisbury", "branch": "V", "steps": 6, "lateral": 1, "isSpouseLine": True, "anchorStep": 5, "desc": "4x Great Grandparent (1788-?)", "bio": "Gay Head, MA."},

            # --- DIAMOND/SMALLEY SPOUSAL OFFSHOOTS ---
            {"id": "Baroness Elsie Ester (Diamond) Vanderhoop", "branch": "V", "steps": 3, "inLaw": True, "anchorStep": 3, "desc": "Great Grandmother (1899-1936)", "bio": "Invested Posthumously as Baroness Consort van der Hoop of the Netherlands by Royal Proclamation issued by Queen Beatrix."},
            {"id": "Durwood Delmond Diamond", "branch": "V", "steps": 4, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "2x Great Grandparent (1878-1947)", "bio": "Served in Military in 1923, St. Louis, MI."},
            {"id": "Elizabeth E. Smalley", "branch": "V", "steps": 4, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "2x Great Grandmother (1880-?)"},
            {"id": "Samuel Smalley", "branch": "V", "steps": 5, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "3x Great Grandparent (1829-1893)", "bio": "Born in New York, migrated to Gay Head."},
            {"id": "Julia Bassett Smalley", "branch": "V", "steps": 5, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "3x Great Grandmother (1838-1888)", "bio": "Owned cottage at 19 Dukes County Ave."},
            {"id": "John Smalley", "branch": "V", "steps": 6, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "4x Great Grandparent", "bio": "New York, USA."},
            {"id": "Amos Peters Smalley", "branch": "V", "steps": 4, "lateral": 2, "isSpouseLine": True, "anchorStep": 3, "desc": "Great Grand Uncle (1877-1961)", "bio": "Legendary Gay Head harpooner credited with being the only person ever to kill a white whale."},
            {"id": "Leander Bassett", "branch": "V", "steps": 6, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "4x Great Grandparent (1810-1879)", "bio": "From Edgartown."},
            {"id": "Huldah Jeffers", "branch": "V", "steps": 6, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "4x Great Grandmother (1807-1879)", "bio": "Aquinnah Wampanoag native."},
            {"id": "James Diamond", "branch": "V", "steps": 5, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "3x Great Grandparent (1820-?)", "bio": "Born in New York. Moved to Aquinnah in 1870."},
            {"id": "Abiah Manning", "branch": "V", "steps": 5, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "3x Great Grandmother (1821-1906)"},
            {"id": "Rosetta Ellis Diamond", "branch": "V", "steps": 4, "lateral": 2, "isSpouseLine": True, "anchorStep": 3, "desc": "Great Grand Aunt (1862-1922)"},
            {"id": "Raymond Frances Madison", "branch": "V", "steps": 4, "lateral": 2, "isSpouseLine": True, "anchorStep": 3, "inLaw": True, "desc": "Spouse (1858-1942)"},

            # Vanderhoop Laterals
            {"id": "Johnny Vanderhoop", "branch": "V", "steps": 1, "lateral": 1, "desc": "Aunt / Uncle"},
            {"id": "Leonard Jr. V.", "branch": "V", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle"},
            {"id": "William (Billy) V.", "branch": "V", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle"},
            {"id": "Edmund V.", "branch": "V", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle"},
            {"id": "Margery V.", "branch": "V", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle"},
            {"id": "Paul V.", "branch": "V", "steps": 1, "lateral": 2, "desc": "First Cousin Once Removed"},
            {"id": "Polly V.", "branch": "V", "steps": 1, "lateral": 2, "desc": "First Cousin Once Removed"},
            {"id": "Paul's Wife", "branch": "V", "steps": 1, "lateral": 2, "inLaw": True},
            {"id": "Maushup V.", "branch": "V", "steps": 0, "lateral": 3, "desc": "Second Cousin"},
            {"id": "Nashawn V.", "branch": "V", "steps": 0, "lateral": 3, "desc": "Second Cousin"},

            # --- LIEBER BRANCH (L) ---
            {"id": "G. Heinrich L. Lieber", "branch": "L", "steps": 3, "lateral": 0, "desc": "Great Grandparent"},
            {"id": "Marie Emilie Ibe", "branch": "L", "steps": 3, "lateral": 0, "inLaw": True},
            {"id": "Manfred Lieber", "branch": "L", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle"},

            # --- BUZUNIS BRANCH (B) ---
            {"id": "George Constantine Buzunis", "branch": "B", "steps": 4, "lateral": 0, "desc": "Great Great Grandparent (1846 - 1912)", "bio": "Levidion, Greece"},
            {"id": "Elenis 'Helen' Georgakopoulis", "branch": "B", "steps": 4, "lateral": 0, "inLaw": True, "desc": "Great Great Grandmother (1867-1912)"},
            {"id": "Theodore Buzunis", "branch": "B", "steps": 3, "lateral": 0, "desc": "Great Grandparent (1885 - 1978)", "bio": "Tripoli, Peloponnese, Greece; Winnipeg, MB"},
            {"id": "Constantina Colilos", "branch": "B", "steps": 3, "lateral": 0, "inLaw": True, "desc": "Great Grandmother (1887-1963)"},
            
            # Buzunis Laterals
            {"id": "William Buzunis", "branch": "B", "steps": 3, "lateral": 1, "desc": "Great Grand Aunt / Uncle"},
            {"id": "Alexander Buzunis", "branch": "B", "steps": 3, "lateral": 1, "desc": "Great Grand Aunt / Uncle"},
            {"id": "Nickolas Buzunis", "branch": "B", "steps": 3, "lateral": 1, "desc": "Great Grand Aunt / Uncle"},
            {"id": "Christopher Buzunis", "branch": "B", "steps": 3, "lateral": 1, "desc": "Great Grand Aunt / Uncle"},
            {"id": "Helen Buzunis", "branch": "B", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle (1911 - 1971)"},
            {"id": "James Effos", "branch": "B", "steps": 2, "lateral": 1, "inLaw": True},
            {"id": "George Buzunis", "branch": "B", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle (1914 - 1988)"},
            {"id": "Christina Buzunis", "branch": "B", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle (1915 - 2005)"},
            {"id": "Teddy Buzunis", "branch": "B", "steps": 1, "lateral": 1, "desc": "Aunt / Uncle"},
            {"id": "Victoria Buzunis", "branch": "B", "steps": 0, "lateral": 2, "desc": "First Cousin"},
            {"id": "Demo Buzunis", "branch": "B", "steps": 1, "lateral": 1, "desc": "Aunt / Uncle"},
            {"id": "Andrea Buzunis", "branch": "B", "steps": 1, "lateral": 1, "desc": "Aunt / Uncle"},
            {"id": "Timothy Buzunis", "branch": "B", "steps": 1, "lateral": 1, "desc": "Aunt / Uncle"},
            {"id": "Kerry", "branch": "B", "steps": 1, "lateral": 1, "inLaw": True},

            # --- GINAKES BRANCH (A) ---
            {"id": "Unknown Ginakes", "branch": "A", "steps": 4, "lateral": 0, "desc": "Great Great Grandparent"},
            {"id": "Andrew Demetrius Ginakes", "branch": "A", "steps": 3, "lateral": 0, "desc": "Great Grandparent (1895 - 1967)", "bio": "Greece; Fargo, ND"},
            {"id": "Arcondo A. Boosalis", "branch": "A", "steps": 3, "lateral": 0, "inLaw": True, "desc": "Great Grandmother (1895-1984)"},
            
            # Ginakes Laterals
            {"id": "Constantine D Ginakes", "branch": "A", "steps": 3, "lateral": 1, "desc": "Great Grand Aunt / Uncle (1908 - 1965)"},
            {"id": "Anastasia Mirras", "branch": "A", "steps": 3, "lateral": 1, "inLaw": True},
            {"id": "Nicholas Ginakes", "branch": "A", "steps": 3, "lateral": 1, "desc": "Great Grand Aunt / Uncle"},
            {"id": "John Ginakes", "branch": "A", "steps": 3, "lateral": 1, "desc": "Great Grand Aunt / Uncle"},
            {"id": "Desmos Andrew Ginakes", "branch": "A", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle (1923 - 1996)"},
            {"id": "Mary Ginakes", "branch": "A", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle (1928 - 2013)"}
        ],
        "links": [
            # Core Lines
            {"source": "Eric Killebrew", "target": "Kyle Killebrew", "type": "main"},
            {"source": "Christina Vanderhoop", "target": "Kyle Killebrew", "type": "main"},
            {"source": "Robert Killebrew", "target": "Eric Killebrew", "type": "main"},
            {"source": "Bonnie Rasmussen", "target": "Eric Killebrew", "type": "main"},
            {"source": "John O. Vanderhoop", "target": "Christina Vanderhoop", "type": "main"},
            {"source": "Waltrud M. Lieber", "target": "Christina Vanderhoop", "type": "main"},
            {"source": "Peter Buzunis", "target": "Antonia Constance Buzunis", "type": "main"},
            {"source": "Anastasia Ginakes", "target": "Antonia Constance Buzunis", "type": "main"},
            {"source": "Theodore Buzunis", "target": "Peter Buzunis", "type": "main"},
            {"source": "George Constantine Buzunis", "target": "Theodore Buzunis", "type": "main"},
            {"source": "Andrew Demetrius Ginakes", "target": "Anastasia Ginakes", "type": "main"},
            {"source": "Unknown Ginakes", "target": "Andrew Demetrius Ginakes", "type": "main"},
            
            # Vanderhoop Main Expansion
            {"source": "William A. V.", "target": "Baron Adriaan vanDel Vanderhoop Sr.", "type": "main"},
            {"source": "Baron Adriaan vanDel Vanderhoop Sr.", "target": "Joan Cornelis Vanderhoop", "type": "main"},
            {"source": "Joan Cornelis Vanderhoop", "target": "Baron Adriaan Vanderhoop I", "type": "main"},
            {"source": "Baron Adriaan Vanderhoop I", "target": "Baron Francois Adrien Vanderhoop", "type": "main"},

            # Marriages
            {"source": "Eric Killebrew", "target": "Christina Vanderhoop", "type": "marriage"},
            {"source": "Eric Killebrew", "target": "Antonia Constance Buzunis", "type": "marriage"},
            {"source": "Robert Killebrew", "target": "Bonnie Rasmussen", "type": "marriage"},
            {"source": "John O. Vanderhoop", "target": "Waltrud M. Lieber", "type": "marriage"},
            {"source": "Leonard V.", "target": "Baroness Elsie Ester (Diamond) Vanderhoop", "type": "marriage"},
            {"source": "Peter Buzunis", "target": "Anastasia Ginakes", "type": "marriage"},
            {"source": "Theodore Buzunis", "target": "Constantina Colilos", "type": "marriage"},
            {"source": "George Constantine Buzunis", "target": "Elenis 'Helen' Georgakopoulis", "type": "marriage"},
            {"source": "Andrew Demetrius Ginakes", "target": "Arcondo A. Boosalis", "type": "marriage"},
            {"source": "Constantine D Ginakes", "target": "Anastasia Mirras", "type": "marriage"},
            {"source": "Antonia Constance Buzunis", "target": "Kyle Killebrew", "type": "inlaw"},
            
            # Vanderhoop Baron Marriages
            {"source": "Baron Adriaan vanDel Vanderhoop Sr.", "target": "Anthonia Immerentia Weveringh", "type": "marriage"},
            {"source": "Joan Cornelis Vanderhoop", "target": "Agnes Maria Dedel", "type": "marriage"},
            {"source": "Baron Adriaan Vanderhoop I", "target": "Susana Sophia Dedel", "type": "marriage"},

            # Robinson/Impson Spouse Line
            {"source": "William H. K.", "target": "Mary Esther Robinson", "type": "marriage"},
            {"source": "Mary Esther Robinson", "target": "James Wesley Robinson", "type": "spouse_main"},
            {"source": "Mary Esther Robinson", "target": "Jane Jincey Impson", "type": "spouse_main"},
            {"source": "James Wesley Robinson", "target": "Niel C. Robinson", "type": "spouse_main"},
            {"source": "James Wesley Robinson", "target": "Huldah Jennie Wood", "type": "spouse_main"},
            {"source": "Niel C. Robinson", "target": "Neal Clark Robeson", "type": "spouse_main"},
            {"source": "Niel C. Robinson", "target": "Ileyvina Robinson", "type": "spouse_main"},
            {"source": "Neal Clark Robeson", "target": "Neal Clark Robeson Sr.", "type": "spouse_main"},
            {"source": "Jane Jincey Impson", "target": "Josiah Impson", "type": "spouse_main"},
            {"source": "Josiah Impson", "target": "Isaac Impson", "type": "spouse_main"},
            {"source": "Isaac Impson", "target": "John Adam Josiah Impson", "type": "spouse_main"},
            {"source": "John Adam Josiah Impson", "target": "John Adam Impson", "type": "spouse_main"},
            {"source": "John Adam Impson", "target": "William John Impson", "type": "spouse_main"},
            
            # Salisbury Spouse Line
            {"source": "Beulah Salisbury", "target": "John Salisbury", "type": "spouse_main"},
            {"source": "Beulah Salisbury", "target": "Naomi Occouch Salisbury", "type": "spouse_main"},

            # Diamond / Smalley Spouse Line
            {"source": "Baroness Elsie Ester (Diamond) Vanderhoop", "target": "Durwood Delmond Diamond", "type": "spouse_main"},
            {"source": "Baroness Elsie Ester (Diamond) Vanderhoop", "target": "Elizabeth E. Smalley", "type": "spouse_main"},
            {"source": "Elizabeth E. Smalley", "target": "Samuel Smalley", "type": "spouse_main"},
            {"source": "Elizabeth E. Smalley", "target": "Julia Bassett Smalley", "type": "spouse_main"},
            {"source": "Samuel Smalley", "target": "John Smalley", "type": "spouse_main"},
            {"source": "Julia Bassett Smalley", "target": "Leander Bassett", "type": "spouse_main"},
            {"source": "Julia Bassett Smalley", "target": "Huldah Jeffers", "type": "spouse_main"},
            {"source": "Durwood Delmond Diamond", "target": "James Diamond", "type": "spouse_main"},
            {"source": "Durwood Delmond Diamond", "target": "Abiah Manning", "type": "spouse_main"},
            {"source": "James Diamond", "target": "Rosetta Ellis Diamond", "type": "leaf"},
            {"source": "Rosetta Ellis Diamond", "target": "Raymond Frances Madison", "type": "inlaw"},
            {"source": "Samuel Smalley", "target": "Amos Peters Smalley", "type": "leaf"},

            # Laterals (Killebrew/Rasmussen/Vanderhoop/Lieber/Buzunis/Ginakes)
            {"source": "Eric Killebrew", "target": "Andrea Nicole Killebrew", "type": "leaf"},
            {"source": "Eric Killebrew", "target": "Eric Scott Killebrew, Jr.", "type": "leaf"},
            {"source": "William H. K.", "target": "Robert Killebrew", "type": "main"},
            {"source": "Daniel Boone K.", "target": "William H. K.", "type": "main"},
            {"source": "George W. K.", "target": "Daniel Boone K.", "type": "main"},
            {"source": "Whitfield K.", "target": "George W. K.", "type": "main"},
            {"source": "Joseph K.", "target": "Whitfield K.", "type": "main"},
            {"source": "Francis K.", "target": "Joseph K.", "type": "main"},
            {"source": "William H. K.", "target": "Ron K.", "type": "leaf"},
            {"source": "William H. K.", "target": "Urma K.", "type": "leaf"},
            {"source": "Robert Killebrew", "target": "Kelly K.", "type": "leaf"},
            {"source": "Robert Killebrew", "target": "Stephen K.", "type": "leaf"},
            {"source": "Stephen K.", "target": "Iman Killebrew", "type": "inlaw"},
            {"source": "Stephen K.", "target": "Alec K.", "type": "leaf"},
            {"source": "Stephen K.", "target": "Gillan K.", "type": "leaf"},
            {"source": "Stephen K.", "target": "Jayce K.", "type": "leaf"},
            {"source": "Robert Killebrew", "target": "Suzie K.", "type": "leaf"},
            {"source": "Suzie K.", "target": "Steve Gerphy", "type": "inlaw"},
            {"source": "Suzie K.", "target": "Matt Gerphy", "type": "leaf"},
            {"source": "Suzie K.", "target": "Chris Gerphy", "type": "leaf"},
            {"source": "Matt Gerphy", "target": "Francesca", "type": "inlaw"},
            {"source": "Matt Gerphy", "target": "Amelia Gerphy", "type": "leaf"},
            {"source": "Matt Gerphy", "target": "Riley Gerphy", "type": "leaf"},
            {"source": "Matt Gerphy", "target": "Oliver Gerphy", "type": "leaf"},
            {"source": "Robert Killebrew", "target": "Tony K.", "type": "leaf"},
            {"source": "Tony K.", "target": "Martha", "type": "inlaw"},
            {"source": "Tony K.", "target": "Will K.", "type": "leaf"},
            {"source": "Tony K.", "target": "Ben K.", "type": "leaf"},
            {"source": "Robert Killebrew", "target": "Keri K.", "type": "leaf"},
            {"source": "Keri K.", "target": "Stacy", "type": "inlaw"},
            {"source": "Robert Killebrew", "target": "Sheri K.", "type": "leaf"},
            {"source": "Sheri K.", "target": "Ray Snisky", "type": "inlaw"},
            {"source": "Sheri K.", "target": "Ellie Snisky", "type": "leaf"},
            {"source": "Sheri K.", "target": "Ava Snisky", "type": "leaf"},
            {"source": "Clinton Rasmussen", "target": "Bonnie Rasmussen", "type": "main"},
            {"source": "James A. R.", "target": "Clinton Rasmussen", "type": "main"},
            {"source": "Rasmus J. R.", "target": "James A. R.", "type": "main"},
            {"source": "Jens Rasmussen", "target": "Rasmus J. R.", "type": "main"},
            {"source": "Clinton Rasmussen", "target": "Richard R.", "type": "leaf"},
            {"source": "Clinton Rasmussen", "target": "Bettie R.", "type": "leaf"},
            {"source": "Clinton Rasmussen", "target": "Rhett R.", "type": "leaf"},
            {"source": "Richard R.", "target": "Oranell", "type": "inlaw"},
            {"source": "Richard R.", "target": "James", "type": "leaf"},
            {"source": "Richard R.", "target": "Rosemary", "type": "leaf"},
            {"source": "Richard R.", "target": "Ruth", "type": "leaf"},
            {"source": "Richard R.", "target": "Karen", "type": "leaf"},
            {"source": "Bettie R.", "target": "Bob", "type": "inlaw"},
            {"source": "Bettie R.", "target": "Michelle", "type": "leaf"},
            {"source": "Leonard V.", "target": "John O. Vanderhoop", "type": "main"},
            {"source": "Edwin DeVries V.", "target": "Leonard V.", "type": "main"},
            {"source": "William A. V.", "target": "Edwin DeVries V.", "type": "main"},
            {"source": "William A. V.", "target": "Beulah Salisbury", "type": "inlaw"},
            {"source": "John O. Vanderhoop", "target": "Johnny Vanderhoop", "type": "leaf"},
            {"source": "Leonard V.", "target": "Leonard Jr. V.", "type": "leaf"},
            {"source": "Leonard V.", "target": "William (Billy) V.", "type": "leaf"},
            {"source": "Leonard V.", "target": "Edmund V.", "type": "leaf"},
            {"source": "Leonard V.", "target": "Margery V.", "type": "leaf"},
            {"source": "Margery V.", "target": "Paul V.", "type": "leaf"},
            {"source": "Margery V.", "target": "Polly V.", "type": "leaf"},
            {"source": "Paul V.", "target": "Paul's Wife", "type": "inlaw"},
            {"source": "Paul V.", "target": "Maushup V.", "type": "leaf"},
            {"source": "Paul V.", "target": "Nashawn V.", "type": "leaf"},
            {"source": "G. Heinrich L. Lieber", "target": "Waltrud M. Lieber", "type": "main"},
            {"source": "G. Heinrich L. Lieber", "target": "Marie Emilie Ibe", "type": "inlaw"},
            {"source": "G. Heinrich L. Lieber", "target": "Manfred Lieber", "type": "leaf"},
            {"source": "George Constantine Buzunis", "target": "William Buzunis", "type": "leaf"},
            {"source": "George Constantine Buzunis", "target": "Alexander Buzunis", "type": "leaf"},
            {"source": "George Constantine Buzunis", "target": "Nickolas Buzunis", "type": "leaf"},
            {"source": "George Constantine Buzunis", "target": "Christopher Buzunis", "type": "leaf"},
            {"source": "Theodore Buzunis", "target": "Helen Buzunis", "type": "leaf"},
            {"source": "Theodore Buzunis", "target": "George Buzunis", "type": "leaf"},
            {"source": "Theodore Buzunis", "target": "Christina Buzunis", "type": "leaf"},
            {"source": "Helen Buzunis", "target": "James Effos", "type": "inlaw"},
            {"source": "Peter Buzunis", "target": "Teddy Buzunis", "type": "leaf"},
            {"source": "Peter Buzunis", "target": "Demo Buzunis", "type": "leaf"},
            {"source": "Peter Buzunis", "target": "Andrea Buzunis", "type": "leaf"},
            {"source": "Peter Buzunis", "target": "Timothy Buzunis", "type": "leaf"},
            {"source": "Teddy Buzunis", "target": "Victoria Buzunis", "type": "leaf"},
            {"source": "Timothy Buzunis", "target": "Kerry", "type": "inlaw"},
            {"source": "Unknown Ginakes", "target": "Constantine D Ginakes", "type": "leaf"},
            {"source": "Unknown Ginakes", "target": "Nicholas Ginakes", "type": "leaf"},
            {"source": "Unknown Ginakes", "target": "John Ginakes", "type": "leaf"},
            {"source": "Andrew Demetrius Ginakes", "target": "Desmos Andrew Ginakes", "type": "leaf"},
            {"source": "Andrew Demetrius Ginakes", "target": "Mary Ginakes", "type": "leaf"}
        ]
    }

    # The hierarchical format required for D3.tree()
    tree_data = {
        "name": "Kyle Killebrew", "year": 1990, "branch": "M", "steps": 0, "lateral": 0, "desc": "You (Present)",
        "children": [
            {
                "name": "Eric Killebrew", "year": 1961, "branch": "K", "steps": 1, "lateral": 0, "desc": "Parent",
                "children": [
                    {"name": "Robert Killebrew", "year": 1930, "branch": "K", "steps": 2, "lateral": 0, "desc": "Grandparent", "children": [
                        {"name": "William H. K.", "year": 1898, "branch": "K", "steps": 3, "lateral": 0, "desc": "Great Grandparent", "children": [
                            {"name": "Mary Esther Robinson", "year": 1902, "branch": "K", "steps": 3, "inLaw": True, "anchorStep": 3, "desc": "Great Grandmother", "children": [
                                {"name": "James Wesley Robinson", "year": 1847, "branch": "K", "steps": 4, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "2x Great Grandparent", "children": [
                                    {"name": "Niel C. Robinson", "year": 1820, "branch": "K", "steps": 5, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "3x Great Grandparent", "children": [
                                        {"name": "Neal Clark Robeson", "year": 1785, "branch": "K", "steps": 6, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "4x Great Grandparent", "children": [
                                            {"name": "Neal Clark Robeson Sr.", "year": 1760, "branch": "K", "steps": 7, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "5x Great Grandparent"}
                                        ]},
                                        {"name": "Ileyvina Robinson", "year": 1788, "branch": "K", "steps": 6, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "4x Great Grandparent"}
                                    ]},
                                    {"name": "Huldah Jennie Wood", "year": 1830, "branch": "K", "steps": 5, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "3x Great Grandparent"}
                                ]},
                                {"name": "Jane Jincey Impson", "year": 1862, "branch": "K", "steps": 4, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "2x Great Grandparent", "children": [
                                    {"name": "Josiah Impson", "year": 1824, "branch": "K", "steps": 5, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "3x Great Grandparent", "children": [
                                        {"name": "Isaac Impson", "year": 1800, "branch": "K", "steps": 6, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "4x Great Grandparent", "children": [
                                            {"name": "John Adam Josiah Impson", "year": 1745, "branch": "K", "steps": 7, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "5x Great Grandparent", "children": [
                                                {"name": "John Adam Impson", "year": 1718, "branch": "K", "steps": 8, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "6x Great Grandparent", "children": [
                                                    {"name": "William John Impson", "year": 1700, "branch": "K", "steps": 9, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "7x Great Grandparent"}
                                                ]}
                                            ]}
                                        ]}
                                    ]}
                                ]}
                            ]},
                            {"name": "Daniel Boone K.", "year": 1860, "branch": "K", "steps": 4, "lateral": 0, "desc": "Great Great Grandparent", "children": [
                                {"name": "George W. K.", "year": 1812, "branch": "K", "steps": 5, "lateral": 0, "desc": "3x Great Grandparent", "children": [
                                    {"name": "Whitfield K.", "year": 1793, "branch": "K", "steps": 6, "lateral": 0, "desc": "4x Great Grandparent", "children": [
                                        {"name": "Joseph K.", "year": 1753, "branch": "K", "steps": 7, "lateral": 0, "desc": "5x Great Grandparent", "children": [
                                            {"name": "Francis K.", "year": 1619, "branch": "K", "steps": 8, "lateral": 0, "desc": "6x Great Grandparent"}
                                        ]}
                                    ]}
                                ]}
                            ]},
                            {"name": "Ron K.", "year": 1928, "branch": "K", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle"},
                            {"name": "Urma K.", "year": 1932, "branch": "K", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle"}
                        ]},
                        {"name": "Kelly K.", "year": 1955, "branch": "K", "steps": 1, "lateral": 1, "desc": "Aunt / Uncle"},
                        {"name": "Stephen K.", "year": 1957, "branch": "K", "steps": 1, "lateral": 1, "desc": "Aunt / Uncle", "children": [
                            {"name": "Iman Killebrew", "year": 1957, "branch": "K", "steps": 1, "lateral": 1, "inLaw": True},
                            {"name": "Alec K.", "year": 1985, "branch": "K", "steps": 0, "lateral": 2, "desc": "First Cousin"},
                            {"name": "Gillan K.", "year": 1987, "branch": "K", "steps": 0, "lateral": 2, "desc": "First Cousin"},
                            {"name": "Jayce K.", "year": 1989, "branch": "K", "steps": 0, "lateral": 2, "desc": "First Cousin"}
                        ]},
                        {"name": "Suzie K.", "year": 1959, "branch": "K", "steps": 1, "lateral": 1, "desc": "Aunt / Uncle", "children": [
                            {"name": "Steve Gerphy", "year": 1959, "branch": "K", "steps": 1, "lateral": 1, "inLaw": True},
                            {"name": "Matt Gerphy", "year": 1985, "branch": "K", "steps": 0, "lateral": 2, "desc": "First Cousin", "children": [
                                {"name": "Francesca", "year": 1985, "branch": "K", "steps": 0, "lateral": 2, "inLaw": True},
                                {"name": "Amelia Gerphy", "year": 2010, "branch": "K", "steps": -1, "lateral": 3, "desc": "First Cousin Once Removed"},
                                {"name": "Riley Gerphy", "year": 2012, "branch": "K", "steps": -1, "lateral": 3, "desc": "First Cousin Once Removed"},
                                {"name": "Oliver Gerphy", "year": 2014, "branch": "K", "steps": -1, "lateral": 3, "desc": "First Cousin Once Removed"}
                            ]},
                            {"name": "Chris Gerphy", "year": 1988, "branch": "K", "steps": 0, "lateral": 2, "desc": "First Cousin"}
                        ]},
                        {"name": "Tony K.", "year": 1963, "branch": "K", "steps": 1, "lateral": 1, "desc": "Aunt / Uncle", "children": [
                            {"name": "Martha", "year": 1963, "branch": "K", "steps": 1, "lateral": 1, "inLaw": True},
                            {"name": "Will K.", "year": 1990, "branch": "K", "steps": 0, "lateral": 2, "desc": "First Cousin"},
                            {"name": "Ben K.", "year": 1992, "branch": "K", "steps": 0, "lateral": 2, "desc": "First Cousin"}
                        ]},
                        {"name": "Keri K.", "year": 1965, "branch": "K", "steps": 1, "lateral": 1, "desc": "Aunt / Uncle", "children": [
                            {"name": "Stacy", "year": 1965, "branch": "K", "steps": 1, "lateral": 1, "inLaw": True}
                        ]},
                        {"name": "Sheri K.", "year": 1967, "branch": "K", "steps": 1, "lateral": 1, "desc": "Aunt / Uncle", "children": [
                            {"name": "Ray Snisky", "year": 1965, "branch": "K", "steps": 1, "lateral": 1, "inLaw": True},
                            {"name": "Ellie Snisky", "year": 1995, "branch": "K", "steps": 0, "lateral": 2, "desc": "First Cousin"},
                            {"name": "Ava Snisky", "year": 1998, "branch": "K", "steps": 0, "lateral": 2, "desc": "First Cousin"}
                        ]}
                    ]},
                    {"name": "Bonnie Rasmussen", "year": 1934, "branch": "R", "steps": 2, "lateral": 0, "desc": "Grandparent", "children": [
                        {"name": "Clinton R.", "year": 1904, "branch": "R", "steps": 3, "lateral": 0, "desc": "Great Grandparent", "children": [
                            {"name": "James A. R.", "year": 1877, "branch": "R", "steps": 4, "lateral": 0, "desc": "Great Great Grandparent", "children": [
                                {"name": "Rasmus J. R.", "year": 1842, "branch": "R", "steps": 5, "lateral": 0, "desc": "3x Great Grandparent", "children": [
                                    {"name": "Jens Rasmussen", "year": 1810, "branch": "R", "steps": 6, "lateral": 0, "desc": "4x Great Grandparent"}
                                ]}
                            ]},
                            {"name": "Richard R.", "year": 1932, "branch": "R", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle", "children": [
                                {"name": "Oranell", "year": 1932, "branch": "R", "steps": 2, "lateral": 1, "inLaw": True},
                                {"name": "James", "year": 1955, "branch": "R", "steps": 1, "lateral": 2, "desc": "First Cousin Once Removed"},
                                {"name": "Rosemary", "year": 1957, "branch": "R", "steps": 1, "lateral": 2, "desc": "First Cousin Once Removed"},
                                {"name": "Ruth", "year": 1959, "branch": "R", "steps": 1, "lateral": 2, "desc": "First Cousin Once Removed"},
                                {"name": "Karen", "year": 1961, "branch": "R", "steps": 1, "lateral": 2, "desc": "First Cousin Once Removed"}
                            ]},
                            {"name": "Bettie R.", "year": 1936, "branch": "R", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle", "children": [
                                {"name": "Bob", "year": 1936, "branch": "R", "steps": 2, "lateral": 1, "inLaw": True},
                                {"name": "Michelle", "year": 1960, "branch": "R", "steps": 1, "lateral": 2, "desc": "First Cousin Once Removed"}
                            ]},
                            {"name": "Rhett R.", "year": 1938, "branch": "R", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle"}
                        ]}
                    ]},
                    {"name": "Antonia Constance Buzunis", "year": 1964, "branch": "B", "steps": 1, "lateral": 0, "desc": "Step Mother", "children": [
                        {"name": "Peter Buzunis", "year": 1917, "branch": "B", "steps": 2, "lateral": 0, "desc": "Grandparent", "children": [
                            {"name": "Theodore Buzunis", "year": 1885, "branch": "B", "steps": 3, "lateral": 0, "desc": "Great Grandparent", "children": [
                                {"name": "George Constantine Buzunis", "year": 1846, "branch": "B", "steps": 4, "lateral": 0, "desc": "Great Great Grandparent", "children": [
                                    {"name": "Elenis 'Helen' Georgakopoulis", "year": 1867, "branch": "B", "steps": 4, "lateral": 0, "inLaw": True},
                                    {"name": "William Buzunis", "year": 1880, "branch": "B", "steps": 3, "lateral": 1, "desc": "Great Grand Aunt / Uncle"},
                                    {"name": "Alexander Buzunis", "year": 1882, "branch": "B", "steps": 3, "lateral": 1, "desc": "Great Grand Aunt / Uncle"},
                                    {"name": "Nickolas Buzunis", "year": 1884, "branch": "B", "steps": 3, "lateral": 1, "desc": "Great Grand Aunt / Uncle"},
                                    {"name": "Christopher Buzunis", "year": 1886, "branch": "B", "steps": 3, "lateral": 1, "desc": "Great Grand Aunt / Uncle"}
                                ]},
                                {"name": "Constantina Colilos", "year": 1887, "branch": "B", "steps": 3, "lateral": 0, "inLaw": True},
                                {"name": "Helen Buzunis", "year": 1911, "branch": "B", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle", "children": [
                                    {"name": "James Effos", "year": 1894, "branch": "B", "steps": 2, "lateral": 1, "inLaw": True}
                                ]},
                                {"name": "George Buzunis", "year": 1914, "branch": "B", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle"},
                                {"name": "Christina Buzunis", "year": 1915, "branch": "B", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle"}
                            ]},
                            {"name": "Teddy Buzunis", "year": 1960, "branch": "B", "steps": 1, "lateral": 1, "desc": "Aunt / Uncle", "children": [
                                {"name": "Victoria Buzunis", "year": 1990, "branch": "B", "steps": 0, "lateral": 2, "desc": "First Cousin"}
                            ]},
                            {"name": "Demo Buzunis", "year": 1962, "branch": "B", "steps": 1, "lateral": 1, "desc": "Aunt / Uncle"},
                            {"name": "Andrea Buzunis", "year": 1968, "branch": "B", "steps": 1, "lateral": 1, "desc": "Aunt / Uncle"},
                            {"name": "Timothy Buzunis", "year": 1970, "branch": "B", "steps": 1, "lateral": 1, "desc": "Aunt / Uncle", "children": [
                                {"name": "Kerry", "year": 1970, "branch": "B", "steps": 1, "lateral": 1, "inLaw": True}
                            ]}
                        ]},
                        {"name": "Anastasia Ginakes", "year": 1925, "branch": "A", "steps": 2, "lateral": 0, "desc": "Grandparent", "children": [
                            {"name": "Andrew Demetrius Ginakes", "year": 1895, "branch": "A", "steps": 3, "lateral": 0, "desc": "Great Grandparent", "children": [
                                {"name": "Unknown Ginakes", "year": 1865, "branch": "A", "steps": 4, "lateral": 0, "desc": "Great Great Grandparent", "children": [
                                    {"name": "Constantine D Ginakes", "year": 1908, "branch": "A", "steps": 3, "lateral": 1, "desc": "Great Grand Aunt / Uncle", "children": [
                                        {"name": "Anastasia Mirras", "year": 1910, "branch": "A", "steps": 3, "lateral": 1, "inLaw": True}
                                    ]},
                                    {"name": "Nicholas Ginakes", "year": 1890, "branch": "A", "steps": 3, "lateral": 1, "desc": "Great Grand Aunt / Uncle"},
                                    {"name": "John Ginakes", "year": 1892, "branch": "A", "steps": 3, "lateral": 1, "desc": "Great Grand Aunt / Uncle"}
                                ]},
                                {"name": "Arcondo A. Boosalis", "year": 1895, "branch": "A", "steps": 3, "lateral": 0, "inLaw": True},
                                {"name": "Desmos Andrew Ginakes", "year": 1923, "branch": "A", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle"},
                                {"name": "Mary Ginakes", "year": 1928, "branch": "A", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle"}
                            ]}
                        ]}
                    ]},
                    {"name": "Andrea Nicole Killebrew", "year": 1985, "branch": "K", "steps": 0, "lateral": 1, "desc": "Sister / Brother"},
                    {"name": "Eric Scott Killebrew, Jr.", "year": 1988, "branch": "K", "steps": 0, "lateral": 1, "desc": "Sister / Brother"}
                ]
            },
            {
                "name": "Christina Vanderhoop", "year": 1961, "branch": "V", "steps": 1, "lateral": 0, "desc": "Mother",
                "children": [
                    {"name": "John O. Vanderhoop", "year": 1934, "branch": "V", "steps": 2, "lateral": 0, "desc": "Grandparent", "children": [
                        {"name": "Leonard V.", "year": 1895, "branch": "V", "steps": 3, "lateral": 0, "desc": "Great Grandparent", "children": [
                            {"name": "Baroness Elsie Ester (Diamond) Vanderhoop", "year": 1899, "branch": "V", "steps": 3, "inLaw": True, "anchorStep": 3, "desc": "Great Grandmother", "children": [
                                {"name": "Durwood Delmond Diamond", "year": 1878, "branch": "V", "steps": 4, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "2x Great Grandparent", "children": [
                                    {"name": "James Diamond", "year": 1820, "branch": "V", "steps": 5, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "3x Great Grandparent", "children": [
                                        {"name": "Rosetta Ellis Diamond", "year": 1862, "branch": "V", "steps": 4, "lateral": 2, "isSpouseLine": True, "anchorStep": 3, "desc": "Great Grand Aunt", "children": [
                                            {"name": "Raymond Frances Madison", "year": 1858, "branch": "V", "steps": 4, "lateral": 2, "isSpouseLine": True, "anchorStep": 3, "inLaw": True}
                                        ]}
                                    ]},
                                    {"name": "Abiah Manning", "year": 1821, "branch": "V", "steps": 5, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "3x Great Grandmother"}
                                ]},
                                {"name": "Elizabeth E. Smalley", "year": 1880, "branch": "V", "steps": 4, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "2x Great Grandmother", "children": [
                                    {"name": "Samuel Smalley", "year": 1829, "branch": "V", "steps": 5, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "3x Great Grandparent", "children": [
                                        {"name": "John Smalley", "year": 1800, "branch": "V", "steps": 6, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "4x Great Grandparent"},
                                        {"name": "Amos Peters Smalley", "year": 1877, "branch": "V", "steps": 4, "lateral": 2, "isSpouseLine": True, "anchorStep": 3, "desc": "Great Grand Uncle"}
                                    ]},
                                    {"name": "Julia Bassett Smalley", "year": 1838, "branch": "V", "steps": 5, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "3x Great Grandmother", "children": [
                                        {"name": "Leander Bassett", "year": 1810, "branch": "V", "steps": 6, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "4x Great Grandparent"},
                                        {"name": "Huldah Jeffers", "year": 1807, "branch": "V", "steps": 6, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "4x Great Grandmother"}
                                    ]}
                                ]}
                            ]},
                            {"name": "Edwin DeVries V.", "year": 1848, "branch": "V", "steps": 4, "lateral": 0, "desc": "Great Great Grandparent", "children": [
                                {"name": "William A. V.", "year": 1816, "branch": "V", "steps": 5, "lateral": 0, "desc": "3x Great Grandparent", "children": [
                                    {"name": "Baron Adriaan vanDel Vanderhoop Sr.", "year": 1778, "branch": "V", "steps": 6, "lateral": 0, "desc": "4x Great Grandparent", "children": [
                                        {"name": "Joan Cornelis Vanderhoop", "year": 1742, "branch": "V", "steps": 7, "lateral": 0, "desc": "5x Great Grandparent", "children": [
                                            {"name": "Baron Adriaan Vanderhoop I", "year": 1701, "branch": "V", "steps": 8, "lateral": 0, "desc": "6x Great Grandparent", "children": [
                                                {"name": "Baron Francois Adrien Vanderhoop", "year": 1675, "branch": "V", "steps": 9, "lateral": 0, "desc": "7x Great Grandparent"}
                                            ]},
                                            {"name": "Susana Sophia Dedel", "year": 1708, "branch": "V", "steps": 8, "lateral": 0, "inLaw": True}
                                        ]},
                                        {"name": "Agnes Maria Dedel", "year": 1742, "branch": "V", "steps": 7, "lateral": 0, "inLaw": True}
                                    ]},
                                    {"name": "Anthonia Immerentia Weveringh", "year": 1775, "branch": "V", "steps": 6, "lateral": 0, "inLaw": True},
                                    {"name": "Beulah Salisbury", "year": 1814, "branch": "V", "steps": 5, "inLaw": True, "anchorStep": 5, "desc": "3x Great Grandmother", "children": [
                                        {"name": "John Salisbury", "year": 1792, "branch": "V", "steps": 6, "lateral": 1, "isSpouseLine": True, "anchorStep": 5, "desc": "4x Great Grandparent"},
                                        {"name": "Naomi Occouch Salisbury", "year": 1788, "branch": "V", "steps": 6, "lateral": 1, "isSpouseLine": True, "anchorStep": 5, "desc": "4x Great Grandparent"}
                                    ]}
                                ]}
                            ]},
                            {"name": "Leonard Jr. V.", "year": 1925, "branch": "V", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle"},
                            {"name": "William (Billy) V.", "year": 1928, "branch": "V", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle"},
                            {"name": "Edmund V.", "year": 1930, "branch": "V", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle"},
                            {"name": "Margery V.", "year": 1932, "branch": "V", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle", "children": [
                                {"name": "Paul V.", "year": 1960, "branch": "V", "steps": 1, "lateral": 2, "desc": "First Cousin Once Removed", "children": [
                                    {"name": "Paul's Wife", "year": 1960, "branch": "V", "steps": 1, "lateral": 2, "inLaw": True},
                                    {"name": "Maushup V.", "year": 1985, "branch": "V", "steps": 0, "lateral": 3, "desc": "Second Cousin"},
                                    {"name": "Nashawn V.", "year": 1987, "branch": "V", "steps": 0, "lateral": 3, "desc": "Second Cousin"}
                                ]},
                                {"name": "Polly V.", "year": 1962, "branch": "V", "steps": 1, "lateral": 2, "desc": "First Cousin Once Removed"}
                            ]}
                        ]},
                        {"name": "Johnny Vanderhoop", "year": 1936, "branch": "V", "steps": 1, "lateral": 1, "desc": "Aunt / Uncle"}
                    ]},
                    {"name": "Waltrud M. Lieber", "year": 1934, "branch": "L", "steps": 2, "lateral": 0, "desc": "Grandparent", "children": [
                        {"name": "G. Heinrich L. Lieber", "year": 1900, "branch": "L", "steps": 3, "lateral": 0, "desc": "Great Grandparent", "children": [
                            {"name": "Marie Emilie Ibe", "year": 1900, "branch": "L", "steps": 3, "lateral": 0, "inLaw": True},
                            {"name": "Manfred Lieber", "year": 1932, "branch": "L", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle"}
                        ]}
                    ]}
                ]
            }
        ]
    }

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
                {"city": "Batavia, Indonesia", "lat": -6.2000, "lon": 106.8166, "people": [
                    {"name": "Baron Francois Adrien Vanderhoop", "years": "1675-1741", "desc": "Born in Batavia, Dutch East Indies."}
                ]},
                {"city": "The Hague, Netherlands", "lat": 52.0705, "lon": 4.3007, "people": [
                    {"name": "Baron Francois Adrien Vanderhoop", "years": "d. 1741", "desc": "Returned from Indonesia, passed away here."},
                    {"name": "Baron Adriaan Vanderhoop I", "years": "1701-1767", "desc": "Born in 's-Gravenhage (The Hague)."},
                    {"name": "Joan Cornelis Vanderhoop", "years": "1742-1825", "desc": "Lived entire life in The Hague."}
                ]},
                {"city": "Amsterdam / Santpoort, Netherlands", "lat": 52.4089, "lon": 4.6300, "people": [
                    {"name": "Baron Adriaan vanDel Vanderhoop Sr.", "years": "1778-1854", "desc": "Lived in Amsterdam and Santpoort Estate."}
                ]},
                {"city": "Paramaribo, Suriname", "lat": 5.8520, "lon": -55.2038, "people": [
                    {"name": "William A. Vanderhoop", "years": "b. 1816", "desc": "Dutch Surinamese immigrant."}
                ]},
                {"city": "Gay Head (Aquinnah), MA", "lat": 41.3368, "lon": -70.8316, "people": [
                    {"name": "William A. Vanderhoop", "years": "d. 1893", "desc": "First Vanderhoop on Martha's Vineyard. Built the homestead."},
                    {"name": "Edwin Devries Vanderhoop", "years": "1848-1923", "desc": "<b>Military:</b> Union Navy gunboat Maheska (Civil War)."}
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