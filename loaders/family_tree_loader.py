"""
=============================================================================
MODULE: loaders/family_tree_loader.py
AUTHOR: Kyle W. Killebrew, PhD
DESCRIPTION: 
    Data extraction and hydration script for the Genealogy Web.
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
            // --- CORE / IMMEDIATE FAMILY ---
            {"id": "Kyle Killebrew", "branch": "M", "steps": 0, "lateral": 0, "desc": "You (Present)", "bio": "1990-Present; Las Vegas, NV"},
            {"id": "Eric Killebrew", "branch": "K", "steps": 1, "lateral": 0, "desc": "Parent", "bio": "1961-Present; Las Vegas, NV"},
            {"id": "Christina Vanderhoop", "branch": "V", "steps": 1, "lateral": 0, "desc": "Parent"},
            {"id": "Antonia Constance Buzunis", "branch": "B", "steps": 1, "lateral": 0, "desc": "Step Mother", "bio": "1964-Present; Las Vegas, NV"},
            {"id": "Andrea Nicole Killebrew", "branch": "K", "steps": 0, "lateral": 1, "desc": "Sister / Brother"},
            {"id": "Eric Scott Killebrew, Jr.", "branch": "K", "steps": 0, "lateral": 1, "desc": "Sister / Brother"},

            // --- GRANDPARENTS ---
            {"id": "Robert Killebrew", "branch": "K", "steps": 2, "lateral": 0, "desc": "Grandparent (1930 - 2017)", "bio": "Hartshorne, OK; Las Vegas, NV"},
            {"id": "Bonnie Rasmussen", "branch": "R", "steps": 2, "lateral": 0, "desc": "Grandparent (1934 - 2020)", "bio": "Monroe, UT"},
            {"id": "John O. Vanderhoop", "branch": "V", "steps": 2, "lateral": 0, "desc": "Grandparent (1934 - 2022)", "bio": "<b>Born:</b> July 15, 1934, Gay Head, MA.<br>Proud member of the Wampanoag tribe of Aquinnah. Earned a degree in Classic Literature at Brown University.<br><b>Military:</b> Retired Major USAF (Germany, Thailand, Vietnam). Awarded Bronze Star & Meritorious Service Medal.<br>Married Waltrud 'Gaby' Lieber in 1965.<br><a href='#' target='_blank'>[View Obituary/Pic]</a>"},
            {"id": "Waltrud M. Lieber", "branch": "L", "steps": 2, "lateral": 0, "desc": "Grandparent (1934 - 2022)", "bio": "Met John in Hofgeismar, Germany in 1961. Married 57 years."},
            {"id": "Peter Buzunis", "branch": "B", "steps": 2, "lateral": 0, "desc": "Grandparent (1917 - 2007)", "bio": "Vanguard, SK, Canada; Winnipeg, MB.<br><a href='#' target='_blank'>[Find A Grave]</a>"},
            {"id": "Anastasia Ginakes", "branch": "A", "steps": 2, "lateral": 0, "desc": "Grandparent (1925 - 2018)", "bio": "Fargo, ND; Winnipeg, MB.<br><a href='#' target='_blank'>[Find A Grave]</a>"},
            
            // --- KILLEBREW BRANCH (K) ---
            {"id": "William H. K.", "branch": "K", "steps": 3, "lateral": 0, "desc": "Great Grandparent (1898 - 1970)", "bio": "Montgomery County, TN; Henderson, NV.<br><a href='#' target='_blank'>[Find A Grave]</a>"},
            {"id": "Mary Esther Robinson", "branch": "K", "steps": 3, "inLaw": True, "anchorStep": 3, "desc": "Great Grandmother (1902-1981)", "bio": "Hartshorne, OK; Henderson, NV.<br><a href='#' target='_blank'>[My Heritage / Gravestone]</a>"},
            {"id": "Daniel Boone K.", "branch": "K", "steps": 4, "lateral": 0, "desc": "Great Great Grandparent (1860 - 1939)", "bio": "Christian County, KY; McAlester, OK"},
            {"id": "George W. K.", "branch": "K", "steps": 5, "lateral": 0, "desc": "3x Great Grandparent (1812 - 1871)", "bio": "Montgomery County, TN. Buried: Osburn-Killebrew Cemetery."},
            {"id": "Whitfield K.", "branch": "K", "steps": 6, "lateral": 0, "desc": "4x Great Grandparent (1793 - 1859)", "bio": "Duplin County, NC"},
            {"id": "Joseph K.", "branch": "K", "steps": 7, "lateral": 0, "desc": "5x Great Grandparent (1753 - 1824)", "bio": "Tarboro, NC; Clarksville, TN"},
            {"id": "Francis K.", "branch": "K", "steps": 8, "lateral": 0, "desc": "6x Great Grandparent (1619 - 1673)", "bio": "Cornwall, England; Westmoreland, Virginia"},
            
            // --- ROBINSON / IMPSON SPOSUAL OFFSHOOTS ---
            {"id": "James Wesley Robinson", "branch": "K", "steps": 4, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "2x Great Grandparent (1847-1916)", "bio": "Civil War Veteran. Married Jane 'Jincey' Impson and became Choctaw citizen."},
            {"id": "Jane Jincey Impson", "branch": "K", "steps": 4, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "2x Great Grandparent (1862-1940)"},
            {"id": "Niel C. Robinson", "branch": "K", "steps": 5, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "3x Great Grandparent (d. 1864)", "bio": "<b>Military:</b> Union Soldier, Kansas 2nd Cavalry. Died of wounds received in action at Roseville, AR.<br><a href='#' target='_blank'>[Registry]</a>"},
            {"id": "Huldah Jennie Wood", "branch": "K", "steps": 5, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "3x Great Grandparent (1830-1880)"},
            {"id": "Neal Clark Robeson", "branch": "K", "steps": 6, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "4x Great Grandparent (1785-1836)"},
            {"id": "Ileyvina Robinson", "branch": "K", "steps": 6, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "4x Great Grandparent (1788-1868)"},
            {"id": "Neal Clark Robeson Sr.", "branch": "K", "steps": 7, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "5x Great Grandparent (1760-1841)"},
            {"id": "Josiah Impson", "branch": "K", "steps": 5, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "3x Great Grandparent (1824-1896)", "bio": "Born on Fannagusha Creek, MS. Survived the 1833 'Trail of Tears' to the Choctaw Nation."},
            {"id": "Isaac Impson", "branch": "K", "steps": 6, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "4x Great Grandparent (1800-1899)", "bio": "English descent. Married into the Choctaw tribe."},
            {"id": "John Adam Josiah Impson", "branch": "K", "steps": 7, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "5x Great Grandparent (1745-1833)"},
            {"id": "John Adam Impson", "branch": "K", "steps": 8, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "6x Great Grandparent (1718-?)"},
            {"id": "William John Impson", "branch": "K", "steps": 9, "lateral": 1, "isSpouseLine": True, "anchorStep": 3, "desc": "7x Great Grandparent (1700-?)"},

            // Killebrew Laterals
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

            // --- RASMUSSEN BRANCH (R) ---
            {"id": "Clinton Rasmussen", "branch": "R", "steps": 3, "lateral": 0, "desc": "Great Grandparent (1904 - 1979)"},
            {"id": "James A. R.", "branch": "R", "steps": 4, "lateral": 0, "desc": "Great Great Grandparent (1877 - 1965)"},
            {"id": "Rasmus J. R.", "branch": "R", "steps": 5, "lateral": 0, "desc": "3x Great Grandparent (1842 - 1920)"},
            {"id": "Jens Rasmussen", "branch": "R", "steps": 6, "lateral": 0, "desc": "4x Great Grandparent (1810 - 1888)", "bio": "Immigrated from Denmark. Weaver. Served in Utah Territorial Militia (Black Hawk War). Wife Maren died of Cholera en route to Utah in 1866."},
            
            // Rasmussen Laterals
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

            // --- VANDERHOOP BRANCH (V) ---
            {"id": "Leonard V.", "branch": "V", "steps": 3, "lateral": 0, "desc": "Great Grandparent (1895 - 1989)", "bio": "Gay Head, Dukes, MA.<br><a href='#' target='_blank'>[Wampanoag Recording: The Cranberry Hunt]</a>"},
            {"id": "Edwin DeVries V.", "branch": "V", "steps": 4, "lateral": 0, "desc": "Great Great Grandparent (1848 - 1923)", "bio": "Served in the Navy on the gunboat Maheska during the Civil War (Union)."},
            {"id": "William A. V.", "branch": "V", "steps": 5, "lateral": 0, "desc": "3x Great Grandparent (~1816 - 1893)", "bio": "Dutch Surinamese Immigrant from Paramaribo. First Vanderhoop on Martha's Vineyard. Built the Vanderhoop homestead in Gay Head."},
            {"id": "Beulah Salisbury", "branch": "V", "steps": 5, "lateral": 0, "inLaw": True, "desc": "3x Great Grandmother (1814-1892)", "bio": "Known as the 'Princess of Aquinnah'. Assisted the Underground Railroad by hiding escaped slaves under a false floor in her family's barn."},
            
            // Vanderhoop Laterals
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

            // --- LIEBER BRANCH (L) ---
            {"id": "G. Heinrich L. Lieber", "branch": "L", "steps": 3, "lateral": 0, "desc": "Great Grandparent"},
            {"id": "Marie Emilie Ibe", "branch": "L", "steps": 3, "lateral": 0, "inLaw": True},
            {"id": "Manfred Lieber", "branch": "L", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle"},

            // --- BUZUNIS BRANCH (B) ---
            {"id": "George Constantine Buzunis", "branch": "B", "steps": 4, "lateral": 0, "desc": "Great Great Grandparent (1846 - 1912)", "bio": "Levidion, Greece"},
            {"id": "Elenis 'Helen' Georgakopoulis", "branch": "B", "steps": 4, "lateral": 0, "inLaw": True, "desc": "Great Great Grandmother (1867-1912)"},
            {"id": "Theodore Buzunis", "branch": "B", "steps": 3, "lateral": 0, "desc": "Great Grandparent (1885 - 1978)", "bio": "Tripoli, Peloponnese, Greece; Winnipeg, MB"},
            {"id": "Constantina Colilos", "branch": "B", "steps": 3, "lateral": 0, "inLaw": True, "desc": "Great Grandmother (1887-1963)"},
            
            // Buzunis Laterals
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

            // --- GINAKES BRANCH (A) ---
            {"id": "Unknown Ginakes", "branch": "A", "steps": 4, "lateral": 0, "desc": "Great Great Grandparent"},
            {"id": "Andrew Demetrius Ginakes", "branch": "A", "steps": 3, "lateral": 0, "desc": "Great Grandparent (1895 - 1967)", "bio": "Greece; Fargo, ND"},
            {"id": "Arcondo A. Boosalis", "branch": "A", "steps": 3, "lateral": 0, "inLaw": True, "desc": "Great Grandmother (1895-1984)"},
            
            // Ginakes Laterals
            {"id": "Constantine D Ginakes", "branch": "A", "steps": 3, "lateral": 1, "desc": "Great Grand Aunt / Uncle (1908 - 1965)"},
            {"id": "Anastasia Mirras", "branch": "A", "steps": 3, "lateral": 1, "inLaw": True},
            {"id": "Nicholas Ginakes", "branch": "A", "steps": 3, "lateral": 1, "desc": "Great Grand Aunt / Uncle"},
            {"id": "John Ginakes", "branch": "A", "steps": 3, "lateral": 1, "desc": "Great Grand Aunt / Uncle"},
            {"id": "Desmos Andrew Ginakes", "branch": "A", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle (1923 - 1996)"},
            {"id": "Mary Ginakes", "branch": "A", "steps": 2, "lateral": 1, "desc": "Great Aunt / Uncle (1928 - 2013)"}
        ],
        "links": [
            // Core Lines
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

            // Marriages
            {"source": "Eric Killebrew", "target": "Christina Vanderhoop", "type": "marriage"},
            {"source": "Eric Killebrew", "target": "Antonia Constance Buzunis", "type": "marriage"},
            {"source": "Robert Killebrew", "target": "Bonnie Rasmussen", "type": "marriage"},
            {"source": "John O. Vanderhoop", "target": "Waltrud M. Lieber", "type": "marriage"},
            {"source": "Peter Buzunis", "target": "Anastasia Ginakes", "type": "marriage"},
            {"source": "Theodore Buzunis", "target": "Constantina Colilos", "type": "marriage"},
            {"source": "George Constantine Buzunis", "target": "Elenis 'Helen' Georgakopoulis", "type": "marriage"},
            {"source": "Andrew Demetrius Ginakes", "target": "Arcondo A. Boosalis", "type": "marriage"},
            {"source": "Constantine D Ginakes", "target": "Anastasia Mirras", "type": "marriage"},
            {"source": "Antonia Constance Buzunis", "target": "Kyle Killebrew", "type": "inlaw"},
            
            // Robinson/Impson Spouse Line
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

            // Laterals (Killebrew/Rasmussen/Vanderhoop/Lieber/Buzunis/Ginakes)
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
                            {"name": "Edwin DeVries V.", "year": 1848, "branch": "V", "steps": 4, "lateral": 0, "desc": "Great Great Grandparent", "children": [
                                {"name": "William A. V.", "year": 1816, "branch": "V", "steps": 5, "lateral": 0, "desc": "3x Great Grandparent", "children": [
                                    {"name": "Beulah Salisbury", "year": 1814, "branch": "V", "steps": 5, "lateral": 0, "inLaw": True}
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