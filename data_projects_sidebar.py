"""
=============================================================================
MODULE: data_projects_sidebar.py (UI Controller)
AUTHOR: Kyle W. Killebrew, PhD
DESCRIPTION: 
    Centralized sidebar component for the isolated Data Science Hub.
    Explicitly excludes links back to the main Career Hub to ensure 
    employers viewing DS projects are kept within the technical ecosystem.
=============================================================================
"""

import streamlit as st

def render_sidebar():
    """
    Renders the custom sidebar with the DS masthead, DS directory, 
    isolated presence links, and unified CSS styling.
    """
    
    # --- 1. GLOBAL SIDEBAR CSS (Styling & Bug Fixes) ---
    st.markdown("""
        <style>
        [data-testid="stSidebarNav"] {display: none;}
        [data-testid="collapsedControl"] { font-family: sans-serif !important; }

        [data-testid="stSidebar"] {
            background-color: #0F172A; /* Deep Slate */
            color: #F8FAFC; /* Crisp White */
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }

        .sidebar-link {
            display: block;
            padding: 0.5rem 1rem;
            color: #F8FAFC !important;
            text-decoration: none;
            font-size: 1.05rem;
            border-radius: 5px;
            margin-bottom: 5px;
            transition: background-color 0.3s, color 0.3s;
        }
        .sidebar-link:hover {
            background-color: #1E293B; 
            color: #38BDF8 !important; /* Sky Blue Accent */
        }
        
        .masthead-container {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-top: -40px; 
            margin-bottom: 30px;
        }
        .masthead-text { line-height: 1.2; }
        .masthead-title {
            font-size: 1.1rem;
            font-weight: bold;
            color: #F8FAFC;
        }
        .masthead-blurb {
            font-size: 0.8rem;
            color: #94A3B8; 
            margin-top: 5px;
        }
        
        .presence-bar {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 30px;
            margin-bottom: 20px;
        }
        .presence-icon {
            color: #94A3B8;
            transition: color 0.3s;
        }
        .presence-icon:hover { color: #38BDF8; }
        
        .sidebar-footer {
            text-align: center;
            font-size: 0.75rem;
            color: #64748B;
            margin-top: 40px;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        # --- 2. MASTHEAD (Brain Logo + DS Blurb) ---
        brain_svg = """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="55" height="55" fill="none" stroke="#38BDF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M50 85 C30 85, 15 70, 15 50 C15 35, 25 20, 40 15 C45 13, 55 13, 60 15 C75 20, 85 35, 85 50 C85 70, 70 85, 50 85 Z" />
          <path d="M30 35 L45 25 L60 30 L70 45 L65 65 L50 75 L35 60 Z" />
          <circle cx="30" cy="35" r="3" fill="#38BDF8"/>
          <circle cx="45" cy="25" r="3" fill="#38BDF8"/>
          <circle cx="60" cy="30" r="3" fill="#38BDF8"/>
          <circle cx="70" cy="45" r="3" fill="#38BDF8"/>
          <circle cx="65" cy="65" r="3" fill="#38BDF8"/>
          <circle cx="50" cy="75" r="3" fill="#38BDF8"/>
          <circle cx="35" cy="60" r="3" fill="#38BDF8"/>
          <circle cx="50" cy="50" r="4" fill="#38BDF8"/>
          <path d="M50 50 L30 35 M50 50 L45 25 M50 50 L60 30 M50 50 L70 45 M50 50 L65 65 M50 50 L50 75 M50 50 L35 60" opacity="0.5"/>
        </svg>
        """
        
        st.markdown(f"""
            <div class="masthead-container">
                <div>{brain_svg}</div>
                <div class="masthead-text">
                    <div class="masthead-title">Data Projects Hub</div>
                    <div class="masthead-blurb">Passion-driven, exploratory data science & interactive modeling.</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.divider()

        # --- 3. DIRECTORY LINKS (Isolated Scope) ---
        # Using native st.page_link for Streamlit's internal router
        st.page_link("data_projects_app.py", label="Data Projects Hub (Home)", icon=None)
        st.page_link("pages/1_oil_predictor_app.py", label="What affects gas prices?", icon=None)
        st.page_link("pages/2_nba_goat_predictor_app.py", label="Who is the real NBA GOAT?", icon=None)
        st.page_link("pages/3_typo_behavior_app.py", label="Why do we make typos?", icon=None)        

        st.divider()

        # --- 4. PRESENCE / SOCIAL BAR (Data-Science Specific) ---
        # Kept GitHub, ORCID, and Kaggle.
        presence_html = """
        <div class="presence-bar">
            <a href="https://github.com/yourprofile" target="_blank" class="presence-icon" title="GitHub">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.332-5.467-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
            </a>
            <a href="https://kaggle.com/yourprofile" target="_blank" class="presence-icon" title="Kaggle">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><path d="M18.825 23.859c-.022.092-.117.141-.281.141h-3.139c-.187 0-.351-.082-.492-.248l-5.178-6.589-1.448 1.374v5.111c0 .235-.117.352-.351.352H5.505c-.236 0-.354-.117-.354-.352V.353c0-.233.118-.353.354-.353h2.431c.234 0 .351.12.351.353v14.343l6.203-6.272c.165-.165.34-.246.526-.246h3.256c.141 0 .235.035.281.106.046.07.035.152-.035.246L11.88 15.006l6.91 8.502c.07.094.081.188.035.351z"/></svg>
            </a>
            <a href="https://orcid.org/your-orcid" target="_blank" class="presence-icon" title="ORCID">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.372 0 0 5.372 0 12s5.372 12 12 12 12-5.372 12-12S18.628 0 12 0zM7.369 4.378c.525 0 .947.431.947.947s-.422.949-.947.949a.95.95 0 0 1-.949-.949c0-.516.424-.947.949-.947zm-.722 3.038h1.444v10.041H6.647V7.416zm3.562 0h3.9c3.712 0 5.344 2.653 5.344 5.025 0 2.578-2.016 5.025-5.325 5.025h-3.919V7.416zm1.444 1.303v7.444h2.297c3.272 0 4.022-2.484 4.022-3.722 0-2.016-1.284-3.722-4.097-3.722h-2.222z"/></svg>
            </a>
        </div>
        """
        st.markdown(presence_html, unsafe_allow_html=True)

        # --- 5. COPYRIGHT FOOTER ---
        st.markdown("""
            <div class="sidebar-footer">
                © 2026 Kyle W. Killebrew.<br>
                Data, models, and resume entirely self-authored.
            </div>
        """, unsafe_allow_html=True)