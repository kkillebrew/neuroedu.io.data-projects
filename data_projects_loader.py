"""
=============================================================================
MODULE: data_projects_loader.py
AUTHOR: Kyle W. Killebrew, PhD
DESCRIPTION: 
    The "Model" layer for the Data Projects hub. Contains concise biographic
    metadata specifically tailored for the Data Science and AI portfolio,
    as well as the project directory data.
=============================================================================
"""

def get_data_bio_metadata():
    """
    Returns a concise bio focused on data science, machine learning, and AI.
    """
    return {
        'name': 'Kyle W. Killebrew, PhD',
        'title': 'Data Scientist & AI Subject Matter Expert',
        'bio': ("Welcome to my interactive Data Science portfolio. I specialize in predictive "
                "analytics, machine learning, and translating complex behavioral and economic "
                "data into actionable insights. This containerized environment hosts my "
                "interactive statistical models, API integrations, and signal processing pipelines. "
                "Select a project below or from the sidebar to explore the live dashboards.")
    }

def get_project_descriptions():
    """
    Returns metadata for the project gateway cards on the main hub.
    """
    return [
        {
            "title": "What effects the price of gas?",
            "desc": "Predictive modeling of crude oil prices using macroeconomic indicators, FRED API integrations, and random forest classifiers.",
            "status": "active",
            "page": "pages/1_oil_predictor_app.py",
            "button_text": "Launch Oil Predictor"
        },
        {
            "title": "Who's the real NBA GOAT?",
            "desc": "Statistical analysis and machine learning models weighing historical player metrics to settle the 'Greatest of All Time' debate.",
            "status": "active", 
            "page": "pages/2_nba_goat_predictor_app.py",
            "button_text": "Launch NBA GOAT Predictor"
        },
        {
            "title": "Does more tech really increase test scores?",
            "desc": "Data-driven evaluation of technological availability in learning environments and their effects on education around the world.",
            "status": "active", 
            "page": "pages/4_tech_ed_app.py",
            "button_text": "Launch Tech in Education"
        }
    ]