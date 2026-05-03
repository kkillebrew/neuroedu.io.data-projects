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
            "title": "🛢️ Macro Oil Predictor",
            "desc": "Predictive modeling of crude oil prices using macroeconomic indicators, FRED API integrations, and Random Forest classifiers.",
            "status": "active",
            "page": "pages/1_oil_predictor_app.py",
            "button_text": "Launch Oil Predictor"
        },
        {
            "title": "🏀 NBA GOAT Predictor",
            "desc": "Statistical analysis and machine learning models weighing historical player metrics to evaluate the 'Greatest of All Time' debate.",
            "status": "development", 
            "page": "pages/2_nba_goat_predictor_app.py",
            "button_text": "Coming Soon"
        },
        {
            "title": "💻 Tech in Education",
            "desc": "Data-driven evaluation of technological interventions in learning environments and their effects on educational outcomes.",
            "status": "development", 
            "page": "pages/3_tech_education_app.py",
            "button_text": "Coming Soon"
        }
    ]