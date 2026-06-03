"""
=============================================================================
MODULE: data_projects_app.py (Data Hub Entry Point)
AUTHOR: Kyle W. Killebrew, PhD
VERSION: 1.0 (Data Science Micro-Frontend Hub)
DESCRIPTION: 
    The landing page for data-projects.neuro-edu.io. Handles the short
    data-focused bio, resume downloads, and links to the analytical spokes.
=============================================================================
"""

import streamlit as st
import os
import sys
import base64 # <-- NEW IMPORT

# --- PATH CONFIGURATION ---
# This tells the script to look one folder up to find the 'loaders' directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_projects_loader import get_data_bio_metadata, get_project_descriptions
from pages.components.genealogy_web_demo import render_genealogy_web # <-- NEW IMPORT

from data_projects_sidebar import apply_global_settings, render_sidebar

########################################
#        APPLY GLOBAL SETTINGS         #
########################################
apply_global_settings("Neuro-Edu | Data Projects Hub")

########################################
# RENDER THE SIDEBAR FOR DATA-PROJECTS #
########################################
render_sidebar()

# --- DATA HYDRATION ---
bio = get_data_bio_metadata()
projects = get_project_descriptions()

# --- MASTHEAD & BIOGRAPHIC LAYOUT ---
current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, "documents", "Neuro-Edu_Logo_Transparent.png")
img_path = os.path.join(current_dir, "documents", "kyle.jpg")

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

b64_logo = get_base64_image(logo_path)
b64_profile = get_base64_image(img_path)

st.markdown(f"""
<style>
.logo-container {{ text-align: center; margin-bottom: 30px; }}
.logo-container img {{ max-width: 320px; width: 100%; height: auto; }}
</style>
<div class="logo-container">
    <img src="data:image/png;base64,{b64_logo}" alt="Neuro-Edu Logo">
</div>
""", unsafe_allow_html=True)

col_img, col_text = st.columns([1, 2.5], gap="large")

with col_img:
    st.markdown(f"""
<style>
.profile-img-container {{
    width: 240px; 
    height: 240px;
    margin: 20px auto 0 auto; 
    overflow: hidden; 
    border-radius: 50%;
    border: 3px solid #334155; 
    background-color: #0F172A;
    display: flex;
    justify-content: center;
    align-items: center;
    box-shadow: 0 4px 10px rgba(0,0,0,0.4); 
}}
.profile-img-container img {{
    width: 100%;
    height: 100%;
    object-fit: cover; 
    object-position: 25% 50%; 
}}
</style>
<div class="profile-img-container">
    <img src="data:image/jpeg;base64,{b64_profile}" alt="{bio['name']}">
</div>
""", unsafe_allow_html=True)

with col_text:
    st.markdown(f"""
<style>
.masthead-title {{ font-size: 3rem; font-weight: bold; color: #F8FAFC; margin-bottom: 5px; line-height: 1.1; }}
.masthead-subtitle {{ font-size: 1.3rem; color: #38BDF8; margin-bottom: 20px; font-weight: 500; }}
.vision-blurb {{ font-size: 1rem; color: #CBD5E1; line-height: 1.6; background-color: transparent; margin-bottom: 20px; }}
</style>
<div class="masthead-title">{bio['name']}</div>
<div class="masthead-subtitle">{bio['title']}</div>
<div class="vision-blurb">{bio['bio']}</div>
""", unsafe_allow_html=True)
    
    # Resume Download Buttons (Shifted into the text column)
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        ds_cv_path = os.path.join(current_dir, "documents", "KWK_Data_Science_Resume_20240520.pdf")
        if os.path.exists(ds_cv_path):
            with open(ds_cv_path, "rb") as f:
                st.download_button("Download Data Science Resume", f.read(), "KWK_Data_Science_Resume.pdf", use_container_width=True)
        else:
            st.button("DS Resume Not Found", disabled=True, use_container_width=True)
            
    with dl_col2:
        ai_cv_path = os.path.join(current_dir, "documents", "KWK_SME_AI_Resume_20260325.pdf")
        if os.path.exists(ai_cv_path):
            with open(ai_cv_path, "rb") as f:
                st.download_button("Download AI SME Resume", f.read(), "KWK_AI_SME_Resume.pdf", use_container_width=True)
        else:
            st.button("AI SME Resume Not Found", disabled=True, use_container_width=True)

st.divider()

# --- FAMILY TREE MODULE ---
st.markdown("<h3 style='text-align: center; color: #F8FAFC; margin-bottom: 10px;'>My Family Tree</h3>", unsafe_allow_html=True)

# Custom CSS to center the radio buttons
st.markdown("""
    <style>
    .stRadio [role=radiogroup] { justify-content: center; }
    </style>
""", unsafe_allow_html=True)

tree_view = st.radio(
    "Select View Mode:",
    ["Ancestral Tree", "Migration Journeys", "DNA Evidence"],
    horizontal=True,
    label_visibility="collapsed"
)

if tree_view == "Ancestral Tree":
    render_genealogy_web()
elif tree_view == "Migration Journeys":
    st.info("Migration maps pipeline is currently in development. Please check back later.")
elif tree_view == "DNA Evidence":
    st.info("DNA sequencing and genomic variance module is currently in development.")

st.divider()

# --- PROJECT GATEWAY ---
st.header("Interactive Models & Dashboards")
st.write("Select a project below to launch the containerized application.")

cols = st.columns(3)

for i, proj in enumerate(projects):
    with cols[i]:
        # Using the same ref-card CSS from the career hub for consistency
        st.markdown(f"""
            <div class="ref-card">
                <div class="ref-name">{proj['title']}</div>
                <span class="ref-desc">{proj['desc']}</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Check status to disable buttons for projects not yet built
        if proj['status'] == 'active':
            if st.button(proj['button_text'], key=f"btn_{i}", use_container_width=True):
                st.switch_page(proj['page'])
        else:
            st.button(proj['button_text'], key=f"btn_{i}", disabled=True, use_container_width=True)

st.divider()