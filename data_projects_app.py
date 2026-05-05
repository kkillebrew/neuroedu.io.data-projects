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

# Ensure local imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_projects_loader import get_data_bio_metadata, get_project_descriptions

# --- DATA HYDRATION ---
bio = get_data_bio_metadata()
projects = get_project_descriptions()

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title=f"{bio['name']} | Data Projects",
    page_icon="📊",
    layout="wide"
)

########################################
# RENDER THE SIDEBAR FOR DATA-PROJECTS #
########################################
from data_projects_sidebar import render_sidebar
    render_sidebar()

# --- MAIN HUB LAYOUT ---
# 1. TOP ROW: Profile Image, Bio, and Resume Downloads
col_img, col_text = st.columns([1, 3], gap="large")

with col_img:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base_dir, "documents", "kyle.jpg")
    
    if os.path.exists(img_path):
        st.image(img_path, width='stretch')
    else:
        st.warning(f"Image not found in: {img_path}")

with col_text:
    st.title(bio['name'])
    st.subheader(bio['title'])
    st.write(bio['bio'])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Resume Download Buttons
    dl_col1, dl_col2 = st.columns(2)
    
    with dl_col1:
        ds_cv_path = os.path.join(base_dir, "documents", "KWK_Data_Science_Resume_20240520.pdf")
        if os.path.exists(ds_cv_path):
            with open(ds_cv_path, "rb") as f:
                st.download_button("📂 Download Data Science Resume", f.read(), "KWK_Data_Science_Resume.pdf", use_container_width=True)
        else:
            st.button("📄 DS Resume Not Found", disabled=True, use_container_width=True)
            
    with dl_col2:
        ai_cv_path = os.path.join(base_dir, "documents", "KWK_SME_AI_Resume_20260325.pdf")
        if os.path.exists(ai_cv_path):
            with open(ai_cv_path, "rb") as f:
                st.download_button("📂 Download AI SME Resume", f.read(), "KWK_AI_SME_Resume.pdf", use_container_width=True)
        else:
            st.button("📄 AI SME Resume Not Found", disabled=True, use_container_width=True)

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