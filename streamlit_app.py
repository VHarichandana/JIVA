import streamlit as st
from utils.helpers import get_custom_css

# Page Configuration
st.set_page_config(
    page_title="JIVA — Multimodal Intelligence for Lung Health Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply Sarvam-Inspired Light Theme CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Session State Page Navigation Management
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home"

current = st.session_state["current_page"]

# Top Bar Header Navigation (Sarvam Style)
nav_brand_col, nav_col1, nav_col2, nav_col3, nav_col4, nav_col_vitals, nav_col5, nav_col6 = st.columns([1.6, 0.8, 1.3, 2.1, 1.4, 1.2, 1.7, 1.0])

with nav_brand_col:
    st.markdown(
        """
        <div style="padding-top: 0.05rem; line-height: 1;">
            <a href="#" class="sarvam-brand">JIVA</a>
        </div>
        """,
        unsafe_allow_html=True
    )

with nav_col1:
    st.markdown(f'<div class="nav-btn{" nav-btn-active" if current == "Home" else ""}">', unsafe_allow_html=True)
    if st.button("Home", use_container_width=True, key="nav_home"):
        st.session_state["current_page"] = "Home"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with nav_col2:
    st.markdown(f'<div class="nav-btn{" nav-btn-active" if current == "VOC Analysis" else ""}">', unsafe_allow_html=True)
    if st.button("VOC Analysis", use_container_width=True, key="nav_voc"):
        st.session_state["current_page"] = "VOC Analysis"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with nav_col3:
    st.markdown(f'<div class="nav-btn{" nav-btn-active" if current == "Audio Analysis" else ""}">', unsafe_allow_html=True)
    if st.button("Audio Analysis", use_container_width=True, key="nav_audio"):
        st.session_state["current_page"] = "Audio Analysis"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with nav_col4:
    st.markdown(f'<div class="nav-btn{" nav-btn-active" if current == "X-ray Analysis" else ""}">', unsafe_allow_html=True)
    if st.button("X-ray Analysis", use_container_width=True, key="nav_xray"):
        st.session_state["current_page"] = "X-ray Analysis"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with nav_col_vitals:
    st.markdown(f'<div class="nav-btn{" nav-btn-active" if current == "Video Analysis" else ""}">', unsafe_allow_html=True)
    if st.button("Video Analysis", use_container_width=True, key="nav_vitals"):
        st.session_state["current_page"] = "Video Analysis"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with nav_col5:
    st.markdown(f'<div class="nav-btn{" nav-btn-active" if current == "Performance" else ""}">', unsafe_allow_html=True)
    if st.button("Performance", use_container_width=True, key="nav_perf"):
        st.session_state["current_page"] = "Performance"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with nav_col6:
    st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
    if st.button("Ask JIVA", use_container_width=True, key="nav_ask"):
        st.session_state["current_page"] = "Home"
        st.session_state["scroll_to_ask_jiva"] = True
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr style='border-color: #e8e8e8; margin-top: 0.5rem; margin-bottom: 2rem;'>", unsafe_allow_html=True)

# Page Router

if current == "Home":
    from pages.home import render_home
    render_home()
elif current == "VOC Analysis":
    from pages.voc_analysis import render_voc_analysis
    render_voc_analysis()
elif current == "Audio Analysis":
    from pages.audio_analysis import render_cardiopulmonary_audio
    render_cardiopulmonary_audio()
elif current == "X-ray Analysis":
    from pages.xray_analysis import render_xray_analysis
    render_xray_analysis()
elif current == "Video Analysis":
    from pages.video_analysis import render_video_analysis
    render_video_analysis()
elif current == "Performance":
    from pages.model_performance import render_model_performance
    render_model_performance()

