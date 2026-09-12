import streamlit as st
from utils.helpers import get_custom_css

st.set_page_config(
    page_title="JIVA - Multimodal Intelligence for Lung Health Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(get_custom_css(), unsafe_allow_html=True)


# =========================================================
# PAGE STATE
# =========================================================

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home"


# =========================================================
# QUERY PARAMETER NAVIGATION
# =========================================================

page_param = st.query_params.get("page")
ask_param = st.query_params.get("ask")

page_map = {
    "home": "Home",
    "voc": "VOC Analysis",
    "audio": "Audio Analysis",
    "xray": "X-ray Analysis",
    "video": "Video Analysis"
    # "performance": "Performance"
}

if page_param in page_map:
    st.session_state["current_page"] = page_map[page_param]

if ask_param == "1":
    st.session_state["current_page"] = "Home"
    st.session_state["scroll_to_ask_jiva"] = True

current = st.session_state["current_page"]


# =========================================================
# NAVBAR TEXT COLOR
# Home -> White
# Other pages -> JIVA MedIntell Blue
# =========================================================

if current == "Home":
    nav_text_color = "#FFFFFF"
    nav_text_shadow = "0 1px 3px rgba(0,70,100,0.42)"
else:
    nav_text_color = "#0477A8"
    nav_text_shadow = "none"


# =========================================================
# ACTIVE PAGE CLASSES
# =========================================================

voc_active = "active" if current == "VOC Analysis" else ""
audio_active = "active" if current == "Audio Analysis" else ""
xray_active = "active" if current == "X-ray Analysis" else ""
video_active = "active" if current == "Video Analysis" else ""
performance_active = "active" if current == "Performance" else ""


# =========================================================
# NAVBAR CSS
# =========================================================

st.html(
    f"""
    <style>

    /* =====================================
       NAVBAR CONTAINER
       ===================================== */

    .jiva-navbar {{
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: space-between;

        gap: 2rem;

        padding: 0.55rem 0 0.9rem 0;
        margin: 0;

        background: transparent !important;

        border: none !important;
        box-shadow: none !important;
    }}


    /* =====================================
       JIVA MEDINTELL BRAND
       ===================================== */

    .jiva-brand {{
        display: inline-flex;
        align-items: center;

        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;

        border: none !important;
        border-radius: 0 !important;

        outline: none !important;
        box-shadow: none !important;

        margin: 0 !important;
        padding: 0 !important;

        color: #0477A8 !important;
        -webkit-text-fill-color: #0477A8 !important;

        text-decoration: none !important;

        font-size: 1.55rem !important;
        font-weight: 900 !important;

        letter-spacing: 0.045em !important;
        line-height: 1 !important;

        white-space: nowrap !important;

        cursor: pointer !important;
    }}


    /* Brand does not change on hover */
    .jiva-brand:hover,
    .jiva-brand:focus,
    .jiva-brand:active,
    .jiva-brand:visited {{
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;

        border: none !important;
        outline: none !important;
        box-shadow: none !important;

        color: #0477A8 !important;
        -webkit-text-fill-color: #0477A8 !important;

        text-decoration: none !important;

        transform: none !important;
    }}


    /* =====================================
       NAVIGATION LINKS CONTAINER
       ===================================== */

    .jiva-nav-links {{
        display: flex;
        align-items: center;
        justify-content: flex-end;

        flex: 1;

        gap: 2.15rem;

        margin: 0;
        padding: 0;

        background: transparent !important;

        border: none !important;
        box-shadow: none !important;

        white-space: nowrap;
    }}


    /* =====================================
       PLAIN NAVIGATION TEXT
       ===================================== */

    .jiva-nav-link {{
        display: inline-flex;
        align-items: center;
        justify-content: center;

        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;

        border: none !important;
        border-radius: 0 !important;

        outline: none !important;
        box-shadow: none !important;

        margin: 0 !important;
        padding: 0 !important;

        color: {nav_text_color} !important;
        -webkit-text-fill-color: {nav_text_color} !important;

        text-decoration: none !important;

        font-size: 0.92rem !important;
        font-weight: 700 !important;

        line-height: 1 !important;

        white-space: nowrap !important;

        cursor: pointer !important;

        text-shadow: {nav_text_shadow} !important;
    }}


    /* =====================================
       NO HOVER EFFECT
       ===================================== */

    .jiva-nav-link:hover,
    .jiva-nav-link:focus,
    .jiva-nav-link:active,
    .jiva-nav-link:visited {{
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;

        border: none !important;
        border-radius: 0 !important;

        outline: none !important;
        box-shadow: none !important;

        color: {nav_text_color} !important;
        -webkit-text-fill-color: {nav_text_color} !important;

        text-decoration: none !important;

        text-shadow: {nav_text_shadow} !important;

        transform: none !important;
    }}


    /* =====================================
       ACTIVE PAGE
       ONLY BOLDER TEXT
       ===================================== */

    .jiva-nav-link.active {{
        background: transparent !important;

        color: {nav_text_color} !important;
        -webkit-text-fill-color: {nav_text_color} !important;

        font-weight: 900 !important;

        border: none !important;
        box-shadow: none !important;
    }}


    /* =====================================
       DIVIDER
       ===================================== */

    .jiva-navbar-divider {{
        border: none !important;

        border-top:
            1px solid rgba(30,120,160,0.18) !important;

        margin-top: 0.3rem !important;
        margin-bottom: 2rem !important;
    }}


    /* =====================================
       RESPONSIVE
       ===================================== */

    @media (max-width: 1200px) {{

        .jiva-navbar {{
            gap: 1rem;
        }}

        .jiva-nav-links {{
            gap: 1.2rem;
        }}

        .jiva-brand {{
            font-size: 1.3rem !important;
        }}

        .jiva-nav-link {{
            font-size: 0.82rem !important;
        }}
    }}

    </style>
    """
)


# =========================================================
# NAVBAR HTML
# Single-line construction prevents Markdown code blocks
# =========================================================

navbar_html = (
    '<div class="jiva-navbar">'
    '<a class="jiva-brand" href="?page=home">JIVA</a>'
    '<div class="jiva-nav-links">'
    f'<a class="jiva-nav-link {voc_active}" href="?page=voc">VOC Analysis</a>'
    f'<a class="jiva-nav-link {audio_active}" href="?page=audio">Audio Analysis</a>'
    f'<a class="jiva-nav-link {xray_active}" href="?page=xray">X-ray Analysis</a>'
    f'<a class="jiva-nav-link {video_active}" href="?page=video">Video Analysis</a>'
    # f'<a class="jiva-nav-link {performance_active}" href="?page=performance">Performance</a>'
    '<a class="jiva-nav-link" href="?page=home&ask=1">Ask JIVA</a>'
    '</div>'
    '</div>'
    '<hr class="jiva-navbar-divider">'
)

st.html(navbar_html)


# =========================================================
# PAGE ROUTER
# =========================================================

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

# elif current == "Performance":
#     from pages.model_performance import render_model_performance
#     render_model_performance()