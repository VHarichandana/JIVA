import streamlit as st
from pathlib import Path
from components.chatbot import render_chatbot

def render_home():
    # Darker Soft Sky Blue Home Background Gradient
    st.markdown(
        '''
        <style>
        .stApp {
            background:
                radial-gradient(
                    circle at 50% 8%,
                    rgba(52, 170, 210, 0.38) 0%,
                    rgba(95, 198, 226, 0.30) 35%,
                    transparent 68%
                ),
                linear-gradient(
                    180deg,
                    #BFE5F3 0%,
                    #CFEAF6 38%,
                    #DDF2F9 68%,
                    #F4FBFD 100%
                ) !important;
        }
        </style>
        ''',
        unsafe_allow_html=True
    )

    # Check if scrolled from nav or hero button
    scroll_to_ask = st.session_state.get("scroll_to_ask_jiva", False)
    if scroll_to_ask:
        st.session_state["scroll_to_ask_jiva"] = False
        st.components.v1.html(
            '''
            <script>
                setTimeout(function() {
                    var el = window.parent.document.getElementById('ask-jiva');
                    if (el) {
                        el.scrollIntoView({behavior: 'smooth', block: 'start'});
                    }
                }, 150);
            </script>
            ''',
            height=0
        )

    # Section 1 — Home Hero
    st.markdown(
        '''
        <div class="sarvam-hero">
            <div class="sarvam-hero-heading">Intelligence for every breath</div>
            <div class="sarvam-hero-sub">JIVA combines breath biomarkers, cardiopulmonary sounds and medical imaging into one connected AI-assisted lung health platform.</div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    col_cta1, col_cta2, _ = st.columns([1.3, 1.3, 5.4], gap="small")
    with col_cta1:
        if st.button("Explore JIVA", type="primary", use_container_width=True, key="hero_explore"):
            st.session_state["current_page"] = "VOC Analysis"
            st.rerun()
    with col_cta2:
        if st.button("Ask JIVA", use_container_width=True, key="hero_ask"):
            st.session_state["current_page"] = "Home"
            st.session_state["scroll_to_ask_jiva"] = True
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Section 2 — Main Module Section (Sarvam Product Cards)
    st.markdown(
        '''
        <div style="margin-bottom: 2rem;">
            <h2 style="font-size: 2.2rem; font-weight: 800; color: #252525; margin-bottom: 0.5rem; letter-spacing: -0.03em;">Built for multimodal lung analysis</h2>
        </div>
        ''',
        unsafe_allow_html=True
    )

    card_col1, card_col2, card_col3, card_col4 = st.columns(4)

    with card_col1:
        st.markdown(
            '''
            <div class="sarvam-card">
                <div class="sarvam-card-corner-accent"></div>
                <div>
                    <div class="sarvam-card-title">VOC Analysis</div>
                    <div class="sarvam-card-desc">Analyze volatile organic compounds in breath to identify patterns associated with Control, Benign and Cancer classes.</div>
                    <div class="sarvam-support-line">Breath biomarkers · Machine learning · Explainability</div>
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )
        if st.button("Explore VOC Analysis", use_container_width=True, key="card_btn_voc"):
            st.session_state["current_page"] = "VOC Analysis"
            st.rerun()

    with card_col2:
        st.markdown(
            '''
            <div class="sarvam-card">
                <div class="sarvam-card-corner-accent"></div>
                <div>
                    <div class="sarvam-card-title">Audio Analysis</div>
                    <div class="sarvam-card-desc">Analyze respiratory and cardiopulmonary recordings using waveform, spectrogram and audio-language intelligence.</div>
                    <div class="sarvam-support-line">Respiratory audio · Signal analysis · StethoLM</div>
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )
        if st.button("Explore Audio Analysis", use_container_width=True, key="card_btn_audio"):
            st.session_state["current_page"] = "Audio Analysis"
            st.rerun()

    with card_col3:
        st.markdown(
            '''
            <div class="sarvam-card">
                <div class="sarvam-card-corner-accent"></div>
                <div>
                    <div class="sarvam-card-title">X-ray Analysis</div>
                    <div class="sarvam-card-desc">Review chest X-ray images through a dedicated medical-imaging workflow prepared for deep-learning integration.</div>
                    <div class="sarvam-support-line">Chest imaging · Visual analysis · Explainability</div>
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )
        if st.button("Explore X-ray Analysis", use_container_width=True, key="card_btn_xray"):
            st.session_state["current_page"] = "X-ray Analysis"
            st.rerun()

    with card_col4:
        st.markdown(
            '''
            <div class="sarvam-card">
                <div class="sarvam-card-corner-accent"></div>
                <div>
                    <div class="sarvam-card-title">Video Analysis</div>
                    <div class="sarvam-card-desc">Non-invasive face video photoplethysmography heart-rate estimation using Spatio-Temporal ViT (PhysFormer).</div>
                    <div class="sarvam-support-line">Video rPPG · Spatio-temporal ViT · Heart Rate</div>
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )
        if st.button("Explore Video Analysis", use_container_width=True, key="card_btn_vitals"):
            st.session_state["current_page"] = "Video Analysis"
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Section 3 — Secondary Feature Grid (Sarvam Compact Grid)
    g_col1, g_col2, g_col3 = st.columns(3)

    with g_col1:
        st.markdown(
            '''
            <div class="sarvam-grid-card" style="margin-bottom: 1.25rem;">
                <div class="sarvam-grid-title">Breath Intelligence</div>
                <div class="sarvam-grid-desc">VOC biomarker patterns transformed into interpretable model predictions.</div>
            </div>
            <div class="sarvam-grid-card">
                <div class="sarvam-grid-title">Explainable AI</div>
                <div class="sarvam-grid-desc">Understand predictions through feature importance, SHAP and model-performance analysis.</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with g_col2:
        st.markdown(
            '''
            <div class="sarvam-grid-card" style="margin-bottom: 1.25rem;">
                <div class="sarvam-grid-title">Audio Intelligence</div>
                <div class="sarvam-grid-desc">Cardiopulmonary recordings transformed into acoustic and language-based findings.</div>
            </div>
            <div class="sarvam-grid-card">
                <div class="sarvam-grid-title">Model Evaluation</div>
                <div class="sarvam-grid-desc">Compare performance using accuracy, precision, recall, F1, MCC and ROC-AUC.</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with g_col3:
        st.markdown(
            '''
            <div class="sarvam-grid-card" style="margin-bottom: 1.25rem;">
                <div class="sarvam-grid-title">Medical Imaging</div>
                <div class="sarvam-grid-desc">Chest images organized for visual AI analysis and future model integration.</div>
            </div>
            <div class="sarvam-grid-card">
                <div class="sarvam-grid-title">Ask JIVA</div>
                <div class="sarvam-grid-desc">Explore results and understand the platform through a conversational interface.</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    # Space between Section 3 feature grid and Section 4
    st.markdown("<div style='height: 3.5rem;'></div>", unsafe_allow_html=True)

    # Section 4 — Large Editorial Image Section ("See beyond a single signal")
    vis_col1, vis_col2 = st.columns([1.1, 1.2])

    with vis_col1:
        st.markdown(
            '''
            <div style="padding-top: 1rem;">
                <h2 style="font-size: 2.5rem; font-weight: 800; color: #252525; margin-bottom: 1.25rem; letter-spacing: -0.04em; line-height: 1.15;">See beyond a single signal</h2>
                <p style="color: #707070; font-size: 1.08rem; line-height: 1.65; margin-bottom: 2rem;">Breath chemistry, respiratory sounds, medical imaging and facial rPPG hemodynamics each reveal a different part of cardiopulmonary health. JIVA brings these signals into one connected workflow.</p>
            </div>
            ''',
            unsafe_allow_html=True
        )
        if st.button("Explore X-ray Analysis", type="primary", key="editorial_explore_xray"):
            st.session_state["current_page"] = "X-ray Analysis"
            st.rerun()

    with vis_col2:
        asset_path = Path("assets/jiva_xray_hero.png")
        if asset_path.exists():
            st.image(str(asset_path), use_container_width=True)
        else:
            st.markdown(
                '''
                <div style="background-color: #fafafa; border: 1px solid #e8e8e8; border-radius: 20px; padding: 5rem 2rem; text-align: center; color: #707070;">
                    <div style="font-size: 1.1rem; font-weight: 600; color: #252525; margin-bottom: 0.5rem;">Radiological Inspection Container</div>
                    <div>assets/jiva_xray_hero.png</div>
                </div>
                ''',
                unsafe_allow_html=True
            )

    st.markdown("<br><hr style='border-color: #e8e8e8; margin-bottom: 3rem;'><br>", unsafe_allow_html=True)

    # Section 5 — Chatbot Section (Sarvam Voice-Agent Style "Ask JIVA")
    st.markdown(
        '''
        <div id="ask-jiva" style="scroll-margin-top: 100px;"></div>
        <div style="margin-bottom: 1.5rem;">
            <h2 style="font-size: 2.2rem; font-weight: 800; color: #252525; margin-bottom: 0.4rem; letter-spacing: -0.03em;">Ask JIVA</h2>
            <p style="color: #707070; font-size: 1.05rem; margin: 0;">Ask about VOC analysis, cardiopulmonary audio, X-ray analysis, model performance or explainability.</p>
        </div>
        ''',
        unsafe_allow_html=True
    )

    render_chatbot()
