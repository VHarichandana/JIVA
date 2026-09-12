import streamlit as st
from pathlib import Path
from components.chatbot import render_chatbot


def render_home():

    st.html("""
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

    .sarvam-card {
        min-height: 300px;
        height: 300px;
        padding: 1.45rem;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.65);
        box-shadow:
            0 10px 28px rgba(28,110,145,0.10),
            inset 0 1px 0 rgba(255,255,255,0.72);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(8px);
        box-sizing: border-box;
    }

    .jiva-card-voc {
        background:
            radial-gradient(
                circle at 90% 10%,
                rgba(15,166,207,0.28),
                transparent 43%
            ),
            linear-gradient(
                145deg,
                rgba(246,253,255,0.98) 0%,
                rgba(190,233,247,0.92) 55%,
                rgba(153,214,237,0.84) 100%
            );
    }

    .jiva-card-audio {
        background:
            radial-gradient(
                circle at 90% 10%,
                rgba(61,137,225,0.25),
                transparent 43%
            ),
            linear-gradient(
                145deg,
                rgba(246,252,255,0.98) 0%,
                rgba(198,228,249,0.92) 55%,
                rgba(166,209,240,0.84) 100%
            );
    }

    .jiva-card-xray {
        background:
            radial-gradient(
                circle at 90% 10%,
                rgba(36,184,193,0.24),
                transparent 43%
            ),
            linear-gradient(
                145deg,
                rgba(246,255,255,0.98) 0%,
                rgba(194,237,238,0.92) 55%,
                rgba(163,220,226,0.84) 100%
            );
    }

    .jiva-card-video {
        background:
            radial-gradient(
                circle at 90% 10%,
                rgba(78,117,222,0.23),
                transparent 43%
            ),
            linear-gradient(
                145deg,
                rgba(247,252,255,0.98) 0%,
                rgba(203,226,249,0.92) 55%,
                rgba(174,208,242,0.84) 100%
            );
    }

    .sarvam-card-corner-accent {
        position: absolute;
        top: 0;
        right: 0;
        width: 70px;
        height: 70px;
        border-radius: 0 20px 0 70px;
        background:
            linear-gradient(
                135deg,
                rgba(0,112,160,0.30),
                rgba(56,190,220,0.08)
            );
    }

    .sarvam-card-title {
        color: #17384A;
        font-size: 1.16rem;
        font-weight: 800;
        margin-bottom: 0.85rem;
        position: relative;
        z-index: 2;
    }

    .sarvam-card-desc {
        color: #3A5968;
        font-size: 0.92rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    .sarvam-support-line {
        color: #0877A5;
        font-size: 0.78rem;
        font-weight: 700;
        line-height: 1.45;
    }

    .sarvam-grid-card {
        min-height: 135px;
        padding: 1.35rem 1.4rem;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.68);
        box-shadow:
            0 8px 22px rgba(31,111,145,0.08),
            inset 0 1px 0 rgba(255,255,255,0.72);
        backdrop-filter: blur(8px);
    }

    .grid-breath {
        background:
            linear-gradient(
                145deg,
                rgba(247,254,255,0.98),
                rgba(181,230,246,0.88)
            );
    }

    .grid-xai {
        background:
            linear-gradient(
                145deg,
                rgba(246,251,255,0.98),
                rgba(194,223,248,0.88)
            );
    }

    .grid-audio {
        background:
            linear-gradient(
                145deg,
                rgba(244,255,254,0.98),
                rgba(184,232,234,0.88)
            );
    }

    .grid-performance {
        background:
            linear-gradient(
                145deg,
                rgba(247,251,255,0.98),
                rgba(195,224,248,0.88)
            );
    }

    .grid-imaging {
        background:
            linear-gradient(
                145deg,
                rgba(245,254,255,0.98),
                rgba(181,228,244,0.88)
            );
    }

    .grid-jiva {
        background:
            linear-gradient(
                145deg,
                rgba(244,252,255,0.98),
                rgba(171,221,242,0.90)
            );
    }

    .sarvam-grid-title {
        color: #17384A;
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 0.55rem;
    }

    .sarvam-grid-desc {
        color: #496674;
        font-size: 0.90rem;
        line-height: 1.55;
    }

    .section-heading {
        font-size: 2.2rem;
        font-weight: 800;
        color: #252525;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
    }

    .editorial-heading {
        font-size: 2.5rem;
        font-weight: 800;
        color: #252525;
        margin-bottom: 1.25rem;
        letter-spacing: -0.04em;
        line-height: 1.15;
    }

    .editorial-text {
        color: #60707A;
        font-size: 1.08rem;
        line-height: 1.65;
        margin-bottom: 2rem;
    }
    </style>
    """)

    scroll_to_ask = st.session_state.get(
        "scroll_to_ask_jiva",
        False
    )

    if scroll_to_ask:
        st.session_state["scroll_to_ask_jiva"] = False

        st.components.v1.html(
            """
            <script>
            setTimeout(function() {
                var el =
                    window.parent.document.getElementById('ask-jiva');

                if (el) {
                    el.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }, 150);
            </script>
            """,
            height=0
        )

    # HERO

    st.html("""
    <div class="sarvam-hero">
        <div class="sarvam-hero-heading">
            Multimodal Intelligence for Lung Health Analysis
        </div>

        <div class="sarvam-hero-sub">
            JIVA combines breath biomarkers, cardiopulmonary sounds
            and medical imaging into one connected AI-assisted
            lung health platform.
        </div>
    </div>
    """)

    col_cta1, col_cta2, _ = st.columns(
        [1.3, 1.3, 5.4],
        gap="small"
    )

    with col_cta1:
        if st.button(
            "Explore JIVA",
            type="primary",
            use_container_width=True,
            key="hero_explore"
        ):
            st.session_state["current_page"] = "VOC Analysis"
            st.rerun()

    with col_cta2:
        if st.button(
            "Ask JIVA",
            use_container_width=True,
            key="hero_ask"
        ):
            st.session_state["current_page"] = "Home"
            st.session_state["scroll_to_ask_jiva"] = True
            st.rerun()

    st.html("<div style='height:3rem;'></div>")

    # BUILT FOR MULTIMODAL LUNG ANALYSIS

    st.html("""
    <div style="margin-bottom:2rem;">
        <div class="section-heading">
            Built for multimodal lung analysis
        </div>
    </div>
    """)

    card_col1, card_col2, card_col3, card_col4 = st.columns(
        4,
        gap="medium"
    )

    with card_col1:
        st.html("""
        <div class="sarvam-card jiva-card-voc">
            <div class="sarvam-card-corner-accent"></div>

            <div class="sarvam-card-title">
                VOC Analysis
            </div>

            <div class="sarvam-card-desc">
                Analyze volatile organic compounds in breath to
                identify patterns associated with Control,
                Benign and Cancer classes.
            </div>

            <div class="sarvam-support-line">
                Breath biomarkers · Machine learning · Explainability
            </div>
        </div>
        """)

        if st.button(
            "Explore VOC Analysis",
            use_container_width=True,
            key="card_btn_voc"
        ):
            st.session_state["current_page"] = "VOC Analysis"
            st.rerun()

    with card_col2:
        st.html("""
        <div class="sarvam-card jiva-card-audio">
            <div class="sarvam-card-corner-accent"></div>

            <div class="sarvam-card-title">
                Audio Analysis
            </div>

            <div class="sarvam-card-desc">
                Analyze respiratory and cardiopulmonary recordings
                using waveform, spectrogram and audio-language
                intelligence.
            </div>

            <div class="sarvam-support-line">
                Respiratory audio · Signal analysis · StethoLM
            </div>
        </div>
        """)

        if st.button(
            "Explore Audio Analysis",
            use_container_width=True,
            key="card_btn_audio"
        ):
            st.session_state["current_page"] = "Audio Analysis"
            st.rerun()

    with card_col3:
        st.html("""
        <div class="sarvam-card jiva-card-xray">
            <div class="sarvam-card-corner-accent"></div>

            <div class="sarvam-card-title">
                X-ray Analysis
            </div>

            <div class="sarvam-card-desc">
                Review chest X-ray images through a dedicated
                medical-imaging workflow prepared for
                deep-learning integration.
            </div>

            <div class="sarvam-support-line">
                Chest imaging · Visual analysis · Explainability
            </div>
        </div>
        """)

        if st.button(
            "Explore X-ray Analysis",
            use_container_width=True,
            key="card_btn_xray"
        ):
            st.session_state["current_page"] = "X-ray Analysis"
            st.rerun()

    with card_col4:
        st.html("""
        <div class="sarvam-card jiva-card-video">
            <div class="sarvam-card-corner-accent"></div>

            <div class="sarvam-card-title">
                Video Analysis
            </div>

            <div class="sarvam-card-desc">
                Non-invasive face video photoplethysmography
                heart-rate estimation using Spatio-Temporal
                ViT (PhysFormer).
            </div>

            <div class="sarvam-support-line">
                Video rPPG · Spatio-temporal ViT · Heart Rate
            </div>
        </div>
        """)

        if st.button(
            "Explore Video Analysis",
            use_container_width=True,
            key="card_btn_vitals"
        ):
            st.session_state["current_page"] = "Video Analysis"
            st.rerun()

    st.html("<div style='height:3rem;'></div>")

    # SECONDARY GRADIENT CARDS

    g_col1, g_col2, g_col3 = st.columns(
        3,
        gap="medium"
    )

    with g_col1:
        st.html("""
        <div class="sarvam-grid-card grid-breath"
             style="margin-bottom:1.25rem;">
            <div class="sarvam-grid-title">
                Breath Intelligence
            </div>

            <div class="sarvam-grid-desc">
                VOC biomarker patterns transformed into
                interpretable model predictions.
            </div>
        </div>
        """)

        st.html("""
        <div class="sarvam-grid-card grid-xai">
            <div class="sarvam-grid-title">
                Explainable AI
            </div>

            <div class="sarvam-grid-desc">
                Understand predictions through feature importance,
                SHAP and model-performance analysis.
            </div>
        </div>
        """)

    with g_col2:
        st.html("""
        <div class="sarvam-grid-card grid-audio"
             style="margin-bottom:1.25rem;">
            <div class="sarvam-grid-title">
                Audio Intelligence
            </div>

            <div class="sarvam-grid-desc">
                Cardiopulmonary recordings transformed into
                acoustic and language-based findings.
            </div>
        </div>
        """)

        st.html("""
        <div class="sarvam-grid-card grid-performance">
            <div class="sarvam-grid-title">
                Model Evaluation
            </div>

            <div class="sarvam-grid-desc">
                Compare performance using accuracy, precision,
                recall, F1, MCC and ROC-AUC.
            </div>
        </div>
        """)

    with g_col3:
        st.html("""
        <div class="sarvam-grid-card grid-imaging"
             style="margin-bottom:1.25rem;">
            <div class="sarvam-grid-title">
                Medical Imaging
            </div>

            <div class="sarvam-grid-desc">
                Chest images organized for visual AI analysis
                and future model integration.
            </div>
        </div>
        """)

        st.html("""
        <div class="sarvam-grid-card grid-jiva">
            <div class="sarvam-grid-title">
                Ask JIVA
            </div>

            <div class="sarvam-grid-desc">
                Explore results and understand the platform
                through a conversational interface.
            </div>
        </div>
        """)

    st.html("<div style='height:4rem;'></div>")

    # SEE BEYOND A SINGLE SIGNAL

    vis_col1, vis_col2 = st.columns(
        [1.1, 1.2],
        gap="large"
    )

    with vis_col1:
        st.html("""
        <div style="padding-top:1rem;">
            <div class="editorial-heading">
                See beyond a single signal
            </div>

            <div class="editorial-text">
                Breath chemistry, respiratory sounds,
                medical imaging and facial rPPG hemodynamics
                each reveal a different part of cardiopulmonary
                health. JIVA brings these signals into one
                connected workflow.
            </div>
        </div>
        """)

        if st.button(
            "Explore X-ray Analysis",
            type="primary",
            key="editorial_explore_xray"
        ):
            st.session_state["current_page"] = "X-ray Analysis"
            st.rerun()

    with vis_col2:
        asset_path = Path(
            "assets/jiva_xray_hero.png"
        )

        if asset_path.exists():
            st.image(
                str(asset_path),
                use_container_width=True
            )

        else:
            st.html("""
            <div style="
                background:
                    linear-gradient(
                        145deg,
                        rgba(245,253,255,0.96),
                        rgba(199,232,244,0.90)
                    );
                border:1px solid rgba(255,255,255,0.68);
                border-radius:20px;
                padding:5rem 2rem;
                text-align:center;
                color:#56707D;
                box-shadow:
                    0 10px 25px rgba(33,118,150,0.08);
            ">
                <div style="
                    font-size:1.1rem;
                    font-weight:700;
                    color:#17384A;
                    margin-bottom:0.5rem;
                ">
                    Radiological Inspection Container
                </div>

                <div>
                    assets/jiva_xray_hero.png
                </div>
            </div>
            """)

    st.html("""
    <div style="height:2rem;"></div>

    <hr style="
        border:none;
        border-top:1px solid rgba(48,140,175,0.18);
        margin-bottom:3rem;
    ">
    """)

    # ASK JIVA

    st.html("""
    <div
        id="ask-jiva"
        style="scroll-margin-top:100px;">
    </div>

    <div style="margin-bottom:1.5rem;">
        <div style="
            font-size:2.2rem;
            font-weight:800;
            color:#252525;
            margin-bottom:0.4rem;
            letter-spacing:-0.03em;
        ">
            Ask JIVA
        </div>

        <div style="
            color:#60707A;
            font-size:1.05rem;
        ">
            Ask about VOC analysis, cardiopulmonary audio,
            X-ray analysis, model performance or explainability.
        </div>
    </div>
    """)

    render_chatbot()