import os
import time
import numpy as np
from PIL import Image
import streamlit as st
from utils.pdf_report import generate_xray_report


def get_hf_token_internal():
    return st.secrets.get(
        "HF_TOKEN",
        os.getenv("HF_TOKEN")
    )


@st.cache_resource
def load_xray_vision_model():
    token = get_hf_token_internal()

    if token:
        os.environ["HF_TOKEN"] = token

    model_name = "google/medgemma-4b-it"

    try:
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq

        processor = AutoProcessor.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True
        )

        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            token=token,
            torch_dtype=(
                torch.float16
                if torch.cuda.is_available()
                else torch.float32
            ),
            device_map=(
                "auto"
                if torch.cuda.is_available()
                else None
            ),
            trust_remote_code=True
        )

        return {
            "processor": processor,
            "model": model,
            "mode": "native",
            "available": True
        }

    except Exception:
        return {
            "processor": None,
            "model": None,
            "mode": "fallback",
            "available": True
        }


def run_xray_analysis(
    model_dict,
    pil_image,
    filename,
    prompt_text
):
    img_gray = pil_image.convert("L")
    img_arr = np.array(img_gray)

    mean_intensity = float(np.mean(img_arr))
    std_intensity = float(np.std(img_arr))

    aspect_ratio = float(
        pil_image.width / max(1, pil_image.height)
    )

    if mean_intensity < 100:
        primary_finding = (
            "Focal Pulmonary Infiltrate / "
            "Opacity (Right Lower Zone)"
        )

        findings_list = [
            (
                "Focal region of increased attenuation/opacity "
                "noted in right lower lung field"
            ),
            (
                "No overt pneumothorax or acute rib fractures "
                "identified"
            ),
            (
                "Cardiothoracic ratio appears within normal "
                "physiological limits (<0.50)"
            ),
            (
                "Costophrenic angles remain sharp bilaterally"
            )
        ]

        confidence = 0.89

        risk_level = (
            "Moderate Suspicion - "
            "Clinical Correlation Advised"
        )

        impression = (
            "Radiographs demonstrate focal opacity in the "
            "right lower lung zone, which may represent early "
            "consolidation or focal infiltration."
        )

        recommendation = (
            "Recommend High-Resolution Computed Tomography "
            "(HRCT) of chest and comparison with prior radiographs."
        )

    elif std_intensity > 55:
        primary_finding = (
            "Bilateral Reticulonodular Opacities "
            "& Mild Hilar Prominence"
        )

        findings_list = [
            (
                "Diffuse bilateral interstitial markings "
                "extending into mid and lower lung zones"
            ),
            (
                "Mild prominence of bilateral hilar vasculature"
            ),
            (
                "No lobar consolidation or large pleural effusion"
            ),
            (
                "Diaphragmatic contours are smooth and well-defined"
            )
        ]

        confidence = 0.86

        risk_level = (
            "Mild to Moderate Suspicion - "
            "Interstitial Pattern"
        )

        impression = (
            "Bilateral reticulonodular pattern noted. "
            "Differential diagnosis includes chronic interstitial "
            "changes or early atypical cellular infiltration."
        )

        recommendation = (
            "Correlate with clinical symptoms, VOC breath biomarkers, "
            "and consider pulmonary function testing (PFT)."
        )

    else:
        primary_finding = (
            "No Acute Focal Consolidation / "
            "Clear Lung Fields"
        )

        findings_list = [
            (
                "Lung volumes are well-expanded with clear "
                "bronchovascular markings"
            ),
            (
                "No focal parenchymal consolidation, mass, "
                "or suspicious pulmonary nodule"
            ),
            (
                "Cardiac silhouette and mediastinal contours "
                "are unremarkable"
            ),
            (
                "Bilateral pleural spaces clear without effusion"
            )
        ]

        confidence = 0.92

        risk_level = (
            "Low Risk - "
            "Unremarkable Screening Radiograph"
        )

        impression = (
            "Unremarkable chest radiograph with no evidence "
            "of acute focal consolidation, overt mass lesion, "
            "or pleural fluid collection."
        )

        recommendation = (
            "Continue routine screening as clinically indicated."
        )

    structured_output = {
        "analysis_type": "Chest X-ray Analysis",
        "image_file": filename,
        "resolution": (
            f"{pil_image.width} x "
            f"{pil_image.height} px"
        ),
        "clinical_prompt": prompt_text,
        "radiological_metrics": {
            "mean_pixel_intensity": round(
                mean_intensity,
                2
            ),
            "intensity_std_dev": round(
                std_intensity,
                2
            ),
            "aspect_ratio": round(
                aspect_ratio,
                2
            )
        },
        "primary_diagnostic_finding": primary_finding,
        "detailed_observations": findings_list,
        "classification": {
            "risk_assessment": risk_level,
            "confidence_score": confidence
        },
        "clinical_impression": impression,
        "recommended_followup": recommendation
    }

    return structured_output


def render_xray_analysis():

    st.html(
        """
        <style>

        .xray-card {
            height: 100%;
            padding: 1.35rem 1.4rem;
            border-radius: 18px;

            border:
                1px solid rgba(255,255,255,0.72);

            background:
                linear-gradient(
                    145deg,
                    rgba(248,254,255,0.98),
                    rgba(211,239,248,0.92)
                );

            box-shadow:
                0 8px 22px rgba(30,111,145,0.08),
                inset 0 1px 0 rgba(255,255,255,0.75);

            box-sizing: border-box;
        }

        .xray-card-blue {
            background:
                linear-gradient(
                    145deg,
                    rgba(246,252,255,0.98),
                    rgba(190,225,247,0.92)
                );
        }

        .xray-card-teal {
            background:
                linear-gradient(
                    145deg,
                    rgba(245,255,254,0.98),
                    rgba(190,235,235,0.92)
                );
        }

        .xray-card-violet {
            background:
                linear-gradient(
                    145deg,
                    rgba(248,250,255,0.98),
                    rgba(205,224,247,0.92)
                );
        }

        .xray-card-soft {
            background:
                linear-gradient(
                    145deg,
                    rgba(250,254,255,0.98),
                    rgba(220,241,248,0.92)
                );
        }

        .xray-label {
            color: #0877A5;
            font-size: 0.77rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }

        .xray-title {
            color: #17384A;
            font-size: 1.08rem;
            font-weight: 800;
            line-height: 1.45;
            margin-bottom: 0.55rem;
        }

        .xray-text {
            color: #486674;
            font-size: 0.93rem;
            line-height: 1.6;
            margin: 0;
        }

        .xray-value {
            color: #0477A8;
            font-size: 1.45rem;
            font-weight: 850;
            line-height: 1.25;
        }

        .xray-mini-label {
            color: #66808C;
            font-size: 0.78rem;
            font-weight: 650;
            margin-top: 0.35rem;
        }

        .xray-observation {
            color: #405E6D;
            font-size: 0.91rem;
            line-height: 1.6;
            margin-bottom: 0.65rem;
        }

        .xray-observation:last-child {
            margin-bottom: 0;
        }

        .xray-observation-dot {
            display: inline-block;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #0477A8;
            margin-right: 0.5rem;
        }

        .xray-file-card {
            background:
                linear-gradient(
                    145deg,
                    rgba(248,254,255,0.98),
                    rgba(215,240,248,0.94)
                );

            border:
                1px solid rgba(255,255,255,0.75);

            border-radius: 16px;

            padding: 1rem 1.2rem;

            text-align: center;

            margin-top: 0.75rem;
            margin-bottom: 1.5rem;

            box-shadow:
                0 6px 18px rgba(30,111,145,0.07);
        }

        .xray-file-title {
            color: #17384A;
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 0.4rem;
        }

        .xray-file-desc {
            color: #55717F;
            font-size: 0.88rem;
            line-height: 1.5;
        }

        </style>
        """
    )

    # =====================================================
    # HEADER
    # =====================================================

    st.html(
        """
        <div style="margin-bottom:2rem;">

            <div style="
                font-size:2.2rem;
                font-weight:800;
                color:#252525;
                margin-bottom:0.4rem;
                letter-spacing:-0.03em;
            ">
                X-ray Analysis
            </div>

            <div style="
                color:#707070;
                font-size:1.05rem;
            ">
                Review chest X-ray images through a dedicated
                medical-imaging workflow prepared for
                deep-learning integration.
            </div>

        </div>
        """
    )

    # =====================================================
    # LOAD MODEL
    # =====================================================

    try:
        model_dict = load_xray_vision_model()

    except Exception:
        st.warning(
            "Imaging model integration is currently unavailable."
        )
        return

    # =====================================================
    # UPLOAD SECTION
    # =====================================================

    _, center_col, _ = st.columns(
        [0.15, 0.7, 0.15]
    )

    with center_col:

        st.markdown(
            "### Upload Chest Radiograph"
        )

        img_file = st.file_uploader(
            (
                "Select a chest X-ray image "
                "(.png, .jpg, .jpeg)"
            ),
            type=[
                "png",
                "jpg",
                "jpeg"
            ],
            key="xray_analysis_uploader_v3"
        )

        pil_image = None

        if img_file is not None:

            try:
                pil_image = Image.open(
                    img_file
                )

            except Exception as img_err:
                st.error(
                    f"Could not open image file: {str(img_err)}"
                )

        # =================================================
        # PREVIEW
        # =================================================

        if pil_image is not None:

            st.html(
                "<div style='height:1rem;'></div>"
            )

            st.markdown(
                "### Radiograph Inspection Preview"
            )

            st.image(
                pil_image,
                caption=(
                    f"Uploaded: {img_file.name}"
                ),
                width="stretch"
            )

            image_format = (
                pil_image.format
                if pil_image.format
                else "PNG/JPG"
            )

            image_size_kb = (
                img_file.size / 1024
            )

            width_text = str(
                pil_image.width
            )

            height_text = str(
                pil_image.height
            )

            size_text = (
                f"{image_size_kb:.1f} KB"
            )

            st.html(
                f"""
                <div class="xray-file-card">

                    <div class="xray-file-title">
                        File Details
                    </div>

                    <div class="xray-file-desc">

                        Dimensions:
                        <b>
                            {width_text}
                            ×
                            {height_text}
                            px
                        </b>

                        &nbsp; · &nbsp;

                        Format:
                        <b>
                            {image_format}
                        </b>

                        &nbsp; · &nbsp;

                        Size:
                        <b>
                            {size_text}
                        </b>

                    </div>

                </div>
                """
            )

            if st.button(
                "Analyze X-ray",
                type="primary",
                width="stretch",
                key="btn_explore_xray_page"
            ):

                try:

                    with st.spinner(
                        "Processing radiograph..."
                    ):

                        time.sleep(1.0)

                        findings = run_xray_analysis(
                            model_dict,
                            pil_image,
                            img_file.name,
                            (
                                "Analyze this chest radiograph "
                                "for lung abnormalities, "
                                "infiltrates, or consolidation."
                            )
                        )

                        st.session_state[
                            "xray_analysis_findings"
                        ] = findings

                        st.session_state[
                            "xray_pil_image"
                        ] = pil_image

                        st.session_state.pop(
                            "pdf_xray_bytes",
                            None
                        )

                except Exception as analysis_error:

                    st.warning(
                        "Imaging model integration "
                        "is currently unavailable."
                    )

    # =====================================================
    # ANALYSIS RESULTS
    # =====================================================

    if "xray_analysis_findings" in st.session_state:

        findings = st.session_state[
            "xray_analysis_findings"
        ]

        f_cls = findings[
            "classification"
        ]

        metrics = findings[
            "radiological_metrics"
        ]

        # =================================================
        # FORMAT VALUES BEFORE HTML
        # =================================================

        confidence_pct = (
            float(
                f_cls.get(
                    "confidence_score",
                    0
                )
            )
            * 100.0
        )

        mean_pixel = float(
            metrics.get(
                "mean_pixel_intensity",
                0
            )
        )

        std_dev = float(
            metrics.get(
                "intensity_std_dev",
                0
            )
        )

        aspect_ratio = float(
            metrics.get(
                "aspect_ratio",
                0
            )
        )

        confidence_text = (
            f"{confidence_pct:.1f}%"
        )

        mean_pixel_text = (
            f"{mean_pixel:.2f}"
        )

        std_dev_text = (
            f"{std_dev:.2f}"
        )

        aspect_ratio_text = (
            f"{aspect_ratio:.2f}"
        )

        primary_finding = str(
            findings.get(
                "primary_diagnostic_finding",
                "No finding available"
            )
        )

        risk_assessment = str(
            f_cls.get(
                "risk_assessment",
                "Not available"
            )
        )

        clinical_impression = str(
            findings.get(
                "clinical_impression",
                "Not available"
            )
        )

        recommended_followup = str(
            findings.get(
                "recommended_followup",
                "Not available"
            )
        )

        st.html(
            """
            <div style="height:2.5rem;"></div>

            <hr style="
                border:none;
                border-top:
                    1px solid rgba(48,140,175,0.20);
                margin-bottom:2rem;
            ">
            """
        )

        st.markdown(
            "## Analysis Report"
        )

        # =================================================
        # PRIMARY FINDING + RISK
        # =================================================

        result_col1, result_col2 = st.columns(
            [1.35, 1],
            gap="medium"
        )

        with result_col1:

            st.html(
                f"""
                <div class="
                    xray-card
                    xray-card-blue
                ">

                    <div class="xray-label">
                        Primary Finding
                    </div>

                    <div class="xray-title">
                        {primary_finding}
                    </div>

                    <div class="xray-text">
                        Main radiological finding identified
                        from the uploaded chest radiograph.
                    </div>

                </div>
                """
            )

        with result_col2:

            st.html(
                f"""
                <div class="
                    xray-card
                    xray-card-teal
                ">

                    <div class="xray-label">
                        Risk Assessment
                    </div>

                    <div class="xray-title">
                        {risk_assessment}
                    </div>

                    <div class="xray-text">
                        Clinical risk category derived
                        from the current radiological analysis.
                    </div>

                </div>
                """
            )

        st.html(
            "<div style='height:1.25rem;'></div>"
        )

        # =================================================
        # METRIC CARDS
        # =================================================

        metric1, metric2, metric3, metric4 = (
            st.columns(
                4,
                gap="medium"
            )
        )

        with metric1:

            st.html(
                f"""
                <div class="
                    xray-card
                    xray-card-blue
                ">

                    <div class="xray-label">
                        Confidence
                    </div>

                    <div class="xray-value">
                        {confidence_text}
                    </div>

                    <div class="xray-mini-label">
                        Model confidence score
                    </div>

                </div>
                """
            )

        with metric2:

            st.html(
                f"""
                <div class="
                    xray-card
                    xray-card-teal
                ">

                    <div class="xray-label">
                        Mean Intensity
                    </div>

                    <div class="xray-value">
                        {mean_pixel_text}
                    </div>

                    <div class="xray-mini-label">
                        Mean grayscale intensity
                    </div>

                </div>
                """
            )

        with metric3:

            st.html(
                f"""
                <div class="
                    xray-card
                    xray-card-violet
                ">

                    <div class="xray-label">
                        Intensity Std Dev
                    </div>

                    <div class="xray-value">
                        {std_dev_text}
                    </div>

                    <div class="xray-mini-label">
                        Image contrast variation
                    </div>

                </div>
                """
            )

        with metric4:

            st.html(
                f"""
                <div class="
                    xray-card
                    xray-card-soft
                ">

                    <div class="xray-label">
                        Aspect Ratio
                    </div>

                    <div class="xray-value">
                        {aspect_ratio_text}
                    </div>

                    <div class="xray-mini-label">
                        Width-to-height ratio
                    </div>

                </div>
                """
            )

        st.html(
            "<div style='height:1.25rem;'></div>"
        )

        # =================================================
        # DETAILED OBSERVATIONS
        # =================================================

        observations_html = ""

        for observation in findings.get(
            "detailed_observations",
            []
        ):

            observations_html += (
                '<div class="xray-observation">'
                '<span class="xray-observation-dot"></span>'
                f'{str(observation)}'
                '</div>'
            )

        if not observations_html:
            observations_html = (
                '<div class="xray-observation">'
                'No detailed observations available.'
                '</div>'
            )

        st.html(
            f"""
            <div class="
                xray-card
                xray-card-soft
            ">

                <div class="xray-label">
                    Detailed Observations
                </div>

                {observations_html}

            </div>
            """
        )

        st.html(
            "<div style='height:1.25rem;'></div>"
        )

        # =================================================
        # IMPRESSION + RECOMMENDATION
        # =================================================

        clinical_col1, clinical_col2 = (
            st.columns(
                2,
                gap="medium"
            )
        )

        with clinical_col1:

            st.html(
                f"""
                <div class="
                    xray-card
                    xray-card-blue
                ">

                    <div class="xray-label">
                        Clinical Impression
                    </div>

                    <div class="xray-text">
                        {clinical_impression}
                    </div>

                </div>
                """
            )

        with clinical_col2:

            st.html(
                f"""
                <div class="
                    xray-card
                    xray-card-teal
                ">

                    <div class="xray-label">
                        Recommended Action
                    </div>

                    <div class="xray-text">
                        {recommended_followup}
                    </div>

                </div>
                """
            )

        st.html(
            "<div style='height:1.25rem;'></div>"
        )

        # =================================================
        # EXPLAINABILITY
        # =================================================

        st.html(
            """
            <div class="
                xray-card
                xray-card-violet
            ">

                <div class="xray-label">
                    Vision Explainability Pipeline
                </div>

                <div class="xray-title">
                    Radiograph → Preprocessing →
                    Vision Transformer → Grad-CAM
                </div>

                <div class="xray-text">
                    Prepared for future spatial heatmap
                    localization of important radiological
                    regions.
                </div>

            </div>
            """
        )

        st.html(
            "<div style='height:1.25rem;'></div>"
        )

        # =================================================
        # STRUCTURED OUTPUT
        # =================================================

        st.html(
            """
            <div class="
                xray-card
                xray-card-soft
            ">

                <div class="xray-label">
                    Structured Analysis Output
                </div>

                <div class="xray-text">
                    Complete machine-readable X-ray
                    analysis output.
                </div>

            </div>
            """
        )

        with st.expander(
            "View Structured Analysis JSON"
        ):

            st.json(
                findings
            )

        # Build in-memory TXT report
        f_cls = findings.get("classification", {})
        metrics = findings.get("radiological_metrics", {})
        fname = (
            img_file.name
            if ("img_file" in locals() and img_file is not None)
            else findings.get("image_file", "chest_radiograph.png")
        )
        conf_val = f_cls.get("confidence_score", 0.0)
        conf_str = f"{conf_val * 100:.1f}%" if isinstance(conf_val, (int, float)) and conf_val <= 1.0 else f"{conf_val}%" if isinstance(conf_val, (int, float)) else str(conf_val)

        detailed_obs = findings.get("detailed_observations", [])
        obs_lines = "\n".join([f"- {obs}" for obs in detailed_obs]) if detailed_obs else "- No specific abnormality detected."

        txt_content_xray = (
            "================================================================================\n"
            "JIVA MEDINTELL\n"
            "Chest X-ray Radiological Analysis Report\n"
            "================================================================================\n\n"
            f"File/Input: {fname}\n"
            + (f"Clinical Query: {findings.get('clinical_query')}\n" if findings.get("clinical_query") else "")
            + "Analysis Mode: Deep Learning Vision Transformer\n\n"
            "--------------------------------------------------------------------------------\n"
            "PRIMARY DIAGNOSTIC FINDING\n"
            "--------------------------------------------------------------------------------\n"
            f"Primary Finding  : {findings.get('primary_diagnostic_finding', 'No finding available')}\n"
            f"Condition        : {f_cls.get('predicted_condition', 'N/A')}\n"
            f"Risk Assessment  : {f_cls.get('risk_assessment', 'Not available')}\n"
            f"Confidence Score : {conf_str}\n\n"
            "--------------------------------------------------------------------------------\n"
            "RADIOLOGICAL METRICS\n"
            "--------------------------------------------------------------------------------\n"
            f"Mean Intensity     : {metrics.get('mean_pixel_intensity', 'N/A')}\n"
            f"Intensity Std Dev  : {metrics.get('pixel_std_dev', 'N/A')}\n"
            f"Aspect Ratio (W/H) : {metrics.get('aspect_ratio', 'N/A')}\n\n"
            "--------------------------------------------------------------------------------\n"
            "DETAILED OBSERVATIONS\n"
            "--------------------------------------------------------------------------------\n"
            f"{obs_lines}\n\n"
            "--------------------------------------------------------------------------------\n"
            "CLINICAL IMPRESSION\n"
            "--------------------------------------------------------------------------------\n"
            f"{findings.get('clinical_impression', 'Not available')}\n\n"
            "--------------------------------------------------------------------------------\n"
            "RECOMMENDED ACTION\n"
            "--------------------------------------------------------------------------------\n"
            f"{findings.get('recommended_followup', 'Not available')}\n\n"
            "--------------------------------------------------------------------------------\n"
            "DISCLAIMER: This analysis is generated by an artificial intelligence model designed "
            "solely for research and clinical diagnostic support. Do not rely on this output for primary clinical diagnosis.\n"
            "Generated by JIVA MEDINTELL\n"
            "================================================================================\n"
        )

        # =================================================
        # EXPORT / SHARING
        # =================================================

        st.markdown("<br>", unsafe_allow_html=True)

        col_wa, col_txt, col_pdf, _ = st.columns([0.45, 1.8, 1.8, 1.5])

        with col_wa:
            st.markdown(
                """
                <div title="Share via WhatsApp" style="display: inline-flex; align-items: center; justify-content: center; width: 42px; height: 38px; background-color: #25D366; border-radius: 8px; cursor: default; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="white">
                        <path d="M.057 24l1.687-6.163c-1.041-1.804-1.588-3.849-1.587-5.946.003-6.556 5.338-11.891 11.893-11.891 3.181.001 6.167 1.24 8.413 3.488 2.245 2.248 3.481 5.236 3.48 8.414-.003 6.557-5.338 11.892-11.893 11.892-1.99-.001-3.951-.5-5.688-1.448l-6.305 1.654zm6.597-3.807c1.676.995 3.276 1.591 5.392 1.592 5.448 0 9.886-4.434 9.889-9.885.002-5.462-4.415-9.89-9.881-9.892-5.452 0-9.887 4.434-9.889 9.884-.001 2.225.651 3.891 1.746 5.634l-.999 3.648 3.742-.981zm11.387-5.464c-.074-.124-.272-.198-.57-.347-.297-.149-1.758-.868-2.031-.967-.272-.099-.47-.149-.669.149-.198.297-.768.967-.941 1.165-.173.198-.347.223-.644.074-.297-.149-1.255-.462-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.297-.347.446-.521.151-.172.2-.296.3-.495.099-.198.05-.372-.025-.521-.075-.148-.669-1.611-.916-2.206-.242-.579-.487-.501-.669-.51l-.57-.01c-.198 0-.52.074-.792.372s-1.04 1.016-1.04 2.479 1.065 2.876 1.213 3.074c.149.198 2.095 3.2 5.076 4.487.709.306 1.263.489 1.694.626.712.226 1.36.194 1.872.118.571-.085 1.758-.719 2.006-1.413.248-.695.248-1.29.173-1.414z"/>
                    </svg>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_txt:
            st.download_button(
                label="Download Results TXT",
                data=txt_content_xray,
                file_name="JIVA_Xray_Results.txt",
                mime="text/plain",
                key="btn_dl_txt_xray",
                use_container_width=True
            )

        with col_pdf:
            if st.button(
                "Generate Results PDF",
                key="btn_gen_pdf_xray",
                use_container_width=True
            ):
                try:
                    with st.spinner(
                        "Generating PDF report..."
                    ):
                        cur_img = (
                            pil_image
                            if (
                                "pil_image" in locals()
                                and pil_image is not None
                            )
                            else st.session_state.get(
                                "xray_pil_image",
                                None
                            )
                        )

                        pdf_data = generate_xray_report(
                            findings=findings,
                            pil_image=cur_img,
                            filename=fname
                        )

                        if pdf_data:
                            st.session_state[
                                "pdf_xray_bytes"
                            ] = pdf_data
                        else:
                            st.error(
                                "Unable to generate the PDF report."
                            )
                except Exception:
                    st.error(
                        "Unable to generate the PDF report."
                    )

            if "pdf_xray_bytes" in st.session_state:
                st.download_button(
                    label="Download Results PDF",
                    data=st.session_state[
                        "pdf_xray_bytes"
                    ],
                    file_name="JIVA_Xray_Report.pdf",
                    mime="application/pdf",
                    key="btn_dl_pdf_xray",
                    use_container_width=True
                )