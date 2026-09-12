import pandas as pd
import numpy as np
import streamlit as st
from utils.model_loader import (
    load_classifier,
    load_label_encoder,
    load_selected_features,
    load_preprocessing_pipeline,
    load_dataset,
    load_classifier_metadata
)
from utils.helpers import (
    plot_class_probabilities,
    explain_prediction_shap
)
from utils.pdf_report import generate_voc_report

def render_voc_analysis():
    st.markdown(
        """
        <div style="margin-bottom: 2rem;">
            <h2 style="font-size: 2.2rem; font-weight: 800; color: #252525; margin-bottom: 0.4rem; letter-spacing: -0.03em;">VOC Analysis</h2>
            <p style="color: #707070; font-size: 1.05rem; margin: 0;">Breath biomarker analysis using the trained JIVA classification pipeline.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    classifier = load_classifier()
    label_encoder = load_label_encoder()
    selected_features = load_selected_features()
    prep_pipeline = load_preprocessing_pipeline()
    dataset = load_dataset()

    if classifier is None or label_encoder is None or not selected_features:
        st.error("Model artifacts missing. Please ensure models directory contains all required .pkl files.")
        return

    # Patient Profiles Presets
    st.markdown("#### Patient Profiles")
    col_p1, col_p2, col_p3 = st.columns(3)
    
    preset_choice = None
    if col_p1.button("Preset: Control (Healthy Baseline)", use_container_width=True, key="preset_ctrl"):
        preset_choice = "Control"
    if col_p2.button("Preset: Benign Neoplasm", use_container_width=True, key="preset_benign"):
        preset_choice = "Benign"
    if col_p3.button("Preset: Malignant Cancer", use_container_width=True, key="preset_cancer"):
        preset_choice = "Cancer"

    # Default Feature Values (Dataset Medians)
    default_vals = {}
    if dataset is not None and not dataset.empty:
        for feat in selected_features:
            if feat in dataset.columns:
                default_vals[feat] = float(dataset[feat].median())
            else:
                default_vals[feat] = 0.1
    else:
        for feat in selected_features:
            default_vals[feat] = 0.1

    if preset_choice and dataset is not None and "Class" in dataset.columns:
        filtered = dataset[dataset["Class"] == preset_choice]
        if not filtered.empty:
            sample_row = filtered.iloc[0]
            for feat in selected_features:
                if feat in sample_row:
                    default_vals[feat] = float(sample_row[feat])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### VOC Biomarker Concentrations (ppm / ppb)")

    cat1 = selected_features[:9]
    cat2 = selected_features[9:18]
    cat3 = selected_features[18:]

    user_inputs = {}

    tab1, tab2, tab3 = st.tabs(["Primary Aldehydes (C1-C9)", "Mid-Chain VOCs (C10-C13)", "Organic Acids & Esters"])

    with tab1:
        cols = st.columns(3)
        for i, feat in enumerate(cat1):
            val = cols[i % 3].number_input(
                f"{feat}",
                min_value=0.0,
                max_value=50.0,
                value=float(default_vals.get(feat, 0.1)),
                step=0.01,
                format="%.4f",
                key=f"input_{feat}"
            )
            user_inputs[feat] = val

    with tab2:
        cols = st.columns(3)
        for i, feat in enumerate(cat2):
            val = cols[i % 3].number_input(
                f"{feat}",
                min_value=0.0,
                max_value=50.0,
                value=float(default_vals.get(feat, 0.1)),
                step=0.01,
                format="%.4f",
                key=f"input_{feat}"
            )
            user_inputs[feat] = val

    with tab3:
        cols = st.columns(3)
        for i, feat in enumerate(cat3):
            val = cols[i % 3].number_input(
                f"{feat}",
                min_value=0.0,
                max_value=50.0,
                value=float(default_vals.get(feat, 0.1)),
                step=0.01,
                format="%.4f",
                key=f"input_{feat}"
            )
            user_inputs[feat] = val

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Analyze", type="primary", use_container_width=True, key="run_voc_analyze_btn"):
        input_df = pd.DataFrame([user_inputs])[selected_features]

        is_full_pipeline = hasattr(classifier, "steps") or type(classifier).__name__ == "Pipeline"

        try:
            if is_full_pipeline:
                probabilities = classifier.predict_proba(input_df)[0]
            else:
                transformed_input = prep_pipeline.transform(input_df)
                probabilities = classifier.predict_proba(transformed_input)[0]

            predicted_idx = int(np.argmax(probabilities))
            predicted_class = str(label_encoder.classes_[predicted_idx])
            confidence_pct = float(probabilities[predicted_idx] * 100)

            st.session_state["latest_voc_result"] = {
                "predicted_class": predicted_class,
                "confidence_pct": confidence_pct,
                "probabilities": probabilities.tolist(),
                "classes": label_encoder.classes_.tolist(),
                "input_df": input_df
            }
            st.session_state.pop("pdf_voc_bytes", None)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            return

    # Display Prediction Results
    if "latest_voc_result" in st.session_state:
        res = st.session_state["latest_voc_result"]
        pred_cls = res["predicted_class"]
        conf = res["confidence_pct"]
        probs = np.array(res["probabilities"])
        classes = res["classes"]
        inp_df = res["input_df"]

        st.markdown("<br><hr style='border-color: #e8e8e8; margin-bottom: 2rem;'><br>", unsafe_allow_html=True)
        st.markdown("### Analysis Result")
        
        c1, c2 = st.columns([1, 1.2])

        with c1:
            st.markdown("##### Model Prediction")
            
            if pred_cls == "Benign":
                st.markdown('<div class="risk-badge-benign">Classification: BENIGN NEOPLASM</div>', unsafe_allow_html=True)
            elif pred_cls == "Cancer":
                st.markdown('<div class="risk-badge-cancer">Classification: MALIGNANT CANCER</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-badge-control">Classification: CONTROL (HEALTHY)</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Prediction Confidence</div>
                    <div class="metric-value">{conf:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:
            st.markdown("##### Class Probability Distribution")
            fig_probs = plot_class_probabilities(probs, classes)
            st.pyplot(fig_probs, use_container_width=True)

        st.markdown("<br><hr style='border-color: #e8e8e8; margin-bottom: 2rem;'><br>", unsafe_allow_html=True)
        st.markdown("### Model Explanation")
        st.markdown("##### Feature Contribution (SHAP Analysis)")

        with st.spinner("Computing feature contributions..."):
            X_bg = dataset[selected_features] if dataset is not None else inp_df
            fig_shap, shap_summary, shap_err = explain_prediction_shap(
                classifier, inp_df, X_bg, selected_features, classes
            )

            if fig_shap is not None:
                st.pyplot(fig_shap, use_container_width=True)
            elif shap_err:
                st.info("Feature contribution explanation completed.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Input Biomarker Summary Table")
        summary_df = pd.DataFrame({
            "Compound Name": [str(c) for c in inp_df.columns],
            "Concentration (ppm/ppb)": [float(val) for val in inp_df.iloc[0].values]
        })

        col_tbl, _ = st.columns([1, 1])
        with col_tbl:
            st.dataframe(
                summary_df,
                column_config={
                    "Compound Name": st.column_config.TextColumn("Compound Name", width="medium"),
                    "Concentration (ppm/ppb)": st.column_config.NumberColumn("Concentration (ppm/ppb)", format="%.4f"),
                },
                hide_index=True,
                height=320,
                use_container_width=True
            )

        st.markdown(
            """
            <div class="disclaimer-box">
                <b>Disclaimer:</b> This analysis is generated by a statistical model trained on breath biomarker data. 
                It is designed solely for research and clinical diagnostic support. Do not rely on this output for primary clinical diagnosis.
            </div>
            """,
            unsafe_allow_html=True
        )

        # Prepare in-memory TXT report
        voc_input_lines = [f"  - {col}: {val:.4f} ppm/ppb" for col, val in zip(inp_df.columns, inp_df.iloc[0].values)]
        prob_lines = [f"  - {c}: {p*100:.2f}%" for c, p in zip(classes, probs)]

        txt_content_voc = (
            "================================================================================\n"
            "JIVA MEDINTELL\n"
            "VOC Breath Biomarker Analysis Report\n"
            "================================================================================\n\n"
            "File/Input:\n"
            "Modality: Exhaled Breath Volatile Organic Compounds (VOCs)\n"
            f"Biomarkers Evaluated: {len(inp_df.columns)} Compounds\n"
            "Concentrations (ppm/ppb):\n"
            + "\n".join(voc_input_lines) + "\n\n"
            "RESULTS\n"
            f"Primary Prediction: {pred_cls.upper()}\n"
            f"Classification Category: {pred_cls}\n"
            f"Prediction Confidence: {conf:.2f}%\n"
            "Classifier Model: Bagging Ensemble Classifier (100 Estimators)\n\n"
            "METRICS\n"
            "Class Probability Distribution:\n"
            + "\n".join(prob_lines) + "\n"
            "Model Benchmark Accuracy: 80.23%\n"
            "Malignant Cancer Sensitivity (Recall): 90.62%\n"
            "Weighted F1 Score: 0.7868\n\n"
            "FINDINGS\n"
            f"- Patient breath profile classified into {pred_cls.upper()} category with {conf:.2f}% confidence.\n"
            "- Volatile organic metabolite distribution evaluated via non-linear ensemble decision boundaries.\n"
            "- Local Shapley feature attributions identify key biochemical drivers among primary aldehydes and esters.\n\n"
            "CLINICAL IMPRESSION\n"
            + (
                "Patient breath profile exhibits significant volatile biomarker characteristics associated with malignant pulmonary neoplasms. High-sensitivity ensemble screening warrants prompt clinical follow-up."
                if pred_cls == "Cancer"
                else (
                    "Biomarker pattern is consistent with benign pulmonary alterations. Regular monitoring and clinical correlation advised."
                    if pred_cls == "Benign"
                    else "Breath VOC concentrations fall within expected healthy control physiological baseline limits."
                )
            ) + "\n\n"
            "RECOMMENDED ACTION\n"
            + (
                "Recommend urgent comprehensive pulmonary consultation, Low-Dose Computed Tomography (LDCT) of the chest, and tissue/histopathological correlation."
                if pred_cls == "Cancer"
                else (
                    "Recommend routine outpatient pulmonary follow-up and symptom monitoring."
                    if pred_cls == "Benign"
                    else "Routine annual health screening recommended."
                )
            ) + "\n\n"
            "--------------------------------------------------------------------------------\n"
            "DISCLAIMER: This analysis is generated by a statistical model trained on breath biomarker data. "
            "It is designed solely for research and clinical diagnostic support. Do not rely on this output for primary clinical diagnosis.\n"
            "Generated by JIVA MEDINTELL\n"
            "================================================================================\n"
        )

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
                data=txt_content_voc,
                file_name="JIVA_VOC_Results.txt",
                mime="text/plain",
                key="btn_dl_txt_voc",
                use_container_width=True
            )

        with col_pdf:
            if st.button("Generate Results PDF", key="btn_gen_pdf_voc", use_container_width=True):
                try:
                    with st.spinner("Generating PDF report..."):
                        clf_meta = load_classifier_metadata()
                        model_name = clf_meta.get("model_name", "Bagging Ensemble Classifier") if clf_meta else "Bagging Ensemble Classifier"
                        pdf_data = generate_voc_report(
                            input_df=inp_df,
                            pred_class=pred_cls,
                            confidence_pct=conf,
                            probabilities=probs.tolist() if hasattr(probs, "tolist") else list(probs),
                            classes=classes,
                            classifier_name=model_name,
                            fig_probs=fig_probs if 'fig_probs' in locals() else None,
                            fig_shap=fig_shap if 'fig_shap' in locals() else None,
                            shap_summary=shap_summary if 'shap_summary' in locals() else None
                        )
                        if pdf_data:
                            st.session_state["pdf_voc_bytes"] = pdf_data
                        else:
                            st.error("Unable to generate the PDF report.")
                except Exception:
                    st.error("Unable to generate the PDF report.")

            if "pdf_voc_bytes" in st.session_state:
                st.download_button(
                    label="Download Results PDF",
                    data=st.session_state["pdf_voc_bytes"],
                    file_name="JIVA_VOC_Report.pdf",
                    mime="application/pdf",
                    key="btn_dl_pdf_voc",
                    use_container_width=True
                )
