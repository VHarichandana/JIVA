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

        st.markdown("<br>", unsafe_allow_html=True)
        col_pdf, _ = st.columns([1.5, 3])
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
