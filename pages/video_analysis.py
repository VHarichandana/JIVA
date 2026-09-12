import os
import io
import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from utils.video_processor import process_face_video, load_physformer_model
from utils.pdf_report import generate_vital_signs_report


def render_video_analysis():
    # Pre-warm model in background singleton cache so analyze is instantaneous
    try:
        load_physformer_model()
    except Exception:
        pass
    # High-contrast gray typography styles for inputs, radio buttons, file uploader, and video preview
    st.markdown(
        """
        <style>
        /* Section Headings */
        .vital-section-heading {
            font-size: 1.2rem;
            font-weight: 700;
            color: #374151 !important; /* Dark Slate Gray */
            margin-bottom: 0.85rem;
            letter-spacing: -0.02em;
        }

        /* Radio Button Label & Options */
        div[data-testid="stRadio"] > label,
        div[data-testid="stRadio"] > label p,
        div[data-testid="stRadio"] > label div p {
            color: #374151 !important; /* Dark Slate Gray */
            -webkit-text-fill-color: #374151 !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] label,
        div[data-testid="stRadio"] div[role="radiogroup"] label div p,
        div[data-testid="stRadio"] div[role="radiogroup"] label p,
        div[data-testid="stRadio"] div[role="radiogroup"] span {
            color: #4B5563 !important; /* Medium Slate Gray */
            -webkit-text-fill-color: #4B5563 !important;
            font-weight: 500 !important;
            font-size: 0.92rem !important;
        }

        /* File Uploader Outer Label */
        div[data-testid="stFileUploader"] > label,
        div[data-testid="stFileUploader"] > label p,
        div[data-testid="stFileUploader"] > label div p {
            color: #374151 !important; /* Dark Slate Gray */
            -webkit-text-fill-color: #374151 !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }

        /* File Uploader Dropzone Container */
        div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] {
            background-color: #ffffff !important;
            border: 1.5px dashed #9CA3AF !important; /* Visible Gray Dashed Border */
            border-radius: 12px !important;
            padding: 1.25rem !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
        }

        /* "Drag and drop file here" and general dropzone text */
        div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] span,
        div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] div,
        div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] p {
            color: #4B5563 !important; /* Medium Slate Gray */
            -webkit-text-fill-color: #4B5563 !important;
            font-weight: 500 !important;
        }

        /* "Limit 200MB per file • MP4, AVI, MOV, MKV, WEBM, MPEG4" */
        div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] small {
            color: #6B7280 !important; /* Slate Gray 500 */
            -webkit-text-fill-color: #6B7280 !important;
            font-weight: 500 !important;
            font-size: 0.82rem !important;
        }

        /* "Browse files" button inside uploader */
        div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button {
            color: #374151 !important;
            -webkit-text-fill-color: #374151 !important;
            background-color: #F3F4F6 !important;
            border: 1px solid #D1D5DB !important;
            font-weight: 600 !important;
        }

        /* File status / No file chosen / uploaded filename */
        div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span,
        div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] small,
        div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] div,
        div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] {
            color: #4B5563 !important;
            -webkit-text-fill-color: #4B5563 !important;
            font-weight: 500 !important;
        }

        /* Checkbox Label */
        div[data-testid="stCheckbox"] label,
        div[data-testid="stCheckbox"] label p,
        div[data-testid="stCheckbox"] span {
            color: #4B5563 !important;
            -webkit-text-fill-color: #4B5563 !important;
            font-weight: 500 !important;
        }

        /* Video Preview Placeholder Box */
        .video-preview-box {
            height: 215px;
            border: 1.5px dashed #9CA3AF;
            border-radius: 12px;
            background-color: #ffffff;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        }

        .video-preview-box-sub {
            color: #6B7280 !important;
            font-size: 0.92rem !important;
            font-weight: 500 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Module Title & Header
    st.markdown(
        """
        <div style="margin-bottom: 2rem;">
            <h2 style="font-size: 2.2rem; font-weight: 800; color: #252525; margin-bottom: 0.4rem; letter-spacing: -0.03em;">Vital Signs — Video rPPG</h2>
            <p style="color: #707070; font-size: 1.05rem; margin: 0;">Non-invasive remote photoplethysmography (rPPG) heart-rate estimation using Spatio-Temporal Vision Transformers (PhysFormer).</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Informational notice
    st.markdown(
        """
        <div style="background-color: #edf1ff; border: 1px solid #c7d2fe; border-left: 4px solid #4b63e6; padding: 1rem 1.25rem; border-radius: 12px; margin-bottom: 1.5rem;">
            <p style="margin: 0; color: #1e293b; font-size: 0.95rem; line-height: 1.5;">
                <b>Multimodal rPPG Pipeline:</b> Patient Face Video &rarr; Face ROI Detection &rarr; PhysFormer Temporal Attention &rarr; rPPG Waveform &rarr; Frequency-Domain FFT/PSD &rarr; Heart Rate (BPM).
                <i>Estimations are investigational and non-invasive.</i>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_input, col_preview = st.columns([1.2, 1.0], gap="large")

    with col_input:
        st.markdown('<div class="vital-section-heading">Patient Video Input</div>', unsafe_allow_html=True)
        
        # Option to upload, record live with camera, or use built-in sample
        input_choice = st.radio(
            "Select Video Source:",
            [
                "Upload Video File (.mp4, .avi, .mov, .webm)",
                "Record Live Video with Camera 📹",
                "Use Benchmark Sample Video"
            ],
            index=0,
            horizontal=True
        )

        uploaded_file = None
        sample_path = None
        recorded_path = None

        if input_choice == "Upload Video File (.mp4, .avi, .mov, .webm)":
            uploaded_file = st.file_uploader(
                "Upload a facial recording (minimum 160 frames, ~5-6 seconds):",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                help="Frontal view of the patient face with steady ambient lighting produces the highest accuracy."
            )

        elif input_choice == "Record Live Video with Camera 📹":
            st.markdown(
                """
                <div style="background-color: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 1.15rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                    <div style="font-weight: 700; color: #374151; font-size: 0.98rem; margin-bottom: 0.35rem;">Live Camera Capture</div>
                    <p style="color: #4B5563; font-size: 0.88rem; margin: 0 0 0.75rem 0; line-height: 1.45;">
                        Sit in front of your webcam with clear, even facial lighting. The camera will record 160 frames (approx. 5.3 seconds) with real-time facial tracking for PhysFormer rPPG analysis.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            auto_analyze = st.checkbox(
                "Automatically analyze immediately after recording",
                value=True,
                key="chk_auto_analyze"
            )

            col_rc1, col_rc2 = st.columns([1.5, 1.0])
            with col_rc1:
                start_rec_clicked = st.button(
                    "🔴 Record 5-Second Video from Camera",
                    type="primary",
                    use_container_width=True,
                    key="btn_start_camera_rec"
                )
            with col_rc2:
                if "recorded_video_path" in st.session_state and os.path.exists(st.session_state["recorded_video_path"]):
                    if st.button("Clear Video", use_container_width=True, key="btn_clear_rec"):
                        st.session_state.pop("recorded_video_path", None)
                        st.rerun()

            if start_rec_clicked:
                from utils.video_processor import record_webcam_video
                with col_preview:
                    live_feed_box = st.empty()
                    rec_badge = st.empty()
                    rec_bar = st.progress(0)

                def on_webcam_frame(curr, total, rgb_frame):
                    pct = curr / total
                    rec_bar.progress(pct)
                    rec_badge.markdown(
                        f"""
                        <div style="background: #fee2e2; border: 1px solid #fca5a5; color: #b91c1c; font-weight: 700; padding: 6px 12px; border-radius: 8px; font-size: 0.88rem; margin-bottom: 8px;">
                            ● RECORDING: Frame {curr}/{total} ({int(pct*100)}%) — Look at camera & remain still
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    live_feed_box.image(rgb_frame, use_container_width=True)

                try:
                    out_vid, total_f, fps_cam = record_webcam_video(
                        output_path="scratch/recorded_camera_video.mp4",
                        target_frames=160,
                        frame_callback=on_webcam_frame
                    )
                    st.session_state["recorded_video_path"] = out_vid
                    rec_badge.markdown(
                        f"""
                        <div style="background: #ecfdf5; border: 1px solid #a7f3d0; color: #047857; font-weight: 700; padding: 6px 12px; border-radius: 8px; font-size: 0.88rem; margin-bottom: 8px;">
                            ✅ Recording complete! Captured {total_f} frames at {fps_cam:.1f} fps.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    rec_bar.empty()
                    time.sleep(0.5)

                    if auto_analyze:
                        st.session_state["auto_analyze_triggered"] = True
                    st.rerun()

                except Exception as cam_err:
                    rec_bar.empty()
                    rec_badge.empty()
                    st.error(f"Camera Recording Error: {cam_err}")

            if "recorded_video_path" in st.session_state and os.path.exists(st.session_state["recorded_video_path"]):
                recorded_path = st.session_state["recorded_video_path"]
                st.success(f"Camera clip ready: `{os.path.basename(recorded_path)}` (160 frames)")

        else:
            # Check for existing sample video
            sample_candidates = [
                "scratch/test_face_video.mp4",
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "scratch", "test_face_video.mp4")
            ]
            for cand in sample_candidates:
                if os.path.exists(cand):
                    sample_path = cand
                    break
            
            if sample_path:
                st.success(f"Benchmark validation sample loaded: `{os.path.basename(sample_path)}`")
            else:
                st.info("Generating standard benchmark validation clip...")
                # Create on-the-fly sample
                try:
                    import cv2
                    os.makedirs("scratch", exist_ok=True)
                    sample_path = "scratch/test_face_video.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(sample_path, fourcc, 30.0, (320, 320))
                    for i in range(180):
                        frame = np.ones((320, 320, 3), dtype=np.uint8) * 235
                        pulse = 1.2 * np.sin(2 * np.pi * 1.25 * (i / 30.0))
                        # Skin tone face circle
                        cv2.ellipse(frame, (160, 160), (80, 100), 0, 0, 360, (140, 160 + int(pulse), 210 + int(pulse)), -1)
                        # Eyes, nose, mouth
                        cv2.circle(frame, (130, 140), 10, (50, 50, 50), -1)
                        cv2.circle(frame, (190, 140), 10, (50, 50, 50), -1)
                        cv2.line(frame, (160, 150), (160, 180), (100, 100, 100), 3)
                        cv2.ellipse(frame, (160, 205), (25, 8), 0, 0, 180, (70, 70, 150), -1)
                        writer.write(frame)
                    writer.release()
                    st.success("Benchmark validation clip ready!")
                except Exception as e:
                    st.error(f"Could not generate benchmark clip: {e}")

        is_pre_cropped = False
        if input_choice != "Record Live Video with Camera 📹":
            is_pre_cropped = st.checkbox(
                "Video is pre-cropped to face region (e.g. VIPL-HR / UBFC benchmark format)",
                value=False,
                help="Check this ONLY if your video file is already tightly cropped around the face, skipping automatic Haar face bounding box search."
            )

        analyze_clicked = st.button("Analyze Video", type="primary", use_container_width=True)

        # Triggered automatically after recording completes if auto-analyze was enabled
        if st.session_state.pop("auto_analyze_triggered", False):
            analyze_clicked = True

    with col_preview:
        st.markdown('<div class="vital-section-heading">Video Preview</div>', unsafe_allow_html=True)
        if uploaded_file is not None:
            st.video(uploaded_file)
        elif input_choice == "Record Live Video with Camera 📹" and recorded_path and os.path.exists(recorded_path):
            with open(recorded_path, "rb") as vf:
                st.video(vf.read())
        elif sample_path is not None and os.path.exists(sample_path):
            with open(sample_path, "rb") as vf:
                st.video(vf.read())
        else:
            st.markdown(
                """
                <div class="video-preview-box">
                    <div style="font-size: 1.05rem; font-weight: 700; color: #374151; margin-bottom: 0.35rem;">Video Preview</div>
                    <div class="video-preview-box-sub">Video preview will display here once loaded or recorded.</div>
                </div>
                """,
                unsafe_allow_html=True
            )


    # Process Video Inference
    if analyze_clicked:
        if input_choice == "Upload Video File (.mp4, .avi, .mov, .webm)":
            target_video = uploaded_file
        elif input_choice == "Record Live Video with Camera 📹":
            target_video = recorded_path if recorded_path else st.session_state.get("recorded_video_path", None)
        else:
            target_video = sample_path

        if target_video is None:
            if input_choice == "Record Live Video with Camera 📹":
                st.warning("Please click '🔴 Record 5-Second Video from Camera' to capture a video clip first.")
            else:
                st.warning("Please upload a video file or select the benchmark sample first.")
        else:
            with st.spinner("PhysFormer Spatio-Temporal ViT is processing vital signs..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(pct, msg):
                    progress_bar.progress(int(pct * 100))
                    status_text.markdown(f"**Status:** {msg}")

                try:
                    start_time = time.time()
                    findings = process_face_video(
                        video_source=target_video,
                        is_pre_cropped=is_pre_cropped,
                        progress_callback=update_progress
                    )
                    elapsed = time.time() - start_time
                    findings["elapsed_time_sec"] = round(elapsed, 2)
                    st.session_state["vital_signs_findings"] = findings
                    st.session_state.pop("pdf_vital_signs_bytes", None)
                    progress_bar.empty()
                    status_text.empty()
                    st.success(f"PhysFormer rPPG analysis completed in {elapsed:.1f}s.")
                except ValueError as ve:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Analysis Error: {str(ve)}")
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Inference Pipeline Error: {str(e)}")

    # Response Display Card
    if "vital_signs_findings" in st.session_state:
        findings = st.session_state["vital_signs_findings"]
        
        st.markdown("<br><hr style='border-color: #e8e8e8; margin-bottom: 2rem;'><br>", unsafe_allow_html=True)
        st.markdown("### Physiological Findings & Heart Rate")

        hr = findings["heart_rate"]
        quality = findings["signal_quality"]
        conf = findings["confidence_score"] * 100
        snr = findings["snr_db"]
        status_label = findings["status_classification"]

        # Main Highlight Box
        st.markdown(
            f"""
            <div style="background-color: #ffffff; border: 1px solid #e8e8e8; border-left: 5px solid #4b63e6; padding: 1.75rem; border-radius: 16px; margin-bottom: 1.5rem; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <div style="font-size: 0.95rem; color: #707070; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem;">PhysFormer rPPG Estimated Heart Rate</div>
                        <div style="font-size: 3.2rem; font-weight: 800; color: #252525; line-height: 1.1;">
                            {hr} <span style="font-size: 1.4rem; font-weight: 600; color: #4b63e6;">BPM</span>
                        </div>
                    </div>
                    <div style="background-color: #edf1ff; border: 1px solid #c7d2fe; padding: 0.6rem 1.2rem; border-radius: 10px; text-align: center;">
                        <div style="font-size: 0.8rem; color: #4b63e6; font-weight: 700; text-transform: uppercase;">Diagnostic Rhythm Status</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #1e293b;">{status_label}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 4 Metric Cards Row
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric(label="Signal Quality", value=quality, delta=f"{snr:+.1f} dB SNR")
        with m_col2:
            st.metric(label="Algorithm Confidence", value=f"{conf:.1f}%")
        with m_col3:
            st.metric(label="Dominant Frequency", value=f"{findings['dominant_freq_hz']} Hz")
        with m_col4:
            st.metric(label="Analyzed Window", value=f"{findings['frames_analyzed']} frames", delta=f"{findings['fps']} fps")

        st.markdown("<br>", unsafe_allow_html=True)

        # Middle Row: Face ROI Detection + rPPG Waveform
        row2_col1, row2_col2 = st.columns([1.0, 1.6], gap="medium")

        with row2_col1:
            st.markdown("##### Face ROI Spatial Localization")
            if "annotated_frame" in findings and findings["annotated_frame"] is not None:
                st.image(
                    findings["annotated_frame"],
                    caption=f"Bounding Box: {findings.get('face_bbox', 'N/A')}",
                    use_container_width=True
                )
            else:
                st.info("No spatial frame visualization available.")

        fig_waveform = None
        with row2_col2:
            st.markdown("##### Recovered rPPG Temporal Waveform")
            time_axis = np.array(findings["time_axis"])
            rppg_wave = np.array(findings["rppg_signal"])
            
            fig_waveform, ax_w = plt.subplots(figsize=(7, 3.8), dpi=120)
            ax_w.plot(time_axis, rppg_wave, color="#4b63e6", linewidth=1.8, label="Filtered rPPG Pulse")
            ax_w.set_title("Temporal rPPG Pulse Signal (Bandpass 0.75 - 2.5 Hz)", fontsize=10, fontweight="bold", color="#1e293b", pad=10)
            ax_w.set_xlabel("Time (seconds)", fontsize=9, color="#64748b")
            ax_w.set_ylabel("Normalized Amplitude", fontsize=9, color="#64748b")
            ax_w.grid(True, linestyle="--", alpha=0.4)
            ax_w.legend(loc="upper right", fontsize=8)
            fig_waveform.tight_layout()
            st.pyplot(fig_waveform)

        # Power Spectrum Density Plot
        st.markdown("##### Frequency Domain Power Spectral Density (PSD)")
        fig_spectrum, ax_s = plt.subplots(figsize=(10, 3.2), dpi=120)
        freqs_bpm = np.array(findings["psd_freqs"]) * 60.0
        psd_vals = np.array(findings["psd_power"])
        
        ax_s.plot(freqs_bpm, psd_vals, color="#2563eb", linewidth=1.8, label="Welch PSD")
        ax_s.axvline(x=hr, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Peak: {hr:.1f} BPM ({findings['dominant_freq_hz']} Hz)")
        ax_s.set_title("Power Spectral Density across Physiological Heart Rate Band (45 - 150 BPM)", fontsize=10, fontweight="bold", color="#1e293b")
        ax_s.set_xlabel("Frequency (Beats Per Minute / BPM)", fontsize=9, color="#64748b")
        ax_s.set_ylabel("Spectral Power Density", fontsize=9, color="#64748b")
        ax_s.grid(True, linestyle="--", alpha=0.4)
        ax_s.legend(loc="upper right", fontsize=8)
        fig_spectrum.tight_layout()
        st.pyplot(fig_spectrum)

        # Clinical Impression & Advisory
        st.markdown("<br>", unsafe_allow_html=True)
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("##### Clinical Impression")
            st.write(findings["clinical_impression"])
        with col_c2:
            st.markdown("##### Recommended Action")
            st.write(findings["recommended_action"])

        # Structured Output JSON
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("View Structured Analysis JSON"):
            # Display findings without heavy arrays for clean viewing
            json_preview = {k: v for k, v in findings.items() if k not in ["annotated_frame", "rppg_signal", "raw_rppg", "time_axis", "psd_freqs", "psd_power"]}
            st.json(json_preview)

        # Investigational Disclaimer
        st.markdown(
            f"""
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 1rem; border-radius: 10px; margin-top: 1.5rem;">
                <p style="margin: 0; color: #64748b; font-size: 0.85rem; line-height: 1.4;">
                    <b>Investigational Medical Disclaimer:</b> {findings['disclaimer']} 
                    Device acceleration: <code>{findings.get('device', 'cpu')}</code>.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # PDF Report Button
        st.markdown("<br>", unsafe_allow_html=True)
        col_pdf, _ = st.columns([1.5, 3])
        with col_pdf:
            if st.button("Generate Results PDF", key="btn_gen_pdf_vitals", use_container_width=True):
                try:
                    with st.spinner("Generating PDF report..."):
                        pdf_data = generate_vital_signs_report(
                            findings=findings,
                            fig_waveform=fig_waveform,
                            fig_spectrum=fig_spectrum,
                            face_image=findings.get("annotated_frame")
                        )
                        if pdf_data:
                            st.session_state["pdf_vital_signs_bytes"] = pdf_data
                        else:
                            st.error("Unable to generate the PDF report.")
                except Exception as e:
                    st.error(f"Unable to generate the PDF report: {e}")

            if "pdf_vital_signs_bytes" in st.session_state:
                st.download_button(
                    label="Download Results PDF",
                    data=st.session_state["pdf_vital_signs_bytes"],
                    file_name="JIVA_Vital_Signs_Report.pdf",
                    mime="application/pdf",
                    key="btn_dl_pdf_vitals",
                    use_container_width=True
                )
