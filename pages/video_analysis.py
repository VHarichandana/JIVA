import os
import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from utils.video_processor import (
    process_face_video,
    load_physformer_model
)
from utils.pdf_report import generate_vital_signs_report


def render_video_analysis():

    # =========================================================
    # PRELOAD PHYFORMER
    # =========================================================

    try:
        load_physformer_model()
    except Exception:
        pass


    # =========================================================
    # PAGE CSS
    # =========================================================

    st.html(
        """
        <style>

        /* ==========================================
           SECTION HEADINGS
           ========================================== */

        .vital-section-heading {
            font-size: 1.2rem;
            font-weight: 700;
            color: #374151 !important;
            margin-bottom: 0.85rem;
            letter-spacing: -0.02em;
        }


        /* ==========================================
           RADIO BUTTONS
           ========================================== */

        div[data-testid="stRadio"] > label,
        div[data-testid="stRadio"] > label p,
        div[data-testid="stRadio"] > label div p {
            color: #374151 !important;
            -webkit-text-fill-color: #374151 !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }

        div[data-testid="stRadio"]
        div[role="radiogroup"] label,

        div[data-testid="stRadio"]
        div[role="radiogroup"] label div p,

        div[data-testid="stRadio"]
        div[role="radiogroup"] label p,

        div[data-testid="stRadio"]
        div[role="radiogroup"] span {
            color: #4B5563 !important;
            -webkit-text-fill-color: #4B5563 !important;
            font-weight: 500 !important;
            font-size: 0.92rem !important;
        }


        /* ==========================================
           FILE UPLOADER
           ========================================== */

        div[data-testid="stFileUploader"] > label,
        div[data-testid="stFileUploader"] > label p,
        div[data-testid="stFileUploader"] > label div p {
            color: #374151 !important;
            -webkit-text-fill-color: #374151 !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }

        div[data-testid="stFileUploader"]
        section[data-testid="stFileUploadDropzone"] {
            background: #FFFFFF !important;
            border: 1.5px dashed #9CA3AF !important;
            border-radius: 12px !important;
            padding: 1.25rem !important;
            box-shadow: none !important;
        }

        div[data-testid="stFileUploader"]
        section[data-testid="stFileUploadDropzone"] span,

        div[data-testid="stFileUploader"]
        section[data-testid="stFileUploadDropzone"] div,

        div[data-testid="stFileUploader"]
        section[data-testid="stFileUploadDropzone"] p {
            color: #4B5563 !important;
            -webkit-text-fill-color: #4B5563 !important;
            font-weight: 500 !important;
        }

        div[data-testid="stFileUploader"]
        section[data-testid="stFileUploadDropzone"] small {
            color: #6B7280 !important;
            -webkit-text-fill-color: #6B7280 !important;
            font-weight: 500 !important;
            font-size: 0.82rem !important;
        }

        div[data-testid="stFileUploader"]
        section[data-testid="stFileUploadDropzone"] button {
            background: #FFFFFF !important;
            background-color: #FFFFFF !important;
            border: 1px solid #9CA3AF !important;
            border-radius: 9px !important;
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
            font-weight: 700 !important;
            opacity: 1 !important;
            visibility: visible !important;
            box-shadow: none !important;
        }

        div[data-testid="stFileUploader"]
        section[data-testid="stFileUploadDropzone"] button * {
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
            opacity: 1 !important;
        }


        /* ==========================================
           CHECKBOX
           ========================================== */

        div[data-testid="stCheckbox"] label,
        div[data-testid="stCheckbox"] label p,
        div[data-testid="stCheckbox"] span {
            color: #4B5563 !important;
            -webkit-text-fill-color: #4B5563 !important;
            font-weight: 500 !important;
        }


        /* ==========================================
           VIDEO PLACEHOLDER
           ========================================== */

        .video-preview-box {
            height: 215px;
            border: 1.5px dashed #9CA3AF;
            border-radius: 12px;
            background: #FFFFFF;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 1.5rem;
            text-align: center;
        }

        .video-preview-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #374151;
            margin-bottom: 0.35rem;
        }

        .video-preview-box-sub {
            color: #6B7280;
            font-size: 0.92rem;
            font-weight: 500;
        }


        /* ==========================================
           RESULT CARDS
           ========================================== */

        .vital-card {
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

        .vital-card-blue {
            background:
                linear-gradient(
                    145deg,
                    rgba(246,252,255,0.98),
                    rgba(190,225,247,0.92)
                );
        }

        .vital-card-teal {
            background:
                linear-gradient(
                    145deg,
                    rgba(245,255,254,0.98),
                    rgba(190,235,235,0.92)
                );
        }

        .vital-card-violet {
            background:
                linear-gradient(
                    145deg,
                    rgba(248,250,255,0.98),
                    rgba(205,224,247,0.92)
                );
        }

        .vital-card-soft {
            background:
                linear-gradient(
                    145deg,
                    rgba(250,254,255,0.98),
                    rgba(220,241,248,0.92)
                );
        }


        /* ==========================================
           RESULT CARD TEXT
           ========================================== */

        .vital-label {
            color: #0877A5;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }

        .vital-value {
            color: #0477A8;
            font-size: 1.75rem;
            font-weight: 850;
            line-height: 1.2;
        }

        .vital-value-large {
            color: #0477A8;
            font-size: 2.6rem;
            font-weight: 900;
            line-height: 1.1;
        }

        .vital-unit {
            color: #486674;
            font-size: 1rem;
            font-weight: 700;
        }

        .vital-title {
            color: #17384A;
            font-size: 1.08rem;
            font-weight: 800;
            line-height: 1.45;
            margin-bottom: 0.5rem;
        }

        .vital-text {
            color: #486674;
            font-size: 0.93rem;
            line-height: 1.6;
            margin: 0;
        }

        .vital-subtext {
            color: #66808C;
            font-size: 0.78rem;
            font-weight: 650;
            margin-top: 0.4rem;
        }


        /* ==========================================
           DISCLAIMER CARD
           ========================================== */

        .vital-disclaimer {
            background:
                linear-gradient(
                    145deg,
                    rgba(248,250,252,0.98),
                    rgba(235,242,247,0.96)
                );

            border: 1px solid #DCE7ED;

            border-radius: 16px;

            padding: 1rem 1.2rem;

            color: #5B6F7B;

            font-size: 0.84rem;

            line-height: 1.55;
        }

        </style>
        """
    )


    # =========================================================
    # HEADER
    # =========================================================

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
                Vital Signs — Video rPPG
            </div>

            <div style="
                color:#707070;
                font-size:1.05rem;
            ">
                Non-invasive remote photoplethysmography
                (rPPG) heart-rate estimation using
                Spatio-Temporal Vision Transformers
                (PhysFormer).
            </div>

        </div>
        """
    )


    # =========================================================
    # INFORMATION CARD
    # =========================================================

    st.html(
        """
        <div style="
            background:
                linear-gradient(
                    145deg,
                    #F2F8FF,
                    #E7F0FF
                );

            border:1px solid #C7DDF5;
            border-left:4px solid #0477A8;

            padding:1rem 1.25rem;

            border-radius:12px;

            margin-bottom:1.5rem;
        ">

            <div style="
                color:#1E3A4A;
                font-size:0.95rem;
                line-height:1.55;
            ">

                <b>Multimodal rPPG Pipeline:</b>

                Patient Face Video →
                Face ROI Detection →
                PhysFormer Temporal Attention →
                rPPG Waveform →
                Frequency-Domain FFT/PSD →
                Heart Rate (BPM).

                <br>

                <i>
                    Estimations are investigational
                    and non-invasive.
                </i>

            </div>

        </div>
        """
    )


    # =========================================================
    # INPUT AND PREVIEW
    # =========================================================

    col_input, col_preview = st.columns(
        [1.2, 1.0],
        gap="large"
    )


    with col_input:

        st.html(
            """
            <div class="vital-section-heading">
                Patient Video Input
            </div>
            """
        )


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


        # =====================================================
        # UPLOAD VIDEO
        # =====================================================

        if input_choice == (
            "Upload Video File (.mp4, .avi, .mov, .webm)"
        ):

            uploaded_file = st.file_uploader(
                (
                    "Upload a facial recording "
                    "(minimum 160 frames, ~5-6 seconds):"
                ),
                type=[
                    "mp4",
                    "avi",
                    "mov",
                    "mkv",
                    "webm"
                ],
                help=(
                    "Frontal view of the patient face "
                    "with steady ambient lighting produces "
                    "the highest accuracy."
                )
            )


        # =====================================================
        # RECORD CAMERA
        # =====================================================

        elif input_choice == (
            "Record Live Video with Camera 📹"
        ):

            st.html(
                """
                <div class="vital-card vital-card-soft"
                     style="margin-bottom:1rem;">

                    <div class="vital-label">
                        Live Camera Capture
                    </div>

                    <div class="vital-text">
                        Sit in front of your webcam with clear,
                        even facial lighting. The camera will
                        record approximately 160 frames for
                        PhysFormer rPPG analysis.
                    </div>

                </div>
                """
            )


            auto_analyze = st.checkbox(
                (
                    "Automatically analyze immediately "
                    "after recording"
                ),
                value=True,
                key="chk_auto_analyze"
            )


            col_rc1, col_rc2 = st.columns(
                [1.5, 1.0]
            )


            with col_rc1:

                start_rec_clicked = st.button(
                    "🔴 Record 5-Second Video",
                    type="primary",
                    width="stretch",
                    key="btn_start_camera_rec"
                )


            with col_rc2:

                if (
                    "recorded_video_path"
                    in st.session_state
                    and os.path.exists(
                        st.session_state[
                            "recorded_video_path"
                        ]
                    )
                ):

                    if st.button(
                        "Clear Video",
                        width="stretch",
                        key="btn_clear_rec"
                    ):

                        st.session_state.pop(
                            "recorded_video_path",
                            None
                        )

                        st.rerun()


            if start_rec_clicked:

                from utils.video_processor import (
                    record_webcam_video
                )


                with col_preview:

                    live_feed_box = st.empty()
                    rec_badge = st.empty()
                    rec_bar = st.progress(0)


                def on_webcam_frame(
                    curr,
                    total,
                    rgb_frame
                ):

                    pct = curr / total

                    rec_bar.progress(
                        pct
                    )

                    rec_badge.html(
                        f"""
                        <div style="
                            background:#FEE2E2;
                            border:1px solid #FCA5A5;
                            color:#B91C1C;
                            font-weight:700;
                            padding:6px 12px;
                            border-radius:8px;
                            font-size:0.88rem;
                            margin-bottom:8px;
                        ">
                            ● RECORDING:
                            Frame {curr}/{total}
                            ({int(pct * 100)}%)
                            — Look at camera and remain still
                        </div>
                        """
                    )

                    live_feed_box.image(
                        rgb_frame,
                        width="stretch"
                    )


                try:

                    out_vid, total_f, fps_cam = (
                        record_webcam_video(
                            output_path=(
                                "scratch/"
                                "recorded_camera_video.mp4"
                            ),
                            target_frames=160,
                            frame_callback=on_webcam_frame
                        )
                    )


                    st.session_state[
                        "recorded_video_path"
                    ] = out_vid


                    fps_cam_text = (
                        f"{float(fps_cam):.1f}"
                    )


                    rec_badge.html(
                        f"""
                        <div style="
                            background:#ECFDF5;
                            border:1px solid #A7F3D0;
                            color:#047857;
                            font-weight:700;
                            padding:6px 12px;
                            border-radius:8px;
                            font-size:0.88rem;
                            margin-bottom:8px;
                        ">
                            Recording complete.
                            Captured {total_f} frames
                            at {fps_cam_text} fps.
                        </div>
                        """
                    )


                    rec_bar.empty()

                    time.sleep(0.5)


                    if auto_analyze:

                        st.session_state[
                            "auto_analyze_triggered"
                        ] = True


                    st.rerun()


                except Exception as cam_err:

                    rec_bar.empty()
                    rec_badge.empty()

                    st.error(
                        f"Camera Recording Error: {cam_err}"
                    )


            if (
                "recorded_video_path"
                in st.session_state
                and os.path.exists(
                    st.session_state[
                        "recorded_video_path"
                    ]
                )
            ):

                recorded_path = (
                    st.session_state[
                        "recorded_video_path"
                    ]
                )

                st.success(
                    "Camera clip ready."
                )


        # =====================================================
        # BENCHMARK VIDEO
        # =====================================================

        else:

            sample_candidates = [
                "scratch/test_face_video.mp4",
                os.path.join(
                    os.path.dirname(
                        os.path.dirname(__file__)
                    ),
                    "scratch",
                    "test_face_video.mp4"
                )
            ]


            for candidate in sample_candidates:

                if os.path.exists(candidate):

                    sample_path = candidate
                    break


            if sample_path:

                st.success(
                    "Benchmark validation sample loaded."
                )

            else:

                st.info(
                    "Generating standard benchmark "
                    "validation clip..."
                )


                try:

                    import cv2

                    os.makedirs(
                        "scratch",
                        exist_ok=True
                    )


                    sample_path = (
                        "scratch/test_face_video.mp4"
                    )


                    fourcc = (
                        cv2.VideoWriter_fourcc(
                            *"mp4v"
                        )
                    )


                    writer = cv2.VideoWriter(
                        sample_path,
                        fourcc,
                        30.0,
                        (320, 320)
                    )


                    for i in range(180):

                        frame = (
                            np.ones(
                                (320, 320, 3),
                                dtype=np.uint8
                            )
                            * 235
                        )


                        pulse = (
                            1.2
                            * np.sin(
                                2
                                * np.pi
                                * 1.25
                                * (i / 30.0)
                            )
                        )


                        cv2.ellipse(
                            frame,
                            (160, 160),
                            (80, 100),
                            0,
                            0,
                            360,
                            (
                                140,
                                160 + int(pulse),
                                210 + int(pulse)
                            ),
                            -1
                        )


                        cv2.circle(
                            frame,
                            (130, 140),
                            10,
                            (50, 50, 50),
                            -1
                        )


                        cv2.circle(
                            frame,
                            (190, 140),
                            10,
                            (50, 50, 50),
                            -1
                        )


                        cv2.line(
                            frame,
                            (160, 150),
                            (160, 180),
                            (100, 100, 100),
                            3
                        )


                        cv2.ellipse(
                            frame,
                            (160, 205),
                            (25, 8),
                            0,
                            0,
                            180,
                            (70, 70, 150),
                            -1
                        )


                        writer.write(
                            frame
                        )


                    writer.release()

                    st.success(
                        "Benchmark validation clip ready."
                    )


                except Exception as generation_error:

                    st.error(
                        "Could not generate benchmark clip: "
                        f"{generation_error}"
                    )


        # =====================================================
        # PRE-CROPPED OPTION
        # =====================================================

        is_pre_cropped = False


        if input_choice != (
            "Record Live Video with Camera 📹"
        ):

            is_pre_cropped = st.checkbox(
                (
                    "Video is pre-cropped to face region "
                    "(e.g. VIPL-HR / UBFC benchmark format)"
                ),
                value=False,
                help=(
                    "Check this only if the video is "
                    "already tightly cropped around the face."
                )
            )


        analyze_clicked = st.button(
            "Analyze Video",
            type="primary",
            width="stretch"
        )


        if st.session_state.pop(
            "auto_analyze_triggered",
            False
        ):

            analyze_clicked = True


    # =========================================================
    # VIDEO PREVIEW
    # =========================================================

    with col_preview:

        st.html(
            """
            <div class="vital-section-heading">
                Video Preview
            </div>
            """
        )


        if uploaded_file is not None:

            st.video(
                uploaded_file
            )


        elif (
            input_choice
            == "Record Live Video with Camera 📹"
            and recorded_path
            and os.path.exists(recorded_path)
        ):

            with open(
                recorded_path,
                "rb"
            ) as video_file:

                st.video(
                    video_file.read()
                )


        elif (
            sample_path is not None
            and os.path.exists(sample_path)
        ):

            with open(
                sample_path,
                "rb"
            ) as video_file:

                st.video(
                    video_file.read()
                )


        else:

            st.html(
                """
                <div class="video-preview-box">

                    <div class="video-preview-title">
                        Video Preview
                    </div>

                    <div class="video-preview-box-sub">
                        Video preview will display here
                        once loaded or recorded.
                    </div>

                </div>
                """
            )


    # =========================================================
    # PROCESS VIDEO
    # =========================================================

    if analyze_clicked:

        if input_choice == (
            "Upload Video File (.mp4, .avi, .mov, .webm)"
        ):

            target_video = uploaded_file


        elif input_choice == (
            "Record Live Video with Camera 📹"
        ):

            target_video = (
                recorded_path
                if recorded_path
                else st.session_state.get(
                    "recorded_video_path",
                    None
                )
            )


        else:

            target_video = sample_path


        if target_video is None:

            if input_choice == (
                "Record Live Video with Camera 📹"
            ):

                st.warning(
                    "Please record a video clip first."
                )

            else:

                st.warning(
                    "Please upload a video file or "
                    "select the benchmark sample first."
                )


        else:

            with st.spinner(
                "PhysFormer is processing vital signs..."
            ):

                progress_bar = st.progress(
                    0
                )

                status_text = st.empty()


                def update_progress(
                    pct,
                    msg
                ):

                    progress_bar.progress(
                        int(pct * 100)
                    )

                    status_text.markdown(
                        f"**Status:** {msg}"
                    )


                try:

                    start_time = time.time()


                    findings = process_face_video(
                        video_source=target_video,
                        is_pre_cropped=is_pre_cropped,
                        progress_callback=update_progress
                    )


                    elapsed = (
                        time.time()
                        - start_time
                    )


                    findings[
                        "elapsed_time_sec"
                    ] = round(
                        elapsed,
                        2
                    )


                    st.session_state[
                        "vital_signs_findings"
                    ] = findings


                    st.session_state.pop(
                        "pdf_vital_signs_bytes",
                        None
                    )


                    progress_bar.empty()
                    status_text.empty()


                    elapsed_text = (
                        f"{elapsed:.1f}"
                    )


                    st.success(
                        "PhysFormer rPPG analysis "
                        f"completed in {elapsed_text}s."
                    )


                except ValueError as value_error:

                    progress_bar.empty()
                    status_text.empty()

                    st.error(
                        "Analysis Error: "
                        f"{value_error}"
                    )


                except Exception as inference_error:

                    progress_bar.empty()
                    status_text.empty()

                    st.error(
                        "Inference Pipeline Error: "
                        f"{inference_error}"
                    )


    # =========================================================
    # RESULT CARDS
    # =========================================================

    if "vital_signs_findings" in st.session_state:

        findings = st.session_state[
            "vital_signs_findings"
        ]


        # =====================================================
        # PREPARE VALUES
        # =====================================================

        hr = float(
            findings.get(
                "heart_rate",
                0
            )
        )

        confidence = float(
            findings.get(
                "confidence_score",
                0
            )
        ) * 100


        snr = float(
            findings.get(
                "snr_db",
                0
            )
        )


        dominant_frequency = float(
            findings.get(
                "dominant_freq_hz",
                0
            )
        )


        fps = float(
            findings.get(
                "fps",
                0
            )
        )


        frames_analyzed = int(
            findings.get(
                "frames_analyzed",
                0
            )
        )


        signal_quality = str(
            findings.get(
                "signal_quality",
                "Unknown"
            )
        )


        status_label = str(
            findings.get(
                "status_classification",
                "Not available"
            )
        )


        clinical_impression = str(
            findings.get(
                "clinical_impression",
                "Not available"
            )
        )


        recommended_action = str(
            findings.get(
                "recommended_action",
                "Not available"
            )
        )


        disclaimer = str(
            findings.get(
                "disclaimer",
                "For investigational use only."
            )
        )


        device = str(
            findings.get(
                "device",
                "cpu"
            )
        )


        hr_text = f"{hr:.1f}"
        confidence_text = f"{confidence:.1f}%"
        snr_text = f"{snr:+.1f} dB"
        frequency_text = f"{dominant_frequency:.2f} Hz"
        frames_text = str(frames_analyzed)
        fps_text = f"{fps:.1f} fps"


        # =====================================================
        # RESULT HEADING
        # =====================================================

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
            "## Physiological Analysis Report"
        )


        # =====================================================
        # HEART RATE + RHYTHM
        # =====================================================

        heart_col, rhythm_col = st.columns(
            [1.3, 1],
            gap="medium"
        )


        with heart_col:

            st.html(
                f"""
                <div class="
                    vital-card
                    vital-card-blue
                ">

                    <div class="vital-label">
                        Estimated Heart Rate
                    </div>

                    <div class="vital-value-large">
                        {hr_text}
                        <span class="vital-unit">
                            BPM
                        </span>
                    </div>

                    <div class="vital-subtext">
                        PhysFormer remote
                        photoplethysmography estimate
                    </div>

                </div>
                """
            )


        with rhythm_col:

            st.html(
                f"""
                <div class="
                    vital-card
                    vital-card-teal
                ">

                    <div class="vital-label">
                        Rhythm Status
                    </div>

                    <div class="vital-title">
                        {status_label}
                    </div>

                    <div class="vital-text">
                        Classification derived from
                        the recovered rPPG pulse signal.
                    </div>

                </div>
                """
            )


        st.html(
            "<div style='height:1.25rem;'></div>"
        )


        # =====================================================
        # METRIC CARDS
        # =====================================================

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
                    vital-card
                    vital-card-blue
                ">

                    <div class="vital-label">
                        Signal Quality
                    </div>

                    <div class="vital-value">
                        {signal_quality}
                    </div>

                    <div class="vital-subtext">
                        {snr_text} SNR
                    </div>

                </div>
                """
            )


        with metric2:

            st.html(
                f"""
                <div class="
                    vital-card
                    vital-card-teal
                ">

                    <div class="vital-label">
                        Confidence
                    </div>

                    <div class="vital-value">
                        {confidence_text}
                    </div>

                    <div class="vital-subtext">
                        Algorithm confidence
                    </div>

                </div>
                """
            )


        with metric3:

            st.html(
                f"""
                <div class="
                    vital-card
                    vital-card-violet
                ">

                    <div class="vital-label">
                        Dominant Frequency
                    </div>

                    <div class="vital-value">
                        {frequency_text}
                    </div>

                    <div class="vital-subtext">
                        Peak physiological frequency
                    </div>

                </div>
                """
            )


        with metric4:

            st.html(
                f"""
                <div class="
                    vital-card
                    vital-card-soft
                ">

                    <div class="vital-label">
                        Analyzed Window
                    </div>

                    <div class="vital-value">
                        {frames_text}
                    </div>

                    <div class="vital-subtext">
                        frames · {fps_text}
                    </div>

                </div>
                """
            )


        st.html(
            "<div style='height:1.4rem;'></div>"
        )


        # =====================================================
        # FACE ROI + WAVEFORM
        # =====================================================

        row2_col1, row2_col2 = st.columns(
            [1.0, 1.6],
            gap="medium"
        )


        with row2_col1:

            st.html(
                """
                <div class="
                    vital-card
                    vital-card-soft
                ">

                    <div class="vital-label">
                        Face ROI Spatial Localization
                    </div>

                    <div class="vital-text">
                        Detected facial region used for
                        rPPG pulse extraction.
                    </div>

                </div>
                """
            )


            st.html(
                "<div style='height:0.7rem;'></div>"
            )


            if (
                "annotated_frame"
                in findings
                and findings[
                    "annotated_frame"
                ] is not None
            ):

                st.image(
                    findings[
                        "annotated_frame"
                    ],
                    caption=(
                        "Bounding Box: "
                        f"{findings.get('face_bbox', 'N/A')}"
                    ),
                    width="stretch"
                )

            else:

                st.info(
                    "No spatial frame visualization available."
                )


        fig_waveform = None


        with row2_col2:

            st.html(
                """
                <div class="
                    vital-card
                    vital-card-blue
                ">

                    <div class="vital-label">
                        Recovered rPPG Waveform
                    </div>

                    <div class="vital-text">
                        Temporal pulse waveform recovered
                        from facial color variations.
                    </div>

                </div>
                """
            )


            time_axis = np.array(
                findings.get(
                    "time_axis",
                    []
                )
            )


            rppg_wave = np.array(
                findings.get(
                    "rppg_signal",
                    []
                )
            )


            if (
                len(time_axis) > 0
                and len(rppg_wave) > 0
            ):

                fig_waveform, ax_w = plt.subplots(
                    figsize=(7, 3.8),
                    dpi=120
                )


                ax_w.plot(
                    time_axis,
                    rppg_wave,
                    linewidth=1.8,
                    label="Filtered rPPG Pulse"
                )


                ax_w.set_title(
                    (
                        "Temporal rPPG Pulse Signal "
                        "(0.75 - 2.5 Hz)"
                    ),
                    fontsize=10,
                    fontweight="bold",
                    pad=10
                )


                ax_w.set_xlabel(
                    "Time (seconds)",
                    fontsize=9
                )


                ax_w.set_ylabel(
                    "Normalized Amplitude",
                    fontsize=9
                )


                ax_w.grid(
                    True,
                    linestyle="--",
                    alpha=0.4
                )


                ax_w.legend(
                    loc="upper right",
                    fontsize=8
                )


                fig_waveform.tight_layout()


                st.pyplot(
                    fig_waveform
                )


            else:

                st.info(
                    "rPPG waveform data is not available."
                )


        # =====================================================
        # PSD
        # =====================================================

        st.html(
            "<div style='height:1.25rem;'></div>"
        )


        st.html(
            """
            <div class="
                vital-card
                vital-card-violet
            ">

                <div class="vital-label">
                    Frequency Domain Analysis
                </div>

                <div class="vital-title">
                    Power Spectral Density
                </div>

                <div class="vital-text">
                    Frequency-domain analysis used to
                    identify the dominant cardiac pulse rate.
                </div>

            </div>
            """
        )


        psd_freqs = np.array(
            findings.get(
                "psd_freqs",
                []
            )
        )


        psd_vals = np.array(
            findings.get(
                "psd_power",
                []
            )
        )


        fig_spectrum = None


        if (
            len(psd_freqs) > 0
            and len(psd_vals) > 0
        ):

            freqs_bpm = (
                psd_freqs * 60.0
            )


            fig_spectrum, ax_s = plt.subplots(
                figsize=(10, 3.2),
                dpi=120
            )


            ax_s.plot(
                freqs_bpm,
                psd_vals,
                linewidth=1.8,
                label="Welch PSD"
            )


            ax_s.axvline(
                x=hr,
                linestyle="--",
                linewidth=1.5,
                label=(
                    f"Peak: {hr_text} BPM "
                    f"({frequency_text})"
                )
            )


            ax_s.set_title(
                (
                    "Power Spectral Density across "
                    "Physiological Heart Rate Band "
                    "(45 - 150 BPM)"
                ),
                fontsize=10,
                fontweight="bold"
            )


            ax_s.set_xlabel(
                "Frequency (BPM)",
                fontsize=9
            )


            ax_s.set_ylabel(
                "Spectral Power Density",
                fontsize=9
            )


            ax_s.grid(
                True,
                linestyle="--",
                alpha=0.4
            )


            ax_s.legend(
                loc="upper right",
                fontsize=8
            )


            fig_spectrum.tight_layout()


            st.pyplot(
                fig_spectrum
            )


        else:

            st.info(
                "Power spectral density data is not available."
            )


        # =====================================================
        # CLINICAL CARDS
        # =====================================================

        st.html(
            "<div style='height:1.25rem;'></div>"
        )


        clinical_col1, clinical_col2 = st.columns(
            2,
            gap="medium"
        )


        with clinical_col1:

            st.html(
                f"""
                <div class="
                    vital-card
                    vital-card-blue
                ">

                    <div class="vital-label">
                        Clinical Impression
                    </div>

                    <div class="vital-text">
                        {clinical_impression}
                    </div>

                </div>
                """
            )


        with clinical_col2:

            st.html(
                f"""
                <div class="
                    vital-card
                    vital-card-teal
                ">

                    <div class="vital-label">
                        Recommended Action
                    </div>

                    <div class="vital-text">
                        {recommended_action}
                    </div>

                </div>
                """
            )


        # =====================================================
        # STRUCTURED OUTPUT CARD
        # =====================================================

        st.html(
            "<div style='height:1.25rem;'></div>"
        )


        st.html(
            """
            <div class="
                vital-card
                vital-card-soft
            ">

                <div class="vital-label">
                    Structured Analysis Output
                </div>

                <div class="vital-text">
                    Complete machine-readable PhysFormer
                    rPPG analysis output.
                </div>

            </div>
            """
        )


        with st.expander(
            "View Structured Analysis JSON"
        ):

            json_preview = {
                key: value
                for key, value in findings.items()
                if key not in [
                    "annotated_frame",
                    "rppg_signal",
                    "raw_rppg",
                    "time_axis",
                    "psd_freqs",
                    "psd_power"
                ]
            }


            st.json(
                json_preview
            )


        # =====================================================
        # DISCLAIMER CARD
        # =====================================================

        st.html(
            "<div style='height:1.25rem;'></div>"
        )


        st.html(
            f"""
            <div class="vital-disclaimer">

                <b>
                    Investigational Medical Disclaimer:
                </b>

                {disclaimer}

                <br><br>

                <b>
                    Device acceleration:
                </b>

                {device}

            </div>
            """
        )


        # Build in-memory TXT report (excluding raw 160-sample waveform and PSD arrays as requested)
        video_src_name = (
            uploaded_file.name
            if ("uploaded_file" in locals() and uploaded_file is not None)
            else "Patient Facial Video Recording"
        )
        conf_val = findings.get("confidence_score", 0.0)
        conf_str = f"{conf_val * 100:.1f}%" if isinstance(conf_val, (int, float)) and conf_val <= 1.0 else f"{conf_val}%" if isinstance(conf_val, (int, float)) else str(conf_val)

        txt_content_video = (
            "================================================================================\n"
            "JIVA MEDINTELL\n"
            "PhysFormer Video Vital Signs Analysis Report\n"
            "================================================================================\n\n"
            f"File/Input: {video_src_name}\n"
            f"Analysis Pipeline: Face Detection / ROI Tracking -> PhysFormer rPPG -> Welch Spectral Density\n"
            f"Device Acceleration: {device}\n\n"
            "--------------------------------------------------------------------------------\n"
            "PHYSIOLOGICAL RESULTS\n"
            "--------------------------------------------------------------------------------\n"
            f"Estimated Heart Rate  : {hr_text} BPM\n"
            f"Rhythm Classification : {status_label}\n"
            f"Confidence Score      : {conf_str}\n"
            f"Signal Quality        : {signal_quality}\n"
            f"SNR (Signal-to-Noise) : {snr_text}\n"
            f"Dominant Frequency    : {frequency_text}\n\n"
            "--------------------------------------------------------------------------------\n"
            "ACQUISITION METRICS\n"
            "--------------------------------------------------------------------------------\n"
            f"Analyzed Window  : {frames_text} frames\n"
            f"Sampling Rate    : {fps_text}\n"
            f"Analyzed Duration: {findings.get('duration_analyzed_sec', 'N/A')} seconds\n"
            f"Face Localization: {'Detected' if findings.get('face_detected') else 'N/A'} (BBox: {findings.get('face_bbox', 'N/A')})\n\n"
            "--------------------------------------------------------------------------------\n"
            "CLINICAL IMPRESSION\n"
            "--------------------------------------------------------------------------------\n"
            f"{clinical_impression}\n\n"
            "--------------------------------------------------------------------------------\n"
            "RECOMMENDED ACTION\n"
            "--------------------------------------------------------------------------------\n"
            f"{recommended_action}\n\n"
            "--------------------------------------------------------------------------------\n"
            f"DISCLAIMER: {disclaimer}\n"
            "Generated by JIVA MEDINTELL\n"
            "================================================================================\n"
        )

        # =====================================================
        # EXPORT / SHARING
        # =====================================================

        st.markdown("<br>", unsafe_allow_html=True)

        col_wa, col_txt, col_pdf, _ = st.columns(
            [0.45, 1.8, 1.8, 1.5],
            gap="small"
        )

        with col_wa:
            st.markdown(
                """
                <div
                    title="Share via WhatsApp"
                    style="
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        width:42px;
                        height:38px;
                        margin-top:1px;
                        background:#25D366;
                        border-radius:8px;
                        cursor:default;
                    "
                >
                    <svg
                        width="21"
                        height="21"
                        viewBox="0 0 24 24"
                        fill="white"
                    >
                        <path d="M.057 24l1.687-6.163c-1.041-1.804-1.588-3.849-1.587-5.946.003-6.556 5.338-11.891 11.893-11.891 3.181.001 6.167 1.24 8.413 3.488 2.245 2.248 3.481 5.236 3.48 8.414-.003 6.557-5.338 11.892-11.893 11.892-1.99-.001-3.951-.5-5.688-1.448l-6.305 1.654zm6.597-3.807c1.676.995 3.276 1.591 5.392 1.592 5.448 0 9.886-4.434 9.889-9.885.002-5.462-4.415-9.89-9.881-9.892-5.452 0-9.887 4.434-9.889 9.884-.001 2.225.651 3.891 1.746 5.634l-.999 3.648 3.742-.981zm11.387-5.464c-.074-.124-.272-.198-.57-.347-.297-.149-1.758-.868-2.031-.967-.272-.099-.47-.149-.669.149-.198.297-.768.967-.941 1.165-.173.198-.347.223-.644.074-.297-.149-1.255-.462-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.297-.347.446-.521.151-.172.2-.296.3-.495.099-.198.05-.372-.025-.521-.075-.148-.669-1.611-.916-2.206-.242-.579-.487-.501-.669-.51l-.57-.01c-.198 0-.52.074-.792.372s-1.04 1.016-1.04 2.479 1.065 2.876 1.213 3.074c.149.198 2.095 3.2 5.076 4.487.709.306 1.263.489 1.694.626.712.226 1.36.194 1.872.118.571-.085 1.758-.719 2.006-1.413.248-.695.248-1.29.173-1.414z"/>
                    </svg>
                </div>
                """,
                unsafe_allow_html=True
            )

        

        with col_txt:
            st.download_button(
                label="Download Results TXT",
                data=txt_content_video,
                file_name="JIVA_Video_Results.txt",
                mime="text/plain",
                key="btn_dl_txt_vitals",
                type="secondary",
                width="stretch"
            )

        with col_pdf:
            if st.button(
                "Generate Results PDF",
                key="btn_gen_pdf_vitals",
                type="secondary",
                width="stretch"
            ):
                try:
                    with st.spinner(
                        "Generating PDF report..."
                    ):
                        pdf_data = (
                            generate_vital_signs_report(
                                findings=findings,
                                fig_waveform=fig_waveform,
                                fig_spectrum=fig_spectrum,
                                face_image=findings.get(
                                    "annotated_frame"
                                )
                            )
                        )

                        if pdf_data:
                            st.session_state[
                                "pdf_vital_signs_bytes"
                            ] = pdf_data
                        else:
                            st.error(
                                "Unable to generate the PDF report."
                            )

                except Exception as pdf_error:
                    st.error(
                        "Unable to generate the PDF report: "
                        f"{pdf_error}"
                    )

            if (
                "pdf_vital_signs_bytes"
                in st.session_state
            ):
                st.download_button(
                    label="Download Results PDF",
                    data=st.session_state[
                        "pdf_vital_signs_bytes"
                    ],
                    file_name=(
                        "JIVA_Vital_Signs_Report.pdf"
                    ),
                    mime="application/pdf",
                    key="btn_dl_pdf_vitals",
                    type="secondary",
                    width="stretch"
                )
