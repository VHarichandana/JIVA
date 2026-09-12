import io
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    HRFlowable,
    KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

# JIVA Palette Colors
COLOR_BRAND_BLUE = colors.HexColor("#4B63E6")
COLOR_HEADER_DARK = colors.HexColor("#1E293B")
COLOR_BODY_DARK = colors.HexColor("#252525")
COLOR_MUTED = colors.HexColor("#64748B")
COLOR_BORDER = colors.HexColor("#E2E8F0")
COLOR_BG_LIGHT = colors.HexColor("#F8FAFC")
COLOR_SOFT_BLUE = colors.HexColor("#EDF1FF")
COLOR_BENIGN = colors.HexColor("#047857")
COLOR_CANCER = colors.HexColor("#B91C1C")
COLOR_CONTROL = colors.HexColor("#2563EB")

def get_report_styles():
    styles = getSampleStyleSheet()
    
    brand_style = ParagraphStyle(
        "JIVABrand",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=COLOR_BRAND_BLUE
    )
    
    brand_sub_style = ParagraphStyle(
        "JIVASubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=COLOR_MUTED
    )
    
    report_title_style = ParagraphStyle(
        "JIVAReportTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=COLOR_HEADER_DARK
    )
    
    meta_style = ParagraphStyle(
        "JIVAMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=COLOR_MUTED
    )
    
    section_h2_style = ParagraphStyle(
        "JIVASectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=COLOR_BRAND_BLUE,
        spaceBefore=10,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        "JIVABody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=COLOR_BODY_DARK
    )
    
    table_hdr_style = ParagraphStyle(
        "JIVATableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=COLOR_HEADER_DARK
    )
    
    table_cell_style = ParagraphStyle(
        "JIVATableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=COLOR_BODY_DARK
    )
    
    disclaimer_style = ParagraphStyle(
        "JIVADisclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=COLOR_MUTED
    )
    
    return {
        "brand": brand_style,
        "brand_sub": brand_sub_style,
        "report_title": report_title_style,
        "meta": meta_style,
        "h2": section_h2_style,
        "body": body_style,
        "th": table_hdr_style,
        "td": table_cell_style,
        "disclaimer": disclaimer_style
    }

def fig_to_flowable(fig, max_width=500, max_height=220):
    """Converts a matplotlib Figure into a ReportLab Image in-memory."""
    if fig is None:
        return None
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        buf.seek(0)
        reader = ImageReader(buf)
        iw, ih = reader.getSize()
        aspect = ih / float(max(1, iw))
        w = min(max_width, iw)
        h = w * aspect
        if h > max_height:
            h = max_height
            w = h / aspect
        return RLImage(buf, width=w, height=h)
    except Exception:
        return None

def pil_to_flowable(pil_img, max_width=320, max_height=260):
    """Converts a PIL Image into a ReportLab Image in-memory."""
    if pil_img is None:
        return None
    try:
        buf = io.BytesIO()
        if pil_img.mode in ("RGBA", "P"):
            converted = pil_img.convert("RGB")
            converted.save(buf, format="JPEG", quality=90)
        else:
            pil_img.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        reader = ImageReader(buf)
        iw, ih = reader.getSize()
        aspect = ih / float(max(1, iw))
        w = min(max_width, iw)
        h = w * aspect
        if h > max_height:
            h = max_height
            w = h / aspect
        return RLImage(buf, width=w, height=h)
    except Exception:
        return None

def build_pdf_header(report_title: str, styles: dict) -> list:
    """Builds the standardized JIVA header banner."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story = [
        Paragraph("JIVA", styles["brand"]),
        Paragraph("AI-Assisted Lung Health Analysis", styles["brand_sub"]),
        Spacer(1, 4),
        HRFlowable(width="100%", thickness=1.5, color=COLOR_BRAND_BLUE, spaceBefore=2, spaceAfter=8),
        Paragraph(f"<b>{report_title}</b>", styles["report_title"]),
        Paragraph(f"Generated: {now_str} &nbsp;|&nbsp; Modality: Multimodal Diagnostic Screening", styles["meta"]),
        Spacer(1, 10),
    ]
    return story

def build_disclaimer_block(styles: dict) -> list:
    """Standardized disclaimer block for all reports."""
    disclaimer_text = (
        "<b>Disclaimer:</b> JIVA is an AI-assisted research system. Results are intended for educational and "
        "research use and should not be considered a medical diagnosis or substitute for professional clinical evaluation."
    )
    story = [
        Spacer(1, 14),
        HRFlowable(width="100%", thickness=0.8, color=COLOR_BORDER, spaceBefore=4, spaceAfter=6),
        Paragraph(disclaimer_text, styles["disclaimer"])
    ]
    return story

def generate_voc_report(
    input_df: pd.DataFrame,
    pred_class: str,
    confidence_pct: float,
    probabilities: list,
    classes: list,
    classifier_name: str = "Bagging Ensemble Classifier",
    fig_probs = None,
    fig_shap = None,
    shap_summary: str = None
) -> bytes:
    """
    Generates an in-memory PDF for VOC Analysis.
    Never crashes; returns PDF bytes or None.
    """
    try:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=letter,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        styles = get_report_styles()
        story = build_pdf_header("VOC Analysis Report", styles)

        # 1. Primary Classification Result Table
        story.append(Paragraph("Diagnostic Classification Summary", styles["h2"]))
        
        result_data = [
            [
                Paragraph("<b>Predicted Class:</b>", styles["th"]),
                Paragraph(f"<b>{pred_class.upper()}</b>", styles["th"]),
                Paragraph("<b>Confidence Score:</b>", styles["th"]),
                Paragraph(f"<b>{confidence_pct:.2f}%</b>", styles["th"])
            ],
            [
                Paragraph("<b>Classifier Model:</b>", styles["th"]),
                Paragraph(f"{classifier_name}", styles["td"]),
                Paragraph("<b>Biomarkers Evaluated:</b>", styles["th"]),
                Paragraph(f"{len(input_df.columns)} VOCs", styles["td"])
            ]
        ]
        res_table = Table(result_data, colWidths=[120, 150, 130, 140])
        res_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), COLOR_BG_LIGHT),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(res_table)
        story.append(Spacer(1, 8))

        # 2. Probability Breakdown Table
        prob_rows = [
            [Paragraph("<b>Class Name</b>", styles["th"]), Paragraph("<b>Probability (%)</b>", styles["th"])]
        ]
        for c, p in zip(classes, probabilities):
            prob_rows.append([
                Paragraph(str(c), styles["td"]),
                Paragraph(f"{float(p)*100:.2f}%", styles["td"])
            ])
        prob_table = Table(prob_rows, colWidths=[200, 150])
        prob_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLOR_SOFT_BLUE),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(prob_table)
        story.append(Spacer(1, 10))

        # 3. Probability Distribution Chart (if available)
        chart_flow = fig_to_flowable(fig_probs, max_width=480, max_height=170)
        if chart_flow:
            story.append(Paragraph("Class Probability Distribution", styles["h2"]))
            story.append(chart_flow)
            story.append(Spacer(1, 10))

        # 4. Input Biomarker Summary Table (4-Column compact layout for 27 VOCs)
        story.append(Paragraph("Input Biomarker Summary (ppm / ppb)", styles["h2"]))
        voc_items = [(str(c), float(input_df[c].iloc[0])) for c in input_df.columns]
        
        table_rows = [
            [
                Paragraph("<b>Compound Name</b>", styles["th"]),
                Paragraph("<b>Concentration</b>", styles["th"]),
                Paragraph("<b>Compound Name</b>", styles["th"]),
                Paragraph("<b>Concentration</b>", styles["th"])
            ]
        ]
        mid = (len(voc_items) + 1) // 2
        for i in range(mid):
            col1_name, col1_val = voc_items[i]
            if i + mid < len(voc_items):
                col2_name, col2_val = voc_items[i + mid]
                col2_val_str = f"{col2_val:.4f}"
            else:
                col2_name, col2_val_str = "", ""
                
            table_rows.append([
                Paragraph(col1_name, styles["td"]),
                Paragraph(f"{col1_val:.4f}", styles["td"]),
                Paragraph(col2_name, styles["td"]),
                Paragraph(col2_val_str, styles["td"])
            ])
            
        bio_table = Table(table_rows, colWidths=[140, 130, 140, 130])
        bio_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLOR_BG_LIGHT),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(bio_table)
        story.append(Spacer(1, 10))

        # 5. SHAP / XAI Explainability (if available)
        shap_flow = fig_to_flowable(fig_shap, max_width=480, max_height=180)
        if shap_flow:
            story.append(Paragraph("Model Explainability (SHAP Feature Contribution)", styles["h2"]))
            story.append(shap_flow)
            if shap_summary:
                story.append(Paragraph(f"<i>{shap_summary}</i>", styles["meta"]))
            story.append(Spacer(1, 6))

        # 6. Disclaimer
        story.extend(build_disclaimer_block(styles))

        doc.build(story)
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None

def generate_audio_report(findings: dict, fig_audio = None) -> bytes:
    """
    Generates an in-memory PDF for Cardiopulmonary Audio Analysis.
    Never crashes; returns PDF bytes or None.
    """
    try:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=letter,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        styles = get_report_styles()
        story = build_pdf_header("Cardiopulmonary Audio Analysis Report", styles)

        diag = findings.get("diagnostic_classification", {})
        acoustics = findings.get("acoustic_features", {})
        
        # 1. Primary Diagnostic Finding
        story.append(Paragraph("Diagnostic Classification", styles["h2"]))
        conf = diag.get("confidence_score", 0.0) * 100
        
        diag_data = [
            [
                Paragraph("<b>Sound Pattern:</b>", styles["th"]),
                Paragraph(f"<b>{diag.get('sound_type', 'N/A')}</b>", styles["th"]),
                Paragraph("<b>Confidence:</b>", styles["th"]),
                Paragraph(f"<b>{conf:.1f}%</b>", styles["th"])
            ],
            [
                Paragraph("<b>Anomaly Status:</b>", styles["th"]),
                Paragraph(f"{diag.get('anomaly_status', 'N/A')}", styles["td"]),
                Paragraph("<b>Severity Level:</b>", styles["th"]),
                Paragraph(f"{diag.get('severity_level', 'N/A')}", styles["td"])
            ]
        ]
        diag_table = Table(diag_data, colWidths=[110, 170, 110, 150])
        diag_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), COLOR_BG_LIGHT),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(diag_table)
        story.append(Spacer(1, 10))

        # 2. Recording & Acoustic Metadata
        story.append(Paragraph("Acoustic Properties & Audio Metadata", styles["h2"]))
        meta_data = [
            [
                Paragraph("<b>Recording File:</b>", styles["th"]),
                Paragraph(f"{findings.get('audio_file', 'N/A')}", styles["td"]),
                Paragraph("<b>Duration (Est.):</b>", styles["th"]),
                Paragraph(f"{acoustics.get('estimated_duration_sec', 'N/A')} s", styles["td"])
            ],
            [
                Paragraph("<b>Dominant Frequency:</b>", styles["th"]),
                Paragraph(f"{acoustics.get('estimated_frequency_hz', 'N/A')} Hz", styles["td"]),
                Paragraph("<b>Relative Amplitude:</b>", styles["th"]),
                Paragraph(f"{acoustics.get('relative_rms_amplitude', 'N/A')}", styles["td"])
            ]
        ]
        if "clinical_prompt" in findings and findings["clinical_prompt"]:
            meta_data.append([
                Paragraph("<b>Clinical Query:</b>", styles["th"]),
                Paragraph(f"{findings['clinical_prompt']}", styles["td"]),
                Paragraph("", styles["th"]),
                Paragraph("", styles["td"])
            ])
            
        meta_table = Table(meta_data, colWidths=[120, 160, 120, 140])
        meta_table.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 10))

        # 3. Clinical Impression & Recommendation
        story.append(Paragraph("Clinical Observations & Findings", styles["h2"]))
        story.append(Paragraph(f"<b>Impression:</b> {findings.get('clinical_impression', 'N/A')}", styles["body"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"<b>Recommended Follow-up:</b> {findings.get('recommended_action', 'N/A')}", styles["body"]))
        story.append(Spacer(1, 10))

        # 4. Waveform & Spectrogram Plot (if available)
        audio_flow = fig_to_flowable(fig_audio, max_width=480, max_height=200)
        if audio_flow:
            story.append(Paragraph("Acoustic Signal & Frequency Spectrogram", styles["h2"]))
            story.append(audio_flow)
            story.append(Spacer(1, 6))

        # 5. Disclaimer
        story.extend(build_disclaimer_block(styles))

        doc.build(story)
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None

def generate_xray_report(findings: dict, pil_image: Image.Image = None, filename: str = "chest_radiograph.png") -> bytes:
    """
    Generates an in-memory PDF for Chest X-ray Analysis.
    Never crashes; returns PDF bytes or None.
    """
    try:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=letter,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        styles = get_report_styles()
        story = build_pdf_header("Chest X-ray Analysis Report", styles)

        cls_info = findings.get("classification", {})
        conf = cls_info.get("confidence_score", 0.0) * 100
        metrics = findings.get("radiological_metrics", {})
        
        # 1. Primary Radiological Finding
        story.append(Paragraph("Radiological Assessment", styles["h2"]))
        diag_data = [
            [
                Paragraph("<b>Primary Finding:</b>", styles["th"]),
                Paragraph(f"<b>{findings.get('primary_diagnostic_finding', 'N/A')}</b>", styles["th"]),
                Paragraph("<b>Confidence:</b>", styles["th"]),
                Paragraph(f"<b>{conf:.1f}%</b>", styles["th"])
            ],
            [
                Paragraph("<b>Risk Status:</b>", styles["th"]),
                Paragraph(f"{cls_info.get('risk_assessment', 'N/A')}", styles["td"]),
                Paragraph("<b>Image File:</b>", styles["th"]),
                Paragraph(f"{findings.get('image_file', filename)}", styles["td"])
            ]
        ]
        diag_table = Table(diag_data, colWidths=[110, 180, 100, 150])
        diag_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), COLOR_BG_LIGHT),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(diag_table)
        story.append(Spacer(1, 10))

        # 2. Detailed Observations
        story.append(Paragraph("Detailed Radiological Observations", styles["h2"]))
        obs_list = findings.get("detailed_observations", [])
        for obs in obs_list:
            story.append(Paragraph(f"&bull; {obs}", styles["body"]))
            story.append(Spacer(1, 3))
        story.append(Spacer(1, 6))

        # 3. Clinical Impression & Recommendation
        story.append(Paragraph(f"<b>Clinical Impression:</b> {findings.get('clinical_impression', 'N/A')}", styles["body"]))
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>Recommended Follow-up:</b> {findings.get('recommended_followup', 'N/A')}", styles["body"]))
        story.append(Spacer(1, 10))

        # 4. Radiological Image Details & Preview
        if pil_image:
            story.append(Paragraph("Radiograph Inspection Preview", styles["h2"]))
            xray_flow = pil_to_flowable(pil_image, max_width=320, max_height=260)
            if xray_flow:
                story.append(xray_flow)
                story.append(Spacer(1, 6))
            
            # Metadata row
            img_meta = [
                [
                    Paragraph("<b>Resolution:</b>", styles["th"]),
                    Paragraph(f"{pil_image.width} x {pil_image.height} px", styles["td"]),
                    Paragraph("<b>Color Mode:</b>", styles["th"]),
                    Paragraph(f"{pil_image.mode}", styles["td"]),
                    Paragraph("<b>Aspect Ratio:</b>", styles["th"]),
                    Paragraph(f"{metrics.get('aspect_ratio', round(pil_image.width/max(1, pil_image.height), 2))}", styles["td"])
                ]
            ]
            meta_tbl = Table(img_meta, colWidths=[80, 100, 80, 100, 80, 100])
            meta_tbl.setStyle(TableStyle([
                ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(meta_tbl)

        # 5. Disclaimer
        story.extend(build_disclaimer_block(styles))

        doc.build(story)
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None

def generate_vital_signs_report(
    findings: dict,
    fig_waveform=None,
    fig_spectrum=None,
    face_image=None
) -> bytes:
    """
    Generates an in-memory PDF for PhysFormer Video Vital Signs (rPPG Heart Rate).
    Never crashes; returns PDF bytes or None.
    """
    try:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=letter,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        styles = get_report_styles()
        story = build_pdf_header("Video Vital Signs (rPPG) Analysis Report", styles)

        hr = findings.get("heart_rate", "N/A")
        unit = findings.get("unit", "BPM")
        quality = findings.get("signal_quality", "N/A")
        conf = findings.get("confidence_score", 0.0) * 100
        snr = findings.get("snr_db", 0.0)
        fps = findings.get("fps", 30.0)
        dom_freq = findings.get("dominant_freq_hz", "N/A")

        # 1. Primary Metrics Table
        story.append(Paragraph("Physiological Vital Signs Summary", styles["h2"]))
        metrics_data = [
            [
                Paragraph("<b>Estimated Heart Rate:</b>", styles["th"]),
                Paragraph(f"<b>{hr} {unit}</b>", styles["th"]),
                Paragraph("<b>Signal Quality:</b>", styles["th"]),
                Paragraph(f"<b>{quality}</b>", styles["th"])
            ],
            [
                Paragraph("<b>Confidence Score:</b>", styles["th"]),
                Paragraph(f"{conf:.1f}%", styles["td"]),
                Paragraph("<b>Signal-to-Noise Ratio:</b>", styles["th"]),
                Paragraph(f"{snr:.1f} dB", styles["td"])
            ],
            [
                Paragraph("<b>Dominant Frequency:</b>", styles["th"]),
                Paragraph(f"{dom_freq} Hz", styles["td"]),
                Paragraph("<b>Effective Video FPS:</b>", styles["th"]),
                Paragraph(f"{fps} fps", styles["td"])
            ],
            [
                Paragraph("<b>Status Classification:</b>", styles["th"]),
                Paragraph(f"{findings.get('status_classification', 'Normal')}", styles["td"]),
                Paragraph("<b>Analyzed Window:</b>", styles["th"]),
                Paragraph(f"{findings.get('frames_analyzed', 160)} frames ({findings.get('duration_analyzed_sec', 5.3)}s)", styles["td"])
            ]
        ]
        metrics_table = Table(metrics_data, colWidths=[130, 140, 130, 140])
        metrics_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), COLOR_BG_LIGHT),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 10))

        # 2. Clinical Impression & Action
        story.append(Paragraph("Clinical Observations & Advisory", styles["h2"]))
        story.append(Paragraph(f"<b>Clinical Impression:</b> {findings.get('clinical_impression', 'N/A')}", styles["body"]))
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>Recommended Action:</b> {findings.get('recommended_action', 'N/A')}", styles["body"]))
        story.append(Spacer(1, 8))

        # 3. Waveform Chart
        if fig_waveform:
            story.append(Paragraph("Recovered rPPG Pulse Waveform", styles["h2"]))
            wave_flow = fig_to_flowable(fig_waveform, max_width=520, max_height=160)
            if wave_flow:
                story.append(wave_flow)
                story.append(Spacer(1, 8))

        # 4. Power Spectrum Chart
        if fig_spectrum:
            story.append(Paragraph("Power Spectral Density (PSD) Frequency Analysis", styles["h2"]))
            spec_flow = fig_to_flowable(fig_spectrum, max_width=520, max_height=160)
            if spec_flow:
                story.append(spec_flow)
                story.append(Spacer(1, 8))

        # 5. Face Detection ROI Snapshot
        if face_image is not None:
            story.append(Paragraph("Face ROI Spatial Localization", styles["h2"]))
            if isinstance(face_image, np.ndarray):
                face_pil = Image.fromarray(face_image)
            else:
                face_pil = face_image
            face_flow = pil_to_flowable(face_pil, max_width=240, max_height=180)
            if face_flow:
                story.append(face_flow)
                story.append(Spacer(1, 6))

        # 6. Safety & AI Disclaimer
        story.extend(build_disclaimer_block(styles))

        doc.build(story)
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None

