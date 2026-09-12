import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def get_custom_css():
    """
    Primary Sarvam.ai-inspired light design system for JIVA.
    Exact colors, typography scale, 18-24px rounded cards, thin 1px borders,
    dark charcoal CTAs, soft blue & soft warm highlights, and generous whitespace.
    """
    return """
    <style>
    /* Sarvam.ai Design Tokens & Variables */
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        --background: #fafafa;
        --surface: #ffffff;
        --text: #252525;
        --muted: #707070;
        --border: #e8e8e8;
        --blue: #4b63e6;
        --soft-blue: #edf1ff;
        --warm: #f3a45b;
        --soft-warm: #fff0e4;
        --dark-button: #2d2e34;
    }
    
    .stApp {
        background-color: var(--background);
        color: var(--text);
        font-family: 'Manrope', 'Inter', system-ui, -apple-system, sans-serif;
        -webkit-font-smoothing: antialiased;
    }
    
    /* Hide Streamlit Header Chrome & Footer */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    footer {
        display: none !important;
    }
    
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 5rem !important;
        max-width: 1180px !important;
    }
    
    /* Top Navigation Header Bar */
    .sarvam-navbar {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.85rem 2rem;
        margin-bottom: 2.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .sarvam-brand {
        font-size: 56px !important;
        font-weight: 800 !important;
        color: var(--blue) !important;
        letter-spacing: -1.5px !important;
        line-height: 1 !important;
        margin: 0 !important;
        text-decoration: none !important;
        white-space: nowrap !important;
        display: inline-block !important;
    }
    
    .sarvam-brand span {
        color: var(--blue);
    }
    
    /* Hero Section - Direct on background */
    .sarvam-hero {
        background-color: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 1.5rem 0 1rem 0 !important;
        margin-bottom: 1rem !important;
        box-shadow: none !important;
    }
    
    html {
        scroll-behavior: smooth !important;
    }
    
    #ask-jiva {
        scroll-margin-top: 100px !important;
    }
    
    .sarvam-hero-heading {
        font-size: 3.2rem;
        font-weight: 800;
        color: var(--text);
        letter-spacing: -0.04em;
        line-height: 1.12;
        margin-bottom: 1.25rem;
        max-width: 850px;
    }
    
    .sarvam-hero-sub {
        font-size: 1.2rem;
        color: var(--muted);
        font-weight: 400;
        line-height: 1.65;
        margin-bottom: 2.25rem;
        max-width: 800px;
    }
    
    /* Sarvam Primary Large Cards */
    .sarvam-card {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 2.25rem;
        height: 100%;
        transition: transform 0.2s ease, border-color 0.2s ease;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
    }
    
    .sarvam-card:hover {
        border-color: var(--blue);
        transform: translateY(-2px);
    }
    
    .sarvam-card-corner-accent {
        position: absolute;
        top: 0;
        right: 0;
        width: 90px;
        height: 90px;
        background: radial-gradient(circle at top right, var(--soft-blue) 0%, transparent 70%);
        pointer-events: none;
    }
    
    .sarvam-card-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.03em;
        margin-bottom: 0.75rem;
    }
    
    .sarvam-card-desc {
        font-size: 1rem;
        color: var(--muted);
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }
    
    .sarvam-support-line {
        font-size: 0.85rem;
        color: var(--blue);
        font-weight: 600;
        background-color: var(--soft-blue);
        padding: 0.45rem 0.9rem;
        border-radius: 8px;
        margin-bottom: 1.75rem;
        display: inline-block;
    }
    
    /* Compact Secondary Grid Cards */
    .sarvam-grid-card {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.75rem;
        height: 100%;
        transition: border-color 0.2s ease;
    }
    
    .sarvam-grid-card:hover {
        border-color: var(--blue);
    }
    
    .sarvam-grid-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }
    
    .sarvam-grid-desc {
        font-size: 0.92rem;
        color: var(--muted);
        line-height: 1.55;
        margin: 0;
    }
    
    /* Large Editorial Section */
    .sarvam-editorial-card {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 3rem;
        margin-bottom: 3.5rem;
    }
    
    /* Sarvam Dark CTA Buttons (Rectangular with 12px rounded corners) */
    div.stButton > button {
        background-color: var(--dark-button) !important;
        color: #ffffff !important;
        border: 1px solid var(--dark-button) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 0.6rem 1.25rem !important;
        white-space: nowrap !important;
        transition: background-color 0.2s ease, color 0.2s ease, border-color 0.2s ease !important;
        transform: none !important;
    }
    
    div.stButton > button:hover {
        background-color: #E5E7EB !important;
        border-color: #E5E7EB !important;
        color: #111111 !important;
        transform: none !important;
    }
    
    /* Top Navbar Button Overrides (Transparent, text-based navigation, no dark box) */
    .nav-btn button {
        background-color: transparent !important;
        background: transparent !important;
        color: #1F2937 !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.5rem 0.65rem !important;
        white-space: nowrap !important;
        width: 100% !important;
        height: 38px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: background-color 0.2s ease, color 0.2s ease !important;
        transform: none !important;
    }
    
    .nav-btn button:hover {
        background-color: rgba(255, 255, 255, 0.45) !important;
        border: none !important;
        box-shadow: none !important;
        color: #111111 !important;
        transform: none !important;
    }

    .nav-btn button:focus,
    .nav-btn button:active {
        background-color: transparent !important;
        color: #3b50c8 !important;
        box-shadow: none !important;
        border: none !important;
    }

    .nav-btn.nav-btn-active button {
        color: #3b50c8 !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #3b50c8 !important;
        border-radius: 0px !important;
        background: transparent !important;
    }

    /* Hero Buttons Side-by-Side Horizontal Row Alignment */
    div[data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
        align-items: center !important;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.35rem;
        text-align: center;
        transition: border-color 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: var(--blue);
    }
    
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: var(--blue);
        letter-spacing: -0.03em;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: var(--muted);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Result Badges */
    .risk-badge-benign {
        background-color: #ecfdf5;
        color: #047857;
        padding: 0.65rem 1.25rem;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
        border: 1px solid #a7f3d0;
    }
    
    .risk-badge-cancer {
        background-color: #fef2f2;
        color: #b91c1c;
        padding: 0.65rem 1.25rem;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
        border: 1px solid #fca5a5;
    }
    
    .risk-badge-control {
        background-color: var(--soft-blue);
        color: var(--blue);
        padding: 0.65rem 1.25rem;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
        border: 1px solid #c7d2fe;
    }
    
    .disclaimer-box {
        background-color: var(--soft-warm);
        border-left: 4px solid var(--warm);
        padding: 1.1rem 1.35rem;
        border-radius: 8px;
        color: #7c2d12;
        font-size: 0.88rem;
        line-height: 1.6;
        margin-top: 2rem;
    }
    
    /* Ask JIVA Chatbot Permanent Text Visibility & Styling */
    .stChatMessage,
    div[data-testid="stChatMessage"],
    div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessageContent"] p,
    div[data-testid="stChatMessageContent"] span,
    div[data-testid="stChatMessageContent"] li,
    div[data-testid="stChatMessageContent"] ul,
    div[data-testid="stChatMessageContent"] ol,
    div[data-testid="stChatMessageContent"] strong,
    div[data-testid="stChatMessageContent"] em,
    div[data-testid="stChatMessageContent"] code,
    div[data-testid="stChatMessageContent"] div,
    div[data-testid="stChatMessageContent"] a {
        color: #111111 !important;
        -webkit-text-fill-color: #111111 !important;
        opacity: 1 !important;
    }
    
    div[data-testid="stChatMessage"] {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        padding: 1.1rem 1.35rem !important;
        margin-bottom: 0.85rem !important;
    }
    
    /* Assistant Chat Bubble Accent */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageContent"]) {
        background-color: var(--soft-blue) !important;
        border-color: #c7d2fe !important;
    }

    /* Chat Input Field Styling & Text Visibility */
    div[data-testid="stChatInput"],
    .stChatInputContainer {
        background-color: #ffffff !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }

    div[data-testid="stChatInput"] textarea,
    .stChatInputContainer textarea {
        color: #111111 !important;
        -webkit-text-fill-color: #111111 !important;
        background-color: #ffffff !important;
        font-size: 0.95rem !important;
        opacity: 1 !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder,
    .stChatInputContainer textarea::placeholder {
        color: #707070 !important;
        -webkit-text-fill-color: #707070 !important;
        opacity: 1 !important;
    }

    /* Global High-Contrast Gray Typography for Inputs & Uploaders */
    div[data-testid="stRadio"] > label,
    div[data-testid="stRadio"] > label p,
    div[data-testid="stRadio"] > label div p {
        color: #374151 !important; /* Dark Slate Gray */
        -webkit-text-fill-color: #374151 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] label,
    div[data-testid="stRadio"] div[role="radiogroup"] label div p,
    div[data-testid="stRadio"] div[role="radiogroup"] label p,
    div[data-testid="stRadio"] div[role="radiogroup"] span {
        color: #4B5563 !important; /* Medium Slate Gray */
        -webkit-text-fill-color: #4B5563 !important;
        font-weight: 500 !important;
    }

    div[data-testid="stFileUploader"] > label,
    div[data-testid="stFileUploader"] > label p,
    div[data-testid="stFileUploader"] > label div p {
        color: #374151 !important; /* Dark Slate Gray */
        -webkit-text-fill-color: #374151 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] {
        background-color: #ffffff !important;
        border: 1.5px dashed #9CA3AF !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
    }

    div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] span,
    div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] div,
    div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] p {
        color: #4B5563 !important; /* Medium Slate Gray */
        -webkit-text-fill-color: #4B5563 !important;
        font-weight: 500 !important;
    }

    div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] small {
        color: #6B7280 !important; /* Slate Gray 500 */
        -webkit-text-fill-color: #6B7280 !important;
        font-weight: 500 !important;
    }

    div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button {
        color: #374151 !important;
        -webkit-text-fill-color: #374151 !important;
        background-color: #F3F4F6 !important;
        border: 1px solid #D1D5DB !important;
        font-weight: 600 !important;
    }

    div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span,
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] small,
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] div,
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] {
        color: #4B5563 !important;
        -webkit-text-fill-color: #4B5563 !important;
        font-weight: 500 !important;
    }

    div[data-testid="stCheckbox"] label,
    div[data-testid="stCheckbox"] label p,
    div[data-testid="stCheckbox"] span {
        color: #4B5563 !important;
        -webkit-text-fill-color: #4B5563 !important;
        font-weight: 500 !important;
    }
    </style>
    """


def get_chatbot_response(query: str) -> str:
    """
    Professional chatbot response engine for JIVA.
    Emoji-free markdown formatting.
    """
    q = query.lower().strip()
    
    if any(k in q for k in ["voc", "breath", "biomarker", "gas", "volatile"]):
        return (
            "VOC Analysis Module\n\n"
            "This module evaluates 27 Volatile Organic Compounds (VOCs) captured in exhaled breath "
            "(such as CH2O, C2H4O, C3H6O, C5H10O). Elevated or altered concentrations of specific VOCs "
            "correlate with metabolic cellular changes associated with pulmonary neoplasms."
        )
    elif any(k in q for k in ["predict", "class", "cancer", "benign", "control", "risk"]):
        return (
            "Multi-Class Prediction Model\n\n"
            "The system classifies subjects into three clinical categories:\n"
            "- Benign: Non-cancerous pulmonary condition\n"
            "- Cancer: Malignant lung carcinoma\n"
            "- Control: Healthy baseline individual\n\n"
            "Predictions are generated using a Bagging Ensemble Classifier trained on clinical VOC profiles."
        )
    elif any(k in q for k in ["metric", "accuracy", "f1", "recall", "performance", "score", "mcc", "roc"]):
        return (
            "Model Performance Summary\n\n"
            "- Best Classification Model: Bagging Ensemble Classifier\n"
            "- Accuracy: Approximately 80.2%\n"
            "- Macro F1: Approximately 0.72\n"
            "- Weighted F1: Approximately 0.79\n"
            "- Cancer Recall: Approximately 90.6%\n"
            "- Best Regressor: MLP Regressor (predicting CH2O concentration levels)."
        )
    elif any(k in q for k in ["shap", "xai", "explain", "importance", "feature"]):
        return (
            "Explainable AI (SHAP / XAI)\n\n"
            "We use SHAP (SHapley Additive exPlanations) KernelExplainer to compute feature contributions. "
            "Model-agnostic SHAP sampling computes local and global feature impact without relying on single-tree assumptions."
        )
    elif any(k in q for k in ["audio", "stetholm", "cough", "breath sound", "auscultation"]):
        return (
            "Cardiopulmonary Audio Analysis\n\n"
            "The Cardiopulmonary Audio Analysis module processes acoustic auscultation recordings (.wav) to detect "
            "adventitious sound anomalies such as wheezes, fine inspiratory crackles, or rhonchi, returning structured clinical observations."
        )
    elif any(k in q for k in ["xray", "x-ray", "image", "ct", "radiology"]):
        return (
            "X-ray Analysis Module\n\n"
            "The X-ray analysis module evaluates chest radiographs (PNG/JPG) for parenchymal abnormalities, "
            "focal opacities, consolidations, or interstitial patterns, returning structured radiological observations."
        )
    elif any(k in q for k in ["workflow", "pipeline", "architecture", "step"]):
        return (
            "System Workflow\n\n"
            "1. Input Data: Submit VOC biomarker concentrations, acoustic audio, or chest radiographs.\n"
            "2. Preprocessing: Missing value imputation, Yeo-Johnson power transform, and Robust Scaling.\n"
            "3. Inference: Single-pass prediction via trained ensemble classifiers.\n"
            "4. Explanation: SHAP feature contribution analysis.\n"
            "5. Multimodal Integration: Connecting biomarker, acoustic, and radiological evidence."
        )
    elif any(k in q for k in ["limit", "disclaimer", "accuracy limit", "caution", "safety"]):
        return (
            "Clinical Limitations & Disclaimers\n\n"
            "- This application is developed strictly for research and diagnostic support purposes.\n"
            "- It should not be used as a standalone diagnostic tool.\n"
            "- Predictions must be verified by licensed medical professionals and clinical diagnostic tests."
        )
    else:
        return (
            "JIVA AI Assistant\n\n"
            "I can answer questions regarding:\n"
            "1. VOC Analysis - Volatile organic compound screening\n"
            "2. Cardiopulmonary Audio - Acoustic sound reasoning\n"
            "3. X-ray Analysis - Radiograph inspection\n"
            "4. Model Performance - Metrics (Accuracy 80.2%, Recall 90.6%)\n"
            "5. SHAP / XAI - Feature contribution explanations\n"
            "6. Application Workflow & Limitations\n\n"
            "How can I assist you?"
        )

def plot_class_probabilities(probabilities, class_names):
    """Generates a probability distribution bar plot styled for light background."""
    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    
    colors = ["#10b981", "#ef4444", "#4b63e6"]
    bars = ax.barh(class_names, probabilities * 100, color=colors, height=0.45, edgecolor="#e8e8e8")
    
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)", color="#707070", fontsize=9, fontweight="500")
    ax.tick_params(colors="#252525", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#e8e8e8")
    ax.spines["bottom"].set_color("#e8e8e8")
    ax.grid(axis="x", linestyle="--", alpha=0.3, color="#e8e8e8")
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", 
                va="center", color="#252525", fontsize=9, fontweight="600")
        
    plt.tight_layout()
    return fig

def plot_confusion_matrix_fig(cm, class_names):
    """Generates a confusion matrix heatmap styled for light theme."""
    fig, ax = plt.subplots(figsize=(5.5, 4.2), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                annot_kws={"size": 11, "weight": "bold", "color": "#1e2033"})
    
    ax.set_xlabel("Predicted Label", color="#707070", fontsize=9, fontweight="500")
    ax.set_ylabel("True Label", color="#707070", fontsize=9, fontweight="500")
    ax.tick_params(colors="#252525", labelsize=9)
    plt.tight_layout()
    return fig

def explain_prediction_shap(classifier, input_df, X_background, feature_names, class_names):
    """
    Computes SHAP explanations safely for Bagging Classifier.
    Model-agnostic KernelExplainer with small background set on light background.
    Never crashes the web app.
    """
    try:
        import shap
        
        if len(X_background) > 15:
            bg_sample = X_background.sample(15, random_state=42)
        else:
            bg_sample = X_background
            
        def predict_fn(data):
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data, columns=feature_names)
            return classifier.predict_proba(data)
            
        explainer = shap.KernelExplainer(predict_fn, bg_sample)
        shap_vals = explainer.shap_values(input_df, nsamples=30)
        
        fig, ax = plt.subplots(figsize=(7, 4.2), facecolor="#ffffff")
        ax.set_facecolor("#ffffff")
        
        if isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
            vals = shap_vals[0]
            mean_imp = np.abs(vals).mean(axis=1)
        elif isinstance(shap_vals, list):
            vals_list = [np.abs(sv[0]) for sv in shap_vals]
            mean_imp = np.mean(vals_list, axis=0)
        else:
            mean_imp = np.abs(shap_vals[0])
            
        top_idx = np.argsort(mean_imp)[-10:]
        top_features = [feature_names[i] for i in top_idx]
        top_values = mean_imp[top_idx]
        
        ax.barh(top_features, top_values, color="#4b63e6", edgecolor="#3b82f6", height=0.55)
        ax.set_xlabel("Mean |SHAP Value| (Impact on Model Output)", color="#707070", fontsize=9)
        ax.tick_params(colors="#252525", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#e8e8e8")
        ax.spines["bottom"].set_color("#e8e8e8")
        ax.grid(axis="x", linestyle="--", alpha=0.3, color="#e8e8e8")
        plt.title("Feature Contribution (Top 10 VOCs)", color="#252525", fontsize=11, fontweight="bold")
        plt.tight_layout()
        
        return fig, "SHAP local contribution computed successfully.", None
        
    except Exception as e:
        fallback_msg = f"SHAP explanation fallback triggered: {str(e)}"
        return None, fallback_msg, str(e)
