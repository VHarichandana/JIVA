import os
import pickle
import pandas as pd
from pathlib import Path
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

@st.cache_resource
def load_classifier():
    path = MODELS_DIR / "best_classifier.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_regressor():
    path = MODELS_DIR / "best_regressor.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_label_encoder():
    path = MODELS_DIR / "label_encoder.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_preprocessing_pipeline():
    path = MODELS_DIR / "preprocessing_pipeline.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_selected_features():
    path = MODELS_DIR / "selected_features.pkl"
    if not path.exists():
        return []
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_classifier_metadata():
    path = MODELS_DIR / "classifier_metadata.pkl"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_regressor_metadata():
    path = MODELS_DIR / "regressor_metadata.pkl"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_dataset():
    path = DATA_DIR / "LungCancer.txt"
    if path.exists():
        try:
            df = pd.read_csv(path, sep="\t")
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception:
            pass
    csv_path = DATA_DIR / "lung_cancer.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    return None

@st.cache_data
def load_classification_metrics():
    path = OUTPUTS_DIR / "classification" / "classification_model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_regression_metrics():
    path = OUTPUTS_DIR / "regression" / "regression_model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None
