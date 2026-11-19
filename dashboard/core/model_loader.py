import os

import joblib
import pandas as pd
import streamlit as st

from core.config import settings


@st.cache_resource(show_spinner=False)
def load_default_model():
    """Loads the scaler/kmeans/pca saved by model_training/train.py."""
    scaler = joblib.load(os.path.join(settings.MODEL_DIR, "scaler.pkl"))
    kmeans = joblib.load(os.path.join(settings.MODEL_DIR, "kmeans.pkl"))
    pca = joblib.load(os.path.join(settings.MODEL_DIR, "pca.pkl"))
    return scaler, kmeans, pca


@st.cache_data(show_spinner=False)
def load_labeled_customers() -> pd.DataFrame:
    return pd.read_csv(os.path.join(settings.MODEL_DIR, "customers_with_clusters.csv"))


@st.cache_data(show_spinner=False)
def load_segment_profiles() -> pd.DataFrame:
    return pd.read_csv(os.path.join(settings.MODEL_DIR, "segment_profiles.csv"))


@st.cache_data(show_spinner=False)
def load_raw_rfm() -> pd.DataFrame:
    """Used by the 'Explore different k' tab to re-cluster live."""
    return pd.read_csv(settings.RFM_PATH)
