"""
Same log-transform + scaling logic as model_training/preprocessing.py,
duplicated here so the dashboard's "Explore different k" tab can
re-cluster live without depending on the model_training/ folder.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RFM_COLUMNS = ["Recency", "Frequency", "Monetary"]


def log_transform(rfm: pd.DataFrame) -> pd.DataFrame:
    transformed = rfm.copy()
    for col in RFM_COLUMNS:
        transformed[col] = np.log1p(transformed[col])
    return transformed


def scale_features(rfm_log: pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_log[RFM_COLUMNS])
    return X_scaled, scaler
