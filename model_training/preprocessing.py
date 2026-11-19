"""
RFM features are almost always heavily right-skewed (a few customers spend
way more / order way more often than the rest), and K-Means is
distance-based, so it's sensitive to that skew and to differing scales.

Fix: log-transform (log1p, safe for zeros) each feature to pull in the long
tail, then standard-scale so Recency/Frequency/Monetary contribute equally
to the distance calculation instead of Monetary (which has the largest raw
range) dominating it.
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
