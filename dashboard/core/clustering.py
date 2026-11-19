"""
Re-runs K-Means + PCA on demand for a user-chosen k — powers the
"Explore different k" tab so the person can see the silhouette score and
resulting segments change interactively, without editing any code.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from core.config import settings
from core.preprocessing import log_transform, scale_features
from core.segment_profiles import build_segment_profiles


def cluster_with_k(rfm: pd.DataFrame, k: int):
    rfm_log = log_transform(rfm)
    X_scaled, _ = scale_features(rfm_log)

    kmeans = KMeans(n_clusters=k, random_state=settings.RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)

    pca = PCA(n_components=2, random_state=settings.RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)

    result = rfm.copy()
    result["Cluster"] = labels
    result["pca_x"] = coords[:, 0]
    result["pca_y"] = coords[:, 1]

    profiles = build_segment_profiles(result)

    return result, profiles, sil_score
