"""
Full clustering pipeline for Customer Segmentation:
RFM table -> log+scale -> pick k via silhouette -> fit K-Means ->
PCA (for 2D visualization) -> segment profiles with suggested actions.

Usage:
    python train.py
    python train.py --k 4   # skip auto-selection, force a specific k
"""

import os
import argparse

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from config import settings
from preprocessing import log_transform, scale_features, RFM_COLUMNS
from cluster_selection import evaluate_k_range, select_best_k
from segment_profiles import build_segment_profiles


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster customers by RFM")
    parser.add_argument("--k", type=int, default=None, help="Force a specific number of clusters")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=== Step 1/5: Load RFM table ===")
    rfm = pd.read_csv(settings.RFM_PATH)
    print(f"Loaded {len(rfm)} customers")

    print("\n=== Step 2/5: Log-transform + scale ===")
    rfm_log = log_transform(rfm)
    X_scaled, scaler = scale_features(rfm_log)

    if args.k:
        best_k = args.k
        print(f"\n=== Step 3/5: Using forced k={best_k} (skipping search) ===")
    else:
        print("\n=== Step 3/5: Selecting k via silhouette score ===")
        evaluation_df = evaluate_k_range(X_scaled)
        best_k = select_best_k(evaluation_df)
        print(f"\nBest k by silhouette score: {best_k}")

    print(f"\n=== Step 4/5: Fitting final K-Means (k={best_k}) ===")
    kmeans = KMeans(n_clusters=best_k, random_state=settings.RANDOM_STATE, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=settings.RANDOM_STATE)
    pca_coords = pca.fit_transform(X_scaled)
    rfm["pca_x"] = pca_coords[:, 0]
    rfm["pca_y"] = pca_coords[:, 1]
    print(f"PCA explained variance (2 components): {pca.explained_variance_ratio_.sum():.1%}")

    print("\n=== Step 5/5: Building segment profiles ===")
    profiles = build_segment_profiles(rfm)
    print(profiles.to_string(index=False))

    # Merge segment_name into the per-customer table too — not just the
    # profiles summary — so any downstream consumer of
    # customers_with_clusters.csv can filter/group by segment_name
    # directly, without needing to separately join segment_profiles.csv.
    cluster_to_name = dict(zip(profiles["Cluster"], profiles["segment_name"]))
    rfm["segment_name"] = rfm["Cluster"].map(cluster_to_name)

    os.makedirs(settings.MODEL_OUTPUT_DIR, exist_ok=True)

    joblib.dump(scaler, os.path.join(settings.MODEL_OUTPUT_DIR, "scaler.pkl"))
    joblib.dump(kmeans, os.path.join(settings.MODEL_OUTPUT_DIR, "kmeans.pkl"))
    joblib.dump(pca, os.path.join(settings.MODEL_OUTPUT_DIR, "pca.pkl"))

    labeled_path = os.path.join(settings.MODEL_OUTPUT_DIR, "customers_with_clusters.csv")
    profiles_path = os.path.join(settings.MODEL_OUTPUT_DIR, "segment_profiles.csv")
    rfm.to_csv(labeled_path, index=False)
    profiles.to_csv(profiles_path, index=False)

    print(f"\nSaved model artifacts to: {settings.MODEL_OUTPUT_DIR}/")
    print(f"  scaler.pkl, kmeans.pkl, pca.pkl")
    print(f"  {labeled_path}")
    print(f"  {profiles_path}")


if __name__ == "__main__":
    main()
