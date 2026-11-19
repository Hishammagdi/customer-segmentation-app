"""
Picks the number of clusters (k) using two complementary checks:
- Silhouette score: how well-separated and internally cohesive the
  clusters are (higher = better, ranges -1 to 1). Used as the primary
  criterion for picking k.
- Inertia (within-cluster sum of squares): used for the classic "elbow"
  plot as a visual cross-check, not the deciding metric on its own since
  inertia always decreases as k increases.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import settings


def evaluate_k_range(X_scaled, min_k: int = None, max_k: int = None) -> pd.DataFrame:
    min_k = min_k or settings.MIN_K
    max_k = max_k or settings.MAX_K

    rows = []
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=settings.RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        sil_score = silhouette_score(X_scaled, labels)

        rows.append({"k": k, "silhouette_score": round(sil_score, 4), "inertia": round(kmeans.inertia_, 2)})
        print(f"k={k}: silhouette={sil_score:.4f}, inertia={kmeans.inertia_:.1f}")

    return pd.DataFrame(rows)


def select_best_k(evaluation_df: pd.DataFrame) -> int:
    best_row = evaluation_df.loc[evaluation_df["silhouette_score"].idxmax()]
    return int(best_row["k"])
