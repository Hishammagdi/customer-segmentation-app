"""
Turns raw cluster numbers (0, 1, 2...) into human-readable segment names
and suggested marketing actions, based on each cluster's RFM profile
RELATIVE TO THE OTHER CLUSTERS FOUND — not fixed thresholds, since what
counts as "high value" depends on the specific customer base.

Method: for each cluster, compute a composite score from its mean Recency
(inverted — lower is better), Frequency, and Monetary, each expressed as a
percentile rank across the clusters. Clusters are then ranked by that
composite score and assigned names from best to worst.
"""

import pandas as pd

# Ordered best -> worst. If there are fewer clusters than labels, the
# middle labels are skipped so the two ends (best/worst) are always used.
SEGMENT_LABELS = [
    ("Champions", "Reward with loyalty perks and early access to new products — protect this relationship."),
    ("Loyal Customers", "Upsell and cross-sell — they already trust the brand and buy often."),
    ("Potential Loyalists", "Encourage repeat purchases with a personalized offer or subscription nudge."),
    ("Needs Attention", "Re-engagement campaign — they were active before but have slowed down."),
    ("At Risk", "Win-back offer before they're gone — highlight what's new since their last visit."),
    ("Lost Customers", "Low-cost reactivation email; deprioritize compared to active segments."),
]


def _pick_labels(n_clusters: int):
    if n_clusters >= len(SEGMENT_LABELS):
        return SEGMENT_LABELS[:n_clusters]
    if n_clusters == 1:
        return [SEGMENT_LABELS[0]]
    # Evenly sample across the ordered list so both ends are represented.
    step = (len(SEGMENT_LABELS) - 1) / (n_clusters - 1)
    indices = [round(i * step) for i in range(n_clusters)]
    return [SEGMENT_LABELS[i] for i in indices]


def build_segment_profiles(rfm_with_clusters: pd.DataFrame) -> pd.DataFrame:
    """
    rfm_with_clusters: the RFM table with a 'Cluster' column already added.
    Returns a profile table: one row per cluster with size, mean R/F/M,
    a composite rank score, an assigned segment name, and a suggested action.
    """
    summary = rfm_with_clusters.groupby("Cluster").agg(
        size=("CustomerID", "count"),
        avg_recency=("Recency", "mean"),
        avg_frequency=("Frequency", "mean"),
        avg_monetary=("Monetary", "mean"),
    ).reset_index()

    # Percentile rank each metric across clusters (0=worst, 1=best).
    summary["recency_rank"] = summary["avg_recency"].rank(ascending=False, pct=True)  # lower recency = better
    summary["frequency_rank"] = summary["avg_frequency"].rank(ascending=True, pct=True)
    summary["monetary_rank"] = summary["avg_monetary"].rank(ascending=True, pct=True)
    summary["composite_score"] = (
        summary["recency_rank"] + summary["frequency_rank"] + summary["monetary_rank"]
    ) / 3

    summary = summary.sort_values("composite_score", ascending=False).reset_index(drop=True)

    labels = _pick_labels(len(summary))
    summary["segment_name"] = [lbl[0] for lbl in labels]
    summary["suggested_action"] = [lbl[1] for lbl in labels]

    summary = summary.round({
        "avg_recency": 1, "avg_frequency": 1, "avg_monetary": 2,
        "recency_rank": 2, "frequency_rank": 2, "monetary_rank": 2, "composite_score": 2,
    })

    return summary[[
        "Cluster", "segment_name", "size", "avg_recency", "avg_frequency",
        "avg_monetary", "composite_score", "suggested_action",
    ]]
