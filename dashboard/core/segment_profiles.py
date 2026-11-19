"""
Same segment-naming logic as model_training/segment_profiles.py,
duplicated here so the dashboard is self-contained.
"""

import pandas as pd

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
    step = (len(SEGMENT_LABELS) - 1) / (n_clusters - 1)
    indices = [round(i * step) for i in range(n_clusters)]
    return [SEGMENT_LABELS[i] for i in indices]


def build_segment_profiles(rfm_with_clusters: pd.DataFrame, id_col: str = "CustomerID") -> pd.DataFrame:
    summary = rfm_with_clusters.groupby("Cluster").agg(
        size=(id_col, "count"),
        avg_recency=("Recency", "mean"),
        avg_frequency=("Frequency", "mean"),
        avg_monetary=("Monetary", "mean"),
    ).reset_index()

    summary["recency_rank"] = summary["avg_recency"].rank(ascending=False, pct=True)
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
        "avg_recency": 1, "avg_frequency": 1, "avg_monetary": 2, "composite_score": 2,
    })

    return summary[[
        "Cluster", "segment_name", "size", "avg_recency", "avg_frequency",
        "avg_monetary", "composite_score", "suggested_action",
    ]]
