"""
Tracks how customers move between segments over time.

Crucially, this reuses the SAME trained scaler + K-Means model from the
Model step (never retrains) — so "Champions" means the same thing at every
snapshot date. Only the RFM inputs change per snapshot; the segment
definitions stay fixed, which is what makes a before/after comparison
meaningful.

Usage:
    python track_migration.py
"""

import os

import joblib
import numpy as np
import pandas as pd

from config import settings
from snapshot_rfm import rfm_as_of, generate_snapshot_dates


def load_model_and_segment_map():
    scaler = joblib.load(os.path.join(settings.MODEL_DIR, "scaler.pkl"))
    kmeans = joblib.load(os.path.join(settings.MODEL_DIR, "kmeans.pkl"))
    profiles = pd.read_csv(os.path.join(settings.MODEL_DIR, "segment_profiles.csv"))

    segment_name_by_cluster = dict(zip(profiles["Cluster"], profiles["segment_name"]))
    # Higher composite_score = healthier segment — used to detect downgrades.
    score_by_segment = dict(zip(profiles["segment_name"], profiles["composite_score"]))

    return scaler, kmeans, segment_name_by_cluster, score_by_segment


def assign_segment(rfm: pd.DataFrame, scaler, kmeans, segment_name_by_cluster) -> pd.DataFrame:
    rfm = rfm.copy()
    rfm_log = rfm[["Recency", "Frequency", "Monetary"]].apply(np.log1p)
    X_scaled = scaler.transform(rfm_log)  # transform only — never re-fit
    rfm["Cluster"] = kmeans.predict(X_scaled)
    rfm["segment_name"] = rfm["Cluster"].map(segment_name_by_cluster)
    return rfm


def build_segment_history(transactions: pd.DataFrame, scaler, kmeans, segment_name_by_cluster) -> pd.DataFrame:
    snapshot_dates = generate_snapshot_dates(transactions, settings.SNAPSHOT_FREQUENCY)
    print(f"Snapshots ({settings.SNAPSHOT_FREQUENCY}): {[d.date() for d in snapshot_dates]}")

    if len(snapshot_dates) < settings.MIN_SNAPSHOTS:
        raise ValueError(
            f"Only {len(snapshot_dates)} snapshot(s) available — need at least "
            f"{settings.MIN_SNAPSHOTS} to track migration. Try a finer "
            f"SNAPSHOT_FREQUENCY (e.g. 'W' instead of 'M')."
        )

    history_frames = []
    for snapshot_date in snapshot_dates:
        rfm = rfm_as_of(transactions, snapshot_date)
        if rfm.empty:
            continue
        rfm = assign_segment(rfm, scaler, kmeans, segment_name_by_cluster)
        rfm["snapshot_date"] = snapshot_date.date()
        history_frames.append(rfm)
        print(f"  {snapshot_date.date()}: {len(rfm)} customers with history so far")

    return pd.concat(history_frames, ignore_index=True)


def build_transition_matrix(history: pd.DataFrame, from_date, to_date) -> pd.DataFrame:
    from_seg = history[history["snapshot_date"] == from_date].set_index("CustomerID")["segment_name"]
    to_seg = history[history["snapshot_date"] == to_date].set_index("CustomerID")["segment_name"]

    # Only customers present at both snapshots can "transition".
    joined = pd.DataFrame({"from_segment": from_seg, "to_segment": to_seg}).dropna()

    matrix = pd.crosstab(joined["from_segment"], joined["to_segment"])
    return matrix, joined


def find_downgraded_customers(joined_transitions: pd.DataFrame, score_by_segment: dict) -> pd.DataFrame:
    joined = joined_transitions.copy()
    joined["from_score"] = joined["from_segment"].map(score_by_segment)
    joined["to_score"] = joined["to_segment"].map(score_by_segment)
    joined["score_change"] = joined["to_score"] - joined["from_score"]

    downgraded = joined[joined["score_change"] < 0].sort_values("score_change")
    return downgraded.reset_index()


def main():
    print("=== Step 1/4: Load transactions + trained model ===")
    transactions = pd.read_csv(settings.TRANSACTIONS_PATH)
    transactions["InvoiceDate"] = pd.to_datetime(transactions["InvoiceDate"])
    transactions["CustomerID"] = transactions["CustomerID"].astype(str)

    scaler, kmeans, segment_name_by_cluster, score_by_segment = load_model_and_segment_map()
    print(f"Segment map: {segment_name_by_cluster}")

    print("\n=== Step 2/4: Building segment history across snapshots ===")
    history = build_segment_history(transactions, scaler, kmeans, segment_name_by_cluster)

    snapshot_dates = sorted(history["snapshot_date"].unique())
    first_date, last_date = snapshot_dates[0], snapshot_dates[-1]

    print(f"\n=== Step 3/4: Transition matrix ({first_date} -> {last_date}) ===")
    matrix, joined = build_transition_matrix(history, first_date, last_date)
    print(matrix)

    print("\n=== Step 4/4: Flagging downgraded customers ===")
    downgraded = find_downgraded_customers(joined, score_by_segment)
    print(f"{len(downgraded)} of {len(joined)} tracked customers downgraded segments "
          f"between {first_date} and {last_date}")
    if not downgraded.empty:
        print(downgraded.head(10).to_string(index=False))

    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    history_path = os.path.join(settings.OUTPUT_DIR, "segment_history.csv")
    matrix_path = os.path.join(settings.OUTPUT_DIR, "transition_matrix.csv")
    downgraded_path = os.path.join(settings.OUTPUT_DIR, "downgraded_customers.csv")

    history.to_csv(history_path, index=False)
    matrix.to_csv(matrix_path)
    downgraded.to_csv(downgraded_path, index=False)

    print(f"\nSaved:\n  {history_path}\n  {matrix_path}\n  {downgraded_path}")


if __name__ == "__main__":
    main()
