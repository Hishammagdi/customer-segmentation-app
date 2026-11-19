"""
Builds a supervised learning dataset for CLV prediction using the standard
calibration/holdout split:

- CALIBRATION period (e.g. the first 70% of the date range): transactions
  here are used to engineer each customer's FEATURES.
- HOLDOUT period (the remaining time): each customer's TOTAL SPEND in this
  period is the TARGET — what we're trying to predict.

This mirrors a real deployment: "given what we know about a customer so
far, predict how much they'll spend going forward."
"""

import pandas as pd

from config import settings


def get_cutoff_date(transactions: pd.DataFrame) -> pd.Timestamp:
    min_date = transactions["InvoiceDate"].min()
    max_date = transactions["InvoiceDate"].max()
    total_days = (max_date - min_date).days
    cutoff = min_date + pd.Timedelta(days=int(total_days * settings.CALIBRATION_FRACTION))
    return cutoff


def build_features(calibration_df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    features = calibration_df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda dates: (cutoff - dates.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
        Tenure=("InvoiceDate", lambda dates: (cutoff - dates.min()).days),
        NumUniqueProducts=("Description", "nunique"),
    ).reset_index()

    features["AvgOrderValue"] = (features["Monetary"] / features["Frequency"]).round(2)
    features["Monetary"] = features["Monetary"].round(2)

    return features


def build_target(holdout_df: pd.DataFrame, customer_ids) -> pd.DataFrame:
    spend = holdout_df.groupby("CustomerID")["TotalPrice"].sum().round(2)
    target = pd.DataFrame({"CustomerID": customer_ids})
    target = target.merge(spend.rename("FutureSpend"), on="CustomerID", how="left")
    target["FutureSpend"] = target["FutureSpend"].fillna(0.0)
    return target


def build_dataset(transactions: pd.DataFrame):
    cutoff = get_cutoff_date(transactions)
    print(f"Calibration/holdout cutoff: {cutoff.date()} "
          f"({settings.CALIBRATION_FRACTION:.0%} of the date range)")

    calibration_df = transactions[transactions["InvoiceDate"] <= cutoff]
    holdout_df = transactions[transactions["InvoiceDate"] > cutoff]
    print(f"Calibration: {len(calibration_df)} rows, {calibration_df['CustomerID'].nunique()} customers")
    print(f"Holdout:     {len(holdout_df)} rows, {holdout_df['CustomerID'].nunique()} customers")

    features = build_features(calibration_df, cutoff)
    target = build_target(holdout_df, features["CustomerID"])

    dataset = features.merge(target, on="CustomerID")
    print(f"Dataset: {len(dataset)} customers with both features and a target")
    print(f"Customers with $0 future spend (churned within holdout): "
          f"{(dataset['FutureSpend'] == 0).sum()} ({(dataset['FutureSpend'] == 0).mean():.1%})")

    return dataset, cutoff
