"""
Builds one row per customer with Recency, Frequency, and Monetary features
— the standard RFM framework used for customer segmentation.

- Recency:  days since the customer's last purchase (lower = more engaged)
- Frequency: number of distinct orders placed
- Monetary: total amount spent
"""

import pandas as pd


def compute_rfm(transactions: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = transactions["InvoiceDate"].max() + pd.Timedelta(days=1)
    print(f"Snapshot date (for Recency): {snapshot_date.date()}")

    rfm = transactions.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda dates: (snapshot_date - dates.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    rfm["Monetary"] = rfm["Monetary"].round(2)

    print(f"RFM table built for {len(rfm)} customers")
    print(rfm[["Recency", "Frequency", "Monetary"]].describe().round(1))

    return rfm
