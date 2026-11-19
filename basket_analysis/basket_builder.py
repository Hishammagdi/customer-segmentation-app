"""
Builds a one-hot encoded basket matrix (one row per invoice, one column
per product, 1 if that product was in that invoice) — the input format
mlxtend's apriori() expects.
"""

import os

import pandas as pd

from config import settings


def build_basket_matrix(transactions: pd.DataFrame) -> pd.DataFrame:
    df = transactions.copy()
    df["Description"] = df["Description"].astype(str).str.strip()
    df = df[df["Description"] != ""]

    # Drop unusually large invoices (bulk/wholesale outliers) so they don't
    # dominate co-occurrence counts.
    basket_sizes = df.groupby("InvoiceNo")["Description"].nunique()
    valid_invoices = basket_sizes[basket_sizes <= settings.MAX_BASKET_SIZE].index
    df = df[df["InvoiceNo"].isin(valid_invoices)]

    basket = (
        df.groupby(["InvoiceNo", "Description"])["Quantity"]
        .sum()
        .unstack(fill_value=0)
    )
    basket_encoded = basket.map(lambda qty: True if qty > 0 else False)

    print(f"Basket matrix: {basket_encoded.shape[0]} invoices x {basket_encoded.shape[1]} distinct products")
    return basket_encoded


def filter_transactions_by_segment(transactions: pd.DataFrame, segment_filter: str = None) -> pd.DataFrame:
    """If SEGMENT_FILTER is set, keeps only transactions from customers in
    that segment (joins with the Model step's labeled customers)."""
    segment_filter = segment_filter or settings.SEGMENT_FILTER
    if not segment_filter:
        return transactions

    customers = pd.read_csv(settings.CUSTOMERS_WITH_CLUSTERS_PATH)
    customers["CustomerID"] = customers["CustomerID"].astype(str)

    # customers_with_clusters.csv only has the numeric 'Cluster' column —
    # segment_profiles.csv (same models/ folder) has the Cluster -> name
    # mapping, so merge that in rather than assuming segment_name already
    # exists on the customers file.
    profiles_path = os.path.join(os.path.dirname(settings.CUSTOMERS_WITH_CLUSTERS_PATH), "segment_profiles.csv")
    profiles = pd.read_csv(profiles_path)
    cluster_to_name = dict(zip(profiles["Cluster"], profiles["segment_name"]))
    customers["segment_name"] = customers["Cluster"].map(cluster_to_name)

    matching_ids = set(customers[customers["segment_name"] == segment_filter]["CustomerID"])
    if not matching_ids:
        available = sorted(customers["segment_name"].dropna().unique())
        raise ValueError(f"No customers found in segment '{segment_filter}'. Available: {available}")

    filtered = transactions[transactions["CustomerID"].astype(str).isin(matching_ids)]
    print(f"Filtered to segment '{segment_filter}': {len(matching_ids)} customers, "
          f"{len(filtered)} transaction rows")
    return filtered
