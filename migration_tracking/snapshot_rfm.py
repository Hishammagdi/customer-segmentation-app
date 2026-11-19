"""
Computes an RFM table as of any historical cutoff date, using only
transactions that happened on or before that date. This is what makes
migration tracking possible: the SAME RFM logic as the Data step, but
re-run "as of" several points in time instead of just once.
"""

import pandas as pd


def rfm_as_of(transactions: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    up_to_cutoff = transactions[transactions["InvoiceDate"] <= cutoff_date]

    if up_to_cutoff.empty:
        return pd.DataFrame(columns=["CustomerID", "Recency", "Frequency", "Monetary"])

    rfm = up_to_cutoff.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda dates: (cutoff_date - dates.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    rfm["Monetary"] = rfm["Monetary"].round(2)
    return rfm


def generate_snapshot_dates(transactions: pd.DataFrame, frequency: str) -> list:
    """
    Returns a list of cutoff dates spanning the transaction history, spaced
    by `frequency` (pandas offset alias, e.g. 'M' for monthly).
    The first snapshot is skipped if it would have too little history to
    be meaningful (less than one full period of data).
    """
    min_date = transactions["InvoiceDate"].min()
    max_date = transactions["InvoiceDate"].max()

    dates = pd.date_range(start=min_date, end=max_date, freq=frequency)
    dates = [d for d in dates if d > min_date]  # skip a snapshot with ~0 history

    if len(dates) == 0 or dates[-1] < max_date:
        dates.append(max_date)  # always include the most recent full snapshot

    return dates
