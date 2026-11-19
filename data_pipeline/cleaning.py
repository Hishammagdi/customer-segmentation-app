"""
Cleans the raw Online Retail transactions dataset (UCI / Kaggle).

Known quirks this handles:
- InvoiceNo starting with 'C' means a CANCELLED order — these have to be
  removed, or "cancel" transactions with negative quantities would corrupt
  the RFM math.
- A meaningful share of rows have a missing CustomerID (guest/anonymous
  checkouts) — these can't be attributed to any customer, so they're dropped.
- Some rows have Quantity <= 0 or UnitPrice <= 0 (data entry errors,
  adjustments) — not real purchases, so they're removed.
- The raw file is often saved with Latin-1 encoding, not UTF-8.
"""

import os
import pandas as pd

from config import settings


def load_raw_data(path: str = None) -> pd.DataFrame:
    path = path or settings.RAW_DATA_PATH
    ext = os.path.splitext(path)[1].lower()

    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin1")

    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    before = len(df)

    # Drop rows with no CustomerID — can't attribute these to a customer.
    df = df.dropna(subset=["CustomerID"])

    # Remove cancelled orders (InvoiceNo starting with 'C').
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Remove non-positive quantity/price rows (returns, adjustments, errors).
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    if settings.COUNTRY_FILTER:
        df = df[df["Country"].str.lower() == settings.COUNTRY_FILTER.lower()]

    df = df.drop_duplicates()

    print(f"Cleaned data: {len(df)} rows (removed {before - len(df)} rows: "
          f"missing CustomerID, cancellations, invalid quantity/price)")
    print(f"Date range: {df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}")
    print(f"Unique customers: {df['CustomerID'].nunique()}")

    return df
