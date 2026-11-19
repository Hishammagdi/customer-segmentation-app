"""
Full data preparation pipeline for the Customer Segmentation project:
raw transactions -> cleaned transactions -> per-customer RFM table.

Usage:
    python prepare_data.py
"""

import os

from config import settings
from cleaning import load_raw_data, clean_data
from rfm import compute_rfm


def main():
    print("=== Step 1/3: Load raw transactions ===")
    df = load_raw_data()

    print("\n=== Step 2/3: Clean transactions ===")
    df = clean_data(df)

    print("\n=== Step 3/3: Build RFM table ===")
    rfm = compute_rfm(df)

    os.makedirs(settings.PROCESSED_DATA_DIR, exist_ok=True)
    transactions_path = os.path.join(settings.PROCESSED_DATA_DIR, "clean_transactions.csv")
    rfm_path = os.path.join(settings.PROCESSED_DATA_DIR, "rfm_table.csv")

    df.to_csv(transactions_path, index=False)
    rfm.to_csv(rfm_path, index=False)

    print(f"\nSaved to:\n  {transactions_path}\n  {rfm_path}")
    print("\nThe RFM table is what the Model step (K-Means + PCA) will cluster.")


if __name__ == "__main__":
    main()
