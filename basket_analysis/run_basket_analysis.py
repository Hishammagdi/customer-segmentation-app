"""
Full product affinity / basket analysis pipeline:
transactions -> (optional segment filter) -> basket matrix -> Apriori ->
association rules -> saved + printed.

Usage:
    python run_basket_analysis.py
    python run_basket_analysis.py --segment Champions
"""

import argparse
import os

import pandas as pd

from config import settings
from basket_builder import build_basket_matrix, filter_transactions_by_segment
from rules import generate_rules, readable_rules


def parse_args():
    parser = argparse.ArgumentParser(description="Product affinity / basket analysis")
    parser.add_argument("--segment", default=None,
                         help="Restrict analysis to one customer segment (e.g. 'Champions')")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=== Step 1/3: Load transactions ===")
    transactions = pd.read_csv(settings.TRANSACTIONS_PATH)
    print(f"Loaded {len(transactions)} transaction rows")

    transactions = filter_transactions_by_segment(transactions, args.segment)

    print("\n=== Step 2/3: Build basket matrix ===")
    basket_matrix = build_basket_matrix(transactions)

    print("\n=== Step 3/3: Mine association rules ===")
    rules = generate_rules(basket_matrix)

    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    suffix = f"_{args.segment}" if args.segment else ""
    rules_path = os.path.join(settings.OUTPUT_DIR, f"association_rules{suffix}.csv")

    if not rules.empty:
        rules.drop(columns=["antecedents", "consequents"]).to_csv(rules_path, index=False)
        print(f"\nSaved {len(rules)} rules to: {rules_path}")

        print("\n=== Top rules by lift ===")
        print(readable_rules(rules).to_string(index=False))
    else:
        print("\nNo rules to save — try adjusting the thresholds in .env")


if __name__ == "__main__":
    main()
