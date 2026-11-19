"""
Runs the Apriori algorithm to find frequent itemsets, then derives
association rules ("customers who buy A also tend to buy B") scored by
support, confidence, and lift.

- Support: how often the itemset appears across all invoices
- Confidence: P(buy B | bought A) — how reliable the rule is
- Lift: how much more likely B is when A is bought, vs. B's base rate.
  Lift > 1 means a real positive association (not just both being popular
  items); lift = 1 means no relationship; lift < 1 means A discourages B.
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from config import settings


def _format_itemset(itemset) -> str:
    return ", ".join(sorted(itemset))


def generate_rules(basket_matrix: pd.DataFrame) -> pd.DataFrame:
    print(f"Running Apriori (min_support={settings.MIN_SUPPORT})...")
    frequent_itemsets = apriori(
        basket_matrix, min_support=settings.MIN_SUPPORT, use_colnames=True
    )
    print(f"Found {len(frequent_itemsets)} frequent itemsets")

    if frequent_itemsets.empty:
        print("No frequent itemsets found — try lowering MIN_SUPPORT.")
        return pd.DataFrame()

    rules = association_rules(
        frequent_itemsets, metric="confidence", min_threshold=settings.MIN_CONFIDENCE
    )
    rules = rules[rules["lift"] >= settings.MIN_LIFT]

    if rules.empty:
        print("No rules passed the confidence/lift thresholds — try lowering MIN_CONFIDENCE/MIN_LIFT.")
        return rules

    rules["antecedents_str"] = rules["antecedents"].apply(_format_itemset)
    rules["consequents_str"] = rules["consequents"].apply(_format_itemset)
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    print(f"{len(rules)} rules found (min_confidence={settings.MIN_CONFIDENCE}, min_lift={settings.MIN_LIFT})")
    return rules


def readable_rules(rules: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    if rules.empty:
        return rules
    cols = ["antecedents_str", "consequents_str", "support", "confidence", "lift"]
    out = rules[cols].head(top_n).copy()
    out.columns = ["If bought", "Then also buys", "Support", "Confidence", "Lift"]
    out["Support"] = out["Support"].round(4)
    out["Confidence"] = out["Confidence"].round(3)
    out["Lift"] = out["Lift"].round(2)
    return out
