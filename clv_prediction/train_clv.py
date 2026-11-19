"""
Trains and compares Random Forest and XGBoost regressors to predict a
customer's future spend (CLV), using a calibration/holdout time split so
evaluation reflects a realistic "predict what happens next" setup rather
than random leakage across time.

Usage:
    python train_clv.py
"""

import os
import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import settings
from feature_target_builder import build_dataset, build_features, get_cutoff_date

FEATURE_COLUMNS = ["Recency", "Frequency", "Monetary", "Tenure", "NumUniqueProducts", "AvgOrderValue"]
TARGET_COLUMN = "FutureSpend"


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_pred = y_pred.clip(min=0)  # spend can't be negative
    return {
        "mae": round(mean_absolute_error(y_test, y_pred), 2),
        "rmse": round(mean_squared_error(y_test, y_pred) ** 0.5, 2),
        "r2": round(r2_score(y_test, y_pred), 4),
    }


def main():
    print("=== Step 1/4: Load transactions + build calibration/holdout dataset ===")
    transactions = pd.read_csv(settings.TRANSACTIONS_PATH)
    transactions["InvoiceDate"] = pd.to_datetime(transactions["InvoiceDate"])
    transactions["CustomerID"] = transactions["CustomerID"].astype(str)

    dataset, cutoff = build_dataset(transactions)

    X = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]

    print("\n=== Step 2/4: Train/test split ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=settings.RANDOM_STATE
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    print("\n=== Step 3/4: Train + compare models ===")
    candidates = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=8, random_state=settings.RANDOM_STATE
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05, random_state=settings.RANDOM_STATE
        ),
    }

    results = {}
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        results[name] = metrics
        fitted[name] = model
        print(f"{name}: MAE=${metrics['mae']}, RMSE=${metrics['rmse']}, R2={metrics['r2']}")

    best_name = min(results, key=lambda n: results[n]["rmse"])
    best_model = fitted[best_name]
    print(f"\nBest model (lowest RMSE): {best_name}")

    print("\n=== Step 4/4: Predicting CLV for ALL current customers ===")
    # Re-featurize using the FULL transaction history (cutoff = most recent
    # date in the data) — this is the practical, deployable output: "how
    # much is each customer likely to spend going forward, starting today."
    full_cutoff = transactions["InvoiceDate"].max()
    current_features = build_features(transactions, full_cutoff)
    current_features["PredictedCLV"] = best_model.predict(
        current_features[FEATURE_COLUMNS]
    ).clip(min=0).round(2)

    os.makedirs(settings.MODEL_OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(settings.MODEL_OUTPUT_DIR, "clv_model.pkl")
    predictions_path = os.path.join(settings.MODEL_OUTPUT_DIR, "clv_predictions.csv")
    report_path = os.path.join(settings.MODEL_OUTPUT_DIR, "clv_metrics_report.json")

    joblib.dump({"model": best_model, "feature_columns": FEATURE_COLUMNS}, model_path)
    current_features.sort_values("PredictedCLV", ascending=False).to_csv(predictions_path, index=False)
    with open(report_path, "w") as f:
        json.dump({"best_model": best_name, "all_results": results,
                   "calibration_cutoff": str(cutoff.date())}, f, indent=2)

    print(f"\nSaved:\n  {model_path}\n  {predictions_path}\n  {report_path}")
    print(f"\nTop 5 predicted CLV customers:")
    print(current_features.sort_values("PredictedCLV", ascending=False)
          [["CustomerID", "PredictedCLV"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
