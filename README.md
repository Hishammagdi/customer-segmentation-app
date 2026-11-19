# Customer Segmentation & Marketing Intelligence Platform

An end-to-end customer analytics platform that turns transaction history into actionable marketing segments. The project combines RFM analysis, K-Means clustering, PCA visualization, an interactive Streamlit dashboard, customer lifetime value prediction, product affinity analysis, and segment migration tracking.

## Business Problem

Treating every customer the same leads to inefficient campaigns and missed retention opportunities. This project identifies meaningful customer groups from purchasing behavior so teams can prioritize retention, loyalty, reactivation, and cross-sell actions.

## What the Platform Does

- Cleans transaction-level retail data and removes invalid or cancelled orders.
- Builds customer-level **RFM features**: Recency, Frequency, and Monetary value.
- Selects a suitable number of clusters using silhouette score.
- Trains K-Means on log-transformed and standardized RFM features.
- Uses PCA only for two-dimensional visualization.
- Assigns adaptive business labels such as Champions, Loyal Customers, Needs Attention, and At Risk.
- Provides a Streamlit dashboard for segment exploration and customer lookup.
- Predicts future customer spend with a time-based calibration/holdout design.
- Finds products frequently purchased together using Apriori association rules.
- Tracks customer movement between segments over historical snapshots without retraining the model.

## Architecture

```text
Online Retail Transactions
            |
            v
Data Cleaning + RFM Feature Engineering
            |
            v
Log Transform + Standardization
            |
            v
K-Means Training + Silhouette Selection
            |
      +-----+------------------+
      |                        |
      v                        v
Streamlit Dashboard       Saved Models
      |                 +--------+---------+
      |                 |                  |
      v                 v                  v
Customer Lookup   CLV Prediction    Migration Tracking
                           |
                           v
                    Basket Analysis
```

## Repository Structure

```text
.
├── data_pipeline/        # Clean transactions and build the RFM table
├── model_training/       # K-Means clustering, PCA, model selection, profiles
├── dashboard/            # Streamlit segment exploration and customer lookup
├── clv_prediction/       # Future-spend regression with temporal holdout
├── basket_analysis/      # Apriori product affinity and association rules
├── migration_tracking/   # Historical segment transitions using the same model
├── .env.example files    # Configuration templates for each module
└── README.md
```

The raw Online Retail dataset and generated transaction tables are intentionally excluded from the repository. Download the dataset separately and place it in the configured data path.

## Quick Start

### 1. Prepare the data

```bash
cd data_pipeline
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
python prepare_data.py
```

The project uses the UCI/Kaggle Online Retail dataset. Download it from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail) and configure `RAW_DATA_PATH` in `.env`.

### 2. Train the segmentation model

```bash
cd ../model_training
pip install -r requirements.txt
cp .env.example .env
python train.py
```

To force a specific number of clusters:

```bash
python train.py --k 4
```

The training step produces reusable artifacts in `model_training/models/`, including the scaler, K-Means model, PCA transformer, customer assignments, and segment profiles.

### 3. Launch the dashboard

```bash
cd ../dashboard
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

The dashboard provides three views:

- **Segment Overview:** PCA scatter plot, segment sizes, RFM summaries, and recommended actions.
- **Explore Different k:** live experimentation with alternative cluster counts and silhouette scores.
- **Customer Lookup:** inspect a customer's segment, RFM values, and suggested marketing action.

## Optional Modules

### Customer Lifetime Value Prediction

```bash
cd clv_prediction
pip install -r requirements.txt
cp .env.example .env
python train_clv.py
```

The module builds features from an earlier calibration period and predicts spend in a later holdout period. It compares Random Forest and XGBoost using MAE, RMSE, and R² before selecting the stronger model.

### Basket Analysis

```bash
cd basket_analysis
pip install -r requirements.txt
cp .env.example .env
python run_basket_analysis.py
python run_basket_analysis.py --segment Champions
```

The output ranks product relationships using support, confidence, and lift.

### Segment Migration Tracking

```bash
cd migration_tracking
pip install -r requirements.txt
cp .env.example .env
python track_migration.py
```

This module intentionally reuses the saved scaler and K-Means model. It transforms historical snapshots with the same model instead of retraining per snapshot, making segment transitions comparable over time.

## Modeling Notes

- RFM is calculated at customer level from cleaned transactions.
- Log transformation reduces the effect of heavy-tailed spending and order-count distributions.
- Standardization prevents one RFM feature from dominating distance calculations.
- Silhouette score provides an objective comparison across candidate cluster counts.
- PCA is used for visualization only; clustering uses the full transformed RFM feature space.
- Segment labels are assigned by relative cluster ranking rather than fixed hard-coded thresholds.
- CLV evaluation uses a temporal split to avoid using future information during training.

## Reproducibility and Data Policy

This repository does not include the original retail dataset, generated transaction tables, local environment files, Python caches, or generated analysis exports. These files can be recreated by running the pipeline locally. Reusable trained model artifacts are included where they are small enough to support the dashboard demonstration.

Never commit API keys, passwords, private customer data, or local `.env` files.

## Future Improvements

- Add a FastAPI serving layer for real-time segment prediction.
- Add automated model monitoring and drift detection.
- Add a campaign recommendation layer based on segment and predicted CLV.
- Add tests and CI checks for data validation and model reproducibility.
- Deploy the Streamlit dashboard with secrets configured through the hosting platform.

## License

This project is provided for portfolio and educational purposes. Add an open-source license before distributing it as a public package.
