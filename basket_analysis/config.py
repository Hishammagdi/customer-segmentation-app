import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    TRANSACTIONS_PATH: str = os.getenv("TRANSACTIONS_PATH", "../data_pipeline/data/processed/clean_transactions.csv")
    CUSTOMERS_WITH_CLUSTERS_PATH: str = os.getenv(
        "CUSTOMERS_WITH_CLUSTERS_PATH", "../model_training/models/customers_with_clusters.csv"
    )
    SEGMENT_FILTER: str = os.getenv("SEGMENT_FILTER", "").strip()

    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "./output")

    MIN_SUPPORT: float = float(os.getenv("MIN_SUPPORT", "0.02"))
    MIN_CONFIDENCE: float = float(os.getenv("MIN_CONFIDENCE", "0.3"))
    MIN_LIFT: float = float(os.getenv("MIN_LIFT", "1.0"))
    MAX_BASKET_SIZE: int = int(os.getenv("MAX_BASKET_SIZE", "50"))


settings = Settings()
