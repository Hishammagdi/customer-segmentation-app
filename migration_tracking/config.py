import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    TRANSACTIONS_PATH: str = os.getenv("TRANSACTIONS_PATH", "../data_pipeline/data/processed/clean_transactions.csv")
    MODEL_DIR: str = os.getenv("MODEL_DIR", "../model_training/models")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "./output")
    SNAPSHOT_FREQUENCY: str = os.getenv("SNAPSHOT_FREQUENCY", "ME")
    MIN_SNAPSHOTS: int = int(os.getenv("MIN_SNAPSHOTS", "2"))


settings = Settings()
