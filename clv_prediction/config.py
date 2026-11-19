import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    TRANSACTIONS_PATH: str = os.getenv("TRANSACTIONS_PATH", "../data_pipeline/data/processed/clean_transactions.csv")
    MODEL_OUTPUT_DIR: str = os.getenv("MODEL_OUTPUT_DIR", "./models")
    CALIBRATION_FRACTION: float = float(os.getenv("CALIBRATION_FRACTION", "0.7"))
    RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))


settings = Settings()
