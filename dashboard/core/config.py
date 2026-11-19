import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    MODEL_DIR: str = os.getenv("MODEL_DIR", "../model_training/models")
    RFM_PATH: str = os.getenv("RFM_PATH", "../data_pipeline/data/processed/rfm_table.csv")
    MIN_K: int = int(os.getenv("MIN_K", "2"))
    MAX_K: int = int(os.getenv("MAX_K", "8"))
    RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))


settings = Settings()
