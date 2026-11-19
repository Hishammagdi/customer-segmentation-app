import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    RAW_DATA_PATH: str = os.getenv("RAW_DATA_PATH", "./data/raw/OnlineRetail.csv")
    PROCESSED_DATA_DIR: str = os.getenv("PROCESSED_DATA_DIR", "./data/processed")
    COUNTRY_FILTER: str = os.getenv("COUNTRY_FILTER", "").strip()


settings = Settings()
