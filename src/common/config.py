import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "./data")
DB_URL = os.getenv("DB_URL", "sqlite:///./db/ncai.sqlite")
NCBC_BASE = os.getenv("NCBC_BASE", "https://www.nccourts.gov/documents/business-court-opinions")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
RATE_SLEEP_SEC = float(os.getenv("RATE_SLEEP_SEC", "1.2"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "73"))

RAW_DIR = os.path.join(DATA_DIR, "raw", "ncbc")
BRONZE_DIR = os.path.join(DATA_DIR, "bronze", "ncbc_jsonl")
SILVER_PATH = os.path.join(DATA_DIR, "silver", "ncbc.parquet")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(BRONZE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_URL.replace("sqlite:///","")), exist_ok=True)
os.makedirs(os.path.dirname(SILVER_PATH), exist_ok=True)
