"""Project-wide constants: paths, random seed and column roles."""
from pathlib import Path

# This file lives in <root>/src/3_Feature_selection/, so the project root
# (the folder that contains `src/`, `data/`, `notebooks/` and `logs/`) is 2 levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Transactions merged with the customers, before any preparation (pre-processing, section 1.1)
MERGED_FILE = DATA_DIR / "preprocessing" / "merge_1_2.csv"
TEST_FILE = DATA_DIR / "test.csv"

RANDOM_STATE = 42

target = "repeat_purchase_90d"
identifiers = ["ID", "customer_id"]

# Unique per customer (they name a customer instead of describing one) or pure noise
excluded = ["company_name", "address", "street", "postal_code", "random_noise"]