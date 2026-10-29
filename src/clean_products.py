
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "products.csv"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUT_FILE = CLEAN_DIR / "products.csv"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Load the original products
df = pd.read_csv(RAW_FILE)

print("Original product rows:", len(df))

# Remove exact duplicate rows
df = df.drop_duplicates()

# Standardize text fields
for column in ["product_id", "product_name", "category"]:
    df[column] = df[column].astype("string").str.strip()

# Convert prices to numeric values
df["price_inr"] = pd.to_numeric(df["price_inr"], errors="coerce")

# Check for invalid or missing required values
required_columns = ["product_id", "product_name", "category", "price_inr"]
df = df.dropna(subset=required_columns)

# Product IDs should uniquely identify products
df = df.drop_duplicates(subset="product_id", keep="last")

# Save the cleaned products
df.to_csv(OUTPUT_FILE, index=False)

print("Cleaned product rows:", len(df))
print("\nProduct counts by category:")
print(df["category"].value_counts().sort_index())
print("\nMissing values:")
print(df.isna().sum())
print("\nSaved cleaned products to:", OUTPUT_FILE)
