
from pathlib import Path
import pandas as pd

# Find the project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "customers.csv"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUT_FILE = CLEAN_DIR / "customers.csv"

# Create the clean folder if it does not exist
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Load the original customer data
df = pd.read_csv(RAW_FILE)

print("Original customer rows:", len(df))

# Standardize city names by trimming whitespace and using title case
df["city"] = df["city"].astype("string").str.strip().str.title()

# Standardize gender values
gender_map = {
    "M": "male",
    "MALE": "male",
    "F": "female",
    "FEMALE": "female",
}

df["gender"] = (
    df["gender"]
    .astype("string")
    .str.strip()
    .str.upper()
    .map(gender_map)
    .fillna("unknown")
)

# Convert ages to numeric; invalid values become missing
df["age"] = pd.to_numeric(df["age"], errors="coerce")

# Treat ages outside the allowed 10–100 range as invalid
df.loc[~df["age"].between(10, 100), "age"] = pd.NA

# Fill missing ages with the median age for the same city
city_medians = df.groupby("city")["age"].transform("median")
df["age"] = df["age"].fillna(city_medians)

# If any age is still missing, use the overall median as a fallback
df["age"] = df["age"].fillna(df["age"].median())

# Convert signup dates to datetime
df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")

# Save the cleaned data
df.to_csv(OUTPUT_FILE, index=False, date_format="%Y-%m-%d")

print("\nCleaned customer rows:", len(df))
print("\nCity counts:")
print(df["city"].value_counts().sort_index())

print("\nGender counts:")
print(df["gender"].value_counts())

print("\nMissing ages after cleaning:", df["age"].isna().sum())
print("Missing genders after cleaning:", df["gender"].isna().sum())
print("Ages outside 10–100:", (~df["age"].between(10, 100)).sum())
print("\nSaved cleaned customers to:", OUTPUT_FILE)
