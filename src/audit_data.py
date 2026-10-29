
from pathlib import Path
import pandas as pd

# Project root is the parent folder of src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# CSV files to inspect
files_to_audit = [
    "customers.csv",
    "products.csv",
    "orders.csv",
    "tickets.csv",
    "policy_eval.csv",
    "agent_scenarios.csv",
]

for filename in files_to_audit:
    file_path = RAW_DIR / filename

    print("\n" + "=" * 60)
    print(f"DATA AUDIT: {filename}")
    print("=" * 60)

    # Stop with a clear message if a file is missing
    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        continue

    # Load the CSV into a Pandas DataFrame
    df = pd.read_csv(file_path)

    # Display basic information about the dataset
    print("\nShape (rows, columns):")
    print(df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nExact duplicate rows:")
    print(df.duplicated().sum())

print("\nDATA AUDIT COMPLETED")
