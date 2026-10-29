import sqlite3
from pathlib import Path
import pandas as pd

# Step 1: Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
DB_DIR = PROJECT_ROOT / "data"
DB_PATH = DB_DIR / "instamart.db"

# Step 2: Define the cleaned CSV files
files = {
    "customers": CLEAN_DIR / "customers.csv",
    "orders": CLEAN_DIR / "orders.csv",
    "products": CLEAN_DIR / "products.csv",
}

def main():
    # Create the data folder if it does not exist
    DB_DIR.mkdir(parents=True, exist_ok=True)

    # Check that all required CSV files exist
    for table_name, csv_path in files.items():
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {csv_path}"
            )

    # Step 3: Connect to the SQLite database
    connection = sqlite3.connect(DB_PATH)

    try:
        # Step 4: Load each cleaned CSV into a database table
        for table_name, csv_path in files.items():
            df = pd.read_csv(csv_path)

            df.to_sql(
                table_name,
                connection,
                if_exists="replace",
                index=False
            )

            print(
                f"Loaded {len(df)} rows into "
                f"'{table_name}' table."
            )

        # Step 5: Verify the tables and row counts
        print("\nDATABASE VERIFICATION")
        print("=" * 40)

        tables = connection.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
        """).fetchall()

        for (table_name,) in tables:
            count = connection.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            ).fetchone()[0]

            print(f"{table_name}: {count} rows")

        print(f"\nDatabase saved at: {DB_PATH}")

    finally:
        connection.close()

if __name__ == "__main__":
    main()
