import sqlite3
from pathlib import Path
import pandas as pd

# Step 1: Find the project folder and database
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "instamart.db"
OUTPUT_PATH = PROJECT_ROOT / "data" / "clean" / "customer_segments.csv"


def main():
    # Step 2: Connect to the SQLite database
    connection = sqlite3.connect(DB_PATH)

    try:
        # Step 3: Get customer details and completed-order behavior
        query = """
            SELECT
                c.customer_id,
                c.city,
                c.age,
                COUNT(o.order_id) AS frequency,
                COALESCE(SUM(o.order_amount), 0) AS monetary,
                MAX(o.order_ts) AS last_order
            FROM customers AS c
            LEFT JOIN orders AS o
                ON c.customer_id = o.customer_id
                AND LOWER(o.status) = 'completed'
            GROUP BY c.customer_id, c.city, c.age
        """

        df = pd.read_sql_query(query, connection)

    finally:
        connection.close()

    # Step 4: Convert last-order dates into datetime values
    df["last_order"] = pd.to_datetime(
        df["last_order"], errors="coerce"
    )

    # Step 5: Calculate recency relative to the latest completed
    # order in the dataset, not today's date
    latest_date = df["last_order"].max()

    if pd.isna(latest_date):
        df["recency_days"] = pd.NA
    else:
        df["recency_days"] = (
            latest_date - df["last_order"]
        ).dt.days

    # Step 6: Create customer segments
    # Customers with no completed orders are inactive
    df["segment"] = "Regular"

    no_orders = df["frequency"] == 0
    df.loc[no_orders, "segment"] = "No Completed Orders"

    active = df.loc[~no_orders]

    if not active.empty:
        # Use dataset-based thresholds
        frequency_median = active["frequency"].median()
        monetary_75 = active["monetary"].quantile(0.75)
        recency_75 = active["recency_days"].quantile(0.75)

        # Customers with recent, high spending are high value
        high_value = (
            (df["frequency"] >= frequency_median)
            & (df["monetary"] >= monetary_75)
            & (~no_orders)
        )
        df.loc[high_value, "segment"] = "High Value"

        # Customers whose last order is relatively old are at risk
        at_risk = (
            (df["recency_days"] >= recency_75)
            & (~no_orders)
        )
        df.loc[at_risk, "segment"] = "At Risk"

    # Step 7: Save customer segmentation results
    df.to_csv(OUTPUT_PATH, index=False)

    # Step 8: Display the results
    print("=" * 55)
    print("INSTAMART CUSTOMER INTELLIGENCE REPORT")
    print("=" * 55)

    print(f"\nTotal customers analyzed: {len(df)}")

    print("\nCustomer segments:")
    print(df["segment"].value_counts().to_string())

    print("\nAverage customer metrics by segment:")
    summary = df.groupby("segment").agg(
        customers=("customer_id", "count"),
        avg_completed_orders=("frequency", "mean"),
        avg_spend=("monetary", "mean"),
        avg_recency_days=("recency_days", "mean"),
    ).round(2)

    print(summary.to_string())

    print(f"\nSaved customer segments to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
