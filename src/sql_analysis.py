
"""Run the seven required SQL analyses and pandas summaries."""

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "instamart.db"
SQL_FILE = ROOT / "sql" / "queries.sql"


def load_queries():
    """Load the seven SQL statements, excluding SQL comments."""
    queries = []
    current = []

    for line in SQL_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("--"):
            continue

        current.append(line)

        if ";" in line:
            query = "\n".join(current).strip().rstrip(";").strip()
            if query:
                queries.append(query)
            current = []

    if "\n".join(current).strip():
        queries.append("\n".join(current).strip())

    return queries


def main():
    queries = load_queries()

    if len(queries) != 7:
        raise ValueError(f"Expected 7 SQL queries, found {len(queries)}")

    with sqlite3.connect(DB) as con:
        print("=" * 60)
        print("PART A: SEVEN SQL ANALYSES")
        print("=" * 60)

        results = []

        for i, query in enumerate(queries, 1):
            df = pd.read_sql_query(query, con)
            results.append(df)

            print(f"\nQ{i}: {len(df)} rows")
            print(df.head(5).to_string(index=False))

        # Q2: Revenue by city and product category.
        q2 = results[1]
        print("\nQ2 revenue total:", f"INR {q2['revenue_inr'].sum():,.2f}")

        largest = q2.loc[q2["revenue_inr"].idxmax()]
        print(
            "Largest city/category:",
            largest["city"], "/", largest["category"],
            f"INR {largest['revenue_inr']:,.2f}",
        )

        # Q6: Compare ranking functions.
        q6 = results[5]
        print(
            "\nQ6 RANK vs ROW_NUMBER differences:",
            int((q6["spend_rank"] != q6["row_number_in_city"]).sum()),
        )

        # Q7: Verify the daily and cumulative revenue output.
        q7 = results[6]
        print("Q7 daily rows:", len(q7))

        if not q7.empty:
            print(
                "Q7 final running revenue:",
                f"INR {q7['running_revenue_inr'].iloc[-1]:,.2f}",
            )

        orders = pd.read_sql_query("SELECT * FROM orders", con)
        customers = pd.read_sql_query("SELECT * FROM customers", con)
        products = pd.read_sql_query("SELECT * FROM products", con)

    # Keep completed orders for revenue, statistics, and correlation analysis.
    completed = orders.loc[
        orders["status"].astype(str).str.lower().eq("completed")
    ].copy()

    for column in [
        "order_amount",
        "quantity",
        "discount_pct",
        "delivery_minutes",
    ]:
        completed[column] = pd.to_numeric(
            completed[column], errors="coerce"
        )

    # Join completed orders to customer cities and product categories.
    revenue = (
        completed
        .merge(customers[["customer_id", "city"]], on="customer_id")
        .merge(products[["product_id", "category"]], on="product_id")
    )

    # Pandas pivot: revenue by city and product category.
    pivot = pd.pivot_table(
        revenue,
        values="order_amount",
        index="city",
        columns="category",
        aggfunc="sum",
        fill_value=0,
        margins=True,
        margins_name="Grand Total",
    )

    print("\nPANDAS PIVOT: REVENUE BY CITY AND CATEGORY (INR)")
    print(pivot.round(2).to_string())
    print(
        "Pivot grand total:",
        f"INR {pivot.loc['Grand Total', 'Grand Total']:,.2f}",
    )

    # City-level revenue, average completed order amount, and order count.
    city_summary = revenue.groupby("city", as_index=False).agg(
        total_revenue_inr=("order_amount", "sum"),
        mean_order_amount_inr=("order_amount", "mean"),
        completed_order_count=("order_id", "count"),
    ).sort_values("total_revenue_inr", ascending=False)

    print("\nCITY SUMMARY")
    print(city_summary.round(2).to_string(index=False))

    # IQR outlier analysis for completed order amounts.
    amounts = completed["order_amount"].dropna()

    q1 = amounts.quantile(0.25)
    median = amounts.median()
    q3 = amounts.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    print("\nCOMPLETED ORDER AMOUNT STATISTICS (INR)")
    print("Count:", len(amounts))
    print(f"Mean: {amounts.mean():.2f}")
    print(f"Median: {median:.2f}")
    print(f"Skewness: {amounts.skew():.2f}")
    print(f"Q1: {q1:.2f}")
    print(f"Q3: {q3:.2f}")
    print(f"Lower fence: {lower:.2f}")
    print(f"Upper fence: {upper:.2f}")
    print("Outliers below lower fence:", int((amounts < lower).sum()))
    print("Outliers above upper fence:", int((amounts > upper).sum()))

    # Correlations: use completed orders with valid values for all four
    # numeric variables. Cancelled orders are excluded by design.
    numeric_cols = [
        "order_amount",
        "discount_pct",
        "quantity",
        "delivery_minutes",
    ]

    correlation_data = completed[numeric_cols].dropna()
    correlations = correlation_data.corr()

    print("\nCOMPLETED-ORDER NUMERIC CORRELATIONS")
    print(correlations.round(3).to_string())
    print("Orders used for correlation:", len(correlation_data))
    print("Orders excluded due to missing correlation values:",
          len(completed) - len(correlation_data))

    # Find the strongest absolute off-diagonal correlation.
    pairs = [
        (a, b, correlations.loc[a, b])
        for i, a in enumerate(numeric_cols)
        for b in numeric_cols[i + 1:]
        if pd.notna(correlations.loc[a, b])
    ]

    if pairs:
        a, b, value = max(pairs, key=lambda item: abs(item[2]))
        print(
            f"Largest absolute pairwise correlation: "
            f"{a} vs {b}: {value:.3f}"
        )

    print(
        "\nInterpretation: Correlation describes the strength and direction "
        "of a linear relationship between two numeric variables. "
        "A positive value indicates that the variables tend to increase "
        "together; a negative value indicates that one tends to decrease "
        "as the other increases. Values near zero indicate a weak linear "
        "relationship. Correlation does not establish causation."
    )

    print("\nSQL and pandas analysis complete.")


if __name__ == "__main__":
    main()
