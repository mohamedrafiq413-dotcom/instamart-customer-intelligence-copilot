from pathlib import Path
import sqlite3

import pandas as pd
from pandas.testing import assert_frame_equal


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"

EXPECTED_REVENUE = 2_998_668.80
EXPECTED_CITIES = {
    "Bengaluru": 197,
    "Chennai": 191,
    "Delhi": 210,
    "Hyderabad": 191,
    "Mumbai": 184,
    "Pune": 227,
}
EXPECTED_GENDERS = {
    "female": 610,
    "male": 560,
    "unknown": 30,
}
EXPECTED_CATEGORIES = {
    "Dairy": 8,
    "Electronics": 8,
    "Grocery": 8,
    "Personal Care": 8,
    "Snacks": 8,
}

checks = []


def check(description, condition, detail=""):
    passed = bool(condition)
    checks.append((description, passed))
    label = "PASS" if passed else "FAIL"
    suffix = f" | {detail}" if detail else ""
    print(f"[{label}] {description}{suffix}")


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required CSV file not found: {path}")
    return pd.read_csv(path)


def normalized_frame(df, id_column):
    """Sort rows by ID and normalize column order for CSV/SQLite comparison."""
    result = df.copy()
    result = result.sort_values(id_column, kind="stable").reset_index(drop=True)
    return result.reindex(sorted(result.columns), axis=1)


def frames_match(csv_df, db_df, id_column):
    """Compare data values while allowing tiny floating-point differences."""
    if set(csv_df.columns) != set(db_df.columns):
        return False, (
            f"Column mismatch: CSV={sorted(csv_df.columns)}, "
            f"SQLite={sorted(db_df.columns)}"
        )

    left = normalized_frame(csv_df, id_column)
    right = normalized_frame(db_df, id_column)

    try:
        assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=False,
            rtol=1e-9,
            atol=1e-9,
        )
        return True, "All rows and values match"
    except AssertionError as error:
        return False, str(error).splitlines()[0]


def main():
    print("=" * 72)
    print("INSTAMART DATA VALIDATION")
    print("=" * 72)

    # ---------------------------------------------------------
    # 1. Required clean CSV files
    # ---------------------------------------------------------
    customers = read_csv(CLEAN / "customers.csv")
    products = read_csv(CLEAN / "products.csv")
    orders = read_csv(CLEAN / "orders.csv")

    check("Clean customers contain 1200 rows", len(customers) == 1200)
    check("Clean products contain 40 rows", len(products) == 40)
    check("Clean orders contain 12309 rows", len(orders) == 12309)

    # ---------------------------------------------------------
    # 2. Primary identifiers and duplicate records
    # ---------------------------------------------------------
    for df, id_column, label in [
        (customers, "customer_id", "Customer"),
        (products, "product_id", "Product"),
        (orders, "order_id", "Order"),
    ]:
        check(
            f"{label} IDs are non-missing and unique",
            df[id_column].notna().all() and df[id_column].is_unique,
        )
        check(
            f"No duplicate {label.lower()} rows",
            not df.duplicated().any(),
        )

    # ---------------------------------------------------------
    # 3. Referential integrity
    # ---------------------------------------------------------
    check(
        "Every order customer exists in customers",
        orders["customer_id"].isin(customers["customer_id"]).all(),
    )
    check(
        "Every order product exists in products",
        orders["product_id"].isin(products["product_id"]).all(),
    )

    # ---------------------------------------------------------
    # 4. Clean customer data
    # ---------------------------------------------------------
    city_counts = customers["city"].value_counts().to_dict()
    gender_counts = customers["gender"].value_counts().to_dict()

    check(
        "Clean city names and counts match expected values",
        city_counts == EXPECTED_CITIES,
        f"Found {city_counts}",
    )
    check(
        "Clean gender values and counts match expected values",
        gender_counts == EXPECTED_GENDERS,
        f"Found {gender_counts}",
    )

    age = pd.to_numeric(customers["age"], errors="coerce")
    check("Customer ages are numeric and non-missing", age.notna().all())
    check(
        "Customer ages are within 18-60",
        age.between(18, 60).all(),
        f"Range {age.min()} to {age.max()}",
    )

    signup_dates = pd.to_datetime(customers["signup_date"], errors="coerce")
    check(
        "All customer signup dates are valid",
        signup_dates.notna().all(),
    )

    # ---------------------------------------------------------
    # 5. Clean product data
    # ---------------------------------------------------------
    prices = pd.to_numeric(products["price_inr"], errors="coerce")
    category_counts = products["category"].value_counts().to_dict()

    check("Product prices are numeric and non-missing", prices.notna().all())
    check("All product prices are positive", (prices > 0).all())
    check(
        "Product categories and counts match expected values",
        category_counts == EXPECTED_CATEGORIES,
        f"Found {category_counts}",
    )

    # ---------------------------------------------------------
    # 6. Clean order statuses and amounts
    # ---------------------------------------------------------
    status_counts = orders["status"].value_counts().to_dict()

    check(
        "Completed order count is 11110",
        status_counts.get("completed", 0) == 11110,
    )
    check(
        "Cancelled order count is 1199",
        status_counts.get("cancelled", 0) == 1199,
    )
    check(
        "No pending orders remain",
        status_counts.get("pending", 0) == 0,
    )
    check(
        "Only expected order statuses remain",
        set(status_counts).issubset({"completed", "cancelled"}),
        f"Found {status_counts}",
    )

    amounts = pd.to_numeric(orders["order_amount"], errors="coerce")
    check("Order amounts are numeric and non-missing", amounts.notna().all())
    check("Order amounts are non-negative", (amounts >= 0).all())

    completed = orders["status"].eq("completed")
    cancelled = orders["status"].eq("cancelled")
    delivery = pd.to_numeric(orders["delivery_minutes"], errors="coerce")

    completed_revenue = round(amounts[completed].sum(), 2)
    check(
        "Completed revenue is INR 2,998,668.80",
        abs(completed_revenue - EXPECTED_REVENUE) < 0.005,
        f"Found INR {completed_revenue:,.2f}",
    )

    check(
        "Completed orders have valid delivery times",
        delivery[completed].notna().all(),
    )
    check(
        "Cancelled orders retain missing delivery times",
        delivery[cancelled].isna().all(),
    )
    check(
        "Exactly 1199 delivery times are missing",
        int(delivery.isna().sum()) == 1199,
    )
    check(
        "Delivery times are non-negative when present",
        (delivery.dropna() >= 0).all(),
    )

    order_dates = pd.to_datetime(orders["order_ts"], errors="coerce")
    check("All order timestamps are valid", order_dates.notna().all())

    # Require the documented timestamp text format.
    timestamp_format_ok = (
        order_dates.notna().all()
        and orders["order_ts"].astype(str).equals(
            order_dates.dt.strftime("%Y-%m-%d %H:%M:%S").rename(
                "order_ts"
            )
        )
    )
    check(
        "Order timestamps use YYYY-MM-DD HH:MM:SS format",
        timestamp_format_ok,
    )

    # ---------------------------------------------------------
    # 7. Raw-data reference checks
    # ---------------------------------------------------------
    raw_customers = pd.read_csv(
    RAW / "customers.csv",
    keep_default_na=False,
)
    raw_orders = pd.read_csv(
    RAW / "orders.csv",
    keep_default_na=False,
)

    raw_cities = raw_customers["city"].astype(str)
    nonblank_cities = raw_cities[raw_cities.str.strip().ne("")]

    check(
        "Raw customers contain 1200 rows",
        len(raw_customers) == 1200,
    )
    check(
        "Raw city spellings count is 24 before normalization",
        nonblank_cities.nunique() == 24,
        f"Found {nonblank_cities.nunique()}",
    )

    raw_genders = raw_customers["gender"].astype(str).str.strip()
    check(
        "Raw nonblank gender spellings count is 10",
        raw_genders[raw_genders.ne("")].nunique() == 10,
    )
    check(
        "Raw customer age blanks count is 61",
        raw_customers["age"].astype(str).str.strip().eq("").sum() == 61,
    )
    raw_ages = pd.to_numeric(raw_customers["age"], errors="coerce")
    check(
        "Raw customer ages outside 10-100 count is 6",
        int(((raw_ages < 10) | (raw_ages > 100)).sum()) == 6,
    )
    check(
        "Raw blank gender count is 30",
        raw_genders.eq("").sum() == 30,
    )

    raw_amounts = pd.to_numeric(raw_orders["order_amount"], errors="coerce")
    raw_delivery = pd.to_numeric(
        raw_orders["delivery_minutes"], errors="coerce"
    )
    raw_status = raw_orders["status"].astype(str).str.strip().str.lower()

    check("Raw orders contain 13084 rows", len(raw_orders) == 13084)
    check(
        "Raw exact duplicate order rows count is 119",
        int(raw_orders.duplicated().sum()) == 119,
    )
    check(
        "Raw distinct order IDs count is 12309",
        raw_orders["order_id"].nunique() == 12309,
    )
    check(
        "Raw pending order count is 656",
        int(raw_status.eq("pending").sum()) == 656,
    )
    check(
        "Raw invalid or missing order amounts count is 120",
        int(raw_amounts.isna().sum()) == 120,
    )
    check(
        "Raw negative delivery time count is 106",
        int((raw_delivery < 0).sum()) == 106,
    )
    check(
        "Raw blank delivery time count is 2541",
        int(
            raw_orders["delivery_minutes"]
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        ) == 2541,
    )

    # ---------------------------------------------------------
    # 8. SQLite database and data consistency
    # ---------------------------------------------------------
    if not DB_PATH.exists():
        check("SQLite database exists", False, str(DB_PATH))
    else:
        check("SQLite database exists", True)

        try:
            with sqlite3.connect(DB_PATH) as connection:
                table_names = {
                    row[0]
                    for row in connection.execute(
                        """
                        SELECT name
                        FROM sqlite_master
                        WHERE type = 'table'
                        """
                    ).fetchall()
                }

                required_tables = {"customers", "products", "orders"}
                check(
                    "SQLite contains all three required tables",
                    required_tables.issubset(table_names),
                    f"Found {sorted(table_names)}",
                )

                for table, csv_df, id_column, expected_rows in [
                    ("customers", customers, "customer_id", 1200),
                    ("products", products, "product_id", 40),
                    ("orders", orders, "order_id", 12309),
                ]:
                    if table not in table_names:
                        check(
                            f"SQLite {table} row count is {expected_rows}",
                            False,
                            "Table is missing",
                        )
                        check(
                            f"SQLite {table} values match cleaned CSV",
                            False,
                            "Table is missing",
                        )
                        continue

                    db_df = pd.read_sql_query(
                        f'SELECT * FROM "{table}"',
                        connection,
                    )

                    check(
                        f"SQLite {table} row count is {expected_rows}",
                        len(db_df) == expected_rows,
                        f"Found {len(db_df)}",
                    )

                    matches, detail = frames_match(
                        csv_df, db_df, id_column
                    )
                    check(
                        f"SQLite {table} values match cleaned CSV",
                        matches,
                        detail,
                    )

        except (sqlite3.Error, ValueError, KeyError) as error:
            check(
                "SQLite database can be read and compared",
                False,
                str(error),
            )

    # ---------------------------------------------------------
    # 9. Final summary
    # ---------------------------------------------------------
    print("=" * 72)
    passed = sum(ok for _, ok in checks)
    failed = len(checks) - passed
    print(f"Validation summary: {passed} passed, {failed} failed")

    if failed:
        print("VALIDATION FAILED - investigate every failed check above")
        raise SystemExit(1)

    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()