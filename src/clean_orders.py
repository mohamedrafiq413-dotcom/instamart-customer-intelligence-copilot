
from pathlib import Path

import pandas as pd


# Locate project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"

ORDERS_FILE = RAW_DIR / "orders.csv"
PRODUCTS_FILE = RAW_DIR / "products.csv"
OUTPUT_FILE = CLEAN_DIR / "orders.csv"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Load the raw datasets
orders = pd.read_csv(ORDERS_FILE)
products = pd.read_csv(PRODUCTS_FILE)

print("Raw order rows:", len(orders))

# 1. Parse timestamps before removing exact duplicate rows
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"],
    errors="coerce",
)

# 2. Remove exact duplicate rows
orders = orders.drop_duplicates()
print("After removing exact duplicates:", len(orders))

# 3. Sort by timestamp so the latest record comes last
orders = orders.sort_values("order_ts")

# 4. Keep the latest record for each order_id
orders = orders.drop_duplicates(
    subset="order_id",
    keep="last",
)
print("After keeping latest order per ID:", len(orders))

# 5. Standardize order status and remove pending orders
orders["status"] = (
    orders["status"]
    .astype("string")
    .str.strip()
    .str.lower()
)

orders = orders[
    orders["status"].isin(["completed", "cancelled"])
].copy()

print("After removing pending orders:", len(orders))

# 6. Convert order amounts to numeric
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"],
    errors="coerce",
)

# 7. Join product details to calculate missing order amounts
orders = orders.merge(
    products[["product_id", "category", "price_inr"]],
    on="product_id",
    how="left",
    validate="many_to_one",
)

calculated_amount = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

orders["order_amount"] = orders["order_amount"].fillna(
    calculated_amount
)

# 8. Convert delivery times to numeric and invalidate negative values
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"],
    errors="coerce",
)

orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes",
] = pd.NA

# 9. Fill missing delivery times for completed orders using the
# median for the same product category among completed orders
completed_mask = orders["status"].eq("completed")

completed_medians = (
    orders.loc[completed_mask]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

orders.loc[completed_mask, "delivery_minutes"] = (
    orders.loc[completed_mask, "delivery_minutes"]
    .fillna(completed_medians)
)

# Cancelled orders must have missing delivery times
orders.loc[
    orders["status"].eq("cancelled"),
    "delivery_minutes",
] = pd.NA

# 10. Keep only the required order columns
orders = orders[
    [
        "order_id",
        "customer_id",
        "product_id",
        "quantity",
        "order_ts",
        "status",
        "discount_pct",
        "order_amount",
        "delivery_minutes",
    ]
].copy()

# 11. Save the cleaned data
orders.to_csv(
    OUTPUT_FILE,
    index=False,
    date_format="%Y-%m-%d %H:%M:%S",
    float_format="%.2f",
)

# 12. Print validation results
print("\nCleaned order rows:", len(orders))
print("\nOrder counts by status:")
print(orders["status"].value_counts())

missing_amounts = int(orders["order_amount"].isna().sum())

missing_completed_delivery = int(
    orders.loc[
        orders["status"].eq("completed"),
        "delivery_minutes",
    ].isna().sum()
)

missing_cancelled_delivery = int(
    orders.loc[
        orders["status"].eq("cancelled"),
        "delivery_minutes",
    ].isna().sum()
)

completed_revenue = orders.loc[
    orders["status"].eq("completed"),
    "order_amount",
].sum()

print("\nMissing order amounts:", missing_amounts)
print(
    "Missing delivery times for completed orders:",
    missing_completed_delivery,
)
print(
    "Missing delivery times for cancelled orders:",
    missing_cancelled_delivery,
)
print(f"\nCompleted revenue: INR {completed_revenue:,.2f}")
print("\nSaved cleaned orders to:", OUTPUT_FILE)

# 13. Validate expected reference results
completed_count = int(orders["status"].eq("completed").sum())
cancelled_count = int(orders["status"].eq("cancelled").sum())
pending_count = int(orders["status"].eq("pending").sum())

checks = {
    "Cleaned order rows": len(orders) == 12309,
    "Completed orders": completed_count == 11110,
    "Cancelled orders": cancelled_count == 1199,
    "No pending orders": pending_count == 0,
    "No missing order amounts": missing_amounts == 0,
    "No missing completed delivery times": (
        missing_completed_delivery == 0
    ),
    "Cancelled delivery times remain missing": (
        missing_cancelled_delivery == 1199
    ),
    "Completed revenue": abs(
        completed_revenue - 2998668.80
    ) < 0.01,
}

print("\nValidation checks:")
for check_name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}: {check_name}")

if not all(checks.values()):
    raise ValueError(
        "Order-cleaning validation failed. "
        "Review the checks above before continuing."
    )

print("\nALL ORDER-CLEANING CHECKS PASSED")
